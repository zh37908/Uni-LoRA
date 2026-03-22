# Copyright 2026-present
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import hashlib
import re
import warnings
from collections import OrderedDict, defaultdict

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

from .config import UniLoRALayerWiseConfig
from .layer import Linear, UniLoRALayerWiseLayer


class UniLoRALayerWiseModel(BaseTuner):
    """
    Creates a UniLoRA local-projection control variant where all target modules
    inside the same transformer block share one local theta_d bank.
    """

    prefix: str = "unilora_layer_wise_"
    tuner_layer_cls = UniLoRALayerWiseLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
    _TRANSFORMER_LAYER_PATTERNS = (
        re.compile(r"^(?P<group>.*?\.encoder\.layer\.\d+)(?:\.|$)"),
        re.compile(r"^(?P<group>.*?\.decoder\.layer\.\d+)(?:\.|$)"),
        re.compile(r"^(?P<group>.*?\.layers\.\d+)(?:\.|$)"),
        re.compile(r"^(?P<group>.*?\.layer\.\d+)(?:\.|$)"),
        re.compile(r"^(?P<group>.*?\.blocks\.\d+)(?:\.|$)"),
        re.compile(r"^(?P<group>.*?\.block\.\d+)(?:\.|$)"),
        re.compile(r"^(?P<group>.*?\.h\.\d+)(?:\.|$)"),
    )

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        if isinstance(config, dict):
            unilora_config = config[adapter_name]
        else:
            unilora_config = config

        if unilora_config.target_modules is None:
            model_config = self.get_model_config(model)
            if model_config and "model_type" in model_config:
                target_modules = self.target_module_mapping.get(model_config["model_type"])
                if target_modules:
                    unilora_config.target_modules = set(target_modules)

        self._warned_fallback_groups: set[str] = set()
        self.group_target_modules = self._collect_target_groups(model, unilora_config)
        self.group_theta_d_sizes = self._build_group_theta_d_sizes(unilora_config.theta_d_length)
        self.group_storage_names = self._build_group_storage_names()

        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

        if self.group_target_modules:
            self._assign_groupwise_indices(adapter_name, unilora_config.proj_seed)

    @staticmethod
    def _stable_int_seed(text: str) -> int:
        digest = hashlib.md5(text.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], byteorder="little", signed=False)

    @classmethod
    def _extract_transformer_layer_group(cls, module_key: str) -> str | None:
        for pattern in cls._TRANSFORMER_LAYER_PATTERNS:
            match = pattern.match(module_key)
            if match is not None:
                return match.group("group")
        return None

    def _resolve_group_key(self, module_key: str) -> str:
        group_key = self._extract_transformer_layer_group(module_key)
        if group_key is not None:
            return group_key

        fallback_group = module_key.rsplit(".", 1)[0]
        if fallback_group not in self._warned_fallback_groups:
            warnings.warn(
                f"Could not infer a transformer-layer group from `{module_key}`. "
                f"Falling back to parent module `{fallback_group}`."
            )
            self._warned_fallback_groups.add(fallback_group)
        return fallback_group

    def _collect_target_groups(self, model, unilora_config) -> OrderedDict[str, list[str]]:
        groups: OrderedDict[str, list[str]] = OrderedDict()
        is_all_linear = getattr(unilora_config, "target_modules", None) == "all-linear"

        for key, module in model.named_modules():
            is_valid = False
            if is_all_linear:
                is_valid = isinstance(module, (nn.Linear, Conv1D))
            elif check_target_module_exists(unilora_config, key):
                is_valid = isinstance(module, (nn.Linear, Conv1D))

            if not is_valid:
                continue

            group_key = self._resolve_group_key(key)
            groups.setdefault(group_key, []).append(key)

        if not groups:
            warnings.warn("No target modules found for UniLoRA-Layer-Wise.")
        return groups

    def _build_group_theta_d_sizes(self, total_d: int) -> dict[str, int]:
        if not self.group_target_modules:
            return {}

        num_groups = len(self.group_target_modules)
        if total_d < num_groups:
            raise ValueError(
                f"`theta_d_length` ({total_d}) must be >= number of transformer groups ({num_groups}) "
                "for per-transformer-layer local projection."
            )

        base = total_d // num_groups
        remainder = total_d % num_groups
        sizes: dict[str, int] = {}
        for idx, group_key in enumerate(self.group_target_modules.keys()):
            sizes[group_key] = base + (1 if idx < remainder else 0)
        return sizes

    def _build_group_storage_names(self) -> dict[str, str]:
        storage_names: dict[str, str] = {}
        used_names: set[str] = set()

        for group_key in self.group_target_modules.keys():
            candidate = re.sub(r"[^0-9a-zA-Z_]+", "_", group_key).strip("_")
            if not candidate:
                candidate = "group"
            if candidate[0].isdigit():
                candidate = f"group_{candidate}"

            unique_candidate = candidate
            suffix = 1
            while unique_candidate in used_names:
                unique_candidate = f"{candidate}_{suffix}"
                suffix += 1

            used_names.add(unique_candidate)
            storage_names[group_key] = unique_candidate

        return storage_names

    @staticmethod
    def _generate_balanced_index(total_length: int, theta_d_length: int, seed: int) -> torch.Tensor:
        if total_length <= 0:
            return torch.empty(0, dtype=torch.long)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        base_count = total_length // theta_d_length
        remainder = total_length % theta_d_length

        chunks = []
        if base_count > 0:
            base = torch.arange(theta_d_length, dtype=torch.long).repeat_interleave(base_count)
            chunks.append(base)
        if remainder > 0:
            extra = torch.randperm(theta_d_length, generator=generator, dtype=torch.long)[:remainder]
            chunks.append(extra)

        index = torch.cat(chunks, dim=0)
        perm = torch.randperm(index.numel(), generator=generator, dtype=torch.long)
        return index[perm]

    def _init_group_theta_d(
        self,
        group_storage_name: str,
        adapter_name: str,
        theta_d_length_local: int,
        init_theta_d_bound: float,
    ) -> None:
        if group_storage_name not in self.unilora_theta_d:
            self.unilora_theta_d[group_storage_name] = nn.ParameterDict({})

        group_bank = self.unilora_theta_d[group_storage_name]
        if adapter_name in group_bank:
            return

        theta = torch.zeros(theta_d_length_local)
        torch.nn.init.uniform_(theta, -init_theta_d_bound, init_theta_d_bound)
        group_bank[adapter_name] = nn.Parameter(theta)

    def _assign_groupwise_indices(self, adapter_name: str, proj_seed: int) -> None:
        modules_by_group: dict[str, list[UniLoRALayerWiseLayer]] = defaultdict(list)
        for module in self.model.modules():
            if isinstance(module, UniLoRALayerWiseLayer) and adapter_name in module.unilora_indices_A:
                group_name = getattr(module, "projection_group_name", None)
                if group_name is not None:
                    modules_by_group[group_name].append(module)

        for group_name, modules in modules_by_group.items():
            modules.sort(key=lambda module: getattr(module, "projection_module_key", ""))
            local_d = self.unilora_theta_d[group_name][adapter_name].numel()
            total_params = sum(
                module.unilora_indices_A[adapter_name].numel() + module.unilora_indices_B[adapter_name].numel()
                for module in modules
            )
            all_indices = self._generate_balanced_index(
                total_length=total_params,
                theta_d_length=local_d,
                seed=self._stable_int_seed(f"{proj_seed}:{group_name}:{adapter_name}"),
            )

            pointer = 0
            for module in modules:
                indices_a = module.unilora_indices_A[adapter_name]
                numel_a = indices_a.numel()
                module.unilora_indices_A[adapter_name] = all_indices[pointer : pointer + numel_a].view_as(indices_a).clone()
                pointer += numel_a

                indices_b = module.unilora_indices_B[adapter_name]
                numel_b = indices_b.numel()
                module.unilora_indices_B[adapter_name] = all_indices[pointer : pointer + numel_b].view_as(indices_b).clone()
                pointer += numel_b

            if pointer != total_params:
                raise RuntimeError("Failed to assign all local projection indices in UniLoRA-Layer-Wise.")

            counts = torch.bincount(all_indices, minlength=local_d)
            inv_sqrt_counts = torch.zeros(local_d, dtype=torch.float32)
            non_zero = counts > 0
            inv_sqrt_counts[non_zero] = 1.0 / torch.sqrt(counts[non_zero].float())

            for module in modules:
                module.update_norm(
                    adapter_name=adapter_name,
                    unilora_scales_A=inv_sqrt_counts[module.unilora_indices_A[adapter_name].long()],
                    unilora_scales_B=inv_sqrt_counts[module.unilora_indices_B[adapter_name].long()],
                )

    def _create_and_replace(
        self,
        unilora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {"fan_in_fan_out": unilora_config.fan_in_fan_out, "bias": bias}

        group_key = self._resolve_group_key(current_key)
        group_storage_name = self.group_storage_names[group_key]
        local_d = self.group_theta_d_sizes[group_key]
        self._init_group_theta_d(
            group_storage_name=group_storage_name,
            adapter_name=adapter_name,
            theta_d_length_local=local_d,
            init_theta_d_bound=unilora_config.init_theta_d_bound,
        )
        group_bank = self.unilora_theta_d[group_storage_name]

        if isinstance(target, Linear):
            target.update_layer(
                adapter_name=adapter_name,
                unilora_theta_d=group_bank,
                r=unilora_config.r,
                theta_d_length_local=local_d,
                unilora_dropout=unilora_config.unilora_dropout,
            )
            target.projection_group_name = group_storage_name
            target.projection_module_key = current_key
        else:
            new_module = self._create_new_module(
                unilora_config=unilora_config,
                unilora_theta_d=group_bank,
                adapter_name=adapter_name,
                target=target,
                local_d=local_d,
                **kwargs,
            )
            new_module.projection_group_name = group_storage_name
            new_module.projection_module_key = current_key
            if adapter_name not in self.active_adapter:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(unilora_config, unilora_theta_d, adapter_name, target, local_d, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = unilora_config.fan_in_fan_out = False
        elif isinstance(target_base_layer, Conv1D):
            kwargs["is_target_conv_1d_layer"] = True
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                    "Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = unilora_config.fan_in_fan_out = True
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `transformers.pytorch_utils.Conv1D`."
            )

        return Linear(
            base_layer=target,
            unilora_theta_d=unilora_theta_d,
            adapter_name=adapter_name,
            r=unilora_config.r,
            theta_d_length_local=local_d,
            unilora_dropout=unilora_config.unilora_dropout,
            **kwargs,
        )

    def _pre_injection_hook(self, model: nn.Module, config, adapter_name: str) -> None:
        self.unilora_theta_d = nn.ModuleDict({})

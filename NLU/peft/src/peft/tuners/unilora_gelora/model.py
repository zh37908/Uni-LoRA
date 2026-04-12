from __future__ import annotations

import warnings
import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from .config import UniLoRAGeLoRAConfig
from .layer import Linear, UniLoRAGeLoRALayer


class UniLoRAGeLoRAModel(BaseTuner):
    """
    UniLoRA-GeLoRA model: UniLoRA with per-module rank allocation.
    """

    prefix: str = "unilora_gelora_"
    tuner_layer_cls = UniLoRAGeLoRALayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

        lora_param_count = 0
        for module in model.named_modules():
            _, layer = module
            if isinstance(layer, UniLoRAGeLoRALayer):
                lora_param_count += layer.unilora_indices_A[adapter_name].numel()
                lora_param_count += layer.unilora_indices_B[adapter_name].numel()

        theta_d_length = config[adapter_name].theta_d_length
        proj_seed = config[adapter_name].proj_seed
        all_elements = self.generate_index(lora_param_count, theta_d_length, proj_seed)
        pointer = 0

        for module in model.modules():
            if isinstance(module, UniLoRAGeLoRALayer):
                num_a = module.unilora_indices_A[adapter_name].numel()
                chunk_a = all_elements[pointer : pointer + num_a]
                module.unilora_indices_A[adapter_name] = chunk_a.view_as(module.unilora_indices_A[adapter_name]).clone()
                pointer += num_a

                num_b = module.unilora_indices_B[adapter_name].numel()
                chunk_b = all_elements[pointer : pointer + num_b]
                module.unilora_indices_B[adapter_name] = chunk_b.view_as(module.unilora_indices_B[adapter_name]).clone()
                pointer += num_b

        if pointer != len(all_elements):
            raise RuntimeError("UniLoRA-GeLoRA index assignment is inconsistent.")

        counts = torch.bincount(all_elements, minlength=theta_d_length)
        inv_sqrt_counts = torch.zeros(theta_d_length, dtype=torch.float32)
        non_zero = counts > 0
        inv_sqrt_counts[non_zero] = 1.0 / torch.sqrt(counts[non_zero].float())

        unilora_modules = [m for m in self.modules() if isinstance(m, UniLoRAGeLoRALayer)]
        for module in unilora_modules:
            scale_a = inv_sqrt_counts[module.unilora_indices_A[adapter_name].long()]
            scale_b = inv_sqrt_counts[module.unilora_indices_B[adapter_name].long()]
            module.update_norm(adapter_name, scale_a, scale_b)

    def generate_index(self, total_length: int, theta_d_length: int, proj_seed: int) -> torch.Tensor:
        if total_length <= 0:
            return torch.empty(0, dtype=torch.long)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(proj_seed)

        base_count = total_length // theta_d_length
        remaining = total_length % theta_d_length
        data = torch.arange(theta_d_length, dtype=torch.long).repeat_interleave(base_count)
        if remaining > 0:
            extras = torch.randperm(theta_d_length, generator=generator)[:remaining]
            data = torch.cat([data, extras], dim=0)
        shuffle = torch.randperm(data.numel(), generator=generator)
        return data[shuffle]

    def _init_unilora_theta_d(self, config: UniLoRAGeLoRAConfig, adapter_name: str) -> None:
        if adapter_name in self.unilora_gelora_theta_d:
            return
        unilora_theta_d = torch.zeros(config.theta_d_length)
        torch.nn.init.uniform_(unilora_theta_d, -config.init_theta_d_bound, config.init_theta_d_bound)
        self.unilora_gelora_theta_d[adapter_name] = unilora_theta_d

    def _pre_injection_hook(self, model: nn.Module, config: UniLoRAGeLoRAConfig, adapter_name: str) -> None:
        self.unilora_gelora_theta_d = nn.ParameterDict({})

    @staticmethod
    def _resolve_rank_from_map(rank_map: dict[str, int] | None, module_key: str, default_rank: int) -> int:
        if not rank_map:
            return default_rank
        if module_key in rank_map:
            return int(rank_map[module_key])
        for key, value in rank_map.items():
            if module_key.endswith(key):
                return int(value)
        return default_rank

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
        kwargs = {
            "fan_in_fan_out": unilora_config.fan_in_fan_out,
            "bias": bias,
        }
        self._init_unilora_theta_d(unilora_config, adapter_name)

        rank_map = getattr(unilora_config, "gelora_rank_map", None)
        resolved_rank = self._resolve_rank_from_map(rank_map, current_key, unilora_config.r)

        if isinstance(target, Linear):
            target.update_layer(
                adapter_name=adapter_name,
                unilora_theta_d=self.unilora_gelora_theta_d,
                r=resolved_rank,
                theta_d_length=unilora_config.theta_d_length,
                unilora_dropout=unilora_config.unilora_dropout,
            )
        else:
            new_module = self._create_new_module(
                unilora_config=unilora_config,
                unilora_theta_d=self.unilora_gelora_theta_d,
                adapter_name=adapter_name,
                target=target,
                resolved_rank=resolved_rank,
                **kwargs,
            )
            if adapter_name not in self.active_adapter:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(unilora_config, unilora_theta_d, adapter_name, target, resolved_rank: int, **kwargs):
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
        new_module = Linear(
            base_layer=target,
            unilora_theta_d=unilora_theta_d,
            adapter_name=adapter_name,
            r=resolved_rank,
            theta_d_length=unilora_config.theta_d_length,
            unilora_dropout=unilora_config.unilora_dropout,
            **kwargs,
        )
        return new_module

    def get_nb_savable_parameters(self, adapter="default") -> tuple[int, int]:
        theta_d_params = 0
        other_params = 0
        for name, param in self.named_parameters():
            if "unilora_gelora_theta_d" in name:
                theta_d_params += param.numel()
            elif "unilora_indices" in name:
                other_params += param.numel()
            elif "unilora_scales" in name:
                other_params += param.numel()

        return theta_d_params, other_params

    def print_savable_parameters(self) -> None:
        theta_d_params, other_params = self.get_nb_savable_parameters()
        print(
            f"UniLoRA-GeLoRA params to-be-saved (float32-equivalent): {theta_d_params:,d} "
            f"|| total params to-be-saved: {(theta_d_params + other_params):,d}"
        )

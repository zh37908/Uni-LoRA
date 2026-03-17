from __future__ import annotations

import re
import warnings

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

from .config import UniLoRAStageRatioConfig
from .layer import Linear, UniLoRALayer


class UniLoRAStageRatioModel(BaseTuner):
    """
    UniLoRA stage-ratio model.
    """

    prefix: str = "unilora_stage_ratio_"
    tuner_layer_cls = UniLoRALayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

        stage_entries = self._collect_stage_entries(model, config[adapter_name])
        if not stage_entries:
            return

        theta_d_length = config[adapter_name].theta_d_length
        proj_seed = config[adapter_name].proj_seed
        stage_ratios = config[adapter_name].stage_theta_d_ratios

        stage_param_counts = [0, 0, 0]
        for entry in stage_entries:
            module = entry["module"]
            stage = entry["stage"]
            stage_param_counts[stage] += module.unilora_indices_A[adapter_name].numel()
            stage_param_counts[stage] += module.unilora_indices_B[adapter_name].numel()

        active_stages = [i for i, cnt in enumerate(stage_param_counts) if cnt > 0]
        stage_lengths = self._allocate_stage_lengths(theta_d_length, stage_ratios, active_stages)
        stage_offsets = [0, stage_lengths[0], stage_lengths[0] + stage_lengths[1]]

        stage_streams = {}
        for stage in active_stages:
            local_cnt = stage_param_counts[stage]
            local_d = stage_lengths[stage]
            if local_d <= 0:
                raise ValueError(
                    f"Stage {stage} got theta_d_length=0 but has {local_cnt} parameters to index. "
                    f"Increase `theta_d_length` or adjust `stage_theta_d_ratios`."
                )
            local_elements = self.generate_index(local_cnt, local_d, proj_seed + 10007 * stage)
            stage_streams[stage] = local_elements + stage_offsets[stage]

        stage_pointers = {stage: 0 for stage in active_stages}
        for entry in stage_entries:
            module = entry["module"]
            stage = entry["stage"]
            stream = stage_streams[stage]
            pointer = stage_pointers[stage]

            num_a = module.unilora_indices_A[adapter_name].numel()
            chunk_a = stream[pointer : pointer + num_a]
            module.unilora_indices_A[adapter_name] = chunk_a.view_as(module.unilora_indices_A[adapter_name]).clone()
            pointer += num_a

            num_b = module.unilora_indices_B[adapter_name].numel()
            chunk_b = stream[pointer : pointer + num_b]
            module.unilora_indices_B[adapter_name] = chunk_b.view_as(module.unilora_indices_B[adapter_name]).clone()
            pointer += num_b

            stage_pointers[stage] = pointer

        for stage in active_stages:
            if stage_pointers[stage] != stage_streams[stage].numel():
                raise RuntimeError(f"Stage {stage} index assignment is inconsistent.")

        all_indices = []
        for entry in stage_entries:
            module = entry["module"]
            all_indices.append(module.unilora_indices_A[adapter_name].reshape(-1).long())
            all_indices.append(module.unilora_indices_B[adapter_name].reshape(-1).long())
        all_indices = torch.cat(all_indices, dim=0)

        counts = torch.bincount(all_indices, minlength=theta_d_length)
        sqrt_counts = torch.zeros_like(counts, dtype=torch.float32)
        non_zero = counts > 0
        sqrt_counts[non_zero] = 1.0 / torch.sqrt(counts[non_zero].float())

        for entry in stage_entries:
            module = entry["module"]
            scale_a = sqrt_counts[module.unilora_indices_A[adapter_name].long()]
            scale_b = sqrt_counts[module.unilora_indices_B[adapter_name].long()]
            module.update_norm(adapter_name, scale_a, scale_b)

    @staticmethod
    def _allocate_stage_lengths(total_d: int, ratios: list[float], active_stages: list[int]) -> list[int]:
        lengths = [0, 0, 0]
        if total_d <= 0:
            raise ValueError("`theta_d_length` must be a positive integer.")
        if not active_stages:
            return lengths

        ratio_sum = sum(ratios[s] for s in active_stages)
        raw = [total_d * ratios[s] / ratio_sum for s in active_stages]
        floors = [int(v) for v in raw]
        remain = total_d - sum(floors)

        frac_order = sorted(range(len(active_stages)), key=lambda i: raw[i] - floors[i], reverse=True)
        for i in frac_order[:remain]:
            floors[i] += 1

        if total_d >= len(active_stages):
            for i in range(len(floors)):
                if floors[i] == 0:
                    donor = max(range(len(floors)), key=lambda j: floors[j])
                    if floors[donor] > 1:
                        floors[donor] -= 1
                        floors[i] = 1

        for idx, stage in enumerate(active_stages):
            lengths[stage] = floors[idx]
        return lengths

    @staticmethod
    def _extract_layer_index_from_key(module_key: str, unilora_config: UniLoRAStageRatioConfig) -> int | None:
        layers_pattern = getattr(unilora_config, "layers_pattern", None)
        if layers_pattern is None or (isinstance(layers_pattern, list) and len(layers_pattern) == 0):
            match = re.match(r".*\.[^.]*\.(\d+)\.", module_key)
            if match is not None:
                return int(match.group(1))
        else:
            patterns = [layers_pattern] if isinstance(layers_pattern, str) else layers_pattern
            for pattern in patterns:
                match = re.match(rf".*\.{pattern}\.(\d+)\.", module_key)
                if match is not None:
                    return int(match.group(1))

        fallback = re.search(r"\.(\d+)\.", module_key)
        if fallback is None:
            return None
        return int(fallback.group(1))

    def _collect_stage_entries(self, model, unilora_config: UniLoRAStageRatioConfig) -> list[dict]:
        entries = []
        for name, module in model.named_modules():
            if not isinstance(module, UniLoRALayer):
                continue
            entries.append(
                {
                    "name": name,
                    "module": module,
                    "layer_index": self._extract_layer_index_from_key(name, unilora_config),
                    "stage": 0,
                }
            )

        if not entries:
            return entries

        has_missing = any(e["layer_index"] is None for e in entries)
        if has_missing:
            warnings.warn(
                "Some target modules have no detectable layer index; using module order to split front/middle/back."
            )
            for i, entry in enumerate(entries):
                entry["stage"] = min(2, (3 * i) // max(1, len(entries)))
            return entries

        unique_layers = sorted({int(e["layer_index"]) for e in entries})
        layer_to_rank = {layer: i for i, layer in enumerate(unique_layers)}
        n_layers = max(1, len(unique_layers))
        for entry in entries:
            rank = layer_to_rank[int(entry["layer_index"])]
            entry["stage"] = min(2, (3 * rank) // n_layers)
        return entries

    @staticmethod
    def generate_index(LoRA_para_cnt, theta_d_length, proj_seed):
        import numpy as np

        total_length = LoRA_para_cnt
        num_unique = theta_d_length
        base_count = total_length // num_unique
        remaining = total_length % num_unique
        rng = np.random.default_rng(proj_seed)
        data = np.repeat(np.arange(num_unique), base_count)
        if remaining > 0:
            extras = rng.choice(num_unique, size=remaining, replace=False)
            data = np.concatenate([data, extras])
        rng.shuffle(data)
        return torch.tensor(data)

    def _init_unilora_theta_d(self, config: UniLoRAStageRatioConfig, adapter_name: str) -> None:
        unilora_theta_d = torch.zeros(config.theta_d_length)
        torch.nn.init.uniform_(unilora_theta_d, -config.init_theta_d_bound, config.init_theta_d_bound)
        self.unilora_stage_ratio_theta_d[adapter_name] = unilora_theta_d

    def _pre_injection_hook(self, model: nn.Module, config: UniLoRAStageRatioConfig, adapter_name: str) -> None:
        self.unilora_stage_ratio_theta_d = nn.ParameterDict({})

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

        if isinstance(target, Linear):
            target.update_layer(
                adapter_name=adapter_name,
                unilora_theta_d=self.unilora_stage_ratio_theta_d,
                r=unilora_config.r,
                theta_d_length=unilora_config.theta_d_length,
                unilora_dropout=unilora_config.unilora_dropout,
            )
        else:
            new_module = self._create_new_module(
                unilora_config=unilora_config,
                unilora_theta_d=self.unilora_stage_ratio_theta_d,
                adapter_name=adapter_name,
                target=target,
                **kwargs,
            )
            if adapter_name not in self.active_adapter:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(unilora_config, unilora_theta_d, adapter_name, target, **kwargs):
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
            r=unilora_config.r,
            theta_d_length=unilora_config.theta_d_length,
            unilora_dropout=unilora_config.unilora_dropout,
            **kwargs,
        )
        return new_module

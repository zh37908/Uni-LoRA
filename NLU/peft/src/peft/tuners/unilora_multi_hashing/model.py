from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

from .config import UniLoRAMultiHashingConfig
from .layer import Linear, UniLoRAMultiHashingLayer


class UniLoRAMultiHashingModel(BaseTuner):
    """
    Creates UniLoRA multi-hashing model from a pretrained transformers model.
    """

    prefix: str = "unilora_multi_hashing_"
    tuner_layer_cls = UniLoRAMultiHashingLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

        lora_param_count = 0
        for module in model.modules():
            if isinstance(module, UniLoRAMultiHashingLayer):
                lora_param_count += module.unilora_indices_A[adapter_name].numel()
                lora_param_count += module.unilora_indices_B[adapter_name].numel()

        theta_d_length = config[adapter_name].theta_d_length
        proj_seed = config[adapter_name].proj_seed
        all_elements = self.generate_index(lora_param_count, theta_d_length, proj_seed)
        pointer = 0

        for module in model.modules():
            if not isinstance(module, UniLoRAMultiHashingLayer):
                continue

            param_numel = module.unilora_indices_A[adapter_name].numel()
            chunk = all_elements[pointer : pointer + param_numel]
            module.unilora_indices_A[adapter_name] = chunk.view_as(module.unilora_indices_A[adapter_name]).clone()
            pointer += param_numel

            param_numel = module.unilora_indices_B[adapter_name].numel()
            chunk = all_elements[pointer : pointer + param_numel]
            module.unilora_indices_B[adapter_name] = chunk.view_as(module.unilora_indices_B[adapter_name]).clone()
            pointer += param_numel

        if pointer != len(all_elements):
            raise RuntimeError("Global UniLoRA multi-hashing index assignment did not consume all generated indices.")

        counts = torch.bincount(all_elements, minlength=theta_d_length)
        sqrt_counts = 1 / torch.sqrt(counts.float())

        index_ls = []
        for module in model.modules():
            if isinstance(module, UniLoRAMultiHashingLayer):
                index_ls.append(module.unilora_indices_A[adapter_name].long())
                index_ls.append(module.unilora_indices_B[adapter_name].long())

        norm_factors = [sqrt_counts[t] for t in index_ls]
        uni_modules = [m for m in self.modules() if isinstance(m, UniLoRAMultiHashingLayer)]
        for module, (scale_a, scale_b) in zip(uni_modules, zip(*[iter(norm_factors)] * 2)):
            module.update_norm(adapter_name, scale_a, scale_b)

    def generate_index(self, lora_param_count: int, theta_d_length: int, proj_seed: int) -> torch.Tensor:
        base_count = lora_param_count // theta_d_length
        remaining = lora_param_count % theta_d_length
        rng = np.random.default_rng(proj_seed)
        data = np.repeat(np.arange(theta_d_length), base_count)
        if remaining > 0:
            extras = rng.choice(theta_d_length, size=remaining, replace=False)
            data = np.concatenate([data, extras])
        rng.shuffle(data)
        return torch.tensor(data)

    def _init_unilora_multi_hashing_parameters(
        self, config: UniLoRAMultiHashingConfig, adapter_name: str
    ) -> None:
        if adapter_name in self.unilora_multi_hashing_theta_d:
            return

        theta_d = torch.empty(config.num_hash_components, config.theta_d_length)
        projection = torch.empty(config.num_hash_components, config.theta_d_length)
        projection_center = 1.0 / float(config.num_hash_components)

        for idx in range(config.num_hash_components):
            torch.nn.init.uniform_(theta_d[idx], -config.init_theta_d_bound, config.init_theta_d_bound)
            torch.nn.init.uniform_(
                projection[idx],
                projection_center - config.init_p_bound,
                projection_center + config.init_p_bound,
            )

        self.unilora_multi_hashing_theta_d[adapter_name] = theta_d
        self.unilora_multi_hashing_P[adapter_name] = projection

    def _pre_injection_hook(self, model: nn.Module, config: UniLoRAMultiHashingConfig, adapter_name: str) -> None:
        self.unilora_multi_hashing_theta_d = nn.ParameterDict({})
        self.unilora_multi_hashing_P = nn.ParameterDict({})

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
        self._init_unilora_multi_hashing_parameters(unilora_config, adapter_name)

        if isinstance(target, Linear):
            target.update_layer(
                adapter_name=adapter_name,
                unilora_multi_hashing_theta_d=self.unilora_multi_hashing_theta_d,
                unilora_multi_hashing_P=self.unilora_multi_hashing_P,
                r=unilora_config.r,
                theta_d_length=unilora_config.theta_d_length,
                unilora_dropout=unilora_config.unilora_dropout,
            )
        else:
            new_module = self._create_new_module(
                unilora_config=unilora_config,
                unilora_multi_hashing_theta_d=self.unilora_multi_hashing_theta_d,
                unilora_multi_hashing_P=self.unilora_multi_hashing_P,
                adapter_name=adapter_name,
                target=target,
                **kwargs,
            )
            if adapter_name not in self.active_adapter:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(
        unilora_config,
        unilora_multi_hashing_theta_d,
        unilora_multi_hashing_P,
        adapter_name,
        target,
        **kwargs,
    ):
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
            unilora_multi_hashing_theta_d=unilora_multi_hashing_theta_d,
            unilora_multi_hashing_P=unilora_multi_hashing_P,
            adapter_name=adapter_name,
            r=unilora_config.r,
            theta_d_length=unilora_config.theta_d_length,
            unilora_dropout=unilora_config.unilora_dropout,
            **kwargs,
        )

    def get_nb_savable_parameters(self, adapter="default") -> tuple[int, int]:
        multi_hashing_params = 0
        other_params = 0
        for name, param in self.named_parameters():
            if "unilora_multi_hashing_" in name:
                multi_hashing_params += param.numel()

        for name, buffer in self.named_buffers():
            if "unilora_indices" in name or "unilora_scales" in name:
                other_params += buffer.numel()

        return multi_hashing_params, other_params

    def print_savable_parameters(self) -> None:
        multi_hashing_params, other_params = self.get_nb_savable_parameters()
        print(
            f"UniLoRA-multi-hashing params to-be-saved (float32-equivalent): {multi_hashing_params:,d} "
            f"|| total params to-be-saved: {(multi_hashing_params + other_params):,d}"
        )

# Copyright 2024-present the HuggingFace Inc. team.
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

import warnings
import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from .config import UniLoRALearnableColumnConfig
from .layer import Linear, UniLoRALayer


class UniLoRALearnableColumnModel(BaseTuner):
    """
    UniLoRA with learnable column-wise projection scales.

    Each column shares a single learnable scalar, and the total number of scale
    parameters equals theta_d_length (shared bank).
    """

    prefix: str = "unilora_learnable_column_"
    tuner_layer_cls = UniLoRALayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

        # --- Global hash index allocation for theta_d (element-level) ---
        lora_param_count = 0
        for _, module in model.named_modules():
            if isinstance(module, UniLoRALayer):
                lora_param_count += module.unilora_indices_A[adapter_name].numel()
                lora_param_count += module.unilora_indices_B[adapter_name].numel()

        theta_d_length = config[adapter_name].theta_d_length
        proj_seed = config[adapter_name].proj_seed
        all_elements = self.generate_index(lora_param_count, theta_d_length, proj_seed)
        pointer = 0

        for _, module in model.named_modules():
            if isinstance(module, UniLoRALayer):
                param_numel = module.unilora_indices_A[adapter_name].numel()
                chunk = all_elements[pointer : pointer + param_numel]
                module.unilora_indices_A[adapter_name] = chunk.view_as(module.unilora_indices_A[adapter_name]).clone()
                pointer += param_numel

                param_numel = module.unilora_indices_B[adapter_name].numel()
                chunk = all_elements[pointer : pointer + param_numel]
                module.unilora_indices_B[adapter_name] = chunk.view_as(module.unilora_indices_B[adapter_name]).clone()
                pointer += param_numel

        assert pointer == len(all_elements)

        # --- Global hash index allocation for column scales ---
        col_param_count = 0
        for _, module in model.named_modules():
            if isinstance(module, UniLoRALayer):
                col_param_count += module.unilora_col_indices_A[adapter_name].numel()
                col_param_count += module.unilora_col_indices_B[adapter_name].numel()

        all_col_elements = self.generate_index(col_param_count, theta_d_length, proj_seed)
        pointer = 0

        for _, module in model.named_modules():
            if isinstance(module, UniLoRALayer):
                param_numel = module.unilora_col_indices_A[adapter_name].numel()
                chunk = all_col_elements[pointer : pointer + param_numel]
                pointer += param_numel

                param_numel_b = module.unilora_col_indices_B[adapter_name].numel()
                chunk_b = all_col_elements[pointer : pointer + param_numel_b]
                pointer += param_numel_b

                module.update_norm(adapter_name, chunk.view_as(module.unilora_col_indices_A[adapter_name]), chunk_b)

        assert pointer == len(all_col_elements)

        # Initialize learnable column scales based on index frequency.
        counts = torch.bincount(all_col_elements, minlength=theta_d_length)
        counts = torch.clamp(counts, min=1)
        sqrt_counts = 1 / torch.sqrt(counts.float())
        self.unilora_scales[adapter_name] = nn.Parameter(sqrt_counts)

    def generate_index(self, total_count, theta_d_length, proj_seed):
        import numpy as np

        base_count = total_count // theta_d_length
        remaining = total_count % theta_d_length
        rng = np.random.default_rng(proj_seed)
        data = np.repeat(np.arange(theta_d_length), base_count)
        if remaining > 0:
            extras = rng.choice(theta_d_length, size=remaining, replace=False)
            data = np.concatenate([data, extras])
        rng.shuffle(data)
        return torch.tensor(data)

    def _init_unilora_theta_d(self, config: UniLoRALearnableColumnConfig, adapter_name: str) -> None:
        unilora_theta_d = torch.zeros(config.theta_d_length)
        torch.nn.init.uniform_(unilora_theta_d, -config.init_theta_d_bound, config.init_theta_d_bound)
        self.unilora_learnable_column_theta_d[adapter_name] = unilora_theta_d

    def _pre_injection_hook(self, model: nn.Module, config: UniLoRALearnableColumnConfig, adapter_name: str) -> None:
        self.unilora_learnable_column_theta_d = nn.ParameterDict({})
        self.unilora_scales = nn.ParameterDict({})

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
                unilora_theta_d=self.unilora_learnable_column_theta_d,
                unilora_scales=self.unilora_scales,
                r=unilora_config.r,
                theta_d_length=unilora_config.theta_d_length,
                unilora_dropout=unilora_config.unilora_dropout,
            )
        else:
            new_module = self._create_new_module(
                unilora_config=unilora_config,
                unilora_theta_d=self.unilora_learnable_column_theta_d,
                unilora_scales=self.unilora_scales,
                adapter_name=adapter_name,
                target=target,
                **kwargs,
            )
            if adapter_name not in self.active_adapter:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(unilora_config, unilora_theta_d, unilora_scales, adapter_name, target, **kwargs):
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
            unilora_scales=unilora_scales,
            adapter_name=adapter_name,
            r=unilora_config.r,
            theta_d_length=unilora_config.theta_d_length,
            unilora_dropout=unilora_config.unilora_dropout,
            **kwargs,
        )
        return new_module

    def get_nb_savable_parameters(self, adapter="default") -> tuple[int, int]:
        """
        Returns the number of savable Uni-LoRA parameters and other savable parameters.
        """
        unilora_params = 0
        other_params = 0
        for name, param in self.named_parameters():
            if "unilora_learnable_column_theta_d" in name:
                unilora_params += param.numel()
            elif "unilora_scales" in name:
                unilora_params += param.numel()

        for name, buffer in self.named_buffers():
            if "unilora_indices" in name or "unilora_col_indices" in name:
                other_params += buffer.numel()

        return unilora_params, other_params

    def print_savable_parameters(self) -> None:
        """
        Prints the number of savable Uni-LoRA parameters and total savable parameters.
        """
        unilora_params, other_params = self.get_nb_savable_parameters()
        print(
            f"Uni-LoRA Learnable-Column params to-be-saved (float32-equivalent): {unilora_params:,d} "
            f"|| total params to-be-saved: {(unilora_params + other_params):,d}"
        )

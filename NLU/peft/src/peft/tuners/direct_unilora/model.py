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
from .config import DirectUniLoRAConfig
from .layer import Linear, DirectUniLoRALayer


class DirectUniLoRAModel(BaseTuner):
    """
    Direct-UniLoRA: project theta_d directly to full delta weight (no low-rank AB).
    """

    prefix: str = "direct_unilora_"
    tuner_layer_cls = DirectUniLoRALayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

        # 1) count required indices for full delta weights
        total_params = 0
        for _, module in model.named_modules():
            if isinstance(module, DirectUniLoRALayer):
                total_params += module.direct_unilora_indices_W[adapter_name].numel()

        # 2) generate global indices
        theta_d_length = config[adapter_name].theta_d_length
        proj_seed = config[adapter_name].proj_seed
        all_elements = self.generate_index(total_params, theta_d_length, proj_seed)
        pointer = 0

        # 3) assign indices back to each layer
        for _, module in model.named_modules():
            if isinstance(module, DirectUniLoRALayer):
                param_numel = module.direct_unilora_indices_W[adapter_name].numel()
                chunk = all_elements[pointer: pointer + param_numel]
                module.direct_unilora_indices_W[adapter_name] = chunk.view_as(
                    module.direct_unilora_indices_W[adapter_name]
                ).clone()
                pointer += param_numel

        assert pointer == len(all_elements)

        # 4) compute index frequency for normalization
        counts = torch.bincount(all_elements, minlength=theta_d_length)
        sqrt_counts = 1 / torch.sqrt(counts.float())

        # 5) update per-layer scales
        for module in [m for m in self.modules() if isinstance(m, DirectUniLoRALayer)]:
            scale_w = sqrt_counts[module.direct_unilora_indices_W[adapter_name].long()]
            module.update_norm(adapter_name, scale_w)

    def generate_index(self, total_length, theta_d_length, proj_seed):
        import numpy as np
        base_count = total_length // theta_d_length
        remaining = total_length % theta_d_length
        rng = np.random.default_rng(proj_seed)
        data = np.repeat(np.arange(theta_d_length), base_count)
        if remaining > 0:
            extras = rng.choice(theta_d_length, size=remaining, replace=False)
            data = np.concatenate([data, extras])
        rng.shuffle(data)
        return torch.tensor(data)

    def _init_unilora_theta_d(self, config: DirectUniLoRAConfig, adapter_name: str) -> None:
        theta_d = torch.zeros(config.theta_d_length)
        torch.nn.init.uniform_(theta_d, -config.init_theta_d_bound, config.init_theta_d_bound)
        self.direct_unilora_theta_d[adapter_name] = theta_d

    def _pre_injection_hook(self, model: nn.Module, config: DirectUniLoRAConfig, adapter_name: str) -> None:
        self.direct_unilora_theta_d = nn.ParameterDict({})

    def _create_and_replace(
        self,
        direct_unilora_config,
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
            "fan_in_fan_out": direct_unilora_config.fan_in_fan_out,
            "bias": bias,
        }
        self._init_unilora_theta_d(direct_unilora_config, adapter_name)

        if isinstance(target, Linear):
            target.update_layer(
                adapter_name=adapter_name,
                direct_unilora_theta_d=self.direct_unilora_theta_d,
                theta_d_length=direct_unilora_config.theta_d_length,
                unilora_dropout=direct_unilora_config.unilora_dropout,
            )
        else:
            new_module = self._create_new_module(
                direct_unilora_config=direct_unilora_config,
                direct_unilora_theta_d=self.direct_unilora_theta_d,
                adapter_name=adapter_name,
                target=target,
                **kwargs,
            )
            if adapter_name not in self.active_adapter:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(direct_unilora_config, direct_unilora_theta_d, adapter_name, target, **kwargs):
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
                kwargs["fan_in_fan_out"] = direct_unilora_config.fan_in_fan_out = False
        elif isinstance(target_base_layer, Conv1D):
            kwargs["is_target_conv_1d_layer"] = True
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                    "Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = direct_unilora_config.fan_in_fan_out = True
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `transformers.pytorch_utils.Conv1D`."
            )
        new_module = Linear(
            base_layer=target,
            direct_unilora_theta_d=direct_unilora_theta_d,
            adapter_name=adapter_name,
            theta_d_length=direct_unilora_config.theta_d_length,
            unilora_dropout=direct_unilora_config.unilora_dropout,
            **kwargs,
        )
        return new_module

    def get_nb_savable_parameters(self, adapter="default") -> tuple[int, int]:
        theta_d_params = 0
        other_params = 0
        for name, param in self.named_parameters():
            if "direct_unilora_theta_d" in name:
                theta_d_params += param.numel()
            elif "direct_unilora_indices" in name:
                other_params += param.numel()
            elif "direct_unilora_scales" in name:
                other_params += param.numel()

        return theta_d_params, other_params

    def print_savable_parameters(self) -> None:
        theta_d_params, other_params = self.get_nb_savable_parameters()
        print(
            f"Direct-UniLoRA params to-be-saved (float32-equivalent): {theta_d_params:,d} "
            f"|| total params to-be-saved: {(theta_d_params + other_params):,d}"
        )

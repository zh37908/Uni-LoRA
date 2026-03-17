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
from .config import UniLoRAGSConfig
from .layer import Linear, UniLoRGSLayer


class UniLoRAGSModel(BaseTuner):
    """
    UniLoRA-GS: use Gumbel-Softmax during training and argmax at inference.
    """

    prefix: str = "unilora_gs_"
    tuner_layer_cls = UniLoRGSLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

        # 1) count required indices for A/B elements
        total_params = 0
        for _, module in model.named_modules():
            if isinstance(module, UniLoRGSLayer):
                total_params += module.r[adapter_name] * module.in_features
                total_params += module.out_features * module.r[adapter_name]

        # 2) generate global indices for seeding logits
        theta_d_length = config[adapter_name].theta_d_length
        proj_seed = config[adapter_name].proj_seed
        all_elements = self.generate_index(total_params, theta_d_length, proj_seed)
        pointer = 0

        for _, module in model.named_modules():
            if isinstance(module, UniLoRGSLayer):
                a_numel = module.r[adapter_name] * module.in_features
                a_chunk = all_elements[pointer: pointer + a_numel]
                pointer += a_numel

                b_numel = module.out_features * module.r[adapter_name]
                b_chunk = all_elements[pointer: pointer + b_numel]
                pointer += b_numel

                indices_A = a_chunk.view(module.r[adapter_name], module.in_features).clone()
                indices_B = b_chunk.view(module.out_features, module.r[adapter_name]).clone()
                module.seed_logits_from_indices(adapter_name, indices_A, indices_B, config[adapter_name].init_logits_bias)

        assert pointer == len(all_elements)

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

    def _init_unilora_theta_d(self, config: UniLoRAGSConfig, adapter_name: str) -> None:
        theta_d = torch.zeros(config.theta_d_length)
        torch.nn.init.uniform_(theta_d, -config.init_theta_d_bound, config.init_theta_d_bound)
        self.unilora_gs_theta_d[adapter_name] = theta_d

    def _pre_injection_hook(self, model: nn.Module, config: UniLoRAGSConfig, adapter_name: str) -> None:
        self.unilora_gs_theta_d = nn.ParameterDict({})

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
                unilora_gs_theta_d=self.unilora_gs_theta_d,
                r=unilora_config.r,
                theta_d_length=unilora_config.theta_d_length,
                init_logits_std=unilora_config.init_logits_std,
                gumbel_tau=unilora_config.gumbel_tau,
                unilora_dropout=unilora_config.unilora_dropout,
            )
        else:
            new_module = self._create_new_module(
                unilora_config=unilora_config,
                unilora_gs_theta_d=self.unilora_gs_theta_d,
                adapter_name=adapter_name,
                target=target,
                **kwargs,
            )
            if adapter_name not in self.active_adapter:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(unilora_config, unilora_gs_theta_d, adapter_name, target, **kwargs):
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
            unilora_gs_theta_d=unilora_gs_theta_d,
            adapter_name=adapter_name,
            r=unilora_config.r,
            theta_d_length=unilora_config.theta_d_length,
            init_logits_std=unilora_config.init_logits_std,
            gumbel_tau=unilora_config.gumbel_tau,
            unilora_dropout=unilora_config.unilora_dropout,
            **kwargs,
        )
        return new_module

    def get_nb_savable_parameters(self, adapter="default") -> tuple[int, int]:
        theta_d_params = 0
        other_params = 0
        for name, param in self.named_parameters():
            if "unilora_gs_theta_d" in name:
                theta_d_params += param.numel()
            elif "unilora_gs_logits" in name:
                other_params += param.numel()

        return theta_d_params, other_params

    def print_savable_parameters(self) -> None:
        theta_d_params, other_params = self.get_nb_savable_parameters()
        print(
            f"UniLoRA-GS params to-be-saved (float32-equivalent): {theta_d_params:,d} "
            f"|| total params to-be-saved: {(theta_d_params + other_params):,d}"
        )

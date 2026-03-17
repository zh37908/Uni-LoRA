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
from .config import UniLoRABlockRoutingConfig
from .layer import Linear, UniLoRABlockRoutingLayer


class UniLoRABlockRoutingModel(BaseTuner):
    prefix: str = "unilora_block_routing_"
    tuner_layer_cls = UniLoRABlockRoutingLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

        # 1) Count total indices (across all layers) to distribute to blocks
        # Wait, indices are now PER BLOCK.
        # Actually, standard UniLoRA logic distributes global indices to ensure uniform usage.
        # Here, indices are local to [0, block_size).
        # We can just generate uniform local indices.
        
        theta_d_length = config[adapter_name].theta_d_length
        num_blocks = config[adapter_name].num_blocks
        block_size = theta_d_length // num_blocks
        proj_seed = config[adapter_name].proj_seed
        
        # We need total parameter count for generating stable random indices
        total_params = 0
        for _, module in model.named_modules():
            if isinstance(module, UniLoRABlockRoutingLayer):
                total_params += module.unilora_indices_A[adapter_name].numel()
                total_params += module.unilora_indices_B[adapter_name].numel()

        # Generate indices in range [0, block_size)
        all_elements = self.generate_index(total_params, block_size, proj_seed)
        pointer = 0

        # Assign indices
        for _, module in model.named_modules():
            if isinstance(module, UniLoRABlockRoutingLayer):
                # A
                numel = module.unilora_indices_A[adapter_name].numel()
                chunk = all_elements[pointer : pointer + numel]
                module.unilora_indices_A[adapter_name] = chunk.view_as(
                    module.unilora_indices_A[adapter_name]
                ).clone()
                pointer += numel
                # B
                numel = module.unilora_indices_B[adapter_name].numel()
                chunk = all_elements[pointer : pointer + numel]
                module.unilora_indices_B[adapter_name] = chunk.view_as(
                    module.unilora_indices_B[adapter_name]
                ).clone()
                pointer += numel
        
        assert pointer == len(all_elements)

        # Calculate normalization (1/sqrt(count))
        # Note: 'count' here is frequency of index within a block.
        # Since we use uniform distribution, we can approximate or compute exact.
        counts = torch.bincount(all_elements, minlength=block_size)
        sqrt_counts = 1.0 / torch.sqrt(counts.float().clamp(min=1.0))

        # Assign scales
        for module in self.modules():
            if isinstance(module, UniLoRABlockRoutingLayer):
                idx_a = module.unilora_indices_A[adapter_name].long()
                idx_b = module.unilora_indices_B[adapter_name].long()
                scale_a = sqrt_counts[idx_a]
                scale_b = sqrt_counts[idx_b]
                module.update_norm(adapter_name, scale_a, scale_b)

    def generate_index(self, total_length, block_size, proj_seed):
        import numpy as np
        rng = np.random.default_rng(proj_seed)
        base_count = total_length // block_size
        remaining = total_length % block_size
        data = np.repeat(np.arange(block_size), base_count)
        if remaining > 0:
            extras = rng.choice(block_size, size=remaining, replace=False)
            data = np.concatenate([data, extras])
        rng.shuffle(data)
        return torch.tensor(data)

    def _init_unilora_theta_d(self, config: UniLoRABlockRoutingConfig, adapter_name: str) -> None:
        theta_d = torch.zeros(config.theta_d_length)
        torch.nn.init.uniform_(theta_d, -config.init_theta_d_bound, config.init_theta_d_bound)
        self.unilora_block_routing_theta_d[adapter_name] = theta_d

    def _pre_injection_hook(self, model: nn.Module, config: UniLoRABlockRoutingConfig, adapter_name: str) -> None:
        self.unilora_block_routing_theta_d = nn.ParameterDict({})

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
                unilora_theta_d=self.unilora_block_routing_theta_d,
                r=unilora_config.r,
                theta_d_length=unilora_config.theta_d_length,
                num_blocks=unilora_config.num_blocks,
                router_tau=unilora_config.router_tau,
                unilora_dropout=unilora_config.unilora_dropout,
            )
        else:
            new_module = self._create_new_module(
                unilora_config=unilora_config,
                unilora_theta_d=self.unilora_block_routing_theta_d,
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
            num_blocks=unilora_config.num_blocks,
            router_tau=unilora_config.router_tau,
            unilora_dropout=unilora_config.unilora_dropout,
            **kwargs,
        )
        return new_module

    def get_nb_savable_parameters(self, adapter="default") -> tuple[int, int]:
        theta_d_params = 0
        other_params = 0
        for name, param in self.named_parameters():
            if "unilora_block_routing_theta_d" in name:
                theta_d_params += param.numel()
            elif "unilora_router_logits" in name:
                other_params += param.numel()
        return theta_d_params, other_params

    def print_savable_parameters(self) -> None:
        theta_d_params, other_params = self.get_nb_savable_parameters()
        print(
            f"UniLoRA-Block-Routing params to-be-saved (float32-equivalent): {theta_d_params:,d} "
            f"|| router params: {other_params:,d} "
            f"|| total params to-be-saved: {(theta_d_params + other_params):,d}"
        )

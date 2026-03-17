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

import re
import warnings

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

from .config import UniLoRALayerWiseConfig
from .layer import Linear, UniLoRALayerWiseLayer


class UniLoRALayerWiseModel(BaseTuner):
    """
    Creates UniLoRA-Layer-Wise model from a pretrained transformers model.
    """

    prefix: str = "unilora_layer_wise_"
    tuner_layer_cls = UniLoRALayerWiseLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        # We need to determine the number of layers to split theta_d_length
        # Before calling super().__init__ (which triggers injection), we count matches.

        # If config is a dictionary, grab the config for this adapter
        if isinstance(config, dict):
            unilora_config = config[adapter_name]
        else:
            unilora_config = config
        
        # 1. Prepare config (handle defaults) - mimicking BaseTuner._prepare_adapter_config
        if unilora_config.target_modules is None:
            model_config = self.get_model_config(model)
            # Basic fallback if model_config has model_type
            if model_config and "model_type" in model_config:
                 target_modules = self.target_module_mapping.get(model_config["model_type"])
                 if target_modules:
                     unilora_config.target_modules = set(target_modules)
        
        # 2. Count target modules
        # This count must match exactly what inject_adapter finds.
        # We iterate named_modules same as inject_adapter.
        self.target_modules_count = 0
        
        # Handle 'all-linear' specially as check_target_module_exists might not handle it directly
        is_all_linear = getattr(unilora_config, "target_modules", None) == "all-linear"

        for key, module in model.named_modules():
            is_valid = False
            if is_all_linear:
                 if isinstance(module, (nn.Linear, Conv1D)):
                     is_valid = True
            elif check_target_module_exists(unilora_config, key):
                 if isinstance(module, (nn.Linear, Conv1D)):
                     is_valid = True
            
            if is_valid:
                 self.target_modules_count += 1
        
        if self.target_modules_count == 0:
            warnings.warn("No target modules found for UniLoRA-Layer-Wise. theta_d allocation might fail.")
            self.theta_d_sizes = []
        else:
            # 3. Calculate per-layer dimensions
            total_d = unilora_config.theta_d_length
            n = self.target_modules_count
            base = total_d // n
            remainder = total_d % n
            # Distribute remainder to first 'remainder' layers
            self.theta_d_sizes = [base + 1] * remainder + [base] * (n - remainder)
        
        self.injection_counter = 0

        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

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

        # Determine size for this layer
        if self.injection_counter < len(self.theta_d_sizes):
            local_d = self.theta_d_sizes[self.injection_counter]
        else:
            # Fallback if count was wrong (shouldn't happen if logic matches)
            local_d = unilora_config.theta_d_length // max(1, self.target_modules_count)
            warnings.warn(f"Injection counter {self.injection_counter} exceeded expected count {self.target_modules_count}. Using fallback size {local_d}.")
        
        self.injection_counter += 1
        
        # Generate deterministic seeds
        # We use current_key to generate a seed for the layer
        # This seed is used for FastFood matrix generation
        import hashlib
        def _stable_int_seed(text: str) -> int:
            h = hashlib.md5(text.encode("utf-8")).digest()
            return int.from_bytes(h[:4], byteorder="little", signed=False)
        
        layer_seed = _stable_int_seed(current_key)

        if isinstance(target, Linear):
            target.update_layer(
                adapter_name=adapter_name,
                r=unilora_config.r,
                theta_d_length_local=local_d,
                proj_seed=unilora_config.proj_seed,
                layer_seed=layer_seed,
                init_theta_d_bound=unilora_config.init_theta_d_bound,
                unilora_dropout=unilora_config.unilora_dropout,
            )
        else:
            new_module = self._create_new_module(
                unilora_config=unilora_config,
                adapter_name=adapter_name,
                target=target,
                local_d=local_d,
                layer_seed=layer_seed,
                **kwargs,
            )
            if adapter_name not in self.active_adapter:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(unilora_config, adapter_name, target, local_d, layer_seed, **kwargs):
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
            adapter_name=adapter_name,
            r=unilora_config.r,
            theta_d_length_local=local_d,
            proj_seed=unilora_config.proj_seed,
            layer_seed=layer_seed,
            init_theta_d_bound=unilora_config.init_theta_d_bound,
            unilora_dropout=unilora_config.unilora_dropout,
            **kwargs,
        )
    
    def _pre_injection_hook(self, model: nn.Module, config, adapter_name: str) -> None:
        # Reset counter before injection starts
        # This hook is called by BaseTuner.__init__ before inject_adapter
        self.injection_counter = 0

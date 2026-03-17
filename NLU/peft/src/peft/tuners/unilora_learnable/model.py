from __future__ import annotations
import warnings
import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING 
from .config import UniLoRALearnableConfig
from .layer import Linear, UniLoRALayer

class UniLoRALearnableModel(BaseTuner):
    """
    Creates UniLoRA model with learnable projection entries from a pretrained transformers model.
    """
    prefix: str = "unilora_learnable_"
    # Specify Layer class, BaseTuner needs it to identify layers
    tuner_layer_cls = UniLoRALayer 
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        # BaseTuner's init calls _create_and_replace to inject layers
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)
        
        # --- UniLoRA specific initialization logic (Global Hash Index Allocation) ---
        # 1. Count total indices needed
        LoRA_para_cnt = 0
        for name, module in model.named_modules():
             if isinstance(module, UniLoRALayer):
               LoRA_para_cnt += module.unilora_indices_A[adapter_name].numel()
               LoRA_para_cnt += module.unilora_indices_B[adapter_name].numel()
        
        # 2. Generate globally uniform indices
        theta_d_length = config[adapter_name].theta_d_length
        proj_seed = config[adapter_name].proj_seed
        all_elements = self.generate_index(LoRA_para_cnt, theta_d_length, proj_seed)
        pointer = 0
        
        # 3. Assign indices back to each layer
        for name, module in model.named_modules():
            if isinstance(module, UniLoRALayer):
                # Process A matrix indices
                param_numel = module.unilora_indices_A[adapter_name].numel()
                chunk = all_elements[pointer: pointer + param_numel]
                module.unilora_indices_A[adapter_name] = chunk.view_as(module.unilora_indices_A[adapter_name]).clone()
                pointer += param_numel
                
                # Process B matrix indices
                param_numel = module.unilora_indices_B[adapter_name].numel()
                chunk = all_elements[pointer: pointer + param_numel]
                module.unilora_indices_B[adapter_name] = chunk.view_as(module.unilora_indices_B[adapter_name]).clone()
                pointer += param_numel
        
        assert pointer == len(all_elements)
        
        # 4. Compute index usage frequency for normalization (Scales)
        counts = torch.bincount(all_elements, minlength=theta_d_length) 
        sqrt_counts = 1/torch.sqrt(counts.float()) 
        
        index_ls = []
        for name, module in model.named_modules():
             if isinstance(module, UniLoRALayer):
               index_ls.append(module.unilora_indices_A[adapter_name].long())
               index_ls.append(module.unilora_indices_B[adapter_name].long())
        
        # Extract corresponding scale factors based on actual indices used
        norm_factors = [sqrt_counts[t] for t in index_ls]
        
        # 5. Update each layer's Scales. In this variant, they become learnable Parameters.
        uni_modules = [m for m in self.modules() if isinstance(m, UniLoRALayer)]
        for module, (scale_a, scale_b) in zip(uni_modules, zip(*[iter(norm_factors)] * 2)):
            module.update_norm(adapter_name, scale_a, scale_b)

    def generate_index(self, LoRA_para_cnt, theta_d_length, proj_seed):
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

    def _init_unilora_theta_d(self, config: UniLoRALearnableConfig, adapter_name: str) -> None:
        unilora_theta_d = torch.zeros(config.theta_d_length)
        torch.nn.init.uniform_(unilora_theta_d, -config.init_theta_d_bound, config.init_theta_d_bound)
        self.unilora_learnable_theta_d[adapter_name] = unilora_theta_d

    def _pre_injection_hook(self, model: nn.Module, config: UniLoRALearnableConfig, adapter_name: str) -> None:
        self.unilora_learnable_theta_d = nn.ParameterDict({})

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
                unilora_theta_d=self.unilora_learnable_theta_d,
                r=unilora_config.r,
                theta_d_length=unilora_config.theta_d_length,
                unilora_dropout=unilora_config.unilora_dropout,
            )
        else:
            new_module = self._create_new_module(
                unilora_config=unilora_config,
                unilora_theta_d=self.unilora_learnable_theta_d,
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

    def get_nb_savable_parameters(self, adapter="default") -> tuple[int, int]:
        """
        Returns the number of savable Uni-LoRA parameters and other savable parameters.
        """
        unilora_params = 0
        other_params = 0
        for name, param in self.named_parameters():
            if "unilora_learnable_theta_d" in name:
                unilora_params += param.numel()
            elif "unilora_scales" in name:
                # In this variant, scales are learnable parameters
                unilora_params += param.numel()
        
        # We also need to count indices, even though they are buffers, 
        # because the original print_savable_parameters seems to care about them 
        # (based on the comments in unilora/model.py).
        # However, indices are not in named_parameters().
        for name, buffer in self.named_buffers():
            if "unilora_indices" in name:
                other_params += buffer.numel()
       
        return unilora_params, other_params

    def print_savable_parameters(self) -> None:
        """
        Prints the number of savable Uni-LoRA parameters and total savable parameters.
        """
        unilora_params, other_params = self.get_nb_savable_parameters()
        print(
            f"Uni-LoRA Learnable params to-be-saved (float32-equivalent): {unilora_params:,d} "
            f"|| total params to-be-saved: {(unilora_params + other_params):,d}"
        )

from __future__ import annotations
import warnings
import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING 
from .config import UniLoRAIsometricControlConfig
from .layer import Linear, UniLoRALayer

class UniLoRAIsometricControlModel(BaseTuner):
    prefix: str = "unilora_isometric_control_"
    tuner_layer_cls = UniLoRALayer 
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)
        
        # Global Hash Index Allocation
        LoRA_para_cnt = 0
        for name, module in model.named_modules():
             if isinstance(module, UniLoRALayer):
               LoRA_para_cnt += module.unilora_indices_A[adapter_name].numel()
               LoRA_para_cnt += module.unilora_indices_B[adapter_name].numel()
        
        theta_d_length = config[adapter_name].theta_d_length
        proj_seed = config[adapter_name].proj_seed
        all_elements = self.generate_index(LoRA_para_cnt, theta_d_length, proj_seed)
        pointer = 0
        
        for name, module in model.named_modules():
            if isinstance(module, UniLoRALayer):
                param_numel = module.unilora_indices_A[adapter_name].numel()
                module.unilora_indices_A[adapter_name] = all_elements[pointer: pointer + param_numel].view_as(module.unilora_indices_A[adapter_name]).clone()
                pointer += param_numel
                
                param_numel = module.unilora_indices_B[adapter_name].numel()
                module.unilora_indices_B[adapter_name] = all_elements[pointer: pointer + param_numel].view_as(module.unilora_indices_B[adapter_name]).clone()
                pointer += param_numel
        
        # Scale Calculation with Isometry Control
        # Formula: Scale = count ** (-(1 - alpha) / 2)
        alpha = config[adapter_name].isometry_alpha
        counts = torch.bincount(all_elements, minlength=theta_d_length).float()
        # Avoid division by zero for unused vectors
        counts = torch.clamp(counts, min=1.0)
        
        exponent = -0.5 * (1.0 - alpha)
        scales = counts ** exponent
        
        uni_modules = [m for m in self.modules() if isinstance(m, UniLoRALayer)]
        for module in uni_modules:
            scale_a = scales[module.unilora_indices_A[adapter_name].long()]
            scale_b = scales[module.unilora_indices_B[adapter_name].long()]
            module.update_norm(adapter_name, scale_a, scale_b)

    def generate_index(self, LoRA_para_cnt, theta_d_length, proj_seed):
        import numpy as np
        rng = np.random.default_rng(proj_seed)
        data = np.repeat(np.arange(theta_d_length), LoRA_para_cnt // theta_d_length)
        remaining = LoRA_para_cnt % theta_d_length
        if remaining > 0:
            data = np.concatenate([data, rng.choice(theta_d_length, size=remaining, replace=False)])
        rng.shuffle(data)
        return torch.tensor(data)

    def _init_unilora_theta_d(self, config, adapter_name):
        unilora_theta_d = torch.zeros(config.theta_d_length)
        torch.nn.init.uniform_(unilora_theta_d, -config.init_theta_d_bound, config.init_theta_d_bound)
        self.unilora_isometric_control_theta_d[adapter_name] = unilora_theta_d

    def _pre_injection_hook(self, model, config, adapter_name):
        self.unilora_isometric_control_theta_d = nn.ParameterDict({})

    def _create_and_replace(self, config, adapter_name, target, target_name, parent, current_key):
        bias = hasattr(target, "bias") and target.bias is not None
        self._init_unilora_theta_d(config, adapter_name)
        new_module = Linear(target, self.unilora_isometric_control_theta_d, adapter_name, config.r, config.theta_d_length, config.unilora_dropout, config.fan_in_fan_out, **{"bias": bias})
        self._replace_module(parent, target_name, new_module, target)

    def get_nb_savable_parameters(self, adapter="default"):
        p = sum(p.numel() for n, p in self.named_parameters() if "unilora_isometric_control_theta_d" in n)
        b = sum(b.numel() for n, b in self.named_buffers() if "unilora_indices" in n)
        return p, b

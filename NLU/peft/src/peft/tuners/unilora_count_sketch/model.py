from __future__ import annotations

import warnings

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

from .config import UniLoRACountSketchConfig
from .layer import Linear, UniLoRACountSketchLayer


class UniLoRACountSketchModel(BaseTuner):
    """
    UniLoRA-CountSketch:
    - uses v independent (theta_d, P) sketches;
    - each non-zero entry in P is fixed random +/-1 (no normalization);
    - inference aggregates by element-wise median across sketches.
    """

    prefix: str = "unilora_count_sketch_"
    tuner_layer_cls = UniLoRACountSketchLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

        theta_d_length = config[adapter_name].theta_d_length
        proj_seed = int(config[adapter_name].proj_seed)
        num_sketches = int(config[adapter_name].num_sketches)

        for sketch_idx in range(num_sketches):
            lora_para_cnt = 0
            for _, module in model.named_modules():
                if isinstance(module, UniLoRACountSketchLayer):
                    key = module._sketch_key(sketch_idx)
                    lora_para_cnt += module.unilora_count_sketch_indices_A[adapter_name][key].numel()
                    lora_para_cnt += module.unilora_count_sketch_indices_B[adapter_name][key].numel()

            all_elements = self.generate_index(lora_para_cnt, theta_d_length, proj_seed + sketch_idx)
            pointer = 0

            for _, module in model.named_modules():
                if isinstance(module, UniLoRACountSketchLayer):
                    key = module._sketch_key(sketch_idx)

                    param_numel = module.unilora_count_sketch_indices_A[adapter_name][key].numel()
                    chunk = all_elements[pointer : pointer + param_numel]
                    module.unilora_count_sketch_indices_A[adapter_name][key] = (
                        chunk.view_as(module.unilora_count_sketch_indices_A[adapter_name][key]).clone()
                    )
                    pointer += param_numel

                    param_numel = module.unilora_count_sketch_indices_B[adapter_name][key].numel()
                    chunk = all_elements[pointer : pointer + param_numel]
                    module.unilora_count_sketch_indices_B[adapter_name][key] = (
                        chunk.view_as(module.unilora_count_sketch_indices_B[adapter_name][key]).clone()
                    )
                    pointer += param_numel

            assert pointer == len(all_elements)

            sign_generator = torch.Generator(device="cpu")
            sign_generator.manual_seed(proj_seed + 10000 + sketch_idx)
            for _, module in model.named_modules():
                if isinstance(module, UniLoRACountSketchLayer):
                    key = module._sketch_key(sketch_idx)
                    sign_a = self._sample_rademacher(
                        module.unilora_count_sketch_indices_A[adapter_name][key].shape, sign_generator
                    )
                    sign_b = self._sample_rademacher(
                        module.unilora_count_sketch_indices_B[adapter_name][key].shape, sign_generator
                    )
                    module.update_sign(adapter_name, sketch_idx, sign_a, sign_b)

    @staticmethod
    def _sample_rademacher(shape, generator: torch.Generator) -> torch.Tensor:
        sign = torch.randint(0, 2, shape, generator=generator, dtype=torch.int8)
        sign = sign * 2 - 1
        return sign.to(torch.float32)

    def generate_index(self, lora_para_cnt, theta_d_length, proj_seed):
        import numpy as np

        total_length = lora_para_cnt
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

    def _init_unilora_count_sketch_theta_d(self, config: UniLoRACountSketchConfig, adapter_name: str) -> None:
        theta_list = nn.ParameterList([])
        for _ in range(config.num_sketches):
            theta = nn.Parameter(torch.zeros(config.theta_d_length))
            torch.nn.init.uniform_(theta, -config.init_theta_d_bound, config.init_theta_d_bound)
            theta_list.append(theta)
        self.unilora_count_sketch_theta_d[adapter_name] = theta_list

    def _pre_injection_hook(self, model: nn.Module, config: UniLoRACountSketchConfig, adapter_name: str) -> None:
        self.unilora_count_sketch_theta_d = nn.ModuleDict({})

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
        self._init_unilora_count_sketch_theta_d(unilora_config, adapter_name)

        if isinstance(target, Linear):
            target.update_layer(
                adapter_name=adapter_name,
                unilora_count_sketch_theta_d=self.unilora_count_sketch_theta_d,
                r=unilora_config.r,
                theta_d_length=unilora_config.theta_d_length,
                num_sketches=unilora_config.num_sketches,
                unilora_dropout=unilora_config.unilora_dropout,
            )
        else:
            new_module = self._create_new_module(
                unilora_config=unilora_config,
                unilora_count_sketch_theta_d=self.unilora_count_sketch_theta_d,
                adapter_name=adapter_name,
                target=target,
                **kwargs,
            )
            if adapter_name not in self.active_adapter:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(unilora_config, unilora_count_sketch_theta_d, adapter_name, target, **kwargs):
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
            unilora_count_sketch_theta_d=unilora_count_sketch_theta_d,
            adapter_name=adapter_name,
            r=unilora_config.r,
            theta_d_length=unilora_config.theta_d_length,
            num_sketches=unilora_config.num_sketches,
            unilora_dropout=unilora_config.unilora_dropout,
            **kwargs,
        )
        return new_module

    def get_nb_savable_parameters(self, adapter="default") -> tuple[int, int]:
        theta_d_params = 0
        other_params = 0
        for name, param in self.named_parameters():
            if "unilora_count_sketch_theta_d" in name:
                theta_d_params += param.numel()
            elif "unilora_count_sketch_indices" in name:
                other_params += param.numel()
            elif "unilora_count_sketch_signs" in name:
                other_params += param.numel()
        return theta_d_params, other_params

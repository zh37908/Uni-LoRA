from __future__ import annotations

import warnings

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

from .config import UniLoRASignConfig
from .layer import Linear, UniLoRASignLayer


class UniLoRASignModel(BaseTuner):
    """
    UniLoRA-Sign:
    keep the same index projection as UniLoRA, but multiply each non-zero projection entry by a fixed random sign.
    """

    prefix: str = "unilora_sign_"
    tuner_layer_cls = UniLoRASignLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

        lora_para_cnt = 0
        for _, module in model.named_modules():
            if isinstance(module, UniLoRASignLayer):
                lora_para_cnt += module.unilora_sign_indices_A[adapter_name].numel()
                lora_para_cnt += module.unilora_sign_indices_B[adapter_name].numel()

        theta_d_length = config[adapter_name].theta_d_length
        proj_seed = config[adapter_name].proj_seed
        all_elements = self.generate_index(lora_para_cnt, theta_d_length, proj_seed)
        pointer = 0

        for _, module in model.named_modules():
            if isinstance(module, UniLoRASignLayer):
                param_numel = module.unilora_sign_indices_A[adapter_name].numel()
                chunk = all_elements[pointer : pointer + param_numel]
                module.unilora_sign_indices_A[adapter_name] = (
                    chunk.view_as(module.unilora_sign_indices_A[adapter_name]).clone()
                )
                pointer += param_numel

                param_numel = module.unilora_sign_indices_B[adapter_name].numel()
                chunk = all_elements[pointer : pointer + param_numel]
                module.unilora_sign_indices_B[adapter_name] = (
                    chunk.view_as(module.unilora_sign_indices_B[adapter_name]).clone()
                )
                pointer += param_numel

        assert pointer == len(all_elements)

        counts = torch.bincount(all_elements, minlength=theta_d_length)
        sqrt_counts = 1 / torch.sqrt(counts.float())

        index_ls = []
        for _, module in model.named_modules():
            if isinstance(module, UniLoRASignLayer):
                index_ls.append(module.unilora_sign_indices_A[adapter_name].long())
                index_ls.append(module.unilora_sign_indices_B[adapter_name].long())

        norm_factors = [sqrt_counts[t] for t in index_ls]
        uni_modules = [m for m in self.modules() if isinstance(m, UniLoRASignLayer)]

        sign_generator = torch.Generator(device="cpu")
        sign_generator.manual_seed(int(proj_seed) + 1)

        for module, (scale_a, scale_b) in zip(uni_modules, zip(*[iter(norm_factors)] * 2)):
            module.update_norm(adapter_name, scale_a, scale_b)

            sign_a = self._sample_rademacher(
                module.unilora_sign_indices_A[adapter_name].shape,
                sign_generator,
            )
            sign_b = self._sample_rademacher(
                module.unilora_sign_indices_B[adapter_name].shape,
                sign_generator,
            )
            module.update_sign(adapter_name, sign_a, sign_b)

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

    def _init_unilora_sign_theta_d(self, config: UniLoRASignConfig, adapter_name: str) -> None:
        unilora_sign_theta_d = torch.zeros(config.theta_d_length)
        torch.nn.init.uniform_(unilora_sign_theta_d, -config.init_theta_d_bound, config.init_theta_d_bound)
        self.unilora_sign_theta_d[adapter_name] = unilora_sign_theta_d

    def _pre_injection_hook(self, model: nn.Module, config: UniLoRASignConfig, adapter_name: str) -> None:
        self.unilora_sign_theta_d = nn.ParameterDict({})

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
        self._init_unilora_sign_theta_d(unilora_config, adapter_name)

        if isinstance(target, Linear):
            target.update_layer(
                adapter_name=adapter_name,
                unilora_sign_theta_d=self.unilora_sign_theta_d,
                r=unilora_config.r,
                theta_d_length=unilora_config.theta_d_length,
                unilora_dropout=unilora_config.unilora_dropout,
            )
        else:
            new_module = self._create_new_module(
                unilora_config=unilora_config,
                unilora_sign_theta_d=self.unilora_sign_theta_d,
                adapter_name=adapter_name,
                target=target,
                **kwargs,
            )
            if adapter_name not in self.active_adapter:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(unilora_config, unilora_sign_theta_d, adapter_name, target, **kwargs):
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
            unilora_sign_theta_d=unilora_sign_theta_d,
            adapter_name=adapter_name,
            r=unilora_config.r,
            theta_d_length=unilora_config.theta_d_length,
            unilora_dropout=unilora_config.unilora_dropout,
            **kwargs,
        )
        return new_module

    def get_nb_savable_parameters(self, adapter="default") -> tuple[int, int]:
        theta_d_params = 0
        other_params = 0
        for name, param in self.named_parameters():
            if "unilora_sign_theta_d" in name:
                theta_d_params += param.numel()
            elif "unilora_sign_indices" in name:
                other_params += param.numel()
            elif "unilora_sign_scales" in name:
                other_params += param.numel()
        return theta_d_params, other_params

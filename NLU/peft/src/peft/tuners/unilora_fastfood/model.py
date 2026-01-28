from __future__ import annotations

import hashlib
import warnings

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

from .config import UniLoRAFastFoodConfig
from .layer import Linear, UniLoRAFastFoodLayer


def _stable_int_seed(text: str) -> int:
    """Stable 32-bit integer seed from a string (independent of PYTHONHASHSEED)."""
    h = hashlib.md5(text.encode("utf-8")).digest()
    return int.from_bytes(h[:4], byteorder="little", signed=False)


class UniLoRAFastFoodModel(BaseTuner):
    """
    Creates UniLoRA-FastFood model from a pretrained transformers model.
    """

    prefix: str = "unilora_fastfood_"
    tuner_layer_cls = UniLoRAFastFoodLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

    def _init_theta_d(self, config: UniLoRAFastFoodConfig, adapter_name: str) -> None:
        theta = torch.zeros(config.theta_d_length)
        torch.nn.init.uniform_(theta, -config.init_theta_d_bound, config.init_theta_d_bound)
        self.unilora_fastfood_theta_d[adapter_name] = theta

    def _pre_injection_hook(self, model: nn.Module, config: UniLoRAFastFoodConfig, adapter_name: str) -> None:
        self.unilora_fastfood_theta_d = nn.ParameterDict({})

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

        self._init_theta_d(unilora_config, adapter_name)

        layer_seed = _stable_int_seed(current_key)

        if isinstance(target, Linear):
            target.update_layer(
                adapter_name=adapter_name,
                unilora_fastfood_theta_d=self.unilora_fastfood_theta_d,
                r=unilora_config.r,
                theta_d_length=unilora_config.theta_d_length,
                proj_seed=unilora_config.proj_seed,
                layer_seed=layer_seed,
                unilora_dropout=unilora_config.unilora_dropout,
            )
        else:
            new_module = self._create_new_module(
                unilora_config=unilora_config,
                unilora_fastfood_theta_d=self.unilora_fastfood_theta_d,
                adapter_name=adapter_name,
                layer_seed=layer_seed,
                target=target,
                **kwargs,
            )
            if adapter_name not in self.active_adapter:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(unilora_config, unilora_fastfood_theta_d, adapter_name, layer_seed: int, target, **kwargs):
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
            unilora_fastfood_theta_d=unilora_fastfood_theta_d,
            adapter_name=adapter_name,
            r=unilora_config.r,
            theta_d_length=unilora_config.theta_d_length,
            proj_seed=unilora_config.proj_seed,
            layer_seed=layer_seed,
            unilora_dropout=unilora_config.unilora_dropout,
            **kwargs,
        )

    def get_nb_savable_parameters(self, adapter="default") -> tuple[int, int]:
        theta_d_params = 0
        other_params = 0
        for name, param in self.named_parameters():
            if "unilora_fastfood_theta_d" in name:
                theta_d_params += param.numel()
            else:
                other_params += 0
        return theta_d_params, other_params


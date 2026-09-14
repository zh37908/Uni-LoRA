from __future__ import annotations

import warnings

import torch
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer

from .layer_multi_gpu import MultiGpuLinear
from .model import UniLoRARoSASnipModel


class UniLoRARoSASnipMultiGpuModel(UniLoRARoSASnipModel):
    """
    UniLoRA-RoSA-SNIP model using a multi-GPU-aware forward path.

    It keeps the same mask generation and SNIP scoring behavior as
    `UniLoRARoSASnipModel`, but injects `MultiGpuLinear` layers.
    """

    prefix: str = "unilora_rosa_snip_multi_gpu_"

    @staticmethod
    def _create_new_module(
        unilora_config,
        unilora_rosa_theta_d,
        unilora_rosa_sparse_theta_D,
        unilora_rosa_sparse_mask,
        unilora_rosa_grad_accum,
        adapter_name,
        target,
        **kwargs,
    ):
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

        return MultiGpuLinear(
            base_layer=target,
            unilora_rosa_theta_d=unilora_rosa_theta_d,
            unilora_rosa_sparse_theta_D=unilora_rosa_sparse_theta_D,
            unilora_rosa_sparse_mask=unilora_rosa_sparse_mask,
            unilora_rosa_grad_accum=unilora_rosa_grad_accum,
            adapter_name=adapter_name,
            r=unilora_config.r,
            theta_d_length=unilora_config.theta_d_length,
            unilora_dropout=unilora_config.unilora_dropout,
            **kwargs,
        )

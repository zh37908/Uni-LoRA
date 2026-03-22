from __future__ import annotations

import hashlib
import warnings

import torch
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

from .layer import Linear, UniLoRASketchDeltaLayer


def _stable_int_seed(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


class UniLoRASketchDeltaModel(BaseTuner):
    """
    UniLoRA variant where the LoRA delta matrices are parameterized by local sketch
    codebooks instead of dense A/B weights.
    """

    prefix: str = "unilora_sketch_delta_"
    tuner_layer_cls = UniLoRASketchDeltaLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

    def _create_and_replace(
        self,
        peft_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        if current_key is None:
            raise ValueError("Current key should not be `None`.")

        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "fan_in_fan_out": peft_config.fan_in_fan_out,
            "bias": bias,
        }
        layer_seed = _stable_int_seed(current_key)

        if isinstance(target, Linear):
            target.update_layer(
                adapter_name=adapter_name,
                r=peft_config.r,
                bits=peft_config.bits,
                groups_per_row=peft_config.groups_per_row,
                init_codebook_bound=peft_config.init_codebook_bound,
                proj_seed=peft_config.proj_seed,
                layer_seed=layer_seed,
                unilora_dropout=peft_config.unilora_dropout,
            )
        else:
            new_module = self._create_new_module(
                peft_config=peft_config,
                adapter_name=adapter_name,
                layer_seed=layer_seed,
                target=target,
                **kwargs,
            )
            if adapter_name not in self.active_adapter:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(peft_config, adapter_name, layer_seed: int, target, **kwargs):
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
                kwargs["fan_in_fan_out"] = peft_config.fan_in_fan_out = False
        elif isinstance(target_base_layer, Conv1D):
            kwargs["is_target_conv_1d_layer"] = True
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                    "Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = peft_config.fan_in_fan_out = True
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently only `torch.nn.Linear` and "
                "`transformers.pytorch_utils.Conv1D` are supported."
            )

        return Linear(
            base_layer=target,
            adapter_name=adapter_name,
            r=peft_config.r,
            bits=peft_config.bits,
            groups_per_row=peft_config.groups_per_row,
            init_codebook_bound=peft_config.init_codebook_bound,
            proj_seed=peft_config.proj_seed,
            layer_seed=layer_seed,
            unilora_dropout=peft_config.unilora_dropout,
            **kwargs,
        )

    def get_nb_savable_parameters(self, adapter: str = "default") -> tuple[int, int]:
        trainable_params = 0
        buffer_params = 0

        for name, param in self.named_parameters():
            if "unilora_sketch_delta_quant_" in name:
                trainable_params += param.numel()

        for name, buffer in self.named_buffers():
            if "unilora_sketch_delta_codes_" in name:
                buffer_params += buffer.numel()

        return trainable_params, buffer_params

from __future__ import annotations

import warnings

import torch
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

from .layer import Linear, UniLoRASketchTuneLayer


class UniLoRASketchTuneModel(BaseTuner):
    """
    SketchTune-style tuner that replaces target linear layers with a trainable
    codebook plus fixed discrete codes.
    """

    prefix: str = "unilora_sketch_tune_"
    tuner_layer_cls = UniLoRASketchTuneLayer
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

        if isinstance(target, Linear):
            target.update_layer(
                adapter_name=adapter_name,
                bits=peft_config.bits,
                groups_per_row=peft_config.groups_per_row,
                bootstrap_method=peft_config.bootstrap_method,
                bootstrap_kmeans_iters=peft_config.bootstrap_kmeans_iters,
                unilora_dropout=peft_config.unilora_dropout,
            )
        else:
            new_module = self._create_new_module(
                peft_config=peft_config,
                adapter_name=adapter_name,
                target=target,
                **kwargs,
            )
            if adapter_name not in self.active_adapter:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(peft_config, adapter_name, target, **kwargs):
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
            bits=peft_config.bits,
            groups_per_row=peft_config.groups_per_row,
            bootstrap_method=peft_config.bootstrap_method,
            bootstrap_kmeans_iters=peft_config.bootstrap_kmeans_iters,
            unilora_dropout=peft_config.unilora_dropout,
            **kwargs,
        )

    def get_nb_savable_parameters(self, adapter: str = "default") -> tuple[int, int]:
        codebook_params = 0
        code_buffers = 0

        for name, param in self.named_parameters():
            if "unilora_sketch_tune_quant_grid" in name:
                codebook_params += param.numel()

        for name, buffer in self.named_buffers():
            if "unilora_sketch_tune_weight_codes" in name:
                code_buffers += buffer.numel()

        return codebook_params, code_buffers

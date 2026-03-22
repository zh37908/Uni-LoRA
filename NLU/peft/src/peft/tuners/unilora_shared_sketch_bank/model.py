from __future__ import annotations

import hashlib
import warnings

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

from ..unilora_sketch_utils import generate_balanced_indices
from .config import UniLoRASharedSketchBankConfig
from .layer import Linear, UniLoRASharedSketchBankLayer


def _stable_int_seed(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


class UniLoRASharedSketchBankModel(BaseTuner):
    """
    UniLoRA variant where all target modules share a global sketch bank and fixed
    discrete assignments decode LoRA delta matrices from that bank.
    """

    prefix: str = "unilora_shared_sketch_bank_"
    tuner_layer_cls = UniLoRASharedSketchBankLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

        if isinstance(config, dict):
            sketch_config = config[adapter_name]
        else:
            sketch_config = config

        self._assign_balanced_shared_codes(adapter_name, sketch_config)

    def _pre_injection_hook(self, model: nn.Module, config: UniLoRASharedSketchBankConfig, adapter_name: str) -> None:
        self.unilora_shared_sketch_bank_params = nn.ParameterDict({})

    def _init_shared_bank(self, config: UniLoRASharedSketchBankConfig, adapter_name: str) -> None:
        if adapter_name in self.unilora_shared_sketch_bank_params:
            return

        codebook_size = 1 << config.bits
        bank = torch.zeros(config.num_banks, codebook_size)
        torch.nn.init.uniform_(bank, -config.init_bank_bound, config.init_bank_bound)
        self.unilora_shared_sketch_bank_params[adapter_name] = nn.Parameter(bank)

    def _assign_balanced_shared_codes(self, adapter_name: str, config: UniLoRASharedSketchBankConfig) -> None:
        layers = [m for m in self.model.modules() if isinstance(m, UniLoRASharedSketchBankLayer)]
        if not layers:
            return

        total_positions = 0
        for layer in layers:
            total_positions += layer.unilora_shared_sketch_bank_codes_A[adapter_name].numel()
            total_positions += layer.unilora_shared_sketch_bank_codes_B[adapter_name].numel()

        logical_ids = generate_balanced_indices(
            total_length=total_positions,
            num_buckets=config.num_banks * (1 << config.bits),
            seed=config.proj_seed,
        )

        pointer = 0
        for layer in layers:
            numel_A = layer.unilora_shared_sketch_bank_codes_A[adapter_name].numel()
            logical_A = logical_ids[pointer : pointer + numel_A].clone()
            pointer += numel_A

            numel_B = layer.unilora_shared_sketch_bank_codes_B[adapter_name].numel()
            logical_B = logical_ids[pointer : pointer + numel_B].clone()
            pointer += numel_B

            layer.set_logical_assignments(
                adapter_name=adapter_name,
                logical_ids_A=logical_A,
                logical_ids_B=logical_B,
            )

        if pointer != logical_ids.numel():
            raise RuntimeError("Failed to assign all balanced shared sketch codes.")

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
        self._init_shared_bank(peft_config, adapter_name)
        layer_seed = _stable_int_seed(current_key)

        if isinstance(target, Linear):
            target.update_layer(
                adapter_name=adapter_name,
                unilora_shared_sketch_bank_params=self.unilora_shared_sketch_bank_params,
                r=peft_config.r,
                bits=peft_config.bits,
                groups_per_row=peft_config.groups_per_row,
                num_banks=peft_config.num_banks,
                proj_seed=peft_config.proj_seed,
                layer_seed=layer_seed,
                unilora_dropout=peft_config.unilora_dropout,
            )
        else:
            new_module = self._create_new_module(
                peft_config=peft_config,
                unilora_shared_sketch_bank_params=self.unilora_shared_sketch_bank_params,
                adapter_name=adapter_name,
                layer_seed=layer_seed,
                target=target,
                **kwargs,
            )
            if adapter_name not in self.active_adapter:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(
        peft_config,
        unilora_shared_sketch_bank_params,
        adapter_name,
        layer_seed: int,
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
            unilora_shared_sketch_bank_params=unilora_shared_sketch_bank_params,
            adapter_name=adapter_name,
            r=peft_config.r,
            bits=peft_config.bits,
            groups_per_row=peft_config.groups_per_row,
            num_banks=peft_config.num_banks,
            proj_seed=peft_config.proj_seed,
            layer_seed=layer_seed,
            unilora_dropout=peft_config.unilora_dropout,
            **kwargs,
        )

    def get_nb_savable_parameters(self, adapter: str = "default") -> tuple[int, int]:
        trainable_params = 0
        buffer_params = 0

        for name, param in self.named_parameters():
            if "unilora_shared_sketch_bank_" in name:
                trainable_params += param.numel()

        for name, buffer in self.named_buffers():
            if "unilora_shared_sketch_bank_" in name:
                buffer_params += buffer.numel()

        return trainable_params, buffer_params

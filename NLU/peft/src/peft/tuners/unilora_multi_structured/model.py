from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

from .._buffer_dict import BufferDict
from .config import UniLoRAMultiStructuredConfig
from .layer import Linear, UniLoRAMultiStructuredLayer


class UniLoRAMultiStructuredModel(BaseTuner):
    """
    UniLoRA variant with global structured sum-of-products parameterization.
    """

    prefix: str = "unilora_multi_structured_"
    tuner_layer_cls = UniLoRAMultiStructuredLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

        cfg = config[adapter_name]
        modules = [m for m in model.modules() if isinstance(m, UniLoRAMultiStructuredLayer)]
        total_ab_params = sum(
            (m.r[adapter_name] * m.in_features) + (m.out_features * m.r[adapter_name]) for m in modules
        )

        m_hat_dim = int(math.ceil(math.sqrt(max(total_ab_params, 1))))
        if cfg.target_trainable_params is not None:
            num_hash_pairs = max(1, int(math.ceil(cfg.target_trainable_params / float(2 * m_hat_dim))))
        else:
            num_hash_pairs = cfg.num_hash_pairs

        self._init_shared_m_hat_banks(adapter_name, m_hat_dim=m_hat_dim, num_hash_pairs=num_hash_pairs, init_bound=cfg.init_bound)
        self._assign_structured_indices(model, adapter_name, m_hat_dim=m_hat_dim)

    def _init_shared_m_hat_banks(self, adapter_name: str, m_hat_dim: int, num_hash_pairs: int, init_bound: float) -> None:
        if adapter_name in self.unilora_multi_structured_left:
            return

        left = torch.empty(num_hash_pairs, m_hat_dim)
        right = torch.empty(num_hash_pairs, m_hat_dim)
        torch.nn.init.uniform_(left, -init_bound, init_bound)
        torch.nn.init.uniform_(right, -init_bound, init_bound)
        self.unilora_multi_structured_left[adapter_name] = left
        self.unilora_multi_structured_right[adapter_name] = right
        self.unilora_multi_structured_meta[adapter_name] = torch.tensor([m_hat_dim, num_hash_pairs], dtype=torch.long)

    def _assign_structured_indices(self, model: nn.Module, adapter_name: str, m_hat_dim: int) -> None:
        modules = [m for m in model.modules() if isinstance(m, UniLoRAMultiStructuredLayer)]
        pointer = 0
        cell_ids = []

        for module in modules:
            a_numel = module.m_hat_row_indices_A[adapter_name].numel()
            b_numel = module.m_hat_row_indices_B[adapter_name].numel()

            a_linear = torch.arange(pointer, pointer + a_numel, dtype=torch.long)
            pointer += a_numel
            b_linear = torch.arange(pointer, pointer + b_numel, dtype=torch.long)
            pointer += b_numel

            a_row = (a_linear // m_hat_dim).remainder(m_hat_dim).view_as(module.m_hat_row_indices_A[adapter_name])
            a_col = (a_linear % m_hat_dim).view_as(module.m_hat_col_indices_A[adapter_name])
            b_row = (b_linear // m_hat_dim).remainder(m_hat_dim).view_as(module.m_hat_row_indices_B[adapter_name])
            b_col = (b_linear % m_hat_dim).view_as(module.m_hat_col_indices_B[adapter_name])

            target_device = module.get_base_layer().weight.device
            module.m_hat_row_indices_A[adapter_name] = a_row.to(device=target_device, dtype=torch.long)
            module.m_hat_col_indices_A[adapter_name] = a_col.to(device=target_device, dtype=torch.long)
            module.m_hat_row_indices_B[adapter_name] = b_row.to(device=target_device, dtype=torch.long)
            module.m_hat_col_indices_B[adapter_name] = b_col.to(device=target_device, dtype=torch.long)
            cell_ids.extend((a_linear % (m_hat_dim * m_hat_dim)).tolist())
            cell_ids.extend((b_linear % (m_hat_dim * m_hat_dim)).tolist())

        if not cell_ids:
            return
        flat_ids = torch.tensor(cell_ids, dtype=torch.long)
        counts = torch.bincount(flat_ids, minlength=m_hat_dim * m_hat_dim).float()
        inv_sqrt = torch.zeros_like(counts)
        non_zero = counts > 0
        inv_sqrt[non_zero] = 1.0 / torch.sqrt(counts[non_zero])

        pointer = 0
        for module in modules:
            a_numel = module.m_hat_row_indices_A[adapter_name].numel()
            b_numel = module.m_hat_row_indices_B[adapter_name].numel()
            scale_a = inv_sqrt[flat_ids[pointer : pointer + a_numel]].view_as(module.m_hat_row_indices_A[adapter_name])
            pointer += a_numel
            scale_b = inv_sqrt[flat_ids[pointer : pointer + b_numel]].view_as(module.m_hat_row_indices_B[adapter_name])
            pointer += b_numel
            module.update_norm(adapter_name, scale_a, scale_b)

    def _pre_injection_hook(self, model: nn.Module, config: UniLoRAMultiStructuredConfig, adapter_name: str) -> None:
        self.unilora_multi_structured_left = nn.ParameterDict({})
        self.unilora_multi_structured_right = nn.ParameterDict({})
        self.unilora_multi_structured_meta = BufferDict({}, persistent=True)

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

        m_hat_dim = int(self.unilora_multi_structured_meta[adapter_name][0].item()) if adapter_name in self.unilora_multi_structured_meta else 1
        if isinstance(target, Linear):
            target.update_layer(
                adapter_name=adapter_name,
                unilora_multi_structured_left=self.unilora_multi_structured_left,
                unilora_multi_structured_right=self.unilora_multi_structured_right,
                r=unilora_config.r,
                m_hat_dim=m_hat_dim,
                unilora_dropout=unilora_config.unilora_dropout,
                layerwise_learnable_scale=unilora_config.layerwise_learnable_scale,
            )
        else:
            new_module = self._create_new_module(
                unilora_config=unilora_config,
                unilora_multi_structured_left=self.unilora_multi_structured_left,
                unilora_multi_structured_right=self.unilora_multi_structured_right,
                adapter_name=adapter_name,
                target=target,
                m_hat_dim=m_hat_dim,
                **kwargs,
            )
            if adapter_name not in self.active_adapter:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(
        unilora_config,
        unilora_multi_structured_left,
        unilora_multi_structured_right,
        adapter_name,
        target,
        m_hat_dim,
        **kwargs,
    ):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = unilora_config.fan_in_fan_out = False
        elif isinstance(target_base_layer, Conv1D):
            kwargs["is_target_conv_1d_layer"] = True
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = unilora_config.fan_in_fan_out = True
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only `torch.nn.Linear` and `transformers.pytorch_utils.Conv1D`."
            )

        return Linear(
            base_layer=target,
            unilora_multi_structured_left=unilora_multi_structured_left,
            unilora_multi_structured_right=unilora_multi_structured_right,
            adapter_name=adapter_name,
            r=unilora_config.r,
            m_hat_dim=m_hat_dim,
            unilora_dropout=unilora_config.unilora_dropout,
            layerwise_learnable_scale=unilora_config.layerwise_learnable_scale,
            **kwargs,
        )

    def get_nb_savable_parameters(self, adapter="default") -> tuple[int, int]:
        structured_params = 0
        other_params = 0
        for name, param in self.named_parameters():
            if "unilora_multi_structured_left" in name or "unilora_multi_structured_right" in name:
                structured_params += param.numel()
            elif "unilora_multi_structured_layer_scale" in name:
                structured_params += param.numel()
            elif "m_hat_" in name:
                other_params += param.numel()

        for name, buffer in self.named_buffers():
            if "m_hat_" in name:
                other_params += buffer.numel()

        return structured_params, other_params

    def print_savable_parameters(self) -> None:
        structured_params, other_params = self.get_nb_savable_parameters()
        print(
            f"UniLoRA-multi-structured params to-be-saved (float32-equivalent): {structured_params:,d} "
            f"|| total params to-be-saved: {(structured_params + other_params):,d}"
        )

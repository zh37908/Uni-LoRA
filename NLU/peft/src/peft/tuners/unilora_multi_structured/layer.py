from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose

from .._buffer_dict import BufferDict


class UniLoRAMultiStructuredLayer(BaseTunerLayer):
    adapter_layer_names = (
        "unilora_multi_structured_left",
        "unilora_multi_structured_right",
        "unilora_multi_structured_layer_scale",
    )
    other_param_names = (
        "r",
        "m_hat_row_indices_A",
        "m_hat_col_indices_A",
        "m_hat_row_indices_B",
        "m_hat_col_indices_B",
        "m_hat_scales_A",
        "m_hat_scales_B",
    )

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.unilora_dropout = nn.ModuleDict({})

        self.m_hat_row_indices_A = BufferDict({}, persistent=True)
        self.m_hat_col_indices_A = BufferDict({}, persistent=True)
        self.m_hat_row_indices_B = BufferDict({}, persistent=True)
        self.m_hat_col_indices_B = BufferDict({}, persistent=True)
        self.m_hat_scales_A = BufferDict({}, persistent=True)
        self.m_hat_scales_B = BufferDict({}, persistent=True)
        self.unilora_multi_structured_layer_scale = nn.ParameterDict({})

        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        else:
            raise ValueError(f"Unsupported base layer type {type(base_layer)}.")

        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def update_layer(
        self,
        adapter_name: str,
        unilora_multi_structured_left,
        unilora_multi_structured_right,
        r: int,
        m_hat_dim: int,
        unilora_dropout: float = 0.0,
        layerwise_learnable_scale: bool = True,
    ):
        if r <= 0:
            raise ValueError(f"`r` {r} should be a positive integer value")

        self.r[adapter_name] = r
        self.unilora_dropout.update(
            nn.ModuleDict({adapter_name: nn.Dropout(p=unilora_dropout) if unilora_dropout > 0.0 else nn.Identity()})
        )

        self.unilora_multi_structured_left = unilora_multi_structured_left
        self.unilora_multi_structured_right = unilora_multi_structured_right
        if adapter_name not in self.unilora_multi_structured_layer_scale:
            self.unilora_multi_structured_layer_scale[adapter_name] = nn.Parameter(
                torch.ones(1), requires_grad=layerwise_learnable_scale
            )

        self.reset_m_hat_parameters(adapter_name, m_hat_dim)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_m_hat_parameters(self, adapter_name: str, m_hat_dim: int):
        self.m_hat_row_indices_A[adapter_name] = torch.randint(
            0, m_hat_dim, (self.r[adapter_name], self.in_features), dtype=torch.long
        )
        self.m_hat_col_indices_A[adapter_name] = torch.randint(
            0, m_hat_dim, (self.r[adapter_name], self.in_features), dtype=torch.long
        )
        self.m_hat_row_indices_B[adapter_name] = torch.randint(
            0, m_hat_dim, (self.out_features, self.r[adapter_name]), dtype=torch.long
        )
        self.m_hat_col_indices_B[adapter_name] = torch.randint(
            0, m_hat_dim, (self.out_features, self.r[adapter_name]), dtype=torch.long
        )

    def update_norm(self, adapter_name: str, scales_a: torch.Tensor, scales_b: torch.Tensor):
        if adapter_name not in self.unilora_multi_structured_left:
            return
        base_layer = self.get_base_layer()
        target_device = base_layer.weight.device
        target_dtype = base_layer.weight.dtype
        self.m_hat_scales_A[adapter_name] = scales_a.to(device=target_device, dtype=target_dtype)
        self.m_hat_scales_B[adapter_name] = scales_b.to(device=target_device, dtype=target_dtype)


class Linear(nn.Linear, UniLoRAMultiStructuredLayer):
    def __init__(
        self,
        base_layer,
        unilora_multi_structured_left,
        unilora_multi_structured_right,
        adapter_name: str,
        r: int,
        m_hat_dim: int,
        unilora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ) -> None:
        super(nn.Linear, self).__init__()
        UniLoRAMultiStructuredLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name=adapter_name,
            unilora_multi_structured_left=unilora_multi_structured_left,
            unilora_multi_structured_right=unilora_multi_structured_right,
            r=r,
            m_hat_dim=m_hat_dim,
            unilora_dropout=unilora_dropout,
            layerwise_learnable_scale=kwargs.get("layerwise_learnable_scale", True),
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter in self.m_hat_row_indices_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += self.get_delta_weight(active_adapter)
                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )
                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.m_hat_row_indices_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def _reconstruct_from_m_hat(self, row_indices: torch.Tensor, col_indices: torch.Tensor, adapter: str) -> torch.Tensor:
        left = self.unilora_multi_structured_left[adapter]
        right = self.unilora_multi_structured_right[adapter]
        # [M, ...] * [M, ...] -> sum_M => [...]
        return (left[:, row_indices] * right[:, col_indices]).sum(dim=0)

    def _get_lora_matrices(self, adapter: str, cast_to_fp32: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        row_a = self.m_hat_row_indices_A[adapter].long()
        col_a = self.m_hat_col_indices_A[adapter].long()
        row_b = self.m_hat_row_indices_B[adapter].long()
        col_b = self.m_hat_col_indices_B[adapter].long()

        if cast_to_fp32:
            left = self.unilora_multi_structured_left[adapter].float()
            right = self.unilora_multi_structured_right[adapter].float()
            a = (left[:, row_a] * right[:, col_a]).sum(dim=0)
            b = (left[:, row_b] * right[:, col_b]).sum(dim=0)
        else:
            a = self._reconstruct_from_m_hat(row_a, col_a, adapter)
            b = self._reconstruct_from_m_hat(row_b, col_b, adapter)

        a = a * self.m_hat_scales_A[adapter]
        b = b * self.m_hat_scales_B[adapter]
        return a, b

    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        device = self.m_hat_row_indices_A[adapter].device
        dtype = self.unilora_multi_structured_left[adapter].dtype
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        a, b = self._get_lora_matrices(adapter, cast_to_fp32=cast_to_fp32)
        delta = transpose(b @ a, self.fan_in_fan_out)
        return delta * self.unilora_multi_structured_layer_scale[adapter]

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.m_hat_row_indices_A.keys():
                    continue
                a, b = self._get_lora_matrices(active_adapter)
                x_cast = x.to(self.unilora_multi_structured_left[active_adapter].dtype)
                dropout = self.unilora_dropout[active_adapter]
                delta = F.linear(F.linear(dropout(x_cast), a), b)
                result = result + delta * self.unilora_multi_structured_layer_scale[active_adapter]
        return result.to(previous_dtype)

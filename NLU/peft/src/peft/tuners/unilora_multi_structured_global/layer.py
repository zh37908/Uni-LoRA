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


class UniLoRAMultiStructuredGlobalLayer(BaseTunerLayer):
    adapter_layer_names = (
        "unilora_multi_structured_global_u",
        "unilora_multi_structured_global_v",
        "unilora_multi_structured_global_layer_scale",
    )
    other_param_names = (
        "r",
        "global_linear_indices_A",
        "global_linear_indices_B",
        "global_scales_A",
        "global_scales_B",
    )

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.global_matrix_dim = {}
        self.unilora_dropout = nn.ModuleDict({})

        self.global_linear_indices_A = BufferDict({}, persistent=True)
        self.global_linear_indices_B = BufferDict({}, persistent=True)
        self.global_scales_A = BufferDict({}, persistent=True)
        self.global_scales_B = BufferDict({}, persistent=True)
        self.unilora_multi_structured_global_layer_scale = nn.ParameterDict({})

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
        unilora_multi_structured_global_u,
        unilora_multi_structured_global_v,
        r: int,
        global_matrix_dim: int,
        unilora_dropout: float = 0.0,
        layerwise_learnable_scale: bool = True,
    ):
        if r <= 0:
            raise ValueError(f"`r` {r} should be a positive integer value")

        self.r[adapter_name] = r
        self.global_matrix_dim[adapter_name] = global_matrix_dim
        self.unilora_dropout.update(
            nn.ModuleDict({adapter_name: nn.Dropout(p=unilora_dropout) if unilora_dropout > 0.0 else nn.Identity()})
        )

        self.unilora_multi_structured_global_u = unilora_multi_structured_global_u
        self.unilora_multi_structured_global_v = unilora_multi_structured_global_v
        if adapter_name not in self.unilora_multi_structured_global_layer_scale:
            self.unilora_multi_structured_global_layer_scale[adapter_name] = nn.Parameter(
                torch.ones(1), requires_grad=layerwise_learnable_scale
            )

        self.reset_global_positions(adapter_name)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_global_positions(self, adapter_name: str):
        self.global_linear_indices_A[adapter_name] = torch.zeros(
            (self.r[adapter_name], self.in_features), dtype=torch.long
        )
        self.global_linear_indices_B[adapter_name] = torch.zeros(
            (self.out_features, self.r[adapter_name]), dtype=torch.long
        )

    def update_norm(self, adapter_name: str, scales_a: torch.Tensor, scales_b: torch.Tensor):
        if adapter_name not in self.unilora_multi_structured_global_u:
            return
        base_layer = self.get_base_layer()
        target_device = base_layer.weight.device
        target_dtype = base_layer.weight.dtype
        self.global_scales_A[adapter_name] = scales_a.to(device=target_device, dtype=target_dtype)
        self.global_scales_B[adapter_name] = scales_b.to(device=target_device, dtype=target_dtype)


class Linear(nn.Linear, UniLoRAMultiStructuredGlobalLayer):
    def __init__(
        self,
        base_layer,
        unilora_multi_structured_global_u,
        unilora_multi_structured_global_v,
        adapter_name: str,
        r: int,
        global_matrix_dim: int,
        unilora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ) -> None:
        super(nn.Linear, self).__init__()
        UniLoRAMultiStructuredGlobalLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name=adapter_name,
            unilora_multi_structured_global_u=unilora_multi_structured_global_u,
            unilora_multi_structured_global_v=unilora_multi_structured_global_v,
            r=r,
            global_matrix_dim=global_matrix_dim,
            unilora_dropout=unilora_dropout,
            layerwise_learnable_scale=kwargs.get("layerwise_learnable_scale", True),
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter in self.global_linear_indices_A.keys():
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
            if active_adapter in self.global_linear_indices_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def _values_from_global_positions(
        self, positions: torch.Tensor, adapter: str, cast_to_fp32: bool = False
    ) -> torch.Tensor:
        matrix_dim = self.global_matrix_dim[adapter]
        rows = torch.div(positions, matrix_dim, rounding_mode="floor")
        cols = torch.remainder(positions, matrix_dim)

        if cast_to_fp32:
            u = self.unilora_multi_structured_global_u[adapter].float()
            v = self.unilora_multi_structured_global_v[adapter].float()
        else:
            u = self.unilora_multi_structured_global_u[adapter]
            v = self.unilora_multi_structured_global_v[adapter]

        return (u[:, rows] * v[:, cols]).sum(dim=0)

    def _get_lora_matrices(self, adapter: str, cast_to_fp32: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        pos_a = self.global_linear_indices_A[adapter].long()
        pos_b = self.global_linear_indices_B[adapter].long()
        a = self._values_from_global_positions(pos_a, adapter, cast_to_fp32=cast_to_fp32)
        b = self._values_from_global_positions(pos_b, adapter, cast_to_fp32=cast_to_fp32)
        a = a * self.global_scales_A[adapter]
        b = b * self.global_scales_B[adapter]
        return a, b

    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        device = self.global_linear_indices_A[adapter].device
        dtype = self.unilora_multi_structured_global_u[adapter].dtype
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        a, b = self._get_lora_matrices(adapter, cast_to_fp32=cast_to_fp32)
        delta = transpose(b @ a, self.fan_in_fan_out)
        return delta * self.unilora_multi_structured_global_layer_scale[adapter]

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
                if active_adapter not in self.global_linear_indices_A.keys():
                    continue
                a, b = self._get_lora_matrices(active_adapter)
                x_cast = x.to(self.unilora_multi_structured_global_u[active_adapter].dtype)
                dropout = self.unilora_dropout[active_adapter]
                delta = F.linear(F.linear(dropout(x_cast), a), b)
                result = result + delta * self.unilora_multi_structured_global_layer_scale[active_adapter]
        return result.to(previous_dtype)

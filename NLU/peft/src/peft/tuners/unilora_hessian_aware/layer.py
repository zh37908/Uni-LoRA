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


class UniLoRAHessianAwareLayer(BaseTunerLayer):
    adapter_layer_names = ("unilora_hessian_aware_theta_d",)
    other_param_names = (
        "unilora_indices_A",
        "unilora_indices_B",
        "unilora_scales_A",
        "unilora_scales_B",
        "unilora_curvature_A",
        "unilora_curvature_B",
    )

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.unilora_dropout = nn.ModuleDict({})

        self.unilora_indices_A = BufferDict({}, persistent=True)
        self.unilora_indices_B = BufferDict({}, persistent=True)
        self.unilora_scales_A = BufferDict({}, persistent=True)
        self.unilora_scales_B = BufferDict({}, persistent=True)
        self.unilora_curvature_A = BufferDict({}, persistent=True)
        self.unilora_curvature_B = BufferDict({}, persistent=True)

        self._disable_adapters = False
        self.merged_adapters = []
        self.capture_curvature_stats = False
        self._last_A = {}
        self._last_B = {}

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        else:
            raise ValueError(f"Unsupported base layer type {type(base_layer)} for UniLoRAHessianAwareLayer.")

        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def update_layer(
        self,
        adapter_name: str,
        unilora_hessian_aware_theta_d,
        r: int,
        theta_d_length: int,
        unilora_dropout: float = 0.0,
    ):
        if r <= 0:
            raise ValueError(f"`r` {r} should be a positive integer value")

        self.r[adapter_name] = r

        if unilora_dropout > 0.0:
            unilora_dropout_layer = nn.Dropout(p=unilora_dropout)
        else:
            unilora_dropout_layer = nn.Identity()
        self.unilora_dropout.update(nn.ModuleDict({adapter_name: unilora_dropout_layer}))

        self.unilora_hessian_aware_theta_d = unilora_hessian_aware_theta_d
        self.reset_unilora_parameters(adapter_name, theta_d_length)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_unilora_parameters(self, adapter_name: str, theta_d_length: int):
        if adapter_name not in self.unilora_hessian_aware_theta_d.keys():
            return

        indices_A = torch.randint(0, theta_d_length, (self.r[adapter_name], self.in_features), dtype=torch.long)
        indices_B = torch.randint(0, theta_d_length, (self.out_features, self.r[adapter_name]), dtype=torch.long)

        self.unilora_indices_A[adapter_name] = indices_A
        self.unilora_indices_B[adapter_name] = indices_B
        self.unilora_curvature_A[adapter_name] = torch.ones_like(indices_A, dtype=torch.float32)
        self.unilora_curvature_B[adapter_name] = torch.ones_like(indices_B, dtype=torch.float32)

    def update_norm(self, adapter_name: str, unilora_scales_A, unilora_scales_B):
        if adapter_name in self.unilora_hessian_aware_theta_d.keys():
            base_layer = self.get_base_layer()
            target_device = base_layer.weight.device
            target_dtype = base_layer.weight.dtype

            self.unilora_scales_A[adapter_name] = unilora_scales_A.to(device=target_device, dtype=target_dtype)
            self.unilora_scales_B[adapter_name] = unilora_scales_B.to(device=target_device, dtype=target_dtype)

    def set_capture_curvature(self, enabled: bool = True) -> None:
        self.capture_curvature_stats = enabled
        if not enabled:
            self._last_A.clear()
            self._last_B.clear()

    def accumulate_curvature_statistics(self, adapter_name: str, ema_momentum: float) -> int:
        updated = 0

        if adapter_name in self._last_A and self._last_A[adapter_name] is not None and self._last_A[adapter_name].grad is not None:
            grad_A = self._last_A[adapter_name].grad.detach().to(self.unilora_curvature_A[adapter_name].device, dtype=torch.float32)
            self.unilora_curvature_A[adapter_name].mul_(ema_momentum).add_(grad_A.square(), alpha=1.0 - ema_momentum)
            updated += 1

        if adapter_name in self._last_B and self._last_B[adapter_name] is not None and self._last_B[adapter_name].grad is not None:
            grad_B = self._last_B[adapter_name].grad.detach().to(self.unilora_curvature_B[adapter_name].device, dtype=torch.float32)
            self.unilora_curvature_B[adapter_name].mul_(ema_momentum).add_(grad_B.square(), alpha=1.0 - ema_momentum)
            updated += 1

        self._last_A[adapter_name] = None
        self._last_B[adapter_name] = None
        return updated

    def get_structure_state(self, adapter_name: str) -> dict[str, torch.Tensor]:
        A, B = self._get_lora_matrices(adapter_name, cast_to_fp32=True)
        return {
            "values_A": A.detach().cpu().reshape(-1).to(torch.float32),
            "values_B": B.detach().cpu().reshape(-1).to(torch.float32),
            "indices_A": self.unilora_indices_A[adapter_name].detach().cpu().reshape(-1).long(),
            "indices_B": self.unilora_indices_B[adapter_name].detach().cpu().reshape(-1).long(),
            "curvature_A": self.unilora_curvature_A[adapter_name].detach().cpu().reshape(-1).clamp_min(1e-8),
            "curvature_B": self.unilora_curvature_B[adapter_name].detach().cpu().reshape(-1).clamp_min(1e-8),
            "shape_A": tuple(self.unilora_indices_A[adapter_name].shape),
            "shape_B": tuple(self.unilora_indices_B[adapter_name].shape),
        }


class Linear(nn.Linear, UniLoRAHessianAwareLayer):
    def __init__(
        self,
        base_layer,
        unilora_hessian_aware_theta_d,
        adapter_name: str,
        r: int,
        theta_d_length: int,
        unilora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ) -> None:
        super(nn.Linear, self).__init__()
        UniLoRAHessianAwareLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, unilora_hessian_aware_theta_d, r, theta_d_length, unilora_dropout)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter in self.unilora_indices_A.keys():
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
            if active_adapter in self.unilora_indices_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def _get_lora_matrices(self, adapter, cast_to_fp32: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        indices_A = self.unilora_indices_A[adapter]
        indices_B = self.unilora_indices_B[adapter]
        theta_d = self.unilora_hessian_aware_theta_d[adapter].to(indices_A.device)
        scales_A = self.unilora_scales_A[adapter]
        scales_B = self.unilora_scales_B[adapter]

        if cast_to_fp32:
            theta_d = theta_d.float()
            scales_A = scales_A.float()
            scales_B = scales_B.float()

        A = theta_d[indices_A.long()] * scales_A
        B = theta_d[indices_B.long()] * scales_B

        if self.capture_curvature_stats and self.training and not cast_to_fp32:
            self._last_A[adapter] = A if A.requires_grad else None
            self._last_B[adapter] = B if B.requires_grad else None
            if A.requires_grad:
                A.retain_grad()
            if B.requires_grad:
                B.retain_grad()

        return A, B

    def get_delta_weight(self, adapter) -> torch.Tensor:
        device = self.unilora_indices_A[adapter].device
        dtype = self.unilora_hessian_aware_theta_d[adapter].dtype
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16
        A, B = self._get_lora_matrices(adapter, cast_to_fp32)
        output_tensor = transpose(B @ A, self.fan_in_fan_out)
        return output_tensor

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
                if active_adapter not in self.unilora_indices_A.keys():
                    continue
                A, B = self._get_lora_matrices(active_adapter)
                x_cast = x.to(self.unilora_hessian_aware_theta_d[active_adapter].dtype)
                dropout = self.unilora_dropout[active_adapter]
                result = result + F.linear(F.linear(dropout(x_cast), A), B)
        return result.to(previous_dtype)

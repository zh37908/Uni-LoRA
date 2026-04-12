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


class UniLoRARoSALayer(BaseTunerLayer):
    adapter_layer_names = ("unilora_rosa_theta_d", "unilora_rosa_sparse_theta_D")
    other_param_names = (
        "r",
        "unilora_indices_A",
        "unilora_indices_B",
        "unilora_scales_A",
        "unilora_scales_B",
        "unilora_theta_D_offsets_A",
        "unilora_theta_D_offsets_B",
    )

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.unilora_dropout = nn.ModuleDict({})

        self.unilora_indices_A = BufferDict({}, persistent=True)
        self.unilora_indices_B = BufferDict({}, persistent=True)
        self.unilora_scales_A = BufferDict({}, persistent=True)
        self.unilora_scales_B = BufferDict({}, persistent=True)
        self.unilora_theta_D_offsets_A = BufferDict({}, persistent=True)
        self.unilora_theta_D_offsets_B = BufferDict({}, persistent=True)

        self._disable_adapters = False
        self.merged_adapters = []
        self.capture_gradient_stats = False
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
            raise ValueError(f"Unsupported base layer type {type(base_layer)} for UniLoRARoSALayer.")

        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def update_layer(
        self,
        adapter_name: str,
        unilora_rosa_theta_d,
        unilora_rosa_sparse_theta_D,
        unilora_rosa_sparse_mask,
        unilora_rosa_grad_accum,
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

        self.unilora_rosa_theta_d = unilora_rosa_theta_d
        self.unilora_rosa_sparse_theta_D = unilora_rosa_sparse_theta_D
        self.unilora_rosa_sparse_mask = unilora_rosa_sparse_mask
        self.unilora_rosa_grad_accum = unilora_rosa_grad_accum

        self.reset_unilora_parameters(adapter_name, theta_d_length)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_sparse_requires_grad(adapter_name, False)
        self.set_adapter(self.active_adapters)

    def reset_unilora_parameters(self, adapter_name: str, theta_d_length: int):
        if adapter_name not in self.unilora_rosa_theta_d.keys():
            return

        indices_A = torch.randint(0, theta_d_length, (self.r[adapter_name], self.in_features), dtype=torch.long)
        indices_B = torch.randint(0, theta_d_length, (self.out_features, self.r[adapter_name]), dtype=torch.long)

        self.unilora_indices_A[adapter_name] = indices_A
        self.unilora_indices_B[adapter_name] = indices_B

    def update_norm(self, adapter_name: str, unilora_scales_A, unilora_scales_B):
        if adapter_name in self.unilora_rosa_theta_d.keys():
            base_layer = self.get_base_layer()
            target_device = base_layer.weight.device
            target_dtype = base_layer.weight.dtype

            self.unilora_scales_A[adapter_name] = unilora_scales_A.to(device=target_device, dtype=target_dtype)
            self.unilora_scales_B[adapter_name] = unilora_scales_B.to(device=target_device, dtype=target_dtype)

    def set_capture_gradient(self, enabled: bool = True) -> None:
        self.capture_gradient_stats = enabled
        if not enabled:
            self._last_A.clear()
            self._last_B.clear()

    def set_sparse_requires_grad(self, adapter_name: str, requires_grad: bool) -> None:
        if adapter_name in self.unilora_rosa_sparse_theta_D:
            self.unilora_rosa_sparse_theta_D[adapter_name].requires_grad_(requires_grad)

    def clear_cached_gradients(self, adapter_name: str) -> None:
        self._last_A[adapter_name] = None
        self._last_B[adapter_name] = None

    def has_sparse_mask(self, adapter_name: str) -> bool:
        if adapter_name not in self.unilora_rosa_sparse_mask:
            return False
        return bool(self.unilora_rosa_sparse_mask[adapter_name].any().item())

    def accumulate_gradient_statistics(self, adapter_name: str) -> int:
        updated = 0
        score = self.unilora_rosa_grad_accum[adapter_name]

        if adapter_name in self._last_A and self._last_A[adapter_name] is not None:
            grad_A = self._last_A[adapter_name].grad
            if grad_A is not None:
                offsets_A = self.unilora_theta_D_offsets_A[adapter_name].reshape(-1)
                grad_A = grad_A.detach().abs().reshape(-1).to(device=score.device, dtype=score.dtype)
                score[offsets_A] = torch.maximum(score[offsets_A], grad_A)
                updated += 1

        if adapter_name in self._last_B and self._last_B[adapter_name] is not None:
            grad_B = self._last_B[adapter_name].grad
            if grad_B is not None:
                offsets_B = self.unilora_theta_D_offsets_B[adapter_name].reshape(-1)
                grad_B = grad_B.detach().abs().reshape(-1).to(device=score.device, dtype=score.dtype)
                score[offsets_B] = torch.maximum(score[offsets_B], grad_B)
                updated += 1

        self.clear_cached_gradients(adapter_name)
        return updated


class Linear(nn.Linear, UniLoRARoSALayer):
    def __init__(
        self,
        base_layer,
        unilora_rosa_theta_d,
        unilora_rosa_sparse_theta_D,
        unilora_rosa_sparse_mask,
        unilora_rosa_grad_accum,
        adapter_name: str,
        r: int,
        theta_d_length: int,
        unilora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ) -> None:
        super(nn.Linear, self).__init__()
        UniLoRARoSALayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            unilora_rosa_theta_d,
            unilora_rosa_sparse_theta_D,
            unilora_rosa_sparse_mask,
            unilora_rosa_grad_accum,
            r,
            theta_d_length,
            unilora_dropout,
        )
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

    def _get_lora_matrices(self, adapter: str, cast_to_fp32: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        indices_A = self.unilora_indices_A[adapter]
        indices_B = self.unilora_indices_B[adapter]
        theta_d = self.unilora_rosa_theta_d[adapter].to(indices_A.device)
        sparse_theta_D = self.unilora_rosa_sparse_theta_D[adapter].to(indices_A.device)
        sparse_mask = self.unilora_rosa_sparse_mask[adapter].to(indices_A.device)
        scales_A = self.unilora_scales_A[adapter]
        scales_B = self.unilora_scales_B[adapter]
        offsets_A = self.unilora_theta_D_offsets_A[adapter]
        offsets_B = self.unilora_theta_D_offsets_B[adapter]

        if cast_to_fp32:
            theta_d = theta_d.float()
            sparse_theta_D = sparse_theta_D.float()
            scales_A = scales_A.float()
            scales_B = scales_B.float()

        base_A = theta_d[indices_A.long()] * scales_A
        base_B = theta_d[indices_B.long()] * scales_B

        sparse_A = sparse_theta_D[offsets_A.long()] * sparse_mask[offsets_A.long()].to(dtype=sparse_theta_D.dtype)
        sparse_B = sparse_theta_D[offsets_B.long()] * sparse_mask[offsets_B.long()].to(dtype=sparse_theta_D.dtype)

        A = base_A + sparse_A
        B = base_B + sparse_B

        if self.capture_gradient_stats and self.training and not cast_to_fp32:
            self._last_A[adapter] = A if A.requires_grad else None
            self._last_B[adapter] = B if B.requires_grad else None
            if A.requires_grad:
                A.retain_grad()
            if B.requires_grad:
                B.retain_grad()

        return A, B

    def get_delta_weight(self, adapter) -> torch.Tensor:
        device = self.unilora_indices_A[adapter].device
        dtype = self.unilora_rosa_theta_d[adapter].dtype
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16
        A, B = self._get_lora_matrices(adapter, cast_to_fp32)
        return transpose(B @ A, self.fan_in_fan_out)

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
                x_cast = x.to(self.unilora_rosa_theta_d[active_adapter].dtype)
                dropout = self.unilora_dropout[active_adapter]
                result = result + F.linear(F.linear(dropout(x_cast), A), B)

        return result.to(previous_dtype)

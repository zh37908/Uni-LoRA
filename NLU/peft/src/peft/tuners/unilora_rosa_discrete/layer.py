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


class UniLoRARoSADiscreteLayer(BaseTunerLayer):
    adapter_layer_names = ("unilora_rosa_discrete_theta_d", "unilora_rosa_discrete_sparse_theta_d")
    other_param_names = (
        "r",
        "unilora_rosa_discrete_indices_A",
        "unilora_rosa_discrete_indices_B",
        "unilora_rosa_discrete_indices_S",
        "unilora_rosa_discrete_scales_A",
        "unilora_rosa_discrete_scales_B",
        "unilora_rosa_discrete_scales_S",
        "unilora_rosa_discrete_sparse_offsets",
    )

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.unilora_dropout = nn.ModuleDict({})

        self.unilora_rosa_discrete_indices_A = BufferDict({}, persistent=True)
        self.unilora_rosa_discrete_indices_B = BufferDict({}, persistent=True)
        self.unilora_rosa_discrete_indices_S = BufferDict({}, persistent=True)

        self.unilora_rosa_discrete_scales_A = BufferDict({}, persistent=True)
        self.unilora_rosa_discrete_scales_B = BufferDict({}, persistent=True)
        self.unilora_rosa_discrete_scales_S = BufferDict({}, persistent=True)

        self.unilora_rosa_discrete_sparse_offsets = BufferDict({}, persistent=True)

        self._disable_adapters = False
        self.merged_adapters = []
        self.capture_sparse_gradient_stats = False
        self._last_sparse_probe = {}

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        else:
            raise ValueError(f"Unsupported base layer type {type(base_layer)} for UniLoRARoSADiscreteLayer.")

        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def update_layer(
        self,
        adapter_name: str,
        unilora_rosa_discrete_theta_d,
        unilora_rosa_discrete_sparse_theta_d,
        unilora_rosa_discrete_sparse_mask,
        unilora_rosa_discrete_grad_accum,
        r: int,
        theta_d_length: int,
        sparse_theta_d_length: int,
        unilora_dropout: float = 0.0,
    ):
        if r <= 0:
            raise ValueError(f"`r` {r} should be a positive integer value")

        self.r[adapter_name] = r
        dropout_layer = nn.Dropout(p=unilora_dropout) if unilora_dropout > 0.0 else nn.Identity()
        self.unilora_dropout.update(nn.ModuleDict({adapter_name: dropout_layer}))

        self.unilora_rosa_discrete_theta_d = unilora_rosa_discrete_theta_d
        self.unilora_rosa_discrete_sparse_theta_d = unilora_rosa_discrete_sparse_theta_d
        self.unilora_rosa_discrete_sparse_mask = unilora_rosa_discrete_sparse_mask
        self.unilora_rosa_discrete_grad_accum = unilora_rosa_discrete_grad_accum

        self.reset_unilora_parameters(adapter_name, theta_d_length, sparse_theta_d_length)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_unilora_parameters(self, adapter_name: str, theta_d_length: int, sparse_theta_d_length: int):
        if adapter_name not in self.unilora_rosa_discrete_theta_d:
            return
        self.unilora_rosa_discrete_indices_A[adapter_name] = torch.randint(
            0, theta_d_length, (self.r[adapter_name], self.in_features), dtype=torch.long
        )
        self.unilora_rosa_discrete_indices_B[adapter_name] = torch.randint(
            0, theta_d_length, (self.out_features, self.r[adapter_name]), dtype=torch.long
        )
        self.unilora_rosa_discrete_indices_S[adapter_name] = torch.randint(
            0, sparse_theta_d_length, (self.out_features, self.in_features), dtype=torch.long
        )

    def update_norm(
        self,
        adapter_name: str,
        scales_a: torch.Tensor,
        scales_b: torch.Tensor,
        scales_s: torch.Tensor,
    ):
        if adapter_name not in self.unilora_rosa_discrete_theta_d:
            return
        base_layer = self.get_base_layer()
        target_device = base_layer.weight.device
        target_dtype = base_layer.weight.dtype
        self.unilora_rosa_discrete_scales_A[adapter_name] = scales_a.to(device=target_device, dtype=target_dtype)
        self.unilora_rosa_discrete_scales_B[adapter_name] = scales_b.to(device=target_device, dtype=target_dtype)
        self.unilora_rosa_discrete_scales_S[adapter_name] = scales_s.to(device=target_device, dtype=target_dtype)

    def set_capture_gradient(self, enabled: bool = True) -> None:
        self.capture_sparse_gradient_stats = enabled
        if not enabled:
            self._last_sparse_probe.clear()

    def clear_cached_gradients(self, adapter_name: str) -> None:
        self._last_sparse_probe[adapter_name] = None

    def has_sparse_mask(self, adapter_name: str) -> bool:
        if adapter_name not in self.unilora_rosa_discrete_sparse_mask:
            return False
        return bool(self.unilora_rosa_discrete_sparse_mask[adapter_name].any().item())

    def accumulate_gradient_statistics(self, adapter_name: str) -> int:
        probe = self._last_sparse_probe.get(adapter_name)
        if probe is None or probe.grad is None:
            self.clear_cached_gradients(adapter_name)
            return 0
        score = self.unilora_rosa_discrete_grad_accum[adapter_name]
        offsets = self.unilora_rosa_discrete_sparse_offsets[adapter_name].reshape(-1)
        grad = probe.grad.detach().abs().reshape(-1).to(device=score.device, dtype=score.dtype)
        score[offsets] = torch.maximum(score[offsets], grad)
        self.clear_cached_gradients(adapter_name)
        return 1


class Linear(nn.Linear, UniLoRARoSADiscreteLayer):
    def __init__(
        self,
        base_layer,
        unilora_rosa_discrete_theta_d,
        unilora_rosa_discrete_sparse_theta_d,
        unilora_rosa_discrete_sparse_mask,
        unilora_rosa_discrete_grad_accum,
        adapter_name: str,
        r: int,
        theta_d_length: int,
        sparse_theta_d_length: int,
        unilora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ) -> None:
        super(nn.Linear, self).__init__()
        UniLoRARoSADiscreteLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            unilora_rosa_discrete_theta_d,
            unilora_rosa_discrete_sparse_theta_d,
            unilora_rosa_discrete_sparse_mask,
            unilora_rosa_discrete_grad_accum,
            r,
            theta_d_length,
            sparse_theta_d_length,
            unilora_dropout,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return
        for active_adapter in adapter_names:
            if active_adapter in self.unilora_rosa_discrete_indices_A:
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
            if active_adapter in self.unilora_rosa_discrete_indices_A:
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def _get_lora_bank(self, adapter: str, device: torch.device, cast_to_fp32: bool = False) -> torch.Tensor:
        bank = self.unilora_rosa_discrete_theta_d[adapter].to(device)
        return bank.float() if cast_to_fp32 else bank

    def _get_sparse_bank(self, adapter: str, device: torch.device, cast_to_fp32: bool = False) -> torch.Tensor:
        bank = self.unilora_rosa_discrete_sparse_theta_d[adapter].to(device)
        return bank.float() if cast_to_fp32 else bank

    def _get_lora_matrices(self, adapter: str, cast_to_fp32: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        indices_a = self.unilora_rosa_discrete_indices_A[adapter]
        indices_b = self.unilora_rosa_discrete_indices_B[adapter]
        scales_a = self.unilora_rosa_discrete_scales_A[adapter]
        scales_b = self.unilora_rosa_discrete_scales_B[adapter]
        if cast_to_fp32:
            scales_a = scales_a.float()
            scales_b = scales_b.float()
        bank = self._get_lora_bank(adapter, indices_a.device, cast_to_fp32=cast_to_fp32)
        a = bank[indices_a.long()] * scales_a
        b = bank[indices_b.long()] * scales_b
        return a, b

    def _get_sparse_matrix(
        self,
        adapter: str,
        cast_to_fp32: bool = False,
        collect_dense_gradients: bool = False,
    ) -> torch.Tensor:
        indices_s = self.unilora_rosa_discrete_indices_S[adapter]
        scales_s = self.unilora_rosa_discrete_scales_S[adapter]
        if cast_to_fp32:
            scales_s = scales_s.float()

        if collect_dense_gradients:
            bank = self._get_sparse_bank(adapter, indices_s.device, cast_to_fp32=True).detach()
            sparse_probe = (bank[indices_s.long()] * scales_s.float()).detach().requires_grad_(True)
            sparse_probe.retain_grad()
            self._last_sparse_probe[adapter] = sparse_probe
            return sparse_probe

        bank = self._get_sparse_bank(adapter, indices_s.device, cast_to_fp32=cast_to_fp32)
        sparse_values = bank[indices_s.long()] * scales_s
        sparse_mask = self.unilora_rosa_discrete_sparse_mask[adapter][
            self.unilora_rosa_discrete_sparse_offsets[adapter].long()
        ].to(dtype=sparse_values.dtype, device=sparse_values.device)
        return sparse_values * sparse_mask

    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        device = self.unilora_rosa_discrete_indices_A[adapter].device
        dtype = self.unilora_rosa_discrete_theta_d[adapter].dtype
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16
        a, b = self._get_lora_matrices(adapter, cast_to_fp32=cast_to_fp32)
        sparse = self._get_sparse_matrix(adapter, cast_to_fp32=cast_to_fp32, collect_dense_gradients=False)
        return transpose((b @ a) + sparse, self.fan_in_fan_out)

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
                if active_adapter not in self.unilora_rosa_discrete_indices_A:
                    continue
                a, b = self._get_lora_matrices(active_adapter)
                x_cast = x.to(self.unilora_rosa_discrete_theta_d[active_adapter].dtype)
                dropout = self.unilora_dropout[active_adapter]
                dropped_x = dropout(x_cast)
                result = result + F.linear(F.linear(dropped_x, a), b)

                collect_dense_gradients = (
                    self.capture_sparse_gradient_stats and self.training and not self.has_sparse_mask(active_adapter)
                )
                sparse = self._get_sparse_matrix(
                    active_adapter,
                    cast_to_fp32=False,
                    collect_dense_gradients=collect_dense_gradients,
                )
                result = result + F.linear(dropped_x, sparse)
        return result.to(previous_dtype)

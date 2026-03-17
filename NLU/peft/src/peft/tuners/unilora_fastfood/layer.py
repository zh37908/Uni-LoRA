# Copyright 2026-present
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose

from .._buffer_dict import BufferDict


def _fwht_1d(x: torch.Tensor) -> torch.Tensor:
    """
    Walsh–Hadamard transform for 1D vector, differentiable via PyTorch autograd.
    Input length must be power of two.
    """
    n = x.numel()
    if n & (n - 1) != 0:
        raise ValueError(f"FWHT requires power-of-two length, got {n}.")
    h = 1
    y = x
    while h < n:
        y = y.view(-1, 2, h)
        a = y[:, 0, :]
        b = y[:, 1, :]
        y = torch.cat([a + b, a - b], dim=1)
        y = y.view(-1, 2 * h)
        h *= 2
    return y.view_as(x)


def _fastfood_project(theta: torch.Tensor, G: torch.Tensor, Pi: torch.Tensor, B: torch.Tensor, divisor: torch.Tensor, out_dim: int):
    """
    FastFood projection producing an approximately Gaussian random vector in R^{out_dim}.
    """
    size = G.numel()
    if theta.numel() > size:
        theta = theta[:size]
    theta_padded = F.pad(theta, pad=(0, size - theta.numel()), value=0.0, mode="constant")

    x = theta_padded * B.to(dtype=theta_padded.dtype)
    x = _fwht_1d(x)
    x = x[Pi]
    x = x * G.to(dtype=x.dtype)
    x = _fwht_1d(x)

    x = x[:out_dim]
    # match the scaling used in the reference implementation
    x = x / (divisor.to(dtype=x.dtype) * math.sqrt(float(out_dim) / float(size)))
    return x


class UniLoRAFastFoodLayer(BaseTunerLayer):
    """
    UniLoRA variant where the projection P is FastFood (approx. Gaussian), mapping theta_d -> vec([A, B]).
    """

    adapter_layer_names = ("unilora_fastfood_theta_d",)

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.unilora_dropout = nn.ModuleDict({})
        self.unilora_fastfood_global_scaling = {}

        # FastFood buffers (per-adapter) for deterministic projections
        self.unilora_fastfood_G = BufferDict({}, persistent=True)
        self.unilora_fastfood_Pi = BufferDict({}, persistent=True)
        self.unilora_fastfood_B = BufferDict({}, persistent=True)
        self.unilora_fastfood_divisor = BufferDict({}, persistent=True)

        # Mark the weight as unmerged
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
            raise ValueError(f"Unsupported base layer type {type(base_layer)} for UniLoRAFastFoodLayer.")

        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def update_layer(
        self,
        adapter_name: str,
        unilora_fastfood_theta_d: nn.ParameterDict,
        r: int,
        theta_d_length: int,
        proj_seed: int,
        layer_seed: int,
        unilora_dropout: float = 0.0,
    ):
        if r <= 0:
            raise ValueError(f"`r` {r} should be a positive integer value")
        if theta_d_length <= 0:
            raise ValueError(f"`theta_d_length` {theta_d_length} should be a positive integer value")

        self.r[adapter_name] = r
        self.unilora_fastfood_global_scaling[adapter_name] = 1.0  # Default, will be updated by Model

        if unilora_dropout > 0.0:
            unilora_dropout_layer = nn.Dropout(p=unilora_dropout)
        else:
            unilora_dropout_layer = nn.Identity()
        self.unilora_dropout.update(nn.ModuleDict({adapter_name: unilora_dropout_layer}))

        # reference to the shared trainable vector
        self.unilora_fastfood_theta_d = unilora_fastfood_theta_d

        self._reset_fastfood_buffers(adapter_name, proj_seed=proj_seed, layer_seed=layer_seed)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def _reset_fastfood_buffers(self, adapter_name: str, proj_seed: int, layer_seed: int):
        if adapter_name not in self.unilora_fastfood_theta_d.keys():
            return

        out_dim = self.r[adapter_name] * self.in_features + self.out_features * self.r[adapter_name]
        size = 1 << int(math.ceil(math.log2(max(1, out_dim))))

        # deterministic, per-layer seed derived from (proj_seed, layer_seed, out_dim)
        seed = (int(proj_seed) + int(layer_seed) + int(out_dim)) % (2**32)
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)

        G = torch.randn(size, generator=g, dtype=torch.float32)
        Pi = torch.randperm(size, generator=g, dtype=torch.long)
        B = (2 * torch.randint(0, 2, (size,), generator=g, dtype=torch.int8) - 1).to(torch.int8)
        divisor = torch.sqrt(torch.tensor(float(size), dtype=torch.float32) * torch.sum(G**2))

        self.unilora_fastfood_G[adapter_name] = G
        self.unilora_fastfood_Pi[adapter_name] = Pi
        self.unilora_fastfood_B[adapter_name] = B
        self.unilora_fastfood_divisor[adapter_name] = divisor


class Linear(nn.Linear, UniLoRAFastFoodLayer):
    def __init__(
        self,
        base_layer,
        unilora_fastfood_theta_d,
        adapter_name: str,
        r: int,
        theta_d_length: int,
        proj_seed: int,
        layer_seed: int,
        unilora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ) -> None:
        super(nn.Linear, self).__init__()
        UniLoRAFastFoodLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name=adapter_name,
            unilora_fastfood_theta_d=unilora_fastfood_theta_d,
            r=r,
            theta_d_length=theta_d_length,
            proj_seed=proj_seed,
            layer_seed=layer_seed,
            unilora_dropout=unilora_dropout,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter in self.unilora_fastfood_G.keys():
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
            if active_adapter in self.unilora_fastfood_G.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def _get_lora_matrices(self, adapter, cast_to_fp32=False) -> Tuple[torch.Tensor, torch.Tensor]:
        theta = self.unilora_fastfood_theta_d[adapter]

        if cast_to_fp32:
            theta = theta.float()

        r = self.r[adapter]
        out_dim_A = r * self.in_features
        out_dim_total = out_dim_A + self.out_features * r

        vec = _fastfood_project(
            theta=theta.to(self.unilora_fastfood_G[adapter].device),
            G=self.unilora_fastfood_G[adapter],
            Pi=self.unilora_fastfood_Pi[adapter],
            B=self.unilora_fastfood_B[adapter],
            divisor=self.unilora_fastfood_divisor[adapter],
            out_dim=out_dim_total,
        )

        # Apply global scaling to ensure global isometry (P^T P = I)
        global_scaling = self.unilora_fastfood_global_scaling.get(adapter, 1.0)
        vec = vec * global_scaling

        A = vec[:out_dim_A].view(r, self.in_features)
        B = vec[out_dim_A:].view(self.out_features, r)
        return A, B

    def get_delta_weight(self, adapter) -> torch.Tensor:
        device = self.unilora_fastfood_G[adapter].device
        dtype = self.unilora_fastfood_theta_d[adapter].dtype
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
                if active_adapter not in self.unilora_fastfood_G.keys():
                    continue
                A, B = self._get_lora_matrices(active_adapter)
                x_cast = x.to(A.dtype)
                dropout = self.unilora_dropout[active_adapter]
                result = result + F.linear(F.linear(dropout(x_cast), A), B)
        return result.to(previous_dtype)


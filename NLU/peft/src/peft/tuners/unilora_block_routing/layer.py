# Copyright 2024-present the HuggingFace Inc. team.
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

import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose
from .._buffer_dict import BufferDict


class UniLoRABlockRoutingLayer(BaseTunerLayer):
    adapter_layer_names = ("unilora_theta_d", "unilora_router_logits")

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.unilora_dropout = nn.ModuleDict({})
        # Indices are persistent buffers
        self.unilora_indices_A = BufferDict({}, persistent=True)
        self.unilora_indices_B = BufferDict({}, persistent=True)
        # Scales (1/sqrt(count))
        self.unilora_scales_A = BufferDict({}, persistent=True)
        self.unilora_scales_B = BufferDict({}, persistent=True)
        
        # Router logits: one vector per adapter_name per layer
        self.unilora_router_logits = nn.ParameterDict({})
        self.unilora_router_tau = {}
        self.unilora_num_blocks = {}
        self.unilora_block_size = {}

        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )

        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def update_layer(
        self,
        adapter_name: str,
        unilora_theta_d,
        r: int,
        theta_d_length: int,
        num_blocks: int,
        router_tau: float,
        unilora_dropout: float = 0.0,
    ):
        if r <= 0:
            raise ValueError(f"`r` {r} should be a positive integer value")
        
        self.r[adapter_name] = r
        if unilora_dropout > 0.0:
            self.unilora_dropout.update(nn.ModuleDict({adapter_name: nn.Dropout(p=unilora_dropout)}))
        else:
            self.unilora_dropout.update(nn.ModuleDict({adapter_name: nn.Identity()}))

        self.unilora_theta_d = unilora_theta_d
        self.unilora_num_blocks[adapter_name] = num_blocks
        self.unilora_block_size[adapter_name] = theta_d_length // num_blocks
        self.unilora_router_tau[adapter_name] = router_tau

        # Init router logits
        logits = torch.zeros(num_blocks)
        nn.init.normal_(logits, mean=0.0, std=0.01)
        self.unilora_router_logits[adapter_name] = nn.Parameter(logits)

        # Init indices
        self.reset_unilora_parameters(adapter_name, self.unilora_block_size[adapter_name])
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_unilora_parameters(self, adapter_name, block_size):
        if adapter_name in self.unilora_theta_d.keys():
            # Indices are within a single block (0 to block_size-1)
            indices_A = torch.randint(0, block_size, (self.r[adapter_name], self.in_features), dtype=torch.long)
            indices_B = torch.randint(0, block_size, (self.out_features, self.r[adapter_name]), dtype=torch.long)
            self.unilora_indices_A[adapter_name] = indices_A
            self.unilora_indices_B[adapter_name] = indices_B

    def update_norm(self, adapter_name, scales_A, scales_B):
        if adapter_name in self.unilora_theta_d.keys():
            base_layer = self.get_base_layer()
            device = base_layer.weight.device
            dtype = base_layer.weight.dtype
            self.unilora_scales_A[adapter_name] = scales_A.to(device=device, dtype=dtype)
            self.unilora_scales_B[adapter_name] = scales_B.to(device=device, dtype=dtype)


class Linear(nn.Linear, UniLoRABlockRoutingLayer):
    def __init__(
        self,
        base_layer,
        unilora_theta_d,
        adapter_name: str,
        r: int,
        theta_d_length: int,
        num_blocks: int,
        router_tau: float,
        unilora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ) -> None:
        super(nn.Linear, self).__init__()
        UniLoRABlockRoutingLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name, unilora_theta_d, r, theta_d_length, num_blocks, router_tau, unilora_dropout,
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
                        raise ValueError(f"NaNs detected in merged weights for {active_adapter}")
                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.unilora_indices_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def _get_lora_matrices(self, adapter, cast_to_fp32=False) -> Tuple[torch.Tensor, torch.Tensor]:
        indices_A = self.unilora_indices_A[adapter].long()
        indices_B = self.unilora_indices_B[adapter].long()
        theta_d = self.unilora_theta_d[adapter].to(indices_A.device)
        if cast_to_fp32:
            theta_d = theta_d.float()

        # Block routing logic
        num_blocks = self.unilora_num_blocks[adapter]
        block_size = self.unilora_block_size[adapter]
        logits = self.unilora_router_logits[adapter]
        
        if self.training:
            tau = self.unilora_router_tau[adapter]
            router_probs = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
            # Soft combination of blocks
            # theta_d is (theta_d_length,) -> reshape (num_blocks, block_size)
            theta_blocks = theta_d.view(num_blocks, block_size)
            # Weighted sum of blocks: (block_size,)
            effective_bank = torch.einsum('k, ki -> i', router_probs, theta_blocks)
        else:
            # Hard routing
            idx = logits.argmax()
            effective_bank = theta_d.view(num_blocks, block_size)[idx]

        A = effective_bank[indices_A] * self.unilora_scales_A[adapter]
        B = effective_bank[indices_B] * self.unilora_scales_B[adapter]

        if cast_to_fp32:
            A = A.float()
            B = B.float()
        return A, B

    def get_delta_weight(self, adapter) -> torch.Tensor:
        device = self.unilora_indices_A[adapter].device
        dtype = self.unilora_theta_d[adapter].dtype
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
                x = x.to(self.unilora_theta_d[active_adapter].dtype)
                dropout = self.unilora_dropout[active_adapter]
                result = result + F.linear(F.linear(dropout(x), A), B)
        result = result.to(previous_dtype)
        return result

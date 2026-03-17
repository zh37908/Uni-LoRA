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
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose
from .._buffer_dict import BufferDict


class DirectUniLoRALayer(BaseTunerLayer):
    # Shared vector bank parameter name
    adapter_layer_names = ("direct_unilora_theta_d",)

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.unilora_dropout = nn.ModuleDict({})
        self.direct_unilora_indices_W = BufferDict({}, persistent=True)
        self.direct_unilora_scales_W = BufferDict({}, persistent=True)

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
        direct_unilora_theta_d,
        theta_d_length: int,
        unilora_dropout: float = 0.0,
    ):
        if unilora_dropout > 0.0:
            unilora_dropout_layer = nn.Dropout(p=unilora_dropout)
        else:
            unilora_dropout_layer = nn.Identity()
        self.unilora_dropout.update(nn.ModuleDict({adapter_name: unilora_dropout_layer}))

        # Store reference to shared parameter bank
        self.direct_unilora_theta_d = direct_unilora_theta_d

        # Initialize indices and move to device
        self.reset_direct_unilora_parameters(adapter_name, theta_d_length)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_direct_unilora_parameters(self, adapter_name, theta_d_length):
        if adapter_name in self.direct_unilora_theta_d.keys():
            indices_W = torch.randint(
                0, theta_d_length, (self.out_features, self.in_features), dtype=torch.long
            )
            self.direct_unilora_indices_W[adapter_name] = indices_W

    def update_norm(self, adapter_name: str, direct_unilora_scales_W):
        if adapter_name in self.direct_unilora_theta_d.keys():
            base_layer = self.get_base_layer()
            target_device = base_layer.weight.device
            target_dtype = base_layer.weight.dtype
            self.direct_unilora_scales_W[adapter_name] = direct_unilora_scales_W.to(
                device=target_device, dtype=target_dtype
            )


class Linear(nn.Linear, DirectUniLoRALayer):
    # Direct-UniLoRA implemented in a dense layer
    def __init__(
        self,
        base_layer,
        direct_unilora_theta_d,
        adapter_name: str,
        theta_d_length: int,
        unilora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ) -> None:
        super(nn.Linear, self).__init__()
        DirectUniLoRALayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name, direct_unilora_theta_d, theta_d_length, unilora_dropout,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter in self.direct_unilora_indices_W.keys():
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
            if active_adapter in self.direct_unilora_indices_W.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        device = self.direct_unilora_indices_W[adapter].device
        dtype = self.direct_unilora_theta_d[adapter].dtype
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        direct_unilora_indices_W = self.direct_unilora_indices_W[adapter]
        direct_unilora_theta_d = self.direct_unilora_theta_d[adapter].to(direct_unilora_indices_W.device)
        if cast_to_fp32:
            direct_unilora_theta_d = direct_unilora_theta_d.float()

        delta_w = direct_unilora_theta_d[direct_unilora_indices_W.long()] * self.direct_unilora_scales_W[adapter]
        if cast_to_fp32:
            delta_w = delta_w.float()

        output_tensor = transpose(delta_w, self.fan_in_fan_out)
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
                if active_adapter not in self.direct_unilora_indices_W.keys():
                    continue

                delta_w = self.get_delta_weight(active_adapter)
                x = x.to(self.direct_unilora_theta_d[active_adapter].dtype)
                dropout = self.unilora_dropout[active_adapter]
                result = result + F.linear(dropout(x), delta_w)

        result = result.to(previous_dtype)
        return result

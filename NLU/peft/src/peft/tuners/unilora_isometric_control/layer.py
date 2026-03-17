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

class UniLoRALayer(BaseTunerLayer):
    adapter_layer_names = ("unilora_theta_d",)

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.unilora_dropout = nn.ModuleDict({})
        self.unilora_indices_A = BufferDict({}, persistent=True)
        self.unilora_indices_B = BufferDict({}, persistent=True)
        self.unilora_scales_A = BufferDict({}, persistent=True)
        self.unilora_scales_B = BufferDict({}, persistent=True)
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

    def update_layer(self, adapter_name, unilora_theta_d, r, theta_d_length, unilora_dropout=0.0):
        if r <= 0: raise ValueError(f"`r` {r} should be a positive integer")
        self.r[adapter_name] = r
        dropout = nn.Dropout(p=unilora_dropout) if unilora_dropout > 0.0 else nn.Identity()
        self.unilora_dropout.update(nn.ModuleDict({adapter_name: dropout}))
        self.unilora_theta_d = unilora_theta_d
        self.reset_unilora_parameters(adapter_name, theta_d_length)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_unilora_parameters(self, adapter_name, theta_d_length):
        if adapter_name in self.unilora_theta_d.keys():
            self.unilora_indices_A[adapter_name] = torch.randint(0, theta_d_length, (self.r[adapter_name], self.in_features), dtype=torch.long)
            self.unilora_indices_B[adapter_name] = torch.randint(0, theta_d_length, (self.out_features, self.r[adapter_name]), dtype=torch.long)

    def update_norm(self, adapter_name, unilora_scales_A, unilora_scales_B):   
        if adapter_name in self.unilora_theta_d.keys():
            base_layer = self.get_base_layer()
            self.unilora_scales_A[adapter_name] = unilora_scales_A.to(device=base_layer.weight.device, dtype=base_layer.weight.dtype)
            self.unilora_scales_B[adapter_name] = unilora_scales_B.to(device=base_layer.weight.device, dtype=base_layer.weight.dtype)

class Linear(nn.Linear, UniLoRALayer):
    def __init__(self, base_layer, unilora_theta_d, adapter_name, r, theta_d_length, unilora_dropout=0.0, fan_in_fan_out=False, is_target_conv_1d_layer=False, **kwargs):
        super(nn.Linear, self).__init__()
        UniLoRALayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, unilora_theta_d, r, theta_d_length, unilora_dropout)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge=False, adapter_names=None):
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names: return
        for active_adapter in adapter_names:
            if active_adapter in self.unilora_indices_A.keys():
                self.get_base_layer().weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self):
        if not self.merged: return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.unilora_indices_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def _get_lora_matrices(self, adapter, cast_to_fp32=False):
        unilora_indices_A = self.unilora_indices_A[adapter] 
        unilora_indices_B = self.unilora_indices_B[adapter] 
        unilora_theta_d = self.unilora_theta_d[adapter].to(unilora_indices_A.device)
        if cast_to_fp32: unilora_theta_d = unilora_theta_d.float()
        A = unilora_theta_d[unilora_indices_A.long()] * self.unilora_scales_A[adapter]
        B = unilora_theta_d[unilora_indices_B.long()] * self.unilora_scales_B[adapter]
        return A, B

    def get_delta_weight(self, adapter):
        A, B = self._get_lora_matrices(adapter)
        return transpose(B @ A, self.fan_in_fan_out)

    def forward(self, x, *args, **kwargs):
        if self.disable_adapters or self.merged:
            return self.base_layer(x, *args, **kwargs)
        result = self.base_layer(x, *args, **kwargs)
        for active_adapter in self.active_adapters:
            if active_adapter not in self.unilora_indices_A.keys(): continue
            A, B = self._get_lora_matrices(active_adapter)
            dropout = self.unilora_dropout[active_adapter]
            result += F.linear(F.linear(dropout(x.to(A.dtype)), A), B)
        return result.to(x.dtype)

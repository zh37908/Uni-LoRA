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


class UniLoRGSLayer(BaseTunerLayer):
    adapter_layer_names = ("unilora_gs_theta_d", "unilora_gs_logits_A", "unilora_gs_logits_B")

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.unilora_dropout = nn.ModuleDict({})

        self.unilora_gs_logits_A = nn.ParameterDict({})
        self.unilora_gs_logits_B = nn.ParameterDict({})
        self.unilora_gs_tau = {}

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
        unilora_gs_theta_d,
        r: int,
        theta_d_length: int,
        init_logits_std: float,
        gumbel_tau: float,
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

        # Store reference to shared parameter bank
        self.unilora_gs_theta_d = unilora_gs_theta_d
        self.unilora_gs_tau[adapter_name] = gumbel_tau

        # Initialize logits
        logits_A = torch.empty((r, self.in_features, theta_d_length))
        logits_B = torch.empty((self.out_features, r, theta_d_length))
        torch.nn.init.normal_(logits_A, mean=0.0, std=init_logits_std)
        torch.nn.init.normal_(logits_B, mean=0.0, std=init_logits_std)
        self.unilora_gs_logits_A[adapter_name] = nn.Parameter(logits_A)
        self.unilora_gs_logits_B[adapter_name] = nn.Parameter(logits_B)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def seed_logits_from_indices(self, adapter_name: str, indices_A, indices_B, init_logits_bias: float):
        if adapter_name not in self.unilora_gs_logits_A:
            return
        if init_logits_bias == 0.0:
            return
        logits_A = self.unilora_gs_logits_A[adapter_name].data
        logits_B = self.unilora_gs_logits_B[adapter_name].data
        self._add_bias_to_logits(logits_A, indices_A, init_logits_bias)
        self._add_bias_to_logits(logits_B, indices_B, init_logits_bias)

    @staticmethod
    def _add_bias_to_logits(logits: torch.Tensor, indices: torch.Tensor, bias: float) -> None:
        # Avoid full zero_ for large logits; only bump selected indices.
        flat_logits = logits.view(-1, logits.shape[-1])
        flat_indices = indices.reshape(-1)
        rows = torch.arange(flat_indices.numel(), device=flat_indices.device)
        flat_logits[rows, flat_indices] += bias


class Linear(nn.Linear, UniLoRGSLayer):
    def __init__(
        self,
        base_layer,
        unilora_gs_theta_d,
        adapter_name: str,
        r: int,
        theta_d_length: int,
        init_logits_std: float,
        gumbel_tau: float,
        unilora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ) -> None:
        super(nn.Linear, self).__init__()
        UniLoRGSLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name, unilora_gs_theta_d, r, theta_d_length, init_logits_std, gumbel_tau, unilora_dropout,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter in self.unilora_gs_logits_A.keys():
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
            if active_adapter in self.unilora_gs_logits_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def _get_lora_matrices(self, adapter, cast_to_fp32=False) -> Tuple[torch.Tensor, torch.Tensor]:
        logits_A = self.unilora_gs_logits_A[adapter]
        logits_B = self.unilora_gs_logits_B[adapter]
        theta_d = self.unilora_gs_theta_d[adapter].to(logits_A.device)

        if cast_to_fp32:
            logits_A = logits_A.float()
            logits_B = logits_B.float()
            theta_d = theta_d.float()

        if self.training:
            tau = self.unilora_gs_tau[adapter]
            probs_A = F.gumbel_softmax(logits_A, tau=tau, hard=False, dim=-1)
            probs_B = F.gumbel_softmax(logits_B, tau=tau, hard=False, dim=-1)
            A = (probs_A * theta_d).sum(dim=-1)
            B = (probs_B * theta_d).sum(dim=-1)
        else:
            idx_A = logits_A.argmax(dim=-1)
            idx_B = logits_B.argmax(dim=-1)
            A = theta_d[idx_A]
            B = theta_d[idx_B]

        return A, B

    def get_delta_weight(self, adapter) -> torch.Tensor:
        device = self.unilora_gs_logits_A[adapter].device
        dtype = self.unilora_gs_theta_d[adapter].dtype
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
                if active_adapter not in self.unilora_gs_logits_A.keys():
                    continue

                A, B = self._get_lora_matrices(active_adapter)
                x = x.to(self.unilora_gs_theta_d[active_adapter].dtype)
                dropout = self.unilora_dropout[active_adapter]
                result = result + F.linear(F.linear(dropout(x), A), B)

        result = result.to(previous_dtype)
        return result

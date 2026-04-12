import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose
from .._buffer_dict import BufferDict


class GeoUniLoRALayer(BaseTunerLayer):
    # Use geo_ul_* names so keys do not contain the substring "unilora_" (breaks adapter name insertion).
    adapter_layer_names = ("geo_ul_shared_theta_d", "geo_ul_innovation_theta_d")

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r_shared = {}
        self.r_innov = {}
        self.group_id = {}
        self.innovation_module_key = {}
        self.unilora_dropout = nn.ModuleDict({})

        self.unilora_indices_shared_A = BufferDict({}, persistent=True)
        self.unilora_indices_shared_B = BufferDict({}, persistent=True)
        self.unilora_scales_shared_A = BufferDict({}, persistent=True)
        self.unilora_scales_shared_B = BufferDict({}, persistent=True)

        self.unilora_indices_innov_A = BufferDict({}, persistent=True)
        self.unilora_indices_innov_B = BufferDict({}, persistent=True)
        self.unilora_scales_innov_A = BufferDict({}, persistent=True)
        self.unilora_scales_innov_B = BufferDict({}, persistent=True)

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
            raise ValueError(f"Unsupported base layer type {type(base_layer)} for Geo-UniLoRA.")

        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def update_layer(
        self,
        adapter_name: str,
        geo_ul_shared_theta_d,
        geo_ul_innovation_theta_d,
        group_id: int,
        innovation_module_key: str,
        r_shared: int,
        r_innov: int,
        shared_theta_d_length: int,
        innovation_theta_d_length: int,
        unilora_dropout: float = 0.0,
    ):
        if r_shared < 0 or r_innov < 0:
            raise ValueError(f"r_shared and r_innov must be non-negative, got r_shared={r_shared}, r_innov={r_innov}")
        if r_shared == 0 and r_innov == 0:
            raise ValueError("Geo-UniLoRA requires at least one active branch per module.")

        self.r_shared[adapter_name] = r_shared
        self.r_innov[adapter_name] = r_innov
        self.group_id[adapter_name] = group_id
        self.innovation_module_key[adapter_name] = innovation_module_key

        if unilora_dropout > 0.0:
            unilora_dropout_layer = nn.Dropout(p=unilora_dropout)
        else:
            unilora_dropout_layer = nn.Identity()
        self.unilora_dropout.update(nn.ModuleDict({adapter_name: unilora_dropout_layer}))

        self.geo_ul_shared_theta_d = geo_ul_shared_theta_d
        self.geo_ul_innovation_theta_d = geo_ul_innovation_theta_d
        self.reset_geo_unilora_parameters(adapter_name, shared_theta_d_length, innovation_theta_d_length)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_geo_unilora_parameters(self, adapter_name, shared_theta_d_length, innovation_theta_d_length):
        if adapter_name in self.r_shared:
            rs = self.r_shared[adapter_name]
            ri = self.r_innov[adapter_name]
            if rs > 0:
                self.unilora_indices_shared_A[adapter_name] = torch.randint(
                    0, shared_theta_d_length, (rs, self.in_features), dtype=torch.long
                )
                self.unilora_indices_shared_B[adapter_name] = torch.randint(
                    0, shared_theta_d_length, (self.out_features, rs), dtype=torch.long
                )
            else:
                self.unilora_indices_shared_A[adapter_name] = torch.empty((0, self.in_features), dtype=torch.long)
                self.unilora_indices_shared_B[adapter_name] = torch.empty((self.out_features, 0), dtype=torch.long)

            if ri > 0:
                self.unilora_indices_innov_A[adapter_name] = torch.randint(
                    0, innovation_theta_d_length, (ri, self.in_features), dtype=torch.long
                )
                self.unilora_indices_innov_B[adapter_name] = torch.randint(
                    0, innovation_theta_d_length, (self.out_features, ri), dtype=torch.long
                )
            else:
                self.unilora_indices_innov_A[adapter_name] = torch.empty((0, self.in_features), dtype=torch.long)
                self.unilora_indices_innov_B[adapter_name] = torch.empty((self.out_features, 0), dtype=torch.long)

    def update_norm_shared(
        self,
        adapter_name: str,
        scales_a: torch.Tensor,
        scales_b: torch.Tensor,
    ):
        if adapter_name in self.r_shared:
            base_layer = self.get_base_layer()
            target_device = base_layer.weight.device
            target_dtype = base_layer.weight.dtype
            self.unilora_scales_shared_A[adapter_name] = scales_a.to(device=target_device, dtype=target_dtype)
            self.unilora_scales_shared_B[adapter_name] = scales_b.to(device=target_device, dtype=target_dtype)

    def update_norm_innov(
        self,
        adapter_name: str,
        scales_a: torch.Tensor,
        scales_b: torch.Tensor,
    ):
        if adapter_name in self.r_innov:
            base_layer = self.get_base_layer()
            target_device = base_layer.weight.device
            target_dtype = base_layer.weight.dtype
            self.unilora_scales_innov_A[adapter_name] = scales_a.to(device=target_device, dtype=target_dtype)
            self.unilora_scales_innov_B[adapter_name] = scales_b.to(device=target_device, dtype=target_dtype)


class Linear(nn.Linear, GeoUniLoRALayer):
    def __init__(
        self,
        base_layer,
        geo_ul_shared_theta_d,
        geo_ul_innovation_theta_d,
        adapter_name: str,
        group_id: int,
        innovation_module_key: str,
        r_shared: int,
        r_innov: int,
        shared_theta_d_length: int,
        innovation_theta_d_length: int,
        unilora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ) -> None:
        super(nn.Linear, self).__init__()
        GeoUniLoRALayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            geo_ul_shared_theta_d,
            geo_ul_innovation_theta_d,
            group_id,
            innovation_module_key,
            r_shared,
            r_innov,
            shared_theta_d_length,
            innovation_theta_d_length,
            unilora_dropout,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def _shared_bank_key(self, adapter: str) -> str:
        gid = self.group_id[adapter]
        return f"{adapter}__g{int(gid)}"

    def _innov_bank_key(self, adapter: str) -> str:
        mkey = self.innovation_module_key[adapter]
        return f"{adapter}__{mkey}"

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter in self.unilora_indices_shared_A.keys():
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
            if active_adapter in self.unilora_indices_shared_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def _get_lora_matrices_shared(self, adapter: str, cast_to_fp32: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        idx_a = self.unilora_indices_shared_A[adapter]
        idx_b = self.unilora_indices_shared_B[adapter]
        bank = self.geo_ul_shared_theta_d[self._shared_bank_key(adapter)].to(idx_a.device)
        if cast_to_fp32:
            bank = bank.float()
        A = bank[idx_a.long()] * self.unilora_scales_shared_A[adapter]
        B = bank[idx_b.long()] * self.unilora_scales_shared_B[adapter]
        if cast_to_fp32:
            A = A.float()
            B = B.float()
        return A, B

    def _get_lora_matrices_innov(self, adapter: str, cast_to_fp32: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        idx_a = self.unilora_indices_innov_A[adapter]
        idx_b = self.unilora_indices_innov_B[adapter]
        bank = self.geo_ul_innovation_theta_d[self._innov_bank_key(adapter)].to(idx_a.device)
        if cast_to_fp32:
            bank = bank.float()
        A = bank[idx_a.long()] * self.unilora_scales_innov_A[adapter]
        B = bank[idx_b.long()] * self.unilora_scales_innov_B[adapter]
        if cast_to_fp32:
            A = A.float()
            B = B.float()
        return A, B

    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        device = self.get_base_layer().weight.device
        dtype = self.get_base_layer().weight.dtype
        if self.r_shared[adapter] > 0:
            dtype = self.geo_ul_shared_theta_d[self._shared_bank_key(adapter)].dtype
        elif self.r_innov[adapter] > 0:
            dtype = self.geo_ul_innovation_theta_d[self._innov_bank_key(adapter)].dtype
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16
        out = torch.zeros(
            (self.out_features, self.in_features),
            device=device,
            dtype=torch.float32 if cast_to_fp32 else dtype,
        )
        if self.r_shared[adapter] > 0:
            A_sh, B_sh = self._get_lora_matrices_shared(adapter, cast_to_fp32)
            out = out + (B_sh @ A_sh)
        if self.r_innov[adapter] > 0:
            A_in, B_in = self._get_lora_matrices_innov(adapter, cast_to_fp32)
            out = out + (B_in @ A_in)
        out = transpose(out, self.fan_in_fan_out)
        return out

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
                if active_adapter not in self.unilora_indices_shared_A.keys():
                    continue

                if self.r_shared[active_adapter] > 0:
                    adapter_dtype = self.geo_ul_shared_theta_d[self._shared_bank_key(active_adapter)].dtype
                else:
                    adapter_dtype = self.geo_ul_innovation_theta_d[self._innov_bank_key(active_adapter)].dtype
                x_cast = x.to(adapter_dtype)
                dropout = self.unilora_dropout[active_adapter]
                xd = dropout(x_cast)
                if self.r_shared[active_adapter] > 0:
                    A_sh, B_sh = self._get_lora_matrices_shared(active_adapter)
                    result = result + F.linear(F.linear(xd, A_sh), B_sh)
                if self.r_innov[active_adapter] > 0:
                    A_in, B_in = self._get_lora_matrices_innov(active_adapter)
                    result = result + F.linear(F.linear(xd, A_in), B_in)

        return result.to(previous_dtype)

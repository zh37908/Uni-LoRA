import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose

from .._buffer_dict import BufferDict


class UniLoRACountSketchLayer(BaseTunerLayer):
    adapter_layer_names = ("unilora_count_sketch_theta_d",)

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.num_sketches = {}
        self.unilora_dropout = nn.ModuleDict({})

        self.unilora_count_sketch_indices_A = nn.ModuleDict({})
        self.unilora_count_sketch_indices_B = nn.ModuleDict({})
        self.unilora_count_sketch_signs_A = nn.ModuleDict({})
        self.unilora_count_sketch_signs_B = nn.ModuleDict({})

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
            raise ValueError(f"Unsupported base layer type {type(base_layer)} for UniLoRACountSketchLayer.")

        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    @staticmethod
    def _sketch_key(sketch_idx: int) -> str:
        return str(sketch_idx)

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def update_layer(
        self,
        adapter_name: str,
        unilora_count_sketch_theta_d,
        r: int,
        theta_d_length: int,
        num_sketches: int,
        unilora_dropout: float = 0.0,
    ):
        if r <= 0:
            raise ValueError(f"`r` {r} should be a positive integer value")
        if num_sketches <= 0:
            raise ValueError(f"`num_sketches` should be > 0, got {num_sketches}")

        self.r[adapter_name] = r
        self.num_sketches[adapter_name] = num_sketches

        if unilora_dropout > 0.0:
            unilora_dropout_layer = nn.Dropout(p=unilora_dropout)
        else:
            unilora_dropout_layer = nn.Identity()
        self.unilora_dropout.update(nn.ModuleDict({adapter_name: unilora_dropout_layer}))

        self.unilora_count_sketch_theta_d = unilora_count_sketch_theta_d
        self.reset_unilora_parameters(adapter_name, theta_d_length, num_sketches)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_unilora_parameters(self, adapter_name: str, theta_d_length: int, num_sketches: int):
        if adapter_name in self.unilora_count_sketch_theta_d.keys():
            self.unilora_count_sketch_indices_A[adapter_name] = BufferDict({}, persistent=True)
            self.unilora_count_sketch_indices_B[adapter_name] = BufferDict({}, persistent=True)
            self.unilora_count_sketch_signs_A[adapter_name] = BufferDict({}, persistent=True)
            self.unilora_count_sketch_signs_B[adapter_name] = BufferDict({}, persistent=True)
            for sketch_idx in range(num_sketches):
                sketch_key = self._sketch_key(sketch_idx)
                indices_A = torch.randint(0, theta_d_length, (self.r[adapter_name], self.in_features), dtype=torch.long)
                indices_B = torch.randint(0, theta_d_length, (self.out_features, self.r[adapter_name]), dtype=torch.long)
                self.unilora_count_sketch_indices_A[adapter_name][sketch_key] = indices_A
                self.unilora_count_sketch_indices_B[adapter_name][sketch_key] = indices_B

    def update_sign(self, adapter_name: str, sketch_idx: int, signs_A: torch.Tensor, signs_B: torch.Tensor):
        if adapter_name in self.unilora_count_sketch_theta_d.keys():
            base_layer = self.get_base_layer()
            target_device = base_layer.weight.device
            target_dtype = base_layer.weight.dtype
            sketch_key = self._sketch_key(sketch_idx)
            self.unilora_count_sketch_signs_A[adapter_name][sketch_key] = signs_A.to(
                device=target_device, dtype=target_dtype
            )
            self.unilora_count_sketch_signs_B[adapter_name][sketch_key] = signs_B.to(
                device=target_device, dtype=target_dtype
            )


class Linear(nn.Linear, UniLoRACountSketchLayer):
    def __init__(
        self,
        base_layer,
        unilora_count_sketch_theta_d,
        adapter_name: str,
        r: int,
        theta_d_length: int,
        num_sketches: int,
        unilora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ) -> None:
        super(nn.Linear, self).__init__()
        UniLoRACountSketchLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name=adapter_name,
            unilora_count_sketch_theta_d=unilora_count_sketch_theta_d,
            r=r,
            theta_d_length=theta_d_length,
            num_sketches=num_sketches,
            unilora_dropout=unilora_dropout,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            sketch0_key = self._sketch_key(0)
            if active_adapter in self.unilora_count_sketch_indices_A and sketch0_key in self.unilora_count_sketch_indices_A[active_adapter]:
                base_layer = self.get_base_layer()
                delta = self.get_delta_weight(active_adapter, use_median=True)
                if safe_merge:
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += delta
                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )
                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += delta
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            sketch0_key = self._sketch_key(0)
            if active_adapter in self.unilora_count_sketch_indices_A and sketch0_key in self.unilora_count_sketch_indices_A[active_adapter]:
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter, use_median=True)

    def _get_lora_matrices(self, adapter: str, sketch_idx: int, cast_to_fp32: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        sketch_key = self._sketch_key(sketch_idx)
        indices_A = self.unilora_count_sketch_indices_A[adapter][sketch_key]
        indices_B = self.unilora_count_sketch_indices_B[adapter][sketch_key]

        theta_d = self.unilora_count_sketch_theta_d[adapter][sketch_idx].to(indices_A.device)
        if cast_to_fp32:
            theta_d = theta_d.float()

        A = theta_d[indices_A.long()] * self.unilora_count_sketch_signs_A[adapter][sketch_key]
        B = theta_d[indices_B.long()] * self.unilora_count_sketch_signs_B[adapter][sketch_key]

        if cast_to_fp32:
            A = A.float()
            B = B.float()
        return A, B

    def get_delta_weight(self, adapter: str, use_median: bool = True) -> torch.Tensor:
        sketch0_key = self._sketch_key(0)
        device = self.unilora_count_sketch_indices_A[adapter][sketch0_key].device
        dtype = self.unilora_count_sketch_theta_d[adapter][0].dtype
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        deltas = []
        for sketch_idx in range(self.num_sketches[adapter]):
            A, B = self._get_lora_matrices(adapter, sketch_idx, cast_to_fp32=cast_to_fp32)
            deltas.append(transpose(B @ A, self.fan_in_fan_out))

        if len(deltas) == 1:
            return deltas[0]

        delta_stack = torch.stack(deltas, dim=0)
        if use_median:
            return delta_stack.median(dim=0).values
        return delta_stack.mean(dim=0)

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
                sketch0_key = self._sketch_key(0)
                if (
                    active_adapter not in self.unilora_count_sketch_indices_A
                    or sketch0_key not in self.unilora_count_sketch_indices_A[active_adapter]
                ):
                    continue

                x_cast = x.to(self.unilora_count_sketch_theta_d[active_adapter][0].dtype)
                dropout = self.unilora_dropout[active_adapter]
                delta_weight = self.get_delta_weight(active_adapter, use_median=not self.training)
                result = result + F.linear(dropout(x_cast), delta_weight)

        return result.to(previous_dtype)

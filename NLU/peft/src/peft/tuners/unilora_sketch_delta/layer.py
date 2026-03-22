import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose

from .._buffer_dict import BufferDict
from ..unilora_sketch_utils import compute_group_setup, decode_local_codebook, generate_balanced_indices, select_code_dtype


class UniLoRASketchDeltaLayer(BaseTunerLayer):
    adapter_layer_names = ("unilora_sketch_delta_quant_A", "unilora_sketch_delta_quant_B")
    other_param_names = ("unilora_sketch_delta_codes_A", "unilora_sketch_delta_codes_B")

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.unilora_dropout = nn.ModuleDict({})

        self.unilora_sketch_delta_quant_A = nn.ParameterDict({})
        self.unilora_sketch_delta_quant_B = nn.ParameterDict({})
        self.unilora_sketch_delta_codes_A = BufferDict({}, persistent=True)
        self.unilora_sketch_delta_codes_B = BufferDict({}, persistent=True)

        self.sketch_delta_codebook_size = {}
        self.sketch_delta_groups_A = {}
        self.sketch_delta_groups_B = {}

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
            raise ValueError(f"Unsupported base layer type {type(base_layer)} for UniLoRASketchDeltaLayer.")

        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        bits: int,
        groups_per_row: int,
        init_codebook_bound: float,
        proj_seed: int,
        layer_seed: int,
        unilora_dropout: float = 0.0,
    ):
        if r <= 0:
            raise ValueError(f"`r` should be positive, got {r}.")

        codebook_size = 1 << bits
        self.r[adapter_name] = r
        self.sketch_delta_codebook_size[adapter_name] = codebook_size

        if unilora_dropout > 0.0:
            dropout_layer = nn.Dropout(p=unilora_dropout)
        else:
            dropout_layer = nn.Identity()
        self.unilora_dropout.update(nn.ModuleDict({adapter_name: dropout_layer}))

        group_size_A, _ = compute_group_setup(self.in_features, groups_per_row)
        group_size_B, _ = compute_group_setup(r, groups_per_row)
        groups_A = (self.in_features + group_size_A - 1) // group_size_A
        groups_B = (r + group_size_B - 1) // group_size_B
        self.sketch_delta_groups_A[adapter_name] = groups_A
        self.sketch_delta_groups_B[adapter_name] = groups_B

        quant_A = torch.empty(r, groups_A, codebook_size)
        quant_B = torch.empty(self.out_features, groups_B, codebook_size)
        torch.nn.init.uniform_(quant_A, -init_codebook_bound, init_codebook_bound)
        torch.nn.init.uniform_(quant_B, -init_codebook_bound, init_codebook_bound)

        numel_codes_A = r * groups_A * group_size_A
        numel_codes_B = self.out_features * groups_B * group_size_B
        codes_A = generate_balanced_indices(
            total_length=numel_codes_A,
            num_buckets=codebook_size,
            seed=proj_seed + layer_seed,
        ).view(r, groups_A, group_size_A)
        codes_B = generate_balanced_indices(
            total_length=numel_codes_B,
            num_buckets=codebook_size,
            seed=proj_seed + layer_seed + 1,
        ).view(self.out_features, groups_B, group_size_B)

        code_dtype = select_code_dtype(codebook_size)
        self.unilora_sketch_delta_quant_A[adapter_name] = nn.Parameter(quant_A)
        self.unilora_sketch_delta_quant_B[adapter_name] = nn.Parameter(quant_B)
        self.unilora_sketch_delta_codes_A[adapter_name] = codes_A.to(code_dtype)
        self.unilora_sketch_delta_codes_B[adapter_name] = codes_B.to(code_dtype)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def _get_lora_matrices(self, adapter: str, cast_to_fp32: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        quant_A = self.unilora_sketch_delta_quant_A[adapter]
        quant_B = self.unilora_sketch_delta_quant_B[adapter]
        codes_A = self.unilora_sketch_delta_codes_A[adapter]
        codes_B = self.unilora_sketch_delta_codes_B[adapter]

        if cast_to_fp32:
            quant_A = quant_A.float()
            quant_B = quant_B.float()

        A = decode_local_codebook(quant_A, codes_A.to(device=quant_A.device), self.in_features)
        B = decode_local_codebook(quant_B, codes_B.to(device=quant_B.device), self.r[adapter])
        return A, B

    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        device = self.unilora_sketch_delta_quant_A[adapter].device
        dtype = self.unilora_sketch_delta_quant_A[adapter].dtype
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16
        A, B = self._get_lora_matrices(adapter, cast_to_fp32=cast_to_fp32)
        return transpose(B @ A, self.kwargs.get("fan_in_fan_out", False))

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter in self.unilora_sketch_delta_quant_A.keys():
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
            if active_adapter in self.unilora_sketch_delta_quant_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)


class Linear(nn.Linear, UniLoRASketchDeltaLayer):
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int,
        bits: int,
        groups_per_row: int,
        init_codebook_bound: float,
        proj_seed: int,
        layer_seed: int,
        unilora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ) -> None:
        super(nn.Linear, self).__init__()
        UniLoRASketchDeltaLayer.__init__(self, base_layer, fan_in_fan_out=fan_in_fan_out, **kwargs)
        self._active_adapter = adapter_name
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
        self.update_layer(
            adapter_name=adapter_name,
            r=r,
            bits=bits,
            groups_per_row=groups_per_row,
            init_codebook_bound=init_codebook_bound,
            proj_seed=proj_seed,
            layer_seed=layer_seed,
            unilora_dropout=unilora_dropout,
        )

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
                if active_adapter not in self.unilora_sketch_delta_quant_A.keys():
                    continue

                A, B = self._get_lora_matrices(active_adapter)
                x_cast = x.to(self.unilora_sketch_delta_quant_A[active_adapter].dtype)
                result = result + F.linear(
                    F.linear(self.unilora_dropout[active_adapter](x_cast), A),
                    B,
                )

        return result.to(previous_dtype)

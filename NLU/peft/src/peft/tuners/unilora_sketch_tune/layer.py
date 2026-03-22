import warnings
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose

from .._buffer_dict import BufferDict


def _select_code_dtype(codebook_size: int) -> torch.dtype:
    if codebook_size <= 2**8:
        return torch.uint8
    if codebook_size <= 2**15:
        return torch.int16
    if codebook_size <= 2**31:
        return torch.int32
    return torch.int64


class UniLoRASketchTuneLayer(BaseTunerLayer):
    adapter_layer_names = ("unilora_sketch_tune_quant_grid",)
    other_param_names = ("unilora_sketch_tune_weight_codes",)

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.unilora_dropout = nn.ModuleDict({})
        self.unilora_sketch_tune_quant_grid = nn.ParameterDict({})
        self.unilora_sketch_tune_weight_codes = BufferDict({}, persistent=True)

        self.sketch_bits: Dict[str, int] = {}
        self.sketch_groups_per_row: Dict[str, int] = {}
        self.sketch_group_size: Dict[str, int] = {}
        self.sketch_codebook_size: Dict[str, int] = {}

        self._disable_adapters = False
        self.merged_adapters = []
        self._base_weight_backup: Optional[torch.Tensor] = None

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        else:
            raise ValueError(
                f"Unsupported base layer type {type(base_layer)} for UniLoRASketchTuneLayer."
            )

        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs
        self.fan_in_fan_out = bool(kwargs.get("fan_in_fan_out", False))

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def update_layer(
        self,
        adapter_name: str,
        bits: int,
        groups_per_row: int,
        bootstrap_method: str,
        bootstrap_kmeans_iters: int,
        unilora_dropout: float = 0.0,
    ):
        if bits <= 0:
            raise ValueError(f"`bits` must be positive, got {bits}.")
        if groups_per_row <= 0:
            raise ValueError(f"`groups_per_row` must be positive, got {groups_per_row}.")

        codebook_size = 1 << bits
        group_size = (self.in_features + groups_per_row - 1) // groups_per_row

        self.sketch_bits[adapter_name] = bits
        self.sketch_groups_per_row[adapter_name] = groups_per_row
        self.sketch_group_size[adapter_name] = group_size
        self.sketch_codebook_size[adapter_name] = codebook_size

        if unilora_dropout > 0.0:
            dropout_layer = nn.Dropout(p=unilora_dropout)
        else:
            dropout_layer = nn.Identity()
        self.unilora_dropout.update(nn.ModuleDict({adapter_name: dropout_layer}))

        quant_grid, weight_codes = self._bootstrap_sketch_from_base_weight(
            bits=bits,
            groups_per_row=groups_per_row,
            bootstrap_method=bootstrap_method,
            bootstrap_kmeans_iters=bootstrap_kmeans_iters,
        )
        self.unilora_sketch_tune_quant_grid[adapter_name] = nn.Parameter(quant_grid)
        self.unilora_sketch_tune_weight_codes[adapter_name] = weight_codes
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def _bootstrap_sketch_from_base_weight(
        self,
        bits: int,
        groups_per_row: int,
        bootstrap_method: str,
        bootstrap_kmeans_iters: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        codebook_size = 1 << bits
        group_size = (self.in_features + groups_per_row - 1) // groups_per_row
        padded_in_features = groups_per_row * group_size
        base_weight = self.get_base_layer().weight

        if base_weight.device.type == "meta":
            quant_grid = torch.empty(
                self.out_features,
                groups_per_row,
                codebook_size,
                device=base_weight.device,
                dtype=torch.float32,
            )
            weight_codes = torch.empty(
                self.out_features,
                groups_per_row,
                group_size,
                device=base_weight.device,
                dtype=_select_code_dtype(codebook_size),
            )
            return quant_grid, weight_codes

        with torch.no_grad():
            effective_weight = transpose(base_weight.detach(), self.fan_in_fan_out)
            effective_weight = effective_weight.to(device="cpu", dtype=torch.float32)
            if effective_weight.shape != (self.out_features, self.in_features):
                raise ValueError(
                    "Unexpected effective weight shape during sketch bootstrap: "
                    f"expected {(self.out_features, self.in_features)}, got {tuple(effective_weight.shape)}."
                )

            padded_weight = F.pad(
                effective_weight,
                pad=(0, padded_in_features - self.in_features),
                mode="constant",
                value=0.0,
            )
            grouped_weight = padded_weight.view(self.out_features, groups_per_row, group_size)

            if bootstrap_method == "uniform":
                quant_grid, weight_codes = self._uniform_bootstrap(grouped_weight, codebook_size)
            elif bootstrap_method == "kmeans":
                quant_grid, weight_codes = self._kmeans_bootstrap(
                    grouped_weight,
                    codebook_size=codebook_size,
                    num_iters=bootstrap_kmeans_iters,
                )
            else:
                raise ValueError(f"Unsupported bootstrap method: {bootstrap_method}")

        target_dtype = base_weight.dtype if base_weight.dtype.is_floating_point else torch.float32
        target_device = base_weight.device
        return (
            quant_grid.to(device=target_device, dtype=target_dtype),
            weight_codes.to(device=target_device),
        )

    @staticmethod
    def _uniform_bootstrap(
        grouped_weight: torch.Tensor, codebook_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if codebook_size == 1:
            quant_grid = grouped_weight.mean(dim=-1, keepdim=True)
            weight_codes = torch.zeros(
                grouped_weight.shape,
                device=grouped_weight.device,
                dtype=_select_code_dtype(codebook_size),
            )
            return quant_grid.contiguous(), weight_codes.contiguous()

        mins = grouped_weight.amin(dim=-1, keepdim=True)
        maxs = grouped_weight.amax(dim=-1, keepdim=True)
        steps = torch.linspace(0.0, 1.0, codebook_size, device=grouped_weight.device, dtype=grouped_weight.dtype)
        quant_grid = mins + (maxs - mins) * steps.view(1, 1, -1)

        distances = torch.abs(grouped_weight.unsqueeze(-1) - quant_grid.unsqueeze(-2))
        weight_codes = distances.argmin(dim=-1).to(_select_code_dtype(codebook_size))
        return quant_grid.contiguous(), weight_codes.contiguous()

    @staticmethod
    def _kmeans_bootstrap(
        grouped_weight: torch.Tensor,
        codebook_size: int,
        num_iters: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out_features, groups_per_row, group_size = grouped_weight.shape
        quant_grid = torch.empty(
            out_features,
            groups_per_row,
            codebook_size,
            device=grouped_weight.device,
            dtype=grouped_weight.dtype,
        )
        weight_codes = torch.empty(
            out_features,
            groups_per_row,
            group_size,
            device=grouped_weight.device,
            dtype=_select_code_dtype(codebook_size),
        )

        step_template = torch.linspace(
            0.0,
            1.0,
            codebook_size,
            device=grouped_weight.device,
            dtype=grouped_weight.dtype,
        )

        for row_idx in range(out_features):
            for group_idx in range(groups_per_row):
                values = grouped_weight[row_idx, group_idx]
                min_val = values.min()
                max_val = values.max()

                if torch.isclose(min_val, max_val):
                    centers = values.new_full((codebook_size,), min_val)
                    codes = torch.zeros(group_size, device=values.device, dtype=weight_codes.dtype)
                else:
                    centers = min_val + (max_val - min_val) * step_template
                    for _ in range(max(1, num_iters)):
                        distances = torch.abs(values.unsqueeze(-1) - centers.unsqueeze(0))
                        assignments = distances.argmin(dim=-1)
                        new_centers = centers.clone()
                        for center_idx in range(codebook_size):
                            mask = assignments == center_idx
                            if mask.any():
                                new_centers[center_idx] = values[mask].mean()
                        centers = new_centers

                    sort_idx = centers.argsort()
                    centers = centers[sort_idx]
                    distances = torch.abs(values.unsqueeze(-1) - centers.unsqueeze(0))
                    codes = distances.argmin(dim=-1).to(weight_codes.dtype)

                quant_grid[row_idx, group_idx] = centers
                weight_codes[row_idx, group_idx] = codes

        return quant_grid.contiguous(), weight_codes.contiguous()

    def _get_reconstructed_weight(self, adapter: str) -> torch.Tensor:
        quant_grid = self.unilora_sketch_tune_quant_grid[adapter]
        codes = self.unilora_sketch_tune_weight_codes[adapter].to(device=quant_grid.device, dtype=torch.long)
        reconstructed = torch.gather(quant_grid, dim=-1, index=codes)
        reconstructed = reconstructed.reshape(self.out_features, -1)
        return reconstructed[:, : self.in_features]

    def _get_storage_weight(self, adapter: str) -> torch.Tensor:
        weight = self._get_reconstructed_weight(adapter)
        return transpose(weight, self.fan_in_fan_out)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return
        if len(adapter_names) > 1:
            raise NotImplementedError("UniLoRA-SketchTune merge currently supports one adapter at a time.")

        active_adapter = adapter_names[0]
        if active_adapter not in self.unilora_sketch_tune_quant_grid:
            return

        base_layer = self.get_base_layer()
        merged_weight = self._get_storage_weight(active_adapter)
        if safe_merge and not torch.isfinite(merged_weight).all():
            raise ValueError(f"NaNs detected in the merged weights for adapter {active_adapter}.")

        self._base_weight_backup = base_layer.weight.data.clone()
        base_layer.weight.data = merged_weight.to(device=base_layer.weight.device, dtype=base_layer.weight.dtype)
        self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self._base_weight_backup is None:
            raise RuntimeError("Cannot unmerge UniLoRA-SketchTune layer because no base weight backup was cached.")

        base_layer = self.get_base_layer()
        base_layer.weight.data = self._base_weight_backup.to(
            device=base_layer.weight.device, dtype=base_layer.weight.dtype
        )
        self._base_weight_backup = None
        self.merged_adapters.clear()


class Linear(nn.Linear, UniLoRASketchTuneLayer):
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        bits: int,
        groups_per_row: int,
        bootstrap_method: str,
        bootstrap_kmeans_iters: int,
        unilora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ) -> None:
        super(nn.Linear, self).__init__()
        UniLoRASketchTuneLayer.__init__(self, base_layer, fan_in_fan_out=fan_in_fan_out, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name=adapter_name,
            bits=bits,
            groups_per_row=groups_per_row,
            bootstrap_method=bootstrap_method,
            bootstrap_kmeans_iters=bootstrap_kmeans_iters,
            unilora_dropout=unilora_dropout,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return self.base_layer(x, *args, **kwargs)

        if self.merged:
            return self.base_layer(x, *args, **kwargs)

        chosen_adapter = None
        for active_adapter in self.active_adapters:
            if active_adapter in self.unilora_sketch_tune_quant_grid:
                chosen_adapter = active_adapter
                break

        if chosen_adapter is None:
            return self.base_layer(x, *args, **kwargs)

        weight = self._get_reconstructed_weight(chosen_adapter)
        compute_dtype = weight.dtype
        x_cast = x.to(compute_dtype)
        x_cast = self.unilora_dropout[chosen_adapter](x_cast)

        bias = self.bias
        if bias is not None:
            bias = bias.to(dtype=compute_dtype)

        result = F.linear(x_cast, weight, bias=bias)
        return result.to(previous_dtype)

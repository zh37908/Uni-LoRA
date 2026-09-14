from __future__ import annotations

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners._buffer_dict import BufferDict
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose


class LoraRoSALayer(BaseTunerLayer):
    adapter_layer_names = ("lora_A", "lora_B", "lora_rosa_sparse_values")
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        self.lora_rosa_selected_indices = BufferDict({}, persistent=True)
        self.lora_rosa_value_offsets = BufferDict({}, persistent=True)
        self.sparse_budget = {}
        self.sparse_active = {}
        self.capture_gradient_stats = False
        self._score_accum = {}
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if not isinstance(base_layer, nn.Linear):
            raise ValueError(f"LoRA-RoSA currently supports torch.nn.Linear only, got {type(base_layer)}.")
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def update_layer(
        self,
        adapter_name: str,
        lora_rosa_sparse_values,
        r: int,
        lora_alpha: int,
        lora_dropout: float = 0.0,
        init_lora_weights: bool | str = True,
    ) -> None:
        if r <= 0:
            raise ValueError(f"`r` should be positive, got {r}.")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.scaling[adapter_name] = lora_alpha / r
        self.lora_dropout[adapter_name] = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)
        self.lora_rosa_sparse_values = lora_rosa_sparse_values
        self.reset_lora_parameters(adapter_name, init_lora_weights)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.sparse_budget[adapter_name] = 0
        self.sparse_active[adapter_name] = False
        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name: str, init_lora_weights: bool | str) -> None:
        if init_lora_weights is False:
            return
        if init_lora_weights is True:
            nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
        elif str(init_lora_weights).lower() == "gaussian":
            nn.init.normal_(self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name])
        else:
            raise ValueError(f"LoRA-RoSA does not support init_lora_weights={init_lora_weights!r}.")
        nn.init.zeros_(self.lora_B[adapter_name].weight)

    def assign_sparse_value_offsets(self, adapter_name: str, start: int, count: int) -> None:
        self.sparse_budget[adapter_name] = int(count)
        device = self.get_base_layer().weight.device
        offsets = torch.arange(start, start + count, dtype=torch.long, device=device)
        self.lora_rosa_value_offsets[adapter_name] = offsets
        self.lora_rosa_selected_indices[adapter_name] = torch.empty(0, dtype=torch.long, device=device)
        self.sparse_active[adapter_name] = False

    def set_capture_gradient(self, enabled: bool = True) -> None:
        self.capture_gradient_stats = enabled
        weight = self.get_base_layer().weight
        weight.requires_grad_(enabled)
        if not enabled:
            weight.grad = None

    def clear_gradient_statistics(self, adapter_name: str) -> None:
        self._score_accum.pop(adapter_name, None)
        self.get_base_layer().weight.grad = None

    def accumulate_gradient_statistics(
        self,
        adapter_name: str,
        score_mode: str,
        grad_acc_mode: str = "max_abs",
    ) -> int:
        grad = self.get_base_layer().weight.grad
        if grad is None:
            return 0

        with torch.no_grad():
            if score_mode == "snip":
                raw_score = grad.detach() * self.get_base_layer().weight.detach()
            else:
                raw_score = grad.detach()
            if grad_acc_mode == "mean_squared":
                score = raw_score.float().square()
            else:
                score = raw_score.float().abs()
            flat_score = score.reshape(-1).cpu()
            previous = self._score_accum.get(adapter_name)
            if previous is None:
                self._score_accum[adapter_name] = flat_score
            elif grad_acc_mode == "max_abs":
                self._score_accum[adapter_name] = torch.maximum(previous, flat_score)
            else:
                # Division by the common number of mask batches is unnecessary
                # for TopK selection, so summing avoids an extra full-size pass.
                self._score_accum[adapter_name] = previous + flat_score
        self.get_base_layer().weight.grad = None
        return 1

    def generate_sparse_mask(
        self,
        adapter_name: str,
        score_mode: str,
        generator: torch.Generator | None = None,
    ) -> dict[str, float]:
        count = int(self.sparse_budget.get(adapter_name, 0))
        num_positions = self.get_base_layer().weight.numel()
        device = self.get_base_layer().weight.device
        count = max(0, min(count, num_positions))

        if count == 0:
            selected = torch.empty(0, dtype=torch.long, device=device)
        elif score_mode == "random":
            selected = torch.randperm(num_positions, generator=generator, device="cpu")[:count].to(device=device)
        else:
            score = self._score_accum.get(adapter_name)
            if score is None or score.numel() != num_positions:
                raise RuntimeError("LoRA-RoSA sparse mask generation did not receive valid base-weight scores.")
            score_max = float(score.max().item())
            if score_max <= 0.0:
                raise RuntimeError(
                    "LoRA-RoSA sparse mask generation found only zero scores; gradient capture did not run correctly."
                )
            selected = torch.topk(score, k=count, largest=True, sorted=False).indices.to(device=device)

        self.lora_rosa_selected_indices[adapter_name] = selected.long()
        self.sparse_active[adapter_name] = count > 0
        self.clear_gradient_statistics(adapter_name)
        return {
            "total_positions": int(num_positions),
            "selected_positions": int(count),
            "selected_density": 0.0 if num_positions == 0 else float(count) / float(num_positions),
            "selected_ratio": 0.0 if num_positions == 0 else float(count) / float(num_positions),
        }

    def has_sparse_mask(self, adapter_name: str) -> bool:
        return adapter_name in self.lora_rosa_selected_indices and self.lora_rosa_selected_indices[adapter_name].numel() > 0

    def set_sparse_requires_grad(self, adapter_name: str, requires_grad: bool) -> None:
        if adapter_name in self.lora_rosa_sparse_values:
            self.lora_rosa_sparse_values[adapter_name].requires_grad_(requires_grad)

    def sync_sparse_active_from_mask(self, adapter_name: str) -> bool:
        active = self.has_sparse_mask(adapter_name)
        self.sparse_active[adapter_name] = active
        return active

    def _sparse_values_for_adapter(self, adapter_name: str) -> torch.Tensor:
        values = self.lora_rosa_sparse_values[adapter_name]
        offsets = self.lora_rosa_value_offsets[adapter_name]
        active_count = self.lora_rosa_selected_indices[adapter_name].numel()
        return values[offsets[:active_count].to(values.device)]

    def _sparse_delta_weight(self, adapter_name: str) -> torch.Tensor:
        base_layer = self.get_base_layer()
        delta = torch.zeros_like(base_layer.weight)
        indices = self.lora_rosa_selected_indices[adapter_name].to(delta.device)
        if indices.numel() == 0:
            return delta
        values = self._sparse_values_for_adapter(adapter_name).to(device=delta.device, dtype=delta.dtype)
        delta.reshape(-1).index_add_(0, indices, values)
        return transpose(delta, False)

    def get_delta_weight(self, adapter_name: str) -> torch.Tensor:
        weight_A = self.lora_A[adapter_name].weight
        weight_B = self.lora_B[adapter_name].weight
        delta = (weight_B @ weight_A) * self.scaling[adapter_name]
        if self.has_sparse_mask(adapter_name):
            delta = delta + self._sparse_delta_weight(adapter_name).to(delta.dtype)
        return delta

    def _sparse_forward(self, x: torch.Tensor, adapter_name: str) -> torch.Tensor:
        selected = self.lora_rosa_selected_indices[adapter_name]
        if selected.numel() == 0:
            return x.new_zeros(*x.shape[:-1], self.out_features)

        values = self._sparse_values_for_adapter(adapter_name)
        dtype = values.dtype
        x_cast = x.to(dtype)
        x_flat = x_cast.reshape(-1, self.in_features)
        selected = selected.to(device=x_flat.device, dtype=torch.long)
        rows = torch.div(selected, self.in_features, rounding_mode="floor")
        cols = selected.remainder(self.in_features)
        contrib = x_flat[:, cols] * values.to(device=x_flat.device, dtype=dtype)
        out = x_flat.new_zeros(x_flat.shape[0], self.out_features)
        out.scatter_add_(1, rows.unsqueeze(0).expand(x_flat.shape[0], -1), contrib)
        return out.reshape(*x.shape[:-1], self.out_features)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return
        for active_adapter in adapter_names:
            if active_adapter not in self.lora_A:
                continue
            base_layer = self.get_base_layer()
            delta = self.get_delta_weight(active_adapter).to(base_layer.weight.dtype)
            merged = base_layer.weight.data.clone() + delta if safe_merge else base_layer.weight.data + delta
            if safe_merge and not torch.isfinite(merged).all():
                raise ValueError(f"NaNs detected in merged LoRA-RoSA weights for adapter {active_adapter}.")
            base_layer.weight.data = merged
            self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while self.merged_adapters:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A:
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter).to(
                    self.get_base_layer().weight.dtype
                )


class Linear(nn.Module, LoraRoSALayer):
    def __init__(
        self,
        base_layer: nn.Module,
        lora_rosa_sparse_values,
        adapter_name: str,
        r: int,
        lora_alpha: int,
        lora_dropout: float = 0.0,
        init_lora_weights: bool | str = True,
        **kwargs,
    ) -> None:
        super().__init__()
        LoraRoSALayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name=adapter_name,
            lora_rosa_sparse_values=lora_rosa_sparse_values,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
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
            result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A:
                    continue
                x_lora = x.to(self.lora_A[active_adapter].weight.dtype)
                result = result + self.lora_B[active_adapter](self.lora_A[active_adapter](self.lora_dropout[active_adapter](x_lora))) * self.scaling[active_adapter]
                if not self.sparse_active.get(active_adapter, False) and self.has_sparse_mask(active_adapter):
                    self.sync_sparse_active_from_mask(active_adapter)
                if self.sparse_active.get(active_adapter, False) and self.has_sparse_mask(active_adapter):
                    result = result + self._sparse_forward(x, active_adapter).to(result_dtype)
            result = result.to(result_dtype)
        return result.to(previous_dtype)

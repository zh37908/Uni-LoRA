import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose

from .._buffer_dict import BufferDict
from ..unilora_sketch_utils import (
    compute_group_setup,
    decode_shared_bank,
    generate_balanced_indices,
    select_code_dtype,
)


class UniLoRASketchRoutedLayer(BaseTunerLayer):
    adapter_layer_names = ("unilora_sketch_routed_experts", "unilora_sketch_routed_router_logits")
    other_param_names = (
        "unilora_sketch_routed_indices_A",
        "unilora_sketch_routed_indices_B",
        "unilora_sketch_routed_codes_A",
        "unilora_sketch_routed_codes_B",
    )

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.unilora_dropout = nn.ModuleDict({})

        self.unilora_sketch_routed_router_logits = nn.ParameterDict({})
        self.unilora_sketch_routed_indices_A = BufferDict({}, persistent=True)
        self.unilora_sketch_routed_indices_B = BufferDict({}, persistent=True)
        self.unilora_sketch_routed_codes_A = BufferDict({}, persistent=True)
        self.unilora_sketch_routed_codes_B = BufferDict({}, persistent=True)

        self.sketch_routed_groups_A = {}
        self.sketch_routed_groups_B = {}
        self.sketch_routed_group_size_A = {}
        self.sketch_routed_group_size_B = {}
        self.sketch_routed_codebook_size = {}
        self.sketch_routed_num_banks = {}
        self.sketch_router_tau = {}
        self.sketch_router_mode = {}
        self.sketch_router_gumbel_hard = {}
        self.sketch_router_hard_eval = {}

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
            raise ValueError(f"Unsupported base layer type {type(base_layer)} for UniLoRASketchRoutedLayer.")

        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def update_layer(
        self,
        adapter_name: str,
        unilora_sketch_routed_experts,
        r: int,
        bits: int,
        groups_per_row: int,
        num_banks: int,
        proj_seed: int,
        layer_seed: int,
        router_tau: float,
        router_mode: str,
        router_gumbel_hard: bool,
        router_hard_eval: bool,
        unilora_dropout: float = 0.0,
    ):
        if r <= 0:
            raise ValueError(f"`r` should be positive, got {r}.")

        codebook_size = 1 << bits
        self.r[adapter_name] = r
        self.unilora_sketch_routed_experts = unilora_sketch_routed_experts
        self.sketch_routed_codebook_size[adapter_name] = codebook_size
        self.sketch_routed_num_banks[adapter_name] = num_banks
        self.sketch_router_tau[adapter_name] = float(router_tau)
        self.sketch_router_mode[adapter_name] = router_mode
        self.sketch_router_gumbel_hard[adapter_name] = bool(router_gumbel_hard)
        self.sketch_router_hard_eval[adapter_name] = bool(router_hard_eval)

        if unilora_dropout > 0.0:
            dropout_layer = nn.Dropout(p=unilora_dropout)
        else:
            dropout_layer = nn.Identity()
        self.unilora_dropout.update(nn.ModuleDict({adapter_name: dropout_layer}))

        group_size_A, _ = compute_group_setup(self.in_features, groups_per_row)
        group_size_B, _ = compute_group_setup(r, groups_per_row)
        groups_A = (self.in_features + group_size_A - 1) // group_size_A
        groups_B = (r + group_size_B - 1) // group_size_B
        self.sketch_routed_groups_A[adapter_name] = groups_A
        self.sketch_routed_groups_B[adapter_name] = groups_B
        self.sketch_routed_group_size_A[adapter_name] = group_size_A
        self.sketch_routed_group_size_B[adapter_name] = group_size_B

        router_logits = torch.empty(unilora_sketch_routed_experts[adapter_name].shape[0])
        nn.init.normal_(router_logits, mean=0.0, std=0.01)
        self.unilora_sketch_routed_router_logits[adapter_name] = nn.Parameter(router_logits)

        numel_A = r * groups_A * group_size_A
        numel_B = self.out_features * groups_B * group_size_B
        logical_A = generate_balanced_indices(numel_A, num_banks * codebook_size, proj_seed + layer_seed)
        logical_B = generate_balanced_indices(numel_B, num_banks * codebook_size, proj_seed + layer_seed + 1)
        self.set_logical_assignments(adapter_name, logical_A, logical_B)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def set_logical_assignments(
        self,
        adapter_name: str,
        logical_ids_A: torch.Tensor,
        logical_ids_B: torch.Tensor,
    ) -> None:
        codebook_size = self.sketch_routed_codebook_size[adapter_name]
        code_dtype = select_code_dtype(codebook_size)

        shape_A = (
            self.r[adapter_name],
            self.sketch_routed_groups_A[adapter_name],
            self.sketch_routed_group_size_A[adapter_name],
        )
        shape_B = (
            self.out_features,
            self.sketch_routed_groups_B[adapter_name],
            self.sketch_routed_group_size_B[adapter_name],
        )

        logical_ids_A = logical_ids_A.view(shape_A)
        logical_ids_B = logical_ids_B.view(shape_B)

        self.unilora_sketch_routed_indices_A[adapter_name] = (logical_ids_A // codebook_size).to(code_dtype)
        self.unilora_sketch_routed_indices_B[adapter_name] = (logical_ids_B // codebook_size).to(code_dtype)
        self.unilora_sketch_routed_codes_A[adapter_name] = (logical_ids_A % codebook_size).to(code_dtype)
        self.unilora_sketch_routed_codes_B[adapter_name] = (logical_ids_B % codebook_size).to(code_dtype)

    def _compute_router_probs(self, adapter: str) -> torch.Tensor:
        logits = self.unilora_sketch_routed_router_logits[adapter]
        tau = max(self.sketch_router_tau[adapter], 1e-6)
        if self.training:
            if self.sketch_router_mode[adapter] == "softmax":
                return torch.softmax(logits / tau, dim=-1)
            if self.sketch_router_mode[adapter] == "gumbel":
                return F.gumbel_softmax(
                    logits,
                    tau=tau,
                    hard=self.sketch_router_gumbel_hard[adapter],
                    dim=-1,
                )
            raise ValueError(f"Unsupported router mode: {self.sketch_router_mode[adapter]}")

        if self.sketch_router_hard_eval[adapter]:
            choice = logits.argmax(dim=-1, keepdim=True)
            return torch.zeros_like(logits).scatter_(-1, choice, 1.0)
        return torch.softmax(logits / tau, dim=-1)

    def _get_effective_bank(self, adapter: str, cast_to_fp32: bool = False) -> torch.Tensor:
        experts = self.unilora_sketch_routed_experts[adapter]
        if cast_to_fp32:
            experts = experts.float()
        router_probs = self._compute_router_probs(adapter).to(device=experts.device, dtype=experts.dtype)
        return torch.einsum("e,ebk->bk", router_probs, experts)

    def _get_lora_matrices(self, adapter: str, cast_to_fp32: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        effective_bank = self._get_effective_bank(adapter, cast_to_fp32=cast_to_fp32)
        indices_A = self.unilora_sketch_routed_indices_A[adapter].to(device=effective_bank.device)
        indices_B = self.unilora_sketch_routed_indices_B[adapter].to(device=effective_bank.device)
        codes_A = self.unilora_sketch_routed_codes_A[adapter].to(device=effective_bank.device)
        codes_B = self.unilora_sketch_routed_codes_B[adapter].to(device=effective_bank.device)

        A = decode_shared_bank(effective_bank, indices_A, codes_A, self.in_features)
        B = decode_shared_bank(effective_bank, indices_B, codes_B, self.r[adapter])
        return A, B

    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        device = self.unilora_sketch_routed_experts[adapter].device
        dtype = self.unilora_sketch_routed_experts[adapter].dtype
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16
        A, B = self._get_lora_matrices(adapter, cast_to_fp32=cast_to_fp32)
        return transpose(B @ A, self.kwargs.get("fan_in_fan_out", False))

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter in self.unilora_sketch_routed_experts.keys():
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
            if active_adapter in self.unilora_sketch_routed_experts.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)


class Linear(nn.Linear, UniLoRASketchRoutedLayer):
    def __init__(
        self,
        base_layer,
        unilora_sketch_routed_experts,
        adapter_name: str,
        r: int,
        bits: int,
        groups_per_row: int,
        num_banks: int,
        proj_seed: int,
        layer_seed: int,
        router_tau: float,
        router_mode: str,
        router_gumbel_hard: bool,
        router_hard_eval: bool,
        unilora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ) -> None:
        super(nn.Linear, self).__init__()
        UniLoRASketchRoutedLayer.__init__(self, base_layer, fan_in_fan_out=fan_in_fan_out, **kwargs)
        self._active_adapter = adapter_name
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
        self.update_layer(
            adapter_name=adapter_name,
            unilora_sketch_routed_experts=unilora_sketch_routed_experts,
            r=r,
            bits=bits,
            groups_per_row=groups_per_row,
            num_banks=num_banks,
            proj_seed=proj_seed,
            layer_seed=layer_seed,
            router_tau=router_tau,
            router_mode=router_mode,
            router_gumbel_hard=router_gumbel_hard,
            router_hard_eval=router_hard_eval,
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
                if active_adapter not in self.unilora_sketch_routed_experts.keys():
                    continue

                A, B = self._get_lora_matrices(active_adapter)
                x_cast = x.to(self.unilora_sketch_routed_experts[active_adapter].dtype)
                result = result + F.linear(
                    F.linear(self.unilora_dropout[active_adapter](x_cast), A),
                    B,
                )

        return result.to(previous_dtype)

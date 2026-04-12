from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose

from .._buffer_dict import BufferDict


class UniLoRAIGULayer(BaseTunerLayer):
    adapter_layer_names = ("unilora_igu_theta_d", "unilora_igu_lora_E")

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.unilora_dropout = nn.ModuleDict({})
        self.unilora_igu_lora_E = nn.ParameterDict({})

        self.unilora_indices_A = BufferDict({}, persistent=True)
        self.unilora_indices_B = BufferDict({}, persistent=True)
        self.unilora_scales_A = BufferDict({}, persistent=True)
        self.unilora_scales_B = BufferDict({}, persistent=True)
        self.unilora_igu_lora_mask = BufferDict({}, persistent=True)
        self.unilora_igu_ranknum = BufferDict({}, persistent=True)
        self.unilora_igu_weight_coeff = BufferDict({}, persistent=True)
        self.unilora_igu_exp_avg_ipt_A = BufferDict({}, persistent=False)
        self.unilora_igu_exp_avg_unc_A = BufferDict({}, persistent=False)
        self.unilora_igu_exp_avg_ipt_B = BufferDict({}, persistent=False)
        self.unilora_igu_exp_avg_unc_B = BufferDict({}, persistent=False)
        self.unilora_igu_exp_avg_ipt_E = BufferDict({}, persistent=False)
        self.unilora_igu_exp_avg_unc_E = BufferDict({}, persistent=False)

        self._disable_adapters = False
        self.merged_adapters = []
        self.capture_rank_stats = False
        self._last_A = {}
        self._last_B = {}

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        else:
            raise ValueError(f"Unsupported base layer type {type(base_layer)} for UniLoRA-IGU.")

        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def update_layer(
        self,
        adapter_name: str,
        unilora_igu_theta_d,
        r: int,
        theta_d_length: int,
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

        self.unilora_igu_theta_d = unilora_igu_theta_d
        self.reset_unilora_parameters(adapter_name, theta_d_length)

        base_layer = self.get_base_layer()
        target_device = base_layer.weight.device
        target_dtype = base_layer.weight.dtype
        self.unilora_igu_lora_E[adapter_name] = nn.Parameter(0.1 * torch.ones((r, 1), device=target_device, dtype=target_dtype))
        self.unilora_igu_lora_mask[adapter_name] = torch.ones((r, 1), device=target_device, dtype=target_dtype)
        self.unilora_igu_ranknum[adapter_name] = torch.tensor(float(r), device=target_device, dtype=target_dtype)
        self.unilora_igu_weight_coeff[adapter_name] = torch.ones(1, device=target_device, dtype=target_dtype)
        self.unilora_igu_exp_avg_ipt_A[adapter_name] = torch.zeros((r, self.in_features), device=target_device, dtype=torch.float32)
        self.unilora_igu_exp_avg_unc_A[adapter_name] = torch.zeros((r, self.in_features), device=target_device, dtype=torch.float32)
        self.unilora_igu_exp_avg_ipt_B[adapter_name] = torch.zeros((self.out_features, r), device=target_device, dtype=torch.float32)
        self.unilora_igu_exp_avg_unc_B[adapter_name] = torch.zeros((self.out_features, r), device=target_device, dtype=torch.float32)
        self.unilora_igu_exp_avg_ipt_E[adapter_name] = torch.zeros((r, 1), device=target_device, dtype=torch.float32)
        self.unilora_igu_exp_avg_unc_E[adapter_name] = torch.zeros((r, 1), device=target_device, dtype=torch.float32)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_unilora_parameters(self, adapter_name: str, theta_d_length: int):
        if adapter_name in self.unilora_igu_theta_d.keys():
            indices_A = torch.randint(0, theta_d_length, (self.r[adapter_name], self.in_features), dtype=torch.long)
            indices_B = torch.randint(0, theta_d_length, (self.out_features, self.r[adapter_name]), dtype=torch.long)
            self.unilora_indices_A[adapter_name] = indices_A
            self.unilora_indices_B[adapter_name] = indices_B

    def update_norm(self, adapter_name: str, unilora_scales_A: torch.Tensor, unilora_scales_B: torch.Tensor):
        if adapter_name in self.unilora_igu_theta_d.keys():
            base_layer = self.get_base_layer()
            target_device = base_layer.weight.device
            target_dtype = base_layer.weight.dtype
            self.unilora_scales_A[adapter_name] = unilora_scales_A.to(device=target_device, dtype=target_dtype)
            self.unilora_scales_B[adapter_name] = unilora_scales_B.to(device=target_device, dtype=target_dtype)

    def set_capture_rank_stats(self, enabled: bool = True) -> None:
        self.capture_rank_stats = enabled
        if not enabled:
            self._last_A.clear()
            self._last_B.clear()

    def clear_cached_rank_stats(self, adapter_name: str) -> None:
        self._last_A[adapter_name] = None
        self._last_B[adapter_name] = None

    def get_active_rank_count(self, adapter_name: str) -> int:
        if adapter_name not in self.unilora_igu_lora_mask:
            return 0
        return int((self.unilora_igu_lora_mask[adapter_name] > 0).sum().item())

    def get_rank_scores(self, adapter_name: str, eps: float = 1e-6) -> torch.Tensor:
        score_e = (self.unilora_igu_exp_avg_ipt_E[adapter_name] + eps) / (self.unilora_igu_exp_avg_unc_E[adapter_name] + eps)
        score_a = (self.unilora_igu_exp_avg_ipt_A[adapter_name] + eps) / (self.unilora_igu_exp_avg_unc_A[adapter_name] + eps)
        score_b = (self.unilora_igu_exp_avg_ipt_B[adapter_name] + eps) / (self.unilora_igu_exp_avg_unc_B[adapter_name] + eps)
        return score_e.view(-1) + score_a.mean(dim=1) + score_b.mean(dim=0)

    def accumulate_rank_statistics(self, adapter_name: str, beta1: float, beta2: float) -> dict[str, int]:
        if adapter_name not in self.unilora_igu_lora_mask:
            return {"updated_ranks": 0, "updated_tensors": 0}

        updated_tensors = 0

        if adapter_name in self._last_A and self._last_A[adapter_name] is not None:
            grad_A = self._last_A[adapter_name].grad
            if grad_A is not None:
                raw_A = (self._last_A[adapter_name].detach() * grad_A.detach()).abs().to(dtype=torch.float32)
                prev_ipt_A = self.unilora_igu_exp_avg_ipt_A[adapter_name]
                new_ipt_A = beta1 * prev_ipt_A + (1.0 - beta1) * raw_A
                new_unc_A = beta2 * self.unilora_igu_exp_avg_unc_A[adapter_name] + (1.0 - beta2) * (raw_A - new_ipt_A).abs()
                self.unilora_igu_exp_avg_ipt_A[adapter_name] = new_ipt_A
                self.unilora_igu_exp_avg_unc_A[adapter_name] = new_unc_A
                updated_tensors += 1

        if adapter_name in self._last_B and self._last_B[adapter_name] is not None:
            grad_B = self._last_B[adapter_name].grad
            if grad_B is not None:
                raw_B = (self._last_B[adapter_name].detach() * grad_B.detach()).abs().to(dtype=torch.float32)
                prev_ipt_B = self.unilora_igu_exp_avg_ipt_B[adapter_name]
                new_ipt_B = beta1 * prev_ipt_B + (1.0 - beta1) * raw_B
                new_unc_B = beta2 * self.unilora_igu_exp_avg_unc_B[adapter_name] + (1.0 - beta2) * (raw_B - new_ipt_B).abs()
                self.unilora_igu_exp_avg_ipt_B[adapter_name] = new_ipt_B
                self.unilora_igu_exp_avg_unc_B[adapter_name] = new_unc_B
                updated_tensors += 1

        lora_E = self.unilora_igu_lora_E[adapter_name]
        if lora_E.grad is not None:
            raw_E = (lora_E.detach() * lora_E.grad.detach()).abs().to(dtype=torch.float32)
            prev_ipt_E = self.unilora_igu_exp_avg_ipt_E[adapter_name]
            new_ipt_E = beta1 * prev_ipt_E + (1.0 - beta1) * raw_E
            new_unc_E = beta2 * self.unilora_igu_exp_avg_unc_E[adapter_name] + (1.0 - beta2) * (raw_E - new_ipt_E).abs()
            self.unilora_igu_exp_avg_ipt_E[adapter_name] = new_ipt_E
            self.unilora_igu_exp_avg_unc_E[adapter_name] = new_unc_E
            updated_tensors += 1

        self.clear_cached_rank_stats(adapter_name)
        if updated_tensors == 0:
            return {"updated_ranks": 0, "updated_tensors": 0}
        return {"updated_ranks": int(self.r[adapter_name]), "updated_tensors": updated_tensors}

    @torch.no_grad()
    def prune_ranks(self, adapter_name: str, rank_indices: list[int]) -> None:
        if not rank_indices:
            return
        base_layer = self.get_base_layer()
        mask = self.unilora_igu_lora_mask[adapter_name].detach().clone()
        index_tensor = torch.tensor(rank_indices, device=mask.device, dtype=torch.long)
        mask[index_tensor] = 0
        self.unilora_igu_lora_mask[adapter_name] = mask.to(device=base_layer.weight.device, dtype=base_layer.weight.dtype)
        self.unilora_igu_lora_E[adapter_name].data[index_tensor] = 0
        active_rank = int((self.unilora_igu_lora_mask[adapter_name] > 0).sum().item())
        self.unilora_igu_ranknum[adapter_name] = torch.tensor(
            float(active_rank),
            device=base_layer.weight.device,
            dtype=base_layer.weight.dtype,
        )


class Linear(nn.Linear, UniLoRAIGULayer):
    def __init__(
        self,
        base_layer,
        unilora_igu_theta_d,
        adapter_name: str,
        r: int,
        theta_d_length: int,
        unilora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ) -> None:
        super(nn.Linear, self).__init__()
        UniLoRAIGULayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, unilora_igu_theta_d, r, theta_d_length, unilora_dropout)
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
            if active_adapter in self.unilora_indices_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def _get_lora_matrices(self, adapter: str, cast_to_fp32: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        indices_A = self.unilora_indices_A[adapter]
        indices_B = self.unilora_indices_B[adapter]
        theta_d = self.unilora_igu_theta_d[adapter].to(indices_A.device)
        scales_A = self.unilora_scales_A[adapter]
        scales_B = self.unilora_scales_B[adapter]

        if cast_to_fp32:
            theta_d = theta_d.float()
            scales_A = scales_A.float()
            scales_B = scales_B.float()

        base_A = theta_d[indices_A.long()] * scales_A
        base_B = theta_d[indices_B.long()] * scales_B
        A = base_A
        B = base_B

        if self.capture_rank_stats and self.training and not cast_to_fp32:
            self._last_A[adapter] = A if A.requires_grad else None
            self._last_B[adapter] = B if B.requires_grad else None
            if A.requires_grad:
                A.retain_grad()
            if B.requires_grad:
                B.retain_grad()

        return A, B

    def get_delta_weight(self, adapter) -> torch.Tensor:
        device = self.unilora_indices_A[adapter].device
        dtype = self.unilora_igu_theta_d[adapter].dtype
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16
        A, B = self._get_lora_matrices(adapter, cast_to_fp32)
        lora_E = self.unilora_igu_lora_E[adapter].to(device=device, dtype=A.dtype)
        lora_mask = self.unilora_igu_lora_mask[adapter].to(device=device, dtype=A.dtype)
        ranknum = self.unilora_igu_ranknum[adapter].to(device=device, dtype=A.dtype)
        weight_coeff = self.unilora_igu_weight_coeff[adapter].to(device=device, dtype=A.dtype)
        scaled_A = A * (lora_E * lora_mask)
        return transpose((B @ scaled_A) * (weight_coeff**3) / (ranknum + 1e-5), self.fan_in_fan_out)

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
                x_cast = x.to(self.unilora_igu_theta_d[active_adapter].dtype)
                dropout = self.unilora_dropout[active_adapter]
                lora_E = self.unilora_igu_lora_E[active_adapter].to(dtype=A.dtype)
                lora_mask = self.unilora_igu_lora_mask[active_adapter].to(dtype=A.dtype)
                ranknum = self.unilora_igu_ranknum[active_adapter].to(dtype=A.dtype)
                weight_coeff = self.unilora_igu_weight_coeff[active_adapter].to(dtype=A.dtype)
                tmp_1 = F.linear(dropout(x_cast), A)
                tmp_2 = tmp_1 * (lora_E * lora_mask).view(*([1] * (tmp_1.dim() - 1)), -1)
                tmp_3 = F.linear(tmp_2, B)
                result = result + tmp_3 * (weight_coeff**3) / (ranknum + 1e-5)

        return result.to(previous_dtype)

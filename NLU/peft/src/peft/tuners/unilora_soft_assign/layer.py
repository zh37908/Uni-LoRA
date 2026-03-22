import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose

from .._buffer_dict import BufferDict


class UniLoRASoftAssignLayer(BaseTunerLayer):
    adapter_layer_names = (
        "unilora_soft_assign_theta_d",
        "unilora_soft_assign_logits_A",
        "unilora_soft_assign_logits_B",
    )
    other_param_names = (
        "unilora_soft_assign_candidate_indices_A",
        "unilora_soft_assign_candidate_indices_B",
        "unilora_soft_assign_scales_A",
        "unilora_soft_assign_scales_B",
    )

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.num_candidates = {}
        self.unilora_dropout = nn.ModuleDict({})

        self.unilora_soft_assign_logits_A = nn.ParameterDict({})
        self.unilora_soft_assign_logits_B = nn.ParameterDict({})
        self.unilora_soft_assign_candidate_indices_A = BufferDict({}, persistent=True)
        self.unilora_soft_assign_candidate_indices_B = BufferDict({}, persistent=True)
        self.unilora_soft_assign_scales_A = BufferDict({}, persistent=True)
        self.unilora_soft_assign_scales_B = BufferDict({}, persistent=True)

        self.unilora_soft_assign_temperature = {}
        self.unilora_soft_assign_assignment_mode = {}
        self.unilora_soft_assign_gumbel_hard = {}
        self.unilora_soft_assign_hard_eval = {}

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
            raise ValueError(f"Unsupported base layer type {type(base_layer)} for UniLoRASoftAssignLayer.")

        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def update_layer(
        self,
        adapter_name: str,
        unilora_soft_assign_theta_d,
        r: int,
        theta_d_length: int,
        num_candidates: int,
        init_logits_std: float,
        temperature: float,
        assignment_mode: str,
        gumbel_hard: bool,
        hard_eval: bool,
        unilora_dropout: float = 0.0,
    ):
        if r <= 0:
            raise ValueError(f"`r` {r} should be a positive integer value")
        if num_candidates <= 0:
            raise ValueError(f"`num_candidates` {num_candidates} should be a positive integer value")
        if temperature <= 0.0:
            raise ValueError(f"`temperature` {temperature} should be > 0")

        self.r[adapter_name] = r
        self.num_candidates[adapter_name] = num_candidates
        self.unilora_soft_assign_temperature[adapter_name] = float(temperature)
        self.unilora_soft_assign_assignment_mode[adapter_name] = assignment_mode
        self.unilora_soft_assign_gumbel_hard[adapter_name] = bool(gumbel_hard)
        self.unilora_soft_assign_hard_eval[adapter_name] = bool(hard_eval)

        if unilora_dropout > 0.0:
            unilora_dropout_layer = nn.Dropout(p=unilora_dropout)
        else:
            unilora_dropout_layer = nn.Identity()
        self.unilora_dropout.update(nn.ModuleDict({adapter_name: unilora_dropout_layer}))

        self.unilora_soft_assign_theta_d = unilora_soft_assign_theta_d

        logits_A = torch.empty((r, self.in_features, num_candidates))
        logits_B = torch.empty((self.out_features, r, num_candidates))
        if init_logits_std > 0.0:
            torch.nn.init.normal_(logits_A, mean=0.0, std=init_logits_std)
            torch.nn.init.normal_(logits_B, mean=0.0, std=init_logits_std)
        else:
            nn.init.zeros_(logits_A)
            nn.init.zeros_(logits_B)

        candidate_indices_A = torch.randint(
            0,
            theta_d_length,
            (r, self.in_features, num_candidates),
            dtype=torch.long,
        )
        candidate_indices_B = torch.randint(
            0,
            theta_d_length,
            (self.out_features, r, num_candidates),
            dtype=torch.long,
        )

        self.unilora_soft_assign_logits_A[adapter_name] = nn.Parameter(logits_A)
        self.unilora_soft_assign_logits_B[adapter_name] = nn.Parameter(logits_B)
        self.unilora_soft_assign_candidate_indices_A[adapter_name] = candidate_indices_A
        self.unilora_soft_assign_candidate_indices_B[adapter_name] = candidate_indices_B

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def set_temperature(self, adapter_name: str, temperature: float) -> None:
        if adapter_name in self.unilora_soft_assign_temperature:
            self.unilora_soft_assign_temperature[adapter_name] = float(temperature)

    def seed_primary_candidates(
        self,
        adapter_name: str,
        indices_A: torch.Tensor,
        indices_B: torch.Tensor,
        init_primary_bias: float,
    ) -> None:
        if adapter_name not in self.unilora_soft_assign_candidate_indices_A:
            return

        candidate_indices_A = self.unilora_soft_assign_candidate_indices_A[adapter_name].clone()
        candidate_indices_B = self.unilora_soft_assign_candidate_indices_B[adapter_name].clone()
        candidate_indices_A[..., 0] = indices_A.to(candidate_indices_A.device)
        candidate_indices_B[..., 0] = indices_B.to(candidate_indices_B.device)
        self.unilora_soft_assign_candidate_indices_A[adapter_name] = candidate_indices_A
        self.unilora_soft_assign_candidate_indices_B[adapter_name] = candidate_indices_B

        if init_primary_bias != 0.0:
            self.unilora_soft_assign_logits_A[adapter_name].data[..., 0] += init_primary_bias
            self.unilora_soft_assign_logits_B[adapter_name].data[..., 0] += init_primary_bias

    def update_norm(
        self,
        adapter_name: str,
        unilora_scales_A: torch.Tensor,
        unilora_scales_B: torch.Tensor,
    ) -> None:
        if adapter_name not in self.unilora_soft_assign_logits_A.keys():
            return

        base_layer = self.get_base_layer()
        target_device = base_layer.weight.device
        target_dtype = base_layer.weight.dtype
        self.unilora_soft_assign_scales_A[adapter_name] = unilora_scales_A.to(device=target_device, dtype=target_dtype)
        self.unilora_soft_assign_scales_B[adapter_name] = unilora_scales_B.to(device=target_device, dtype=target_dtype)

    def _compute_probs(self, logits: torch.Tensor, adapter: str) -> torch.Tensor:
        temperature = max(self.unilora_soft_assign_temperature[adapter], 1e-6)
        if self.training:
            if self.unilora_soft_assign_assignment_mode[adapter] == "softmax":
                return torch.softmax(logits / temperature, dim=-1)
            if self.unilora_soft_assign_assignment_mode[adapter] == "gumbel":
                return F.gumbel_softmax(
                    logits,
                    tau=temperature,
                    hard=self.unilora_soft_assign_gumbel_hard[adapter],
                    dim=-1,
                )
            raise ValueError(
                f"Unsupported assignment mode: {self.unilora_soft_assign_assignment_mode[adapter]}"
            )

        if self.unilora_soft_assign_hard_eval[adapter]:
            hard_choice = logits.argmax(dim=-1, keepdim=True)
            return torch.zeros_like(logits).scatter_(-1, hard_choice, 1.0)
        return torch.softmax(logits / temperature, dim=-1)

    def _get_lora_matrices(self, adapter: str, cast_to_fp32: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        logits_A = self.unilora_soft_assign_logits_A[adapter]
        logits_B = self.unilora_soft_assign_logits_B[adapter]
        candidate_indices_A = self.unilora_soft_assign_candidate_indices_A[adapter]
        candidate_indices_B = self.unilora_soft_assign_candidate_indices_B[adapter]
        scales_A = self.unilora_soft_assign_scales_A[adapter]
        scales_B = self.unilora_soft_assign_scales_B[adapter]
        theta_d = self.unilora_soft_assign_theta_d[adapter].to(logits_A.device)

        if cast_to_fp32:
            logits_A = logits_A.float()
            logits_B = logits_B.float()
            scales_A = scales_A.float()
            scales_B = scales_B.float()
            theta_d = theta_d.float()

        probs_A = self._compute_probs(logits_A, adapter)
        probs_B = self._compute_probs(logits_B, adapter)
        candidate_values_A = theta_d[candidate_indices_A.long()] * scales_A
        candidate_values_B = theta_d[candidate_indices_B.long()] * scales_B
        A = (probs_A * candidate_values_A).sum(dim=-1)
        B = (probs_B * candidate_values_B).sum(dim=-1)
        return A, B

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter in self.unilora_soft_assign_logits_A.keys():
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
            if active_adapter in self.unilora_soft_assign_logits_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)


class Linear(nn.Linear, UniLoRASoftAssignLayer):
    def __init__(
        self,
        base_layer,
        unilora_soft_assign_theta_d,
        adapter_name: str,
        r: int,
        theta_d_length: int,
        num_candidates: int,
        init_logits_std: float,
        temperature: float,
        assignment_mode: str,
        gumbel_hard: bool,
        hard_eval: bool,
        unilora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ) -> None:
        super(nn.Linear, self).__init__()
        UniLoRASoftAssignLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name=adapter_name,
            unilora_soft_assign_theta_d=unilora_soft_assign_theta_d,
            r=r,
            theta_d_length=theta_d_length,
            num_candidates=num_candidates,
            init_logits_std=init_logits_std,
            temperature=temperature,
            assignment_mode=assignment_mode,
            gumbel_hard=gumbel_hard,
            hard_eval=hard_eval,
            unilora_dropout=unilora_dropout,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def get_delta_weight(self, adapter) -> torch.Tensor:
        device = self.unilora_soft_assign_logits_A[adapter].device
        dtype = self.unilora_soft_assign_theta_d[adapter].dtype
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16
        A, B = self._get_lora_matrices(adapter, cast_to_fp32)
        return transpose(B @ A, self.fan_in_fan_out)

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
                if active_adapter not in self.unilora_soft_assign_logits_A.keys():
                    continue

                A, B = self._get_lora_matrices(active_adapter)
                x_cast = x.to(self.unilora_soft_assign_theta_d[active_adapter].dtype)
                dropout = self.unilora_dropout[active_adapter]
                result = result + F.linear(F.linear(dropout(x_cast), A), B)

        return result.to(previous_dtype)

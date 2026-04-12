import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.unilora_soft_assign.layer import UniLoRASoftAssignLayer
from peft.utils.other import transpose


class UniLoRASoftWeightSharingLayer(UniLoRASoftAssignLayer):
    """
    Marker subclass of UniLoRA-SoftAssign layer. The injected `Linear` below must inherit
    from this class so `isinstance(module, UniLoRASoftWeightSharingLayer)` succeeds in the
    soft-weight-sharing model (loss collection, finalize, export).
    """


class Linear(nn.Linear, UniLoRASoftWeightSharingLayer):
    """Same as UniLoRA-SoftAssign `Linear` but with correct MRO for soft weight-sharing."""

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
        UniLoRASoftWeightSharingLayer.__init__(self, base_layer, **kwargs)
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

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

from .config import UniLoRASoftAssignConfig
from .layer import Linear, UniLoRASoftAssignLayer


class UniLoRASoftAssignModel(BaseTuner):
    """
    UniLoRA-SoftAssign: each A/B entry softly routes to a small candidate subset
    of the shared theta_d bank.
    """

    prefix: str = "unilora_soft_assign_"
    tuner_layer_cls = UniLoRASoftAssignLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

        if isinstance(config, dict):
            unilora_config = config[adapter_name]
        else:
            unilora_config = config

        total_params = 0
        soft_assign_modules = []
        for _, module in model.named_modules():
            if isinstance(module, UniLoRASoftAssignLayer):
                soft_assign_modules.append(module)
                total_params += module.r[adapter_name] * module.in_features
                total_params += module.out_features * module.r[adapter_name]

        primary_indices = self.generate_index(total_params, unilora_config.theta_d_length, unilora_config.proj_seed)
        pointer = 0
        for module in soft_assign_modules:
            a_numel = module.r[adapter_name] * module.in_features
            a_chunk = primary_indices[pointer : pointer + a_numel]
            pointer += a_numel

            b_numel = module.out_features * module.r[adapter_name]
            b_chunk = primary_indices[pointer : pointer + b_numel]
            pointer += b_numel

            module.seed_primary_candidates(
                adapter_name,
                a_chunk.view(module.r[adapter_name], module.in_features).clone(),
                b_chunk.view(module.out_features, module.r[adapter_name]).clone(),
                unilora_config.init_primary_bias,
            )

        assert pointer == len(primary_indices)

        if soft_assign_modules:
            all_indices = torch.cat(
                [
                    module.unilora_soft_assign_candidate_indices_A[adapter_name].reshape(-1).long()
                    for module in soft_assign_modules
                ]
                + [
                    module.unilora_soft_assign_candidate_indices_B[adapter_name].reshape(-1).long()
                    for module in soft_assign_modules
                ],
                dim=0,
            )
            counts = torch.bincount(all_indices, minlength=unilora_config.theta_d_length)
            inv_sqrt_counts = torch.zeros(unilora_config.theta_d_length, dtype=torch.float32)
            non_zero = counts > 0
            inv_sqrt_counts[non_zero] = 1.0 / torch.sqrt(counts[non_zero].float())

            for module in soft_assign_modules:
                scale_A = inv_sqrt_counts[module.unilora_soft_assign_candidate_indices_A[adapter_name].long()]
                scale_B = inv_sqrt_counts[module.unilora_soft_assign_candidate_indices_B[adapter_name].long()]
                module.update_norm(adapter_name, scale_A, scale_B)

    def generate_index(self, total_length, theta_d_length, proj_seed):
        import numpy as np

        base_count = total_length // theta_d_length
        remaining = total_length % theta_d_length
        rng = np.random.default_rng(proj_seed)
        data = np.repeat(np.arange(theta_d_length), base_count)
        if remaining > 0:
            extras = rng.choice(theta_d_length, size=remaining, replace=False)
            data = np.concatenate([data, extras])
        rng.shuffle(data)
        return torch.tensor(data)

    def _init_unilora_theta_d(self, config: UniLoRASoftAssignConfig, adapter_name: str) -> None:
        theta_d = torch.zeros(config.theta_d_length)
        torch.nn.init.uniform_(theta_d, -config.init_theta_d_bound, config.init_theta_d_bound)
        self.unilora_soft_assign_theta_d[adapter_name] = theta_d

    def _pre_injection_hook(self, model: nn.Module, config: UniLoRASoftAssignConfig, adapter_name: str) -> None:
        self.unilora_soft_assign_theta_d = nn.ParameterDict({})

    def _create_and_replace(
        self,
        unilora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "fan_in_fan_out": unilora_config.fan_in_fan_out,
            "bias": bias,
        }
        self._init_unilora_theta_d(unilora_config, adapter_name)

        if isinstance(target, Linear):
            target.update_layer(
                adapter_name=adapter_name,
                unilora_soft_assign_theta_d=self.unilora_soft_assign_theta_d,
                r=unilora_config.r,
                theta_d_length=unilora_config.theta_d_length,
                num_candidates=unilora_config.num_candidates,
                init_logits_std=unilora_config.init_logits_std,
                temperature=unilora_config.temperature,
                assignment_mode=unilora_config.assignment_mode,
                gumbel_hard=unilora_config.gumbel_hard,
                hard_eval=unilora_config.hard_eval,
                unilora_dropout=unilora_config.unilora_dropout,
            )
        else:
            new_module = self._create_new_module(
                unilora_config=unilora_config,
                unilora_soft_assign_theta_d=self.unilora_soft_assign_theta_d,
                adapter_name=adapter_name,
                target=target,
                **kwargs,
            )
            if adapter_name not in self.active_adapter:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(unilora_config, unilora_soft_assign_theta_d, adapter_name, target, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = unilora_config.fan_in_fan_out = False
        elif isinstance(target_base_layer, Conv1D):
            kwargs["is_target_conv_1d_layer"] = True
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                    "Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = unilora_config.fan_in_fan_out = True
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `transformers.pytorch_utils.Conv1D`."
            )
        return Linear(
            base_layer=target,
            unilora_soft_assign_theta_d=unilora_soft_assign_theta_d,
            adapter_name=adapter_name,
            r=unilora_config.r,
            theta_d_length=unilora_config.theta_d_length,
            num_candidates=unilora_config.num_candidates,
            init_logits_std=unilora_config.init_logits_std,
            temperature=unilora_config.temperature,
            assignment_mode=unilora_config.assignment_mode,
            gumbel_hard=unilora_config.gumbel_hard,
            hard_eval=unilora_config.hard_eval,
            unilora_dropout=unilora_config.unilora_dropout,
            **kwargs,
        )

    def get_nb_savable_parameters(self, adapter="default") -> tuple[int, int]:
        trainable_params = 0
        buffer_params = 0
        for name, param in self.named_parameters():
            if "unilora_soft_assign_theta_d" in name or "unilora_soft_assign_logits" in name:
                trainable_params += param.numel()

        for name, buffer in self.named_buffers():
            if "unilora_soft_assign_candidate_indices" in name or "unilora_soft_assign_scales" in name:
                buffer_params += buffer.numel()

        return trainable_params, buffer_params

    def print_savable_parameters(self) -> None:
        trainable_params, buffer_params = self.get_nb_savable_parameters()
        print(
            f"UniLoRA-SoftAssign params to-be-saved (float32-equivalent): {trainable_params:,d} "
            f"|| total params to-be-saved: {(trainable_params + buffer_params):,d}"
        )

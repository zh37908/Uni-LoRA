from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

from .._buffer_dict import BufferDict
from .config import UniLoRARoSADiscreteConfig
from .layer import Linear, UniLoRARoSADiscreteLayer


class UniLoRARoSADiscreteModel(BaseTuner):
    """
    UniLoRA-RoSA with separate banks for low-rank and sparse branches.
    """

    prefix: str = "unilora_rosa_discrete_"
    tuner_layer_cls = UniLoRARoSADiscreteLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

    def __init__(
        self,
        model,
        config,
        adapter_name,
        low_cpu_mem_usage: bool = False,
        state_dict=None,
    ) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

        cfg = self.peft_config[adapter_name]
        modules = [module for module in model.modules() if isinstance(module, UniLoRARoSADiscreteLayer)]
        total_a_params = sum(module.unilora_rosa_discrete_indices_A[adapter_name].numel() for module in modules)
        total_b_params = sum(module.unilora_rosa_discrete_indices_B[adapter_name].numel() for module in modules)
        total_lora_params = total_a_params + total_b_params
        total_sparse_params = sum(module.unilora_rosa_discrete_indices_S[adapter_name].numel() for module in modules)

        self._init_sparse_metadata(total_sparse_params=total_sparse_params, adapter_name=adapter_name)

        lora_elements = self.generate_index(total_lora_params, cfg.theta_d_length, cfg.proj_seed)
        sparse_elements = self.generate_index(total_sparse_params, cfg.sparse_theta_d_length, cfg.sparse_proj_seed)

        pointer_lora = 0
        pointer_sparse = 0
        for module in modules:
            target_device = module.get_base_layer().weight.device

            num_a = module.unilora_rosa_discrete_indices_A[adapter_name].numel()
            chunk_a = lora_elements[pointer_lora : pointer_lora + num_a]
            module.unilora_rosa_discrete_indices_A[adapter_name] = chunk_a.view_as(
                module.unilora_rosa_discrete_indices_A[adapter_name]
            ).clone().to(device=target_device, dtype=torch.long)
            pointer_lora += num_a

            num_b = module.unilora_rosa_discrete_indices_B[adapter_name].numel()
            chunk_b = lora_elements[pointer_lora : pointer_lora + num_b]
            module.unilora_rosa_discrete_indices_B[adapter_name] = chunk_b.view_as(
                module.unilora_rosa_discrete_indices_B[adapter_name]
            ).clone().to(device=target_device, dtype=torch.long)
            pointer_lora += num_b

            num_s = module.unilora_rosa_discrete_indices_S[adapter_name].numel()
            chunk_s = sparse_elements[pointer_sparse : pointer_sparse + num_s]
            sparse_offsets = torch.arange(pointer_sparse, pointer_sparse + num_s, dtype=torch.long)
            module.unilora_rosa_discrete_indices_S[adapter_name] = chunk_s.view_as(
                module.unilora_rosa_discrete_indices_S[adapter_name]
            ).clone().to(device=target_device, dtype=torch.long)
            module.unilora_rosa_discrete_sparse_offsets[adapter_name] = sparse_offsets.view_as(
                module.unilora_rosa_discrete_indices_S[adapter_name]
            ).clone().to(device=target_device, dtype=torch.long)
            pointer_sparse += num_s

        assert pointer_lora == len(lora_elements)
        assert pointer_sparse == len(sparse_elements)

        lora_counts = torch.bincount(lora_elements, minlength=cfg.theta_d_length)
        lora_inv_sqrt_counts = torch.zeros(cfg.theta_d_length, dtype=torch.float32)
        non_zero_lora = lora_counts > 0
        lora_inv_sqrt_counts[non_zero_lora] = 1.0 / torch.sqrt(lora_counts[non_zero_lora].float())

        sparse_counts = torch.bincount(sparse_elements, minlength=cfg.sparse_theta_d_length)
        sparse_inv_sqrt_counts = torch.zeros(cfg.sparse_theta_d_length, dtype=torch.float32)
        non_zero_sparse = sparse_counts > 0
        sparse_inv_sqrt_counts[non_zero_sparse] = 1.0 / torch.sqrt(sparse_counts[non_zero_sparse].float())

        for module in modules:
            scale_a = lora_inv_sqrt_counts[module.unilora_rosa_discrete_indices_A[adapter_name].detach().cpu().long()]
            scale_b = lora_inv_sqrt_counts[module.unilora_rosa_discrete_indices_B[adapter_name].detach().cpu().long()]
            scale_s = sparse_inv_sqrt_counts[module.unilora_rosa_discrete_indices_S[adapter_name].detach().cpu().long()]
            module.update_norm(adapter_name, scale_a, scale_b, scale_s)

    def _iter_unilora_modules(self):
        return [module for module in self.model.modules() if isinstance(module, UniLoRARoSADiscreteLayer)]

    def has_sparse_masks(self, adapter_name: str = "default") -> bool:
        if adapter_name not in self.unilora_rosa_discrete_sparse_mask:
            return False
        return bool(self.unilora_rosa_discrete_sparse_mask[adapter_name].any().item())

    def enable_gradient_capture(self, enabled: bool = True) -> None:
        for module in self._iter_unilora_modules():
            module.set_capture_gradient(enabled)

    def clear_gradient_statistics(self, adapter_name: str = "default") -> None:
        if adapter_name in self.unilora_rosa_discrete_grad_accum:
            self.unilora_rosa_discrete_grad_accum[adapter_name].zero_()
        for module in self._iter_unilora_modules():
            module.clear_cached_gradients(adapter_name)

    def accumulate_gradient_statistics(self, adapter_name: str = "default") -> dict[str, int]:
        updated_modules = 0
        updated_tensors = 0
        for module in self._iter_unilora_modules():
            updated = module.accumulate_gradient_statistics(adapter_name)
            if updated > 0:
                updated_modules += 1
                updated_tensors += updated
        return {"updated_modules": updated_modules, "updated_tensors": updated_tensors}

    def should_collect_gradients(self, global_step: int, adapter_name: str = "default") -> bool:
        config: UniLoRARoSADiscreteConfig = self.peft_config[adapter_name]
        if config.rosa_density <= 0.0 or config.rosa_mask_steps <= 0 or self.has_sparse_masks(adapter_name):
            return False
        return config.rosa_warmup_steps <= global_step < (config.rosa_warmup_steps + config.rosa_mask_steps)

    def should_generate_masks(self, next_global_step: int, adapter_name: str = "default") -> bool:
        config: UniLoRARoSADiscreteConfig = self.peft_config[adapter_name]
        if config.rosa_density <= 0.0 or config.rosa_mask_steps <= 0 or self.has_sparse_masks(adapter_name):
            return False
        return next_global_step >= (config.rosa_warmup_steps + config.rosa_mask_steps)

    def get_sparse_structure_stats(self, adapter_name: str = "default") -> dict[str, float]:
        if adapter_name not in self.unilora_rosa_discrete_sparse_mask:
            return {"total_positions": 0, "selected_positions": 0, "selected_density": 0.0}
        total_positions = int(self.unilora_rosa_discrete_sparse_mask[adapter_name].numel())
        selected_positions = int(self.unilora_rosa_discrete_sparse_mask[adapter_name].sum().item())
        density = 0.0 if total_positions == 0 else float(selected_positions) / float(total_positions)
        return {
            "total_positions": total_positions,
            "selected_positions": selected_positions,
            "selected_density": density,
        }

    @torch.no_grad()
    def generate_sparse_masks(self, adapter_name: str = "default", density: float | None = None) -> dict[str, float]:
        config: UniLoRARoSADiscreteConfig = self.peft_config[adapter_name]
        density = config.rosa_density if density is None else density
        if adapter_name not in self.unilora_rosa_discrete_grad_accum:
            return {"skipped": True, "reason": "no_unilora_modules"}

        flat_scores = self.unilora_rosa_discrete_grad_accum[adapter_name].detach().clone()
        num_positions = int(flat_scores.numel())
        if num_positions == 0:
            return {"skipped": True, "reason": "empty_sparse_projection"}
        num_selected = int(math.ceil(num_positions * density))
        num_selected = max(0, min(num_positions, num_selected))

        sparse_mask = torch.zeros_like(flat_scores, dtype=torch.bool)
        if num_selected > 0:
            topk = torch.topk(flat_scores, k=num_selected, largest=True, sorted=False).indices
            sparse_mask[topk] = True

        self.unilora_rosa_discrete_sparse_mask[adapter_name] = sparse_mask.to(
            device=self.unilora_rosa_discrete_sparse_mask[adapter_name].device, dtype=torch.bool
        )
        self.enable_gradient_capture(False)
        self.clear_gradient_statistics(adapter_name)

        stats = self.get_sparse_structure_stats(adapter_name)
        stats.update(
            {
                "selected_positions": int(num_selected),
                "selected_ratio": 0.0 if num_positions == 0 else float(num_selected) / float(num_positions),
                "score_max": float(flat_scores.max().item()) if flat_scores.numel() > 0 else 0.0,
                "score_mean": float(flat_scores.mean().item()) if flat_scores.numel() > 0 else 0.0,
            }
        )
        return stats

    def generate_index(self, total_length: int, theta_d_length: int, proj_seed: int):
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

    def _init_unilora_theta_d(self, config: UniLoRARoSADiscreteConfig, adapter_name: str) -> None:
        if adapter_name not in self.unilora_rosa_discrete_theta_d:
            theta_d = torch.zeros(config.theta_d_length)
            torch.nn.init.uniform_(theta_d, -config.init_theta_d_bound, config.init_theta_d_bound)
            self.unilora_rosa_discrete_theta_d[adapter_name] = theta_d
        if adapter_name not in self.unilora_rosa_discrete_sparse_theta_d:
            sparse_theta_d = torch.zeros(config.sparse_theta_d_length)
            torch.nn.init.uniform_(sparse_theta_d, -config.init_theta_d_bound, config.init_theta_d_bound)
            self.unilora_rosa_discrete_sparse_theta_d[adapter_name] = sparse_theta_d

    def _init_sparse_metadata(self, total_sparse_params: int, adapter_name: str) -> None:
        if adapter_name not in self.unilora_rosa_discrete_sparse_mask:
            self.unilora_rosa_discrete_sparse_mask[adapter_name] = torch.zeros(total_sparse_params, dtype=torch.bool)
        if adapter_name not in self.unilora_rosa_discrete_grad_accum:
            self.unilora_rosa_discrete_grad_accum[adapter_name] = torch.zeros(total_sparse_params, dtype=torch.float32)

    def _pre_injection_hook(self, model: nn.Module, config: UniLoRARoSADiscreteConfig, adapter_name: str) -> None:
        self.unilora_rosa_discrete_theta_d = nn.ParameterDict({})
        self.unilora_rosa_discrete_sparse_theta_d = nn.ParameterDict({})
        self.unilora_rosa_discrete_sparse_mask = BufferDict({}, persistent=True)
        self.unilora_rosa_discrete_grad_accum = BufferDict({}, persistent=False)

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
                unilora_rosa_discrete_theta_d=self.unilora_rosa_discrete_theta_d,
                unilora_rosa_discrete_sparse_theta_d=self.unilora_rosa_discrete_sparse_theta_d,
                unilora_rosa_discrete_sparse_mask=self.unilora_rosa_discrete_sparse_mask,
                unilora_rosa_discrete_grad_accum=self.unilora_rosa_discrete_grad_accum,
                r=unilora_config.r,
                theta_d_length=unilora_config.theta_d_length,
                sparse_theta_d_length=unilora_config.sparse_theta_d_length,
                unilora_dropout=unilora_config.unilora_dropout,
            )
        else:
            new_module = self._create_new_module(
                unilora_config=unilora_config,
                unilora_rosa_discrete_theta_d=self.unilora_rosa_discrete_theta_d,
                unilora_rosa_discrete_sparse_theta_d=self.unilora_rosa_discrete_sparse_theta_d,
                unilora_rosa_discrete_sparse_mask=self.unilora_rosa_discrete_sparse_mask,
                unilora_rosa_discrete_grad_accum=self.unilora_rosa_discrete_grad_accum,
                adapter_name=adapter_name,
                target=target,
                **kwargs,
            )
            if adapter_name not in self.active_adapter:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(
        unilora_config,
        unilora_rosa_discrete_theta_d,
        unilora_rosa_discrete_sparse_theta_d,
        unilora_rosa_discrete_sparse_mask,
        unilora_rosa_discrete_grad_accum,
        adapter_name,
        target,
        **kwargs,
    ):
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
            unilora_rosa_discrete_theta_d=unilora_rosa_discrete_theta_d,
            unilora_rosa_discrete_sparse_theta_d=unilora_rosa_discrete_sparse_theta_d,
            unilora_rosa_discrete_sparse_mask=unilora_rosa_discrete_sparse_mask,
            unilora_rosa_discrete_grad_accum=unilora_rosa_discrete_grad_accum,
            adapter_name=adapter_name,
            r=unilora_config.r,
            theta_d_length=unilora_config.theta_d_length,
            sparse_theta_d_length=unilora_config.sparse_theta_d_length,
            unilora_dropout=unilora_config.unilora_dropout,
            **kwargs,
        )

    def get_nb_savable_parameters(self, adapter="default") -> tuple[int, int]:
        compressed_params = 0
        other_params = 0
        for name, param in self.named_parameters():
            if "unilora_rosa_discrete_theta_d" in name or "unilora_rosa_discrete_sparse_theta_d" in name:
                compressed_params += param.numel()
        for name, buffer in self.named_buffers():
            if "unilora_rosa_discrete_" in name and "grad_accum" not in name:
                other_params += buffer.numel()
        return compressed_params, other_params

    def print_savable_parameters(self) -> None:
        compressed_params, other_params = self.get_nb_savable_parameters()
        print(
            f"UniLoRA-RoSA-Discrete params to-be-saved (float32-equivalent): {compressed_params:,d} "
            f"|| total params to-be-saved: {(compressed_params + other_params):,d}"
        )

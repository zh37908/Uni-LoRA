from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

from .._buffer_dict import BufferDict
from .config import UniLoRARoSAGlobalConfig
from .layer import Linear, UniLoRARoSAGlobalLayer


class UniLoRARoSAGlobalModel(BaseTuner):
    """
    UniLoRA-RoSA with a single global theta_D layout over all A/B/S entries.
    """

    prefix: str = "unilora_rosa_global_"
    tuner_layer_cls = UniLoRARoSAGlobalLayer
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
        modules = [module for module in model.modules() if isinstance(module, UniLoRARoSAGlobalLayer)]
        total_a_params = sum(module.unilora_rosa_global_indices_A[adapter_name].numel() for module in modules)
        total_b_params = sum(module.unilora_rosa_global_indices_B[adapter_name].numel() for module in modules)
        total_s_params = sum(module.unilora_rosa_global_indices_S[adapter_name].numel() for module in modules)
        total_params = total_a_params + total_b_params + total_s_params

        self._init_sparse_metadata(total_params=total_params, total_sparse_params=total_s_params, adapter_name=adapter_name)
        all_elements = self.generate_index(total_params, cfg.theta_d_length, cfg.proj_seed)

        pointer = 0
        sparse_offsets = []
        for module in modules:
            target_device = module.get_base_layer().weight.device

            num_a = module.unilora_rosa_global_indices_A[adapter_name].numel()
            chunk_a = all_elements[pointer : pointer + num_a]
            offsets_a = torch.arange(pointer, pointer + num_a, dtype=torch.long)
            module.unilora_rosa_global_indices_A[adapter_name] = chunk_a.view_as(
                module.unilora_rosa_global_indices_A[adapter_name]
            ).clone().to(device=target_device, dtype=torch.long)
            module.unilora_rosa_global_theta_D_offsets_A[adapter_name] = offsets_a.view_as(
                module.unilora_rosa_global_indices_A[adapter_name]
            ).clone().to(device=target_device, dtype=torch.long)
            pointer += num_a

            num_b = module.unilora_rosa_global_indices_B[adapter_name].numel()
            chunk_b = all_elements[pointer : pointer + num_b]
            offsets_b = torch.arange(pointer, pointer + num_b, dtype=torch.long)
            module.unilora_rosa_global_indices_B[adapter_name] = chunk_b.view_as(
                module.unilora_rosa_global_indices_B[adapter_name]
            ).clone().to(device=target_device, dtype=torch.long)
            module.unilora_rosa_global_theta_D_offsets_B[adapter_name] = offsets_b.view_as(
                module.unilora_rosa_global_indices_B[adapter_name]
            ).clone().to(device=target_device, dtype=torch.long)
            pointer += num_b

            num_s = module.unilora_rosa_global_indices_S[adapter_name].numel()
            chunk_s = all_elements[pointer : pointer + num_s]
            offsets_s = torch.arange(pointer, pointer + num_s, dtype=torch.long)
            module.unilora_rosa_global_indices_S[adapter_name] = chunk_s.view_as(
                module.unilora_rosa_global_indices_S[adapter_name]
            ).clone().to(device=target_device, dtype=torch.long)
            module.unilora_rosa_global_theta_D_offsets_S[adapter_name] = offsets_s.view_as(
                module.unilora_rosa_global_indices_S[adapter_name]
            ).clone().to(device=target_device, dtype=torch.long)
            sparse_offsets.append(offsets_s)
            pointer += num_s

        assert pointer == len(all_elements)

        counts = torch.bincount(all_elements, minlength=cfg.theta_d_length)
        inv_sqrt_counts = torch.zeros(cfg.theta_d_length, dtype=torch.float32)
        non_zero = counts > 0
        inv_sqrt_counts[non_zero] = 1.0 / torch.sqrt(counts[non_zero].float())

        for module in modules:
            scale_a = inv_sqrt_counts[module.unilora_rosa_global_indices_A[adapter_name].detach().cpu().long()]
            scale_b = inv_sqrt_counts[module.unilora_rosa_global_indices_B[adapter_name].detach().cpu().long()]
            scale_s = inv_sqrt_counts[module.unilora_rosa_global_indices_S[adapter_name].detach().cpu().long()]
            module.update_norm(adapter_name, scale_a, scale_b, scale_s)

        if sparse_offsets:
            self.unilora_rosa_global_sparse_offsets[adapter_name] = torch.cat(sparse_offsets).to(dtype=torch.long)

    def _iter_unilora_modules(self):
        return [module for module in self.model.modules() if isinstance(module, UniLoRARoSAGlobalLayer)]

    def has_sparse_masks(self, adapter_name: str = "default") -> bool:
        if adapter_name not in self.unilora_rosa_global_sparse_mask:
            return False
        return bool(self.unilora_rosa_global_sparse_mask[adapter_name].any().item())

    def enable_gradient_capture(self, enabled: bool = True) -> None:
        for module in self._iter_unilora_modules():
            module.set_capture_gradient(enabled)

    def clear_gradient_statistics(self, adapter_name: str = "default") -> None:
        if adapter_name in self.unilora_rosa_global_grad_accum:
            self.unilora_rosa_global_grad_accum[adapter_name].zero_()
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
        config: UniLoRARoSAGlobalConfig = self.peft_config[adapter_name]
        if config.rosa_density <= 0.0 or config.rosa_mask_steps <= 0 or self.has_sparse_masks(adapter_name):
            return False
        return config.rosa_warmup_steps <= global_step < (config.rosa_warmup_steps + config.rosa_mask_steps)

    def should_generate_masks(self, next_global_step: int, adapter_name: str = "default") -> bool:
        config: UniLoRARoSAGlobalConfig = self.peft_config[adapter_name]
        if config.rosa_density <= 0.0 or config.rosa_mask_steps <= 0 or self.has_sparse_masks(adapter_name):
            return False
        return next_global_step >= (config.rosa_warmup_steps + config.rosa_mask_steps)

    def get_sparse_structure_stats(self, adapter_name: str = "default") -> dict[str, float]:
        if adapter_name not in self.unilora_rosa_global_sparse_offsets:
            return {"total_positions": 0, "selected_positions": 0, "selected_density": 0.0}

        sparse_offsets = self.unilora_rosa_global_sparse_offsets[adapter_name]
        total_positions = int(sparse_offsets.numel())
        selected_positions = int(self.unilora_rosa_global_sparse_mask[adapter_name][sparse_offsets].sum().item())
        density = 0.0 if total_positions == 0 else float(selected_positions) / float(total_positions)
        return {
            "total_positions": total_positions,
            "selected_positions": selected_positions,
            "selected_density": density,
        }

    @torch.no_grad()
    def generate_sparse_masks(self, adapter_name: str = "default", density: float | None = None) -> dict[str, float]:
        config: UniLoRARoSAGlobalConfig = self.peft_config[adapter_name]
        density = config.rosa_density if density is None else density
        if adapter_name not in self.unilora_rosa_global_sparse_offsets:
            return {"skipped": True, "reason": "no_unilora_modules"}

        sparse_offsets = self.unilora_rosa_global_sparse_offsets[adapter_name].long()
        if sparse_offsets.numel() == 0:
            return {"skipped": True, "reason": "empty_sparse_projection"}

        flat_scores = self.unilora_rosa_global_grad_accum[adapter_name][sparse_offsets].detach().clone()
        num_positions = int(flat_scores.numel())
        num_selected = int(math.ceil(num_positions * density))
        num_selected = max(0, min(num_positions, num_selected))

        full_mask = torch.zeros_like(self.unilora_rosa_global_sparse_mask[adapter_name], dtype=torch.bool)
        if num_selected > 0:
            topk_local = torch.topk(flat_scores, k=num_selected, largest=True, sorted=False).indices
            chosen_offsets = sparse_offsets[topk_local]
            full_mask[chosen_offsets] = True

        self.unilora_rosa_global_sparse_mask[adapter_name] = full_mask.to(
            device=self.unilora_rosa_global_sparse_mask[adapter_name].device, dtype=torch.bool
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

    def _init_unilora_theta_d(self, config: UniLoRARoSAGlobalConfig, adapter_name: str) -> None:
        if adapter_name in self.unilora_rosa_global_theta_d:
            return
        theta_d = torch.zeros(config.theta_d_length)
        torch.nn.init.uniform_(theta_d, -config.init_theta_d_bound, config.init_theta_d_bound)
        self.unilora_rosa_global_theta_d[adapter_name] = theta_d

    def _init_sparse_metadata(self, total_params: int, total_sparse_params: int, adapter_name: str) -> None:
        if adapter_name not in self.unilora_rosa_global_sparse_mask:
            self.unilora_rosa_global_sparse_mask[adapter_name] = torch.zeros(total_params, dtype=torch.bool)
        if adapter_name not in self.unilora_rosa_global_grad_accum:
            self.unilora_rosa_global_grad_accum[adapter_name] = torch.zeros(total_params, dtype=torch.float32)
        if adapter_name not in self.unilora_rosa_global_sparse_offsets:
            self.unilora_rosa_global_sparse_offsets[adapter_name] = torch.empty(total_sparse_params, dtype=torch.long)

    def _pre_injection_hook(self, model: nn.Module, config: UniLoRARoSAGlobalConfig, adapter_name: str) -> None:
        self.unilora_rosa_global_theta_d = nn.ParameterDict({})
        self.unilora_rosa_global_sparse_mask = BufferDict({}, persistent=True)
        self.unilora_rosa_global_sparse_offsets = BufferDict({}, persistent=True)
        self.unilora_rosa_global_grad_accum = BufferDict({}, persistent=False)

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
                unilora_rosa_global_theta_d=self.unilora_rosa_global_theta_d,
                unilora_rosa_global_sparse_mask=self.unilora_rosa_global_sparse_mask,
                unilora_rosa_global_grad_accum=self.unilora_rosa_global_grad_accum,
                r=unilora_config.r,
                theta_d_length=unilora_config.theta_d_length,
                unilora_dropout=unilora_config.unilora_dropout,
            )
        else:
            new_module = self._create_new_module(
                unilora_config=unilora_config,
                unilora_rosa_global_theta_d=self.unilora_rosa_global_theta_d,
                unilora_rosa_global_sparse_mask=self.unilora_rosa_global_sparse_mask,
                unilora_rosa_global_grad_accum=self.unilora_rosa_global_grad_accum,
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
        unilora_rosa_global_theta_d,
        unilora_rosa_global_sparse_mask,
        unilora_rosa_global_grad_accum,
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
            unilora_rosa_global_theta_d=unilora_rosa_global_theta_d,
            unilora_rosa_global_sparse_mask=unilora_rosa_global_sparse_mask,
            unilora_rosa_global_grad_accum=unilora_rosa_global_grad_accum,
            adapter_name=adapter_name,
            r=unilora_config.r,
            theta_d_length=unilora_config.theta_d_length,
            unilora_dropout=unilora_config.unilora_dropout,
            **kwargs,
        )

    def get_nb_savable_parameters(self, adapter="default") -> tuple[int, int]:
        theta_d_params = 0
        other_params = 0
        for name, param in self.named_parameters():
            if "unilora_rosa_global_theta_d" in name:
                theta_d_params += param.numel()
        for name, buffer in self.named_buffers():
            if "unilora_rosa_global_" in name and "grad_accum" not in name:
                other_params += buffer.numel()
        return theta_d_params, other_params

    def print_savable_parameters(self) -> None:
        theta_d_params, other_params = self.get_nb_savable_parameters()
        print(
            f"UniLoRA-RoSA-Global params to-be-saved (float32-equivalent): {theta_d_params:,d} "
            f"|| total params to-be-saved: {(theta_d_params + other_params):,d}"
        )

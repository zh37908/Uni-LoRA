from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

from .._buffer_dict import BufferDict
from .config import UniLoRARoSACompressionConfig
from .layer import Linear, UniLoRARoSACompressionLayer


class UniLoRARoSACompressionModel(BaseTuner):
    """
    UniLoRA-RoSA-Compression:
    - dense part is standard UniLoRA RoSA implementation
    - sparse residual is parameterized by a compressed sparse bank
      (sparse_theta_d_length), while sparse offsets are still selected via per-offset top-k mask.
    """

    prefix: str = "unilora_rosa_compression_"
    tuner_layer_cls = UniLoRARoSACompressionLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

        lora_para_cnt = 0
        for _, module in model.named_modules():
            if isinstance(module, UniLoRARoSACompressionLayer):
                lora_para_cnt += module.unilora_indices_A[adapter_name].numel()
                lora_para_cnt += module.unilora_indices_B[adapter_name].numel()

        cfg: UniLoRARoSACompressionConfig = config[adapter_name]
        theta_d_length = cfg.theta_d_length
        sparse_theta_d_length = cfg.sparse_theta_d_length
        proj_seed = cfg.proj_seed

        self._init_sparse_theta_d_bank(lora_para_cnt, sparse_theta_d_length, adapter_name)

        # 1) Generate dense indices mapping (offsets -> theta_d indices)
        all_elements_dense = self.generate_index(lora_para_cnt, theta_d_length, proj_seed)
        pointer = 0

        for _, module in model.named_modules():
            if isinstance(module, UniLoRARoSACompressionLayer):
                # A
                param_numel = module.unilora_indices_A[adapter_name].numel()
                chunk = all_elements_dense[pointer : pointer + param_numel]
                target_device = module.get_base_layer().weight.device
                offset_chunk = torch.arange(pointer, pointer + param_numel, dtype=torch.long)
                module.unilora_indices_A[adapter_name] = chunk.view_as(module.unilora_indices_A[adapter_name]).clone().to(
                    device=target_device, dtype=torch.long
                )
                module.unilora_theta_D_offsets_A[adapter_name] = offset_chunk.view_as(
                    module.unilora_indices_A[adapter_name]
                ).clone().to(device=target_device, dtype=torch.long)
                pointer += param_numel

                # B
                param_numel = module.unilora_indices_B[adapter_name].numel()
                chunk = all_elements_dense[pointer : pointer + param_numel]
                offset_chunk = torch.arange(pointer, pointer + param_numel, dtype=torch.long)
                module.unilora_indices_B[adapter_name] = chunk.view_as(module.unilora_indices_B[adapter_name]).clone().to(
                    device=target_device, dtype=torch.long
                )
                module.unilora_theta_D_offsets_B[adapter_name] = offset_chunk.view_as(
                    module.unilora_indices_B[adapter_name]
                ).clone().to(device=target_device, dtype=torch.long)
                pointer += param_numel

        assert pointer == len(all_elements_dense)

        # Update dense scaling (inverse-sqrt of counts of each theta_d index)
        counts_dense = torch.bincount(all_elements_dense, minlength=theta_d_length)
        inv_sqrt_counts_dense = torch.zeros(theta_d_length, dtype=torch.float32)
        non_zero = counts_dense > 0
        inv_sqrt_counts_dense[non_zero] = 1.0 / torch.sqrt(counts_dense[non_zero].float())

        for _, module in model.named_modules():
            if isinstance(module, UniLoRARoSACompressionLayer):
                scale_a = inv_sqrt_counts_dense[module.unilora_indices_A[adapter_name].detach().cpu().long()]
                scale_b = inv_sqrt_counts_dense[module.unilora_indices_B[adapter_name].detach().cpu().long()]
                module.update_norm(adapter_name, scale_a, scale_b)
                module.set_sparse_requires_grad(adapter_name, False)

        # 2) Generate sparse compression indices mapping (offsets -> sparse bank indices)
        all_elements_sparse = self.generate_index(lora_para_cnt, sparse_theta_d_length, proj_seed)
        pointer = 0
        for _, module in model.named_modules():
            if isinstance(module, UniLoRARoSACompressionLayer):
                param_numel = module.unilora_rosa_sparse_indices_A[adapter_name].numel()
                chunk = all_elements_sparse[pointer : pointer + param_numel]
                target_device = module.get_base_layer().weight.device
                module.unilora_rosa_sparse_indices_A[adapter_name] = chunk.view_as(
                    module.unilora_rosa_sparse_indices_A[adapter_name]
                ).clone().to(device=target_device, dtype=torch.long)
                pointer += param_numel

                param_numel = module.unilora_rosa_sparse_indices_B[adapter_name].numel()
                chunk = all_elements_sparse[pointer : pointer + param_numel]
                module.unilora_rosa_sparse_indices_B[adapter_name] = chunk.view_as(
                    module.unilora_rosa_sparse_indices_B[adapter_name]
                ).clone().to(device=target_device, dtype=torch.long)
                pointer += param_numel

        assert pointer == len(all_elements_sparse)

        counts_sparse = torch.bincount(all_elements_sparse, minlength=sparse_theta_d_length)
        inv_sqrt_counts_sparse = torch.zeros(sparse_theta_d_length, dtype=torch.float32)
        non_zero = counts_sparse > 0
        inv_sqrt_counts_sparse[non_zero] = 1.0 / torch.sqrt(counts_sparse[non_zero].float())

        for _, module in model.named_modules():
            if isinstance(module, UniLoRARoSACompressionLayer):
                sparse_scale_a = inv_sqrt_counts_sparse[
                    module.unilora_rosa_sparse_indices_A[adapter_name].detach().cpu().long()
                ]
                sparse_scale_b = inv_sqrt_counts_sparse[
                    module.unilora_rosa_sparse_indices_B[adapter_name].detach().cpu().long()
                ]
                module.update_sparse_norm(adapter_name, sparse_scale_a, sparse_scale_b)
                module.set_sparse_requires_grad(adapter_name, False)

    def _iter_unilora_modules(self):
        return [module for module in self.model.modules() if isinstance(module, UniLoRARoSACompressionLayer)]

    def has_sparse_masks(self, adapter_name: str = "default") -> bool:
        if adapter_name not in self.unilora_rosa_sparse_mask:
            return False
        return bool(self.unilora_rosa_sparse_mask[adapter_name].any().item())

    def enable_gradient_capture(self, enabled: bool = True) -> None:
        for module in self._iter_unilora_modules():
            module.set_capture_gradient(enabled)

    def set_sparse_requires_grad(self, adapter_name: str = "default", requires_grad: bool = True) -> None:
        if adapter_name in self.unilora_rosa_sparse_theta_d:
            self.unilora_rosa_sparse_theta_d[adapter_name].requires_grad_(requires_grad)
        for module in self._iter_unilora_modules():
            module.set_sparse_requires_grad(adapter_name, requires_grad)

    def clear_gradient_statistics(self, adapter_name: str = "default") -> None:
        if adapter_name in self.unilora_rosa_grad_accum:
            self.unilora_rosa_grad_accum[adapter_name].zero_()
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
        config: UniLoRARoSACompressionConfig = self.peft_config[adapter_name]
        if config.rosa_density <= 0.0 or config.rosa_mask_steps <= 0 or self.has_sparse_masks(adapter_name):
            return False
        return config.rosa_warmup_steps <= global_step < (config.rosa_warmup_steps + config.rosa_mask_steps)

    def should_generate_masks(self, next_global_step: int, adapter_name: str = "default") -> bool:
        config: UniLoRARoSACompressionConfig = self.peft_config[adapter_name]
        if config.rosa_density <= 0.0 or config.rosa_mask_steps <= 0 or self.has_sparse_masks(adapter_name):
            return False
        return next_global_step >= (config.rosa_warmup_steps + config.rosa_mask_steps)

    def get_sparse_structure_stats(self, adapter_name: str = "default") -> dict[str, float]:
        if adapter_name not in self.unilora_rosa_sparse_mask:
            return {"total_positions": 0, "selected_positions": 0, "selected_density": 0.0}

        total_positions = int(self.unilora_rosa_sparse_mask[adapter_name].numel())
        selected_positions = int(self.unilora_rosa_sparse_mask[adapter_name].sum().item())
        density = 0.0 if total_positions == 0 else float(selected_positions) / float(total_positions)
        return {
            "total_positions": total_positions,
            "selected_positions": selected_positions,
            "selected_density": density,
        }

    @torch.no_grad()
    def generate_sparse_masks(self, adapter_name: str = "default", density: float | None = None) -> dict[str, float]:
        config: UniLoRARoSACompressionConfig = self.peft_config[adapter_name]
        density = config.rosa_density if density is None else density

        if adapter_name not in self.unilora_rosa_grad_accum:
            return {"skipped": True, "reason": "no_unilora_modules"}

        flat_scores = self.unilora_rosa_grad_accum[adapter_name].detach().clone()
        num_positions = int(flat_scores.numel())
        if num_positions == 0:
            return {"skipped": True, "reason": "empty_projection"}

        num_selected = int(math.ceil(num_positions * density))
        num_selected = max(0, min(num_positions, num_selected))

        sparse_mask = torch.zeros_like(flat_scores, dtype=torch.bool)
        if num_selected > 0:
            topk = torch.topk(flat_scores, k=num_selected, largest=True, sorted=False).indices
            sparse_mask[topk] = True

        # Keep mask as the per-offset gate.
        self.unilora_rosa_sparse_mask[adapter_name] = sparse_mask.to(
            device=self.unilora_rosa_sparse_mask[adapter_name].device, dtype=torch.bool
        )

        # Sparse bank values are not multiplied by the mask; gating is done in forward via sparse_mask[offsets].
        self.set_sparse_requires_grad(adapter_name, num_selected > 0)
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

    def generate_index(self, lora_para_cnt, theta_d_length, proj_seed):
        import numpy as np

        total_length = lora_para_cnt
        num_unique = theta_d_length
        base_count = total_length // num_unique
        remaining = total_length % num_unique
        rng = np.random.default_rng(proj_seed)
        data = np.repeat(np.arange(num_unique), base_count)
        if remaining > 0:
            extras = rng.choice(num_unique, size=remaining, replace=False)
            data = np.concatenate([data, extras])
        rng.shuffle(data)
        return torch.tensor(data)

    def _init_unilora_theta_d(self, config: UniLoRARoSACompressionConfig, adapter_name: str) -> None:
        if adapter_name in self.unilora_rosa_theta_d:
            return
        unilora_theta_d = torch.zeros(config.theta_d_length)
        torch.nn.init.uniform_(unilora_theta_d, -config.init_theta_d_bound, config.init_theta_d_bound)
        self.unilora_rosa_theta_d[adapter_name] = unilora_theta_d

    def _init_sparse_theta_d_bank(self, lora_para_cnt: int, sparse_theta_d_length: int, adapter_name: str) -> None:
        if adapter_name in self.unilora_rosa_sparse_theta_d:
            return

        sparse_bank = torch.zeros(sparse_theta_d_length)
        self.unilora_rosa_sparse_theta_d[adapter_name] = sparse_bank
        self.unilora_rosa_sparse_mask[adapter_name] = torch.zeros(lora_para_cnt, dtype=torch.bool)
        self.unilora_rosa_grad_accum[adapter_name] = torch.zeros(lora_para_cnt, dtype=torch.float32)

    def _pre_injection_hook(self, model: nn.Module, config: UniLoRARoSACompressionConfig, adapter_name: str) -> None:
        self.unilora_rosa_theta_d = nn.ParameterDict({})
        self.unilora_rosa_sparse_theta_d = nn.ParameterDict({})
        self.unilora_rosa_sparse_mask = BufferDict({}, persistent=True)

        # Name must not end with "score" due to special wrapping in PeftModel SEQ_CLS.
        self.unilora_rosa_grad_accum = BufferDict({}, persistent=False)

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
                unilora_rosa_theta_d=self.unilora_rosa_theta_d,
                unilora_rosa_sparse_theta_d=self.unilora_rosa_sparse_theta_d,
                unilora_rosa_sparse_mask=self.unilora_rosa_sparse_mask,
                unilora_rosa_grad_accum=self.unilora_rosa_grad_accum,
                r=unilora_config.r,
                theta_d_length=unilora_config.theta_d_length,
                sparse_theta_d_length=unilora_config.sparse_theta_d_length,
                unilora_dropout=unilora_config.unilora_dropout,
            )
        else:
            new_module = self._create_new_module(
                unilora_config=unilora_config,
                adapter_name=adapter_name,
                target=target,
                unilora_rosa_theta_d=self.unilora_rosa_theta_d,
                unilora_rosa_sparse_theta_d=self.unilora_rosa_sparse_theta_d,
                unilora_rosa_sparse_mask=self.unilora_rosa_sparse_mask,
                unilora_rosa_grad_accum=self.unilora_rosa_grad_accum,
                **kwargs,
            )
            if adapter_name not in self.active_adapter:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(
        unilora_config,
        adapter_name: str,
        target,
        unilora_rosa_theta_d,
        unilora_rosa_sparse_theta_d,
        unilora_rosa_sparse_mask,
        unilora_rosa_grad_accum,
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

        new_module = Linear(
            base_layer=target,
            unilora_rosa_theta_d=unilora_rosa_theta_d,
            unilora_rosa_sparse_theta_d=unilora_rosa_sparse_theta_d,
            unilora_rosa_sparse_mask=unilora_rosa_sparse_mask,
            unilora_rosa_grad_accum=unilora_rosa_grad_accum,
            adapter_name=adapter_name,
            r=unilora_config.r,
            theta_d_length=unilora_config.theta_d_length,
            sparse_theta_d_length=unilora_config.sparse_theta_d_length,
            unilora_dropout=unilora_config.unilora_dropout,
            **kwargs,
        )
        return new_module

    def get_nb_savable_parameters(self, adapter="default") -> tuple[int, int]:
        theta_d_params = 0
        sparse_params = 0
        other_params = 0

        for name, param in self.named_parameters():
            if "unilora_rosa_theta_d" in name:
                theta_d_params += param.numel()
            elif "unilora_rosa_sparse_theta_d" in name:
                sparse_params += param.numel()
            elif "unilora_indices" in name or "unilora_scales" in name or "unilora_theta_D_offsets" in name:
                other_params += param.numel()

        for name, buffer in self.named_buffers():
            if (
                "unilora_indices" in name
                or "unilora_scales" in name
                or "unilora_theta_D_offsets" in name
                or "unilora_rosa_sparse_indices" in name
                or "unilora_rosa_sparse_scales" in name
            ):
                other_params += buffer.numel()
            elif "unilora_rosa_sparse_mask" in name:
                other_params += buffer.numel()

        return theta_d_params + sparse_params, other_params

    def print_savable_parameters(self) -> None:
        unilora_params, other_params = self.get_nb_savable_parameters()
        print(
            f"UniLoRA-RoSA-Compression params to-be-saved (float32-equivalent): {unilora_params:,d} "
            f"|| total params to-be-saved: {(unilora_params + other_params):,d}"
        )


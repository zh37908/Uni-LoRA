from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

from .config import UniLoRAIGUConfig
from .layer import Linear, UniLoRAIGULayer


class UniLoRAIGUModel(BaseTuner):
    """
    UniLoRA-IGU model: compressed UniLoRA A/B with explicit IGU-style lora_E rank gating.
    """

    prefix: str = "unilora_igu_"
    tuner_layer_cls = UniLoRAIGULayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

        lora_param_count = 0
        for _, layer in model.named_modules():
            if isinstance(layer, UniLoRAIGULayer):
                lora_param_count += layer.unilora_indices_A[adapter_name].numel()
                lora_param_count += layer.unilora_indices_B[adapter_name].numel()

        theta_d_length = config[adapter_name].theta_d_length
        proj_seed = config[adapter_name].proj_seed
        all_elements = self.generate_index(lora_param_count, theta_d_length, proj_seed)
        pointer = 0

        for _, module in model.named_modules():
            if isinstance(module, UniLoRAIGULayer):
                num_a = module.unilora_indices_A[adapter_name].numel()
                chunk_a = all_elements[pointer : pointer + num_a]
                module.unilora_indices_A[adapter_name] = chunk_a.view_as(module.unilora_indices_A[adapter_name]).clone()
                pointer += num_a

                num_b = module.unilora_indices_B[adapter_name].numel()
                chunk_b = all_elements[pointer : pointer + num_b]
                module.unilora_indices_B[adapter_name] = chunk_b.view_as(module.unilora_indices_B[adapter_name]).clone()
                pointer += num_b

        if pointer != len(all_elements):
            raise RuntimeError("UniLoRA-IGU index assignment is inconsistent.")

        counts = torch.bincount(all_elements, minlength=theta_d_length)
        inv_sqrt_counts = torch.zeros(theta_d_length, dtype=torch.float32)
        non_zero = counts > 0
        inv_sqrt_counts[non_zero] = 1.0 / torch.sqrt(counts[non_zero].float())

        for _, module in model.named_modules():
            if isinstance(module, UniLoRAIGULayer):
                scale_a = inv_sqrt_counts[module.unilora_indices_A[adapter_name].long()]
                scale_b = inv_sqrt_counts[module.unilora_indices_B[adapter_name].long()]
                module.update_norm(adapter_name, scale_a, scale_b)

        self._total_steps = None

    def _iter_unilora_modules(self):
        return [module for module in self.model.modules() if isinstance(module, UniLoRAIGULayer)]

    def generate_index(self, total_length: int, theta_d_length: int, proj_seed: int) -> torch.Tensor:
        if total_length <= 0:
            return torch.empty(0, dtype=torch.long)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(proj_seed)

        base_count = total_length // theta_d_length
        remaining = total_length % theta_d_length
        data = torch.arange(theta_d_length, dtype=torch.long).repeat_interleave(base_count)
        if remaining > 0:
            extras = torch.randperm(theta_d_length, generator=generator)[:remaining]
            data = torch.cat([data, extras], dim=0)
        shuffle = torch.randperm(data.numel(), generator=generator)
        return data[shuffle]

    def _init_unilora_theta_d(self, config: UniLoRAIGUConfig, adapter_name: str) -> None:
        if adapter_name in self.unilora_igu_theta_d:
            return
        unilora_theta_d = torch.zeros(config.theta_d_length)
        torch.nn.init.uniform_(unilora_theta_d, -config.init_theta_d_bound, config.init_theta_d_bound)
        self.unilora_igu_theta_d[adapter_name] = unilora_theta_d

    def _pre_injection_hook(self, model: nn.Module, config: UniLoRAIGUConfig, adapter_name: str) -> None:
        self.unilora_igu_theta_d = nn.ParameterDict({})

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
                unilora_igu_theta_d=self.unilora_igu_theta_d,
                r=unilora_config.r,
                theta_d_length=unilora_config.theta_d_length,
                unilora_dropout=unilora_config.unilora_dropout,
            )
        else:
            new_module = self._create_new_module(
                unilora_config=unilora_config,
                unilora_igu_theta_d=self.unilora_igu_theta_d,
                adapter_name=adapter_name,
                target=target,
                **kwargs,
            )
            if adapter_name not in self.active_adapter:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(unilora_config, unilora_igu_theta_d, adapter_name, target, **kwargs):
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
            unilora_igu_theta_d=unilora_igu_theta_d,
            adapter_name=adapter_name,
            r=unilora_config.r,
            theta_d_length=unilora_config.theta_d_length,
            unilora_dropout=unilora_config.unilora_dropout,
            **kwargs,
        )
        return new_module

    def set_total_step(self, total_step: int) -> None:
        self._total_steps = int(total_step)

    def enable_gradient_capture(self, enabled: bool = True) -> None:
        for module in self._iter_unilora_modules():
            module.set_capture_rank_stats(enabled)

    def should_update_importance(self, global_step: int, adapter_name: str = "default") -> bool:
        cfg: UniLoRAIGUConfig = self.peft_config[adapter_name]
        if self._total_steps is None:
            return True
        return int(global_step) < (int(self._total_steps) - int(cfg.igu_final_warmup))

    def accumulate_rank_statistics(
        self, adapter_name: str = "default", beta1: float | None = None, beta2: float | None = None
    ) -> dict[str, int]:
        cfg: UniLoRAIGUConfig = self.peft_config[adapter_name]
        beta1 = cfg.igu_beta1 if beta1 is None else beta1
        beta2 = cfg.igu_beta2 if beta2 is None else beta2

        updated_modules = 0
        updated_ranks = 0
        for module in self._iter_unilora_modules():
            info = module.accumulate_rank_statistics(adapter_name, beta1=beta1, beta2=beta2)
            if info["updated_tensors"] > 0:
                updated_modules += 1
                updated_ranks += info["updated_ranks"]
        return {"updated_modules": updated_modules, "updated_ranks": updated_ranks}

    @torch.no_grad()
    def set_weight_coeffs(self, weight_coeff_value: float = 1.0, adapter_name: str = "default") -> None:
        for module in self._iter_unilora_modules():
            if adapter_name not in module.unilora_igu_weight_coeff:
                continue
            coeff = module.unilora_igu_weight_coeff[adapter_name].detach().clone()
            coeff[0] = float(weight_coeff_value)
            module.unilora_igu_weight_coeff[adapter_name] = coeff.to(
                device=coeff.device, dtype=coeff.dtype
            )

    def compute_orth_regu(self, adapter_name: str = "default") -> torch.Tensor:
        regu_loss = None
        num_param = 0
        for module in self._iter_unilora_modules():
            if adapter_name not in module.unilora_indices_A:
                continue
            prev_capture = module.capture_rank_stats
            module.capture_rank_stats = False
            A, B = module._get_lora_matrices(adapter_name)
            module.capture_rank_stats = prev_capture
            para_cov_a = A @ A.T
            para_cov_b = B.T @ B
            loss_a = torch.norm(para_cov_a - torch.diag(torch.diag(para_cov_a)), p="fro")
            loss_b = torch.norm(para_cov_b - torch.diag(torch.diag(para_cov_b)), p="fro")
            if regu_loss is None:
                regu_loss = loss_a + loss_b
            else:
                regu_loss = regu_loss + loss_a + loss_b
            num_param += 2
        if regu_loss is None or num_param == 0:
            param = next(self.parameters())
            return param.new_zeros(())
        return regu_loss / num_param

    def get_rank_structure_stats(self, adapter_name: str = "default") -> dict[str, float]:
        total_rank = 0
        active_rank = 0
        per_module = {}
        for module_name, module in self.model.named_modules():
            if not isinstance(module, UniLoRAIGULayer):
                continue
            total_rank += int(module.r[adapter_name])
            current_active = module.get_active_rank_count(adapter_name)
            active_rank += current_active
            per_module[module_name] = current_active

        active_ratio = 0.0 if total_rank == 0 else float(active_rank) / float(total_rank)
        return {
            "total_rank": int(total_rank),
            "active_rank": int(active_rank),
            "active_ratio": active_ratio,
            "num_modules": int(len(per_module)),
            "per_module_active_rank": per_module,
        }

    def _target_total_rank(self, adapter_name: str = "default") -> int:
        cfg: UniLoRAIGUConfig = self.peft_config[adapter_name]
        stats = self.get_rank_structure_stats(adapter_name)
        total_rank = int(stats["total_rank"])
        num_modules = int(stats["num_modules"])
        per_module_floor = max(0, int(cfg.igu_r_min))
        target_total = int(cfg.igu_target_rank) * max(1, num_modules)
        target_total = max(per_module_floor * max(1, num_modules), min(total_rank, target_total))
        return target_total

    def schedule_threshold(self, step: int, adapter_name: str = "default") -> tuple[int, bool]:
        cfg: UniLoRAIGUConfig = self.peft_config[adapter_name]
        stats = self.get_rank_structure_stats(adapter_name)
        total_rank = int(stats["total_rank"])
        target_rank = self._target_total_rank(adapter_name)
        total_step = self._total_steps
        initial_warmup = max(0, int(cfg.igu_init_warmup))
        final_warmup = max(0, int(cfg.igu_final_warmup))
        mask_interval = max(1, int(cfg.igu_mask_interval))

        if total_step is None or total_step <= (initial_warmup + final_warmup + 1):
            return target_rank, step >= total_step if total_step is not None else False

        if step <= initial_warmup:
            return total_rank, False
        if step > total_step - final_warmup:
            return target_rank, True

        mul_coeff = 1.0 - (step - initial_warmup) / float(total_step - final_warmup - initial_warmup)
        curr_rank = target_rank + (total_rank - target_rank) * (mul_coeff**3)
        curr_rank = int(curr_rank)
        curr_rank = max(target_rank, min(total_rank, curr_rank))
        return curr_rank, (step % mask_interval == 0)

    @torch.no_grad()
    def update_and_mask(self, global_step: int, adapter_name: str = "default") -> dict[str, object]:
        cfg: UniLoRAIGUConfig = self.peft_config[adapter_name]
        curr_rank, should_mask = self.schedule_threshold(global_step, adapter_name=adapter_name)
        stats_before = self.get_rank_structure_stats(adapter_name)
        active_rank_before = int(stats_before["active_rank"])
        info = {
            "step": int(global_step),
            "target_rank": int(curr_rank),
            "active_rank_before": int(active_rank_before),
            "active_rank_after": int(active_rank_before),
            "masked_ranks": 0,
            "mask_applied": False,
            "reset_optimizer": False,
        }
        if not should_mask or active_rank_before <= curr_rank:
            return info

        eps = float(cfg.igu_eps)
        r_min = max(0, int(cfg.igu_r_min))
        candidates = []
        module_entries = []
        for module_name, module in self.model.named_modules():
            if not isinstance(module, UniLoRAIGULayer):
                continue
            mask = module.unilora_igu_lora_mask[adapter_name].detach().clone()
            score = module.get_rank_scores(adapter_name, eps=eps).detach().float()
            # `lora_mask` is stored as shape (r, 1); flatten it first so we collect
            # 1-D rank indices instead of 2-D coordinate pairs.
            active_indices = torch.nonzero(mask.reshape(-1) > 0, as_tuple=False).reshape(-1)
            module_entries.append((module_name, module, mask, active_indices))
            if active_indices.numel() <= r_min:
                continue
            for rank_idx in active_indices.tolist():
                candidates.append((float(score[rank_idx].item()), module_name, rank_idx))

        removable = max(0, active_rank_before - curr_rank)
        if removable == 0 or not candidates:
            return info

        candidates.sort(key=lambda item: item[0])
        module_kept = {module_name: len(active_indices) for module_name, _, _, active_indices in module_entries}
        removed = []
        removed_count = 0
        for _, module_name, rank_idx in candidates:
            if removed_count >= removable:
                break
            if module_kept[module_name] <= r_min:
                continue
            module_kept[module_name] -= 1
            removed.append((module_name, rank_idx))
            removed_count += 1

        if not removed:
            return info

        removed_by_module = {}
        for module_name, rank_idx in removed:
            removed_by_module.setdefault(module_name, []).append(rank_idx)

        for module_name, module, mask, _active_indices in module_entries:
            rank_indices = removed_by_module.get(module_name)
            if not rank_indices:
                continue
            module.prune_ranks(adapter_name, sorted(rank_indices))

        stats_after = self.get_rank_structure_stats(adapter_name)
        info.update(
            {
                "active_rank_after": int(stats_after["active_rank"]),
                "masked_ranks": int(len(removed)),
                "mask_applied": True,
                "reset_optimizer": bool(cfg.igu_reset_optimizer_on_mask),
                "removed_by_module": {k: sorted(v) for k, v in removed_by_module.items()},
            }
        )
        return info

    def get_nb_savable_parameters(self, adapter="default") -> tuple[int, int]:
        theta_d_params = 0
        lora_e_params = 0
        other_params = 0
        for name, param in self.named_parameters():
            if "unilora_igu_theta_d" in name:
                theta_d_params += param.numel()
            elif "unilora_igu_lora_E" in name:
                lora_e_params += param.numel()

        for name, buffer in self.named_buffers():
            if (
                "unilora_indices" in name
                or "unilora_scales" in name
                or "unilora_igu_lora_mask" in name
                or "unilora_igu_ranknum" in name
                or "unilora_igu_weight_coeff" in name
            ):
                other_params += buffer.numel()

        return theta_d_params + lora_e_params, other_params

    def print_savable_parameters(self) -> None:
        unilora_params, other_params = self.get_nb_savable_parameters()
        print(
            f"UniLoRA-IGU params to-be-saved (float32-equivalent): {unilora_params:,d} "
            f"|| total params to-be-saved: {(unilora_params + other_params):,d}"
        )

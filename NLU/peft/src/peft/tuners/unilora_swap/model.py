from __future__ import annotations

import warnings

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

from .config import UniLoRASwapConfig
from .layer import Linear, UniLoRALayer


class UniLoRASwapModel(BaseTuner):
    """
    UniLoRA variant with dynamic bucket reassignment.

    The model uses the standard random UniLoRA initialization, then periodically
    performs a split-and-merge swap based on |theta_d| * |optimizer exp_avg|.
    """

    prefix: str = "unilora_swap_"
    tuner_layer_cls = UniLoRALayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

        swap_config = config[adapter_name] if isinstance(config, dict) else config
        all_indices = self._assign_initial_indices(adapter_name, swap_config.theta_d_length, swap_config.proj_seed)
        if all_indices.numel() > 0:
            self.refresh_unilora_scales(adapter_name, theta_d_length=swap_config.theta_d_length)

    def _iter_unilora_modules(self) -> list[UniLoRALayer]:
        return [module for module in self.model.modules() if isinstance(module, UniLoRALayer)]

    def _assign_initial_indices(self, adapter_name: str, theta_d_length: int, proj_seed: int) -> torch.Tensor:
        lora_param_count = 0
        modules = self._iter_unilora_modules()
        for module in modules:
            lora_param_count += module.unilora_indices_A[adapter_name].numel()
            lora_param_count += module.unilora_indices_B[adapter_name].numel()

        if lora_param_count == 0:
            return torch.empty(0, dtype=torch.long)

        all_elements = self.generate_index(lora_param_count, theta_d_length, proj_seed)
        pointer = 0
        for module in modules:
            num_a = module.unilora_indices_A[adapter_name].numel()
            chunk_a = all_elements[pointer : pointer + num_a]
            module.unilora_indices_A[adapter_name] = chunk_a.view_as(module.unilora_indices_A[adapter_name]).clone()
            pointer += num_a

            num_b = module.unilora_indices_B[adapter_name].numel()
            chunk_b = all_elements[pointer : pointer + num_b]
            module.unilora_indices_B[adapter_name] = chunk_b.view_as(module.unilora_indices_B[adapter_name]).clone()
            pointer += num_b

        if pointer != all_elements.numel():
            raise RuntimeError("UniLoRA-Swap index assignment is inconsistent.")
        return all_elements

    def _collect_all_indices(self, adapter_name: str) -> torch.Tensor:
        indices = []
        for module in self._iter_unilora_modules():
            if adapter_name not in module.unilora_indices_A:
                continue
            indices.append(module.unilora_indices_A[adapter_name].reshape(-1).long().cpu())
            indices.append(module.unilora_indices_B[adapter_name].reshape(-1).long().cpu())
        if not indices:
            return torch.empty(0, dtype=torch.long)
        return torch.cat(indices, dim=0)

    def refresh_unilora_scales(self, adapter_name: str, theta_d_length: int | None = None) -> torch.Tensor:
        theta_d = self.unilora_swap_theta_d[adapter_name]
        all_indices = self._collect_all_indices(adapter_name)
        if theta_d_length is None:
            theta_d_length = theta_d.numel()

        counts = torch.bincount(all_indices, minlength=theta_d_length)
        inv_sqrt_counts = torch.zeros(theta_d_length, dtype=torch.float32)
        non_zero = counts > 0
        inv_sqrt_counts[non_zero] = 1.0 / torch.sqrt(counts[non_zero].float())

        for module in self._iter_unilora_modules():
            if adapter_name not in module.unilora_indices_A:
                continue
            scale_a = inv_sqrt_counts[module.unilora_indices_A[adapter_name].long().cpu()]
            scale_b = inv_sqrt_counts[module.unilora_indices_B[adapter_name].long().cpu()]
            module.update_norm(adapter_name, scale_a, scale_b)
        return counts

    @staticmethod
    def generate_index(total_length: int, num_unique: int, proj_seed: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(proj_seed)

        base_count = total_length // num_unique
        remaining = total_length % num_unique
        data = torch.arange(num_unique, dtype=torch.long).repeat_interleave(base_count)
        if remaining > 0:
            extras = torch.randperm(num_unique, generator=generator)[:remaining]
            data = torch.cat([data, extras], dim=0)
        shuffle = torch.randperm(data.numel(), generator=generator)
        return data[shuffle]

    def _init_unilora_theta_d(self, config: UniLoRASwapConfig, adapter_name: str) -> None:
        if adapter_name in self.unilora_swap_theta_d:
            return
        unilora_theta_d = torch.zeros(config.theta_d_length)
        torch.nn.init.uniform_(unilora_theta_d, -config.init_theta_d_bound, config.init_theta_d_bound)
        self.unilora_swap_theta_d[adapter_name] = unilora_theta_d

    def _pre_injection_hook(self, model: nn.Module, config: UniLoRASwapConfig, adapter_name: str) -> None:
        self.unilora_swap_theta_d = nn.ParameterDict({})

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
                unilora_theta_d=self.unilora_swap_theta_d,
                r=unilora_config.r,
                theta_d_length=unilora_config.theta_d_length,
                unilora_dropout=unilora_config.unilora_dropout,
            )
        else:
            new_module = self._create_new_module(
                unilora_config=unilora_config,
                unilora_theta_d=self.unilora_swap_theta_d,
                adapter_name=adapter_name,
                target=target,
                **kwargs,
            )
            if adapter_name not in self.active_adapter:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(unilora_config, unilora_theta_d, adapter_name, target, **kwargs):
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
            unilora_theta_d=unilora_theta_d,
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
            if "unilora_swap_theta_d" in name:
                theta_d_params += param.numel()
            elif "unilora_indices" in name or "unilora_scales" in name:
                other_params += param.numel()
        return theta_d_params, other_params

    def print_savable_parameters(self) -> None:
        theta_d_params, other_params = self.get_nb_savable_parameters()
        print(
            f"UniLoRA-Swap params to-be-saved (float32-equivalent): {theta_d_params:,d} "
            f"|| total params to-be-saved: {(theta_d_params + other_params):,d}"
        )

    def get_swap_callback(self, adapter_name: str = "default"):
        from .swap_callback import UniLoRASwapCallback

        return UniLoRASwapCallback(adapter_name=adapter_name)

    @torch.no_grad()
    def perform_swap(
        self,
        optimizer,
        adapter_name: str = "default",
        dead_bucket_count: int | None = None,
        split_ratio: float | None = None,
    ) -> dict[str, float | int | bool]:
        if optimizer is None:
            return {"swapped": False, "reason": "missing_optimizer"}
        if adapter_name not in self.unilora_swap_theta_d:
            return {"swapped": False, "reason": "missing_adapter"}

        config: UniLoRASwapConfig = self.peft_config[adapter_name]
        theta_d = self.unilora_swap_theta_d[adapter_name]
        theta_d_length = theta_d.numel()
        if theta_d_length <= 1:
            return {"swapped": False, "reason": "theta_too_short"}

        dead_bucket_count = config.swap_dead_bucket_count if dead_bucket_count is None else dead_bucket_count
        split_ratio = config.swap_split_ratio if split_ratio is None else split_ratio
        dead_bucket_count = min(dead_bucket_count, theta_d_length - 1)
        if dead_bucket_count <= 1:
            return {"swapped": False, "reason": "dead_bucket_count_too_small"}

        optimizer_state = optimizer.state.get(theta_d)
        if optimizer_state is None or "exp_avg" not in optimizer_state:
            return {"swapped": False, "reason": "missing_exp_avg"}

        value_score = theta_d.detach().abs().to(device="cpu", dtype=torch.float32)
        grad_score = optimizer_state["exp_avg"].detach().abs().to(device="cpu", dtype=torch.float32)
        importance = value_score * grad_score
        if torch.all(importance == 0):
            return {"swapped": False, "reason": "all_scores_zero"}

        all_counts_before = torch.bincount(self._collect_all_indices(adapter_name), minlength=theta_d_length)
        dead_candidates = torch.argsort(importance, descending=False)
        dead_buckets = dead_candidates[:dead_bucket_count].tolist()
        sink_bucket = dead_buckets[0]
        free_buckets = dead_buckets[1:]
        if not free_buckets:
            return {"swapped": False, "reason": "no_free_buckets"}

        dead_set = set(dead_buckets)
        overloaded_buckets = [idx for idx in torch.argsort(importance, descending=True).tolist() if idx not in dead_set]
        overloaded_buckets = overloaded_buckets[: len(free_buckets)]
        if not overloaded_buckets:
            return {"swapped": False, "reason": "no_overloaded_buckets"}

        self._merge_dead_buckets(adapter_name, sink_bucket=sink_bucket, free_buckets=free_buckets)

        generator = torch.Generator(device="cpu")
        optimizer_step = optimizer_state.get("step", 0)
        if isinstance(optimizer_step, torch.Tensor):
            optimizer_step = int(optimizer_step.item())
        generator.manual_seed(int(config.proj_seed) + int(optimizer_step))
        split_pairs = []
        for overloaded_bucket, free_bucket in zip(overloaded_buckets, free_buckets):
            moved_count = self._split_bucket_assignments(
                adapter_name=adapter_name,
                source_bucket=overloaded_bucket,
                target_bucket=free_bucket,
                split_ratio=split_ratio,
                generator=generator,
            )
            if moved_count <= 0:
                continue

            source_count_after = int(self._count_bucket_assignments(adapter_name, overloaded_bucket))
            target_count_after = int(self._count_bucket_assignments(adapter_name, free_bucket))
            old_count = int(all_counts_before[overloaded_bucket].item())
            if old_count <= 0:
                continue

            old_value = theta_d.data[overloaded_bucket].clone()
            theta_d.data[overloaded_bucket] = old_value * (source_count_after / old_count) ** 0.5
            theta_d.data[free_bucket] = old_value * (target_count_after / old_count) ** 0.5
            split_pairs.append((overloaded_bucket, free_bucket))

        if not split_pairs:
            return {"swapped": False, "reason": "no_assignments_moved"}

        counts_after = self.refresh_unilora_scales(adapter_name, theta_d_length=theta_d_length)
        if config.swap_reset_optimizer_state:
            self._reset_optimizer_state(optimizer_state, split_pairs)

        return {
            "swapped": True,
            "num_pairs": len(split_pairs),
            "sink_bucket": int(sink_bucket),
            "num_freed_buckets": len(free_buckets),
            "importance_max": float(importance.max().item()),
            "importance_min": float(importance.min().item()),
            "count_max_after": int(counts_after.max().item()) if counts_after.numel() > 0 else 0,
        }

    def _merge_dead_buckets(self, adapter_name: str, sink_bucket: int, free_buckets: list[int]) -> None:
        free_bucket_set = set(free_buckets)
        for module in self._iter_unilora_modules():
            if adapter_name not in module.unilora_indices_A:
                continue
            for tensor_name in ("unilora_indices_A", "unilora_indices_B"):
                indices = getattr(module, tensor_name)[adapter_name]
                mask = torch.zeros_like(indices, dtype=torch.bool)
                for free_bucket in free_bucket_set:
                    mask |= indices == free_bucket
                indices[mask] = sink_bucket

    def _split_bucket_assignments(
        self,
        adapter_name: str,
        source_bucket: int,
        target_bucket: int,
        split_ratio: float,
        generator: torch.Generator,
    ) -> int:
        locations = []
        total_matches = 0
        for module in self._iter_unilora_modules():
            if adapter_name not in module.unilora_indices_A:
                continue
            for tensor_name in ("unilora_indices_A", "unilora_indices_B"):
                indices = getattr(module, tensor_name)[adapter_name]
                flat_positions = (indices.reshape(-1) == source_bucket).nonzero(as_tuple=False).reshape(-1).cpu()
                if flat_positions.numel() == 0:
                    continue
                locations.append((indices, flat_positions))
                total_matches += int(flat_positions.numel())

        if total_matches <= 1:
            return 0

        move_count = int(round(total_matches * split_ratio))
        move_count = max(1, min(total_matches - 1, move_count))
        chosen = torch.randperm(total_matches, generator=generator)[:move_count]
        move_mask = torch.zeros(total_matches, dtype=torch.bool)
        move_mask[chosen] = True

        moved = 0
        cursor = 0
        for indices, flat_positions in locations:
            count = flat_positions.numel()
            local_move_mask = move_mask[cursor : cursor + count]
            if local_move_mask.any():
                selected_positions = flat_positions[local_move_mask].to(device=indices.device)
                indices.view(-1)[selected_positions] = target_bucket
                moved += int(selected_positions.numel())
            cursor += count
        return moved

    def _count_bucket_assignments(self, adapter_name: str, bucket_id: int) -> int:
        total = 0
        for module in self._iter_unilora_modules():
            if adapter_name not in module.unilora_indices_A:
                continue
            total += int((module.unilora_indices_A[adapter_name] == bucket_id).sum().item())
            total += int((module.unilora_indices_B[adapter_name] == bucket_id).sum().item())
        return total

    @staticmethod
    def _reset_optimizer_state(optimizer_state: dict, split_pairs: list[tuple[int, int]]) -> None:
        for state_name in ("exp_avg", "exp_avg_sq"):
            if state_name not in optimizer_state:
                continue
            state_tensor = optimizer_state[state_name]
            for source_bucket, target_bucket in split_pairs:
                state_tensor[source_bucket] = 0
                state_tensor[target_bucket] = 0

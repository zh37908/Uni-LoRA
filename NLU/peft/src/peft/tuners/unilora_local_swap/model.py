from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

from .._buffer_dict import BufferDict
from .config import UniLoRALocalSwapConfig
from .layer import Linear, UniLoRALocalSwapLayer


class UniLoRALocalSwapModel(BaseTuner):
    """
    UniLoRA variant with local swap-based bucket reassignment.
    """

    prefix: str = "unilora_local_swap_"
    tuner_layer_cls = UniLoRALocalSwapLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

        local_swap_config = config[adapter_name] if isinstance(config, dict) else config
        all_indices = self._assign_initial_indices(adapter_name, local_swap_config.theta_d_length, local_swap_config.proj_seed)
        total_positions = int(all_indices.numel())
        self._init_grad_ema(total_positions, adapter_name)
        self._assign_position_offsets(adapter_name)
        if all_indices.numel() > 0:
            self.refresh_unilora_scales(adapter_name, theta_d_length=local_swap_config.theta_d_length)

    def _iter_unilora_modules(self) -> list[UniLoRALocalSwapLayer]:
        return [module for module in self.model.modules() if isinstance(module, UniLoRALocalSwapLayer)]

    def enable_gradient_capture(self, enabled: bool = True) -> None:
        for module in self._iter_unilora_modules():
            module.set_capture_gradient(enabled)

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
            target_device = module.get_base_layer().weight.device
            module.unilora_indices_A[adapter_name] = chunk_a.view_as(module.unilora_indices_A[adapter_name]).clone().to(
                device=target_device, dtype=torch.long
            )
            pointer += num_a

            num_b = module.unilora_indices_B[adapter_name].numel()
            chunk_b = all_elements[pointer : pointer + num_b]
            module.unilora_indices_B[adapter_name] = chunk_b.view_as(module.unilora_indices_B[adapter_name]).clone().to(
                device=target_device, dtype=torch.long
            )
            pointer += num_b

        if pointer != all_elements.numel():
            raise RuntimeError("UniLoRA-LocalSwap index assignment is inconsistent.")
        return all_elements

    def _assign_position_offsets(self, adapter_name: str) -> None:
        pointer = 0
        for module in self._iter_unilora_modules():
            num_a = module.unilora_indices_A[adapter_name].numel()
            offsets_a = torch.arange(pointer, pointer + num_a, dtype=torch.long)
            pointer += num_a
            num_b = module.unilora_indices_B[adapter_name].numel()
            offsets_b = torch.arange(pointer, pointer + num_b, dtype=torch.long)
            pointer += num_b

            target_device = module.get_base_layer().weight.device
            module.unilora_local_swap_offsets_A[adapter_name] = offsets_a.view_as(
                module.unilora_indices_A[adapter_name]
            ).clone().to(device=target_device, dtype=torch.long)
            module.unilora_local_swap_offsets_B[adapter_name] = offsets_b.view_as(
                module.unilora_indices_B[adapter_name]
            ).clone().to(device=target_device, dtype=torch.long)

    def _init_grad_ema(self, total_positions: int, adapter_name: str) -> None:
        if adapter_name in self.unilora_local_swap_grad_ema:
            current = self.unilora_local_swap_grad_ema[adapter_name]
            if int(current.numel()) == total_positions:
                return
        self.unilora_local_swap_grad_ema[adapter_name] = torch.zeros(total_positions, dtype=torch.float32)

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
        theta_d = self.unilora_local_swap_theta_d[adapter_name]
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

    def _init_unilora_theta_d(self, config: UniLoRALocalSwapConfig, adapter_name: str) -> None:
        if adapter_name in self.unilora_local_swap_theta_d:
            return
        unilora_theta_d = torch.zeros(config.theta_d_length)
        torch.nn.init.uniform_(unilora_theta_d, -config.init_theta_d_bound, config.init_theta_d_bound)
        self.unilora_local_swap_theta_d[adapter_name] = unilora_theta_d

    def _pre_injection_hook(self, model: nn.Module, config: UniLoRALocalSwapConfig, adapter_name: str) -> None:
        self.unilora_local_swap_theta_d = nn.ParameterDict({})
        self.unilora_local_swap_grad_ema = BufferDict({}, persistent=False)

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
                unilora_local_swap_theta_d=self.unilora_local_swap_theta_d,
                unilora_local_swap_grad_ema=self.unilora_local_swap_grad_ema,
                r=unilora_config.r,
                theta_d_length=unilora_config.theta_d_length,
                unilora_dropout=unilora_config.unilora_dropout,
            )
        else:
            new_module = self._create_new_module(
                unilora_config=unilora_config,
                unilora_local_swap_theta_d=self.unilora_local_swap_theta_d,
                unilora_local_swap_grad_ema=self.unilora_local_swap_grad_ema,
                adapter_name=adapter_name,
                target=target,
                **kwargs,
            )
            if adapter_name not in self.active_adapter:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(unilora_config, unilora_local_swap_theta_d, unilora_local_swap_grad_ema, adapter_name, target, **kwargs):
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
            unilora_local_swap_theta_d=unilora_local_swap_theta_d,
            unilora_local_swap_grad_ema=unilora_local_swap_grad_ema,
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
            if "unilora_local_swap_theta_d" in name:
                theta_d_params += param.numel()
            elif "unilora_indices" in name or "unilora_scales" in name or "unilora_local_swap_offsets" in name:
                other_params += param.numel()

        for name, buffer in self.named_buffers():
            if "unilora_indices" in name or "unilora_scales" in name or "unilora_local_swap_offsets" in name:
                other_params += buffer.numel()

        return theta_d_params, other_params

    def print_savable_parameters(self) -> None:
        theta_d_params, other_params = self.get_nb_savable_parameters()
        print(
            f"UniLoRA-LocalSwap params to-be-saved (float32-equivalent): {theta_d_params:,d} "
            f"|| total params to-be-saved: {(theta_d_params + other_params):,d}"
        )

    def get_swap_callback(self, adapter_name: str = "default"):
        from .swap_callback import UniLoRALocalSwapCallback

        return UniLoRALocalSwapCallback(adapter_name=adapter_name)

    def accumulate_local_swap_statistics(
        self, adapter_name: str = "default", ema_momentum: float | None = None
    ) -> dict[str, int]:
        if ema_momentum is None:
            ema_momentum = self.peft_config[adapter_name].local_swap_grad_ema_momentum

        updated_modules = 0
        updated_tensors = 0
        for module in self._iter_unilora_modules():
            updated = module.accumulate_local_swap_statistics(adapter_name, ema_momentum)
            if updated > 0:
                updated_modules += 1
                updated_tensors += updated
        return {"updated_modules": updated_modules, "updated_tensors": updated_tensors}

    def _gather_flat_state(self, adapter_name: str) -> tuple[torch.Tensor, list[tuple[UniLoRALocalSwapLayer, str, torch.Tensor, tuple[int, ...]]]]:
        num_positions = int(self.unilora_local_swap_grad_ema[adapter_name].numel())
        assignments = torch.empty(num_positions, dtype=torch.long)
        layout = []

        for module in self._iter_unilora_modules():
            if adapter_name not in module.unilora_indices_A:
                continue
            offsets_a = module.unilora_local_swap_offsets_A[adapter_name].detach().cpu().reshape(-1).long()
            offsets_b = module.unilora_local_swap_offsets_B[adapter_name].detach().cpu().reshape(-1).long()
            indices_a = module.unilora_indices_A[adapter_name].detach().cpu().reshape(-1).long()
            indices_b = module.unilora_indices_B[adapter_name].detach().cpu().reshape(-1).long()

            assignments[offsets_a] = indices_a
            assignments[offsets_b] = indices_b

            layout.append((module, "A", offsets_a, tuple(module.unilora_indices_A[adapter_name].shape)))
            layout.append((module, "B", offsets_b, tuple(module.unilora_indices_B[adapter_name].shape)))

        return assignments, layout

    @staticmethod
    def _alignment_ratio(bucket_grad_sum: torch.Tensor, bucket_abs_grad_sum: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        ratio = torch.zeros_like(bucket_grad_sum, dtype=torch.float32)
        non_zero = bucket_abs_grad_sum > eps
        ratio[non_zero] = bucket_grad_sum[non_zero].abs() / bucket_abs_grad_sum[non_zero].clamp_min(eps)
        return ratio

    @staticmethod
    def _bucket_positions(assignments: torch.Tensor) -> dict[int, torch.Tensor]:
        if assignments.numel() == 0:
            return {}
        sorted_pos = torch.argsort(assignments)
        sorted_bucket = assignments[sorted_pos]
        unique_buckets, counts = torch.unique_consecutive(sorted_bucket, return_counts=True)
        bucket_positions = {}
        cursor = 0
        for bucket_id, count in zip(unique_buckets.tolist(), counts.tolist()):
            bucket_positions[int(bucket_id)] = sorted_pos[cursor : cursor + count]
            cursor += count
        return bucket_positions

    @staticmethod
    def _select_candidate_positions(
        positions_j: torch.Tensor,
        grad_ema: torch.Tensor,
        bucket_grad_sum: torch.Tensor,
        bucket_abs_grad_sum: torch.Tensor,
        bucket_id: int,
        max_candidates: int,
        eps: float = 1e-12,
    ) -> list[int]:
        if positions_j.numel() == 0:
            return []
        grad_j = grad_ema[positions_j]
        g_before = bucket_grad_sum[bucket_id]
        h_before = bucket_abs_grad_sum[bucket_id]
        ratio_before = abs(float(g_before.item())) / max(float(h_before.item()), eps)
        g_after = g_before - grad_j
        h_after = h_before - grad_j.abs()
        valid = h_after > eps
        removal_gain = torch.full_like(grad_j, fill_value=-1e9, dtype=torch.float32)
        removal_gain[valid] = g_after[valid].abs() / h_after[valid].clamp_min(eps) - ratio_before
        take = min(max_candidates, int(positions_j.numel()))
        topk = torch.topk(removal_gain, k=take, largest=True).indices
        return positions_j[topk].tolist()

    @staticmethod
    def _sample_target_buckets(
        generator: torch.Generator,
        candidate_buckets: torch.Tensor,
        num_samples: int,
    ) -> list[int]:
        if candidate_buckets.numel() == 0:
            return []
        if candidate_buckets.numel() <= num_samples:
            return candidate_buckets.tolist()
        chosen = torch.randperm(candidate_buckets.numel(), generator=generator)[:num_samples]
        return candidate_buckets[chosen].tolist()

    def _apply_flat_assignments(
        self,
        adapter_name: str,
        assignments: torch.Tensor,
        layout: list[tuple[UniLoRALocalSwapLayer, str, torch.Tensor, tuple[int, ...]]],
    ) -> None:
        for module, tensor_name, offsets, shape in layout:
            updated = assignments[offsets].view(shape)
            target_device = module.get_base_layer().weight.device
            if tensor_name == "A":
                module.unilora_indices_A[adapter_name] = updated.to(device=target_device, dtype=torch.long)
            else:
                module.unilora_indices_B[adapter_name] = updated.to(device=target_device, dtype=torch.long)

    @staticmethod
    def _refit_bucket_values(
        theta_cpu: torch.Tensor,
        bucket_counts: torch.Tensor,
        bucket_left: int,
        bucket_right: int,
    ) -> tuple[float, float]:
        count_left = max(int(bucket_counts[bucket_left].item()), 1)
        count_right = max(int(bucket_counts[bucket_right].item()), 1)
        eff_left = float(theta_cpu[bucket_left].item()) / math.sqrt(count_left)
        eff_right = float(theta_cpu[bucket_right].item()) / math.sqrt(count_right)
        new_eff_left = ((count_left - 1) * eff_left + eff_right) / count_left
        new_eff_right = ((count_right - 1) * eff_right + eff_left) / count_right
        return new_eff_left * math.sqrt(count_left), new_eff_right * math.sqrt(count_right)

    @staticmethod
    def _reset_optimizer_state(optimizer_state: dict, bucket_ids: set[int]) -> None:
        for state_name in ("exp_avg", "exp_avg_sq"):
            if state_name not in optimizer_state:
                continue
            state_tensor = optimizer_state[state_name]
            for bucket_id in bucket_ids:
                state_tensor[bucket_id] = 0

    @staticmethod
    def _max_swaps_from_ratio(num_positions: int, update_ratio: float) -> int:
        if num_positions <= 0:
            return 0
        max_changed_positions = max(1, int(math.ceil(num_positions * update_ratio)))
        return max(1, int(math.ceil(max_changed_positions / 2.0)))

    @torch.no_grad()
    def perform_local_swap(
        self,
        optimizer=None,
        adapter_name: str = "default",
    ) -> dict[str, float | int | bool]:
        if adapter_name not in self.unilora_local_swap_theta_d:
            return {"swapped": False, "reason": "missing_adapter"}

        config: UniLoRALocalSwapConfig = self.peft_config[adapter_name]
        theta_d = self.unilora_local_swap_theta_d[adapter_name]
        theta_d_length = theta_d.numel()
        if theta_d_length <= 1:
            return {"swapped": False, "reason": "theta_too_short"}

        grad_ema = self.unilora_local_swap_grad_ema[adapter_name].detach().cpu().to(torch.float32)
        if grad_ema.numel() == 0:
            return {"swapped": False, "reason": "missing_statistics"}
        if torch.count_nonzero(grad_ema).item() == 0:
            return {"swapped": False, "reason": "all_grad_ema_zero"}

        assignments, layout = self._gather_flat_state(adapter_name)
        if assignments.numel() == 0:
            return {"swapped": False, "reason": "empty_projection"}
        max_swaps_this_round = self._max_swaps_from_ratio(
            num_positions=int(assignments.numel()),
            update_ratio=config.local_swap_update_ratio,
        )

        theta_cpu = theta_d.detach().cpu().to(torch.float32).clone()
        generator = torch.Generator(device="cpu")
        optimizer_step = 0
        if optimizer is not None:
            optimizer_state = optimizer.state.get(theta_d)
            if optimizer_state is not None:
                optimizer_step = optimizer_state.get("step", 0)
                if isinstance(optimizer_step, torch.Tensor):
                    optimizer_step = int(optimizer_step.item())
        generator.manual_seed(int(config.proj_seed) + int(optimizer_step))

        accepted_swaps = []
        candidate_evaluations = 0

        for _ in range(max_swaps_this_round):
            bucket_counts = torch.bincount(assignments, minlength=theta_d_length).to(torch.float32)
            bucket_grad_sum = torch.zeros(theta_d_length, dtype=torch.float32)
            bucket_abs_grad_sum = torch.zeros(theta_d_length, dtype=torch.float32)
            bucket_grad_sum.index_add_(0, assignments, grad_ema)
            bucket_abs_grad_sum.index_add_(0, assignments, grad_ema.abs())

            nonempty = bucket_counts > 0
            if nonempty.sum().item() <= 1:
                break

            bucket_ratio = self._alignment_ratio(bucket_grad_sum, bucket_abs_grad_sum)
            conflict = torch.full_like(bucket_ratio, fill_value=-1.0)
            conflict[nonempty] = 1.0 - bucket_ratio[nonempty]
            bad_bucket_count = max(1, int(math.ceil(theta_d_length * config.local_swap_bad_bucket_frac)))
            bad_bucket_count = min(bad_bucket_count, int(nonempty.sum().item()))
            bad_buckets = torch.topk(conflict, k=bad_bucket_count, largest=True).indices.tolist()
            bucket_positions = self._bucket_positions(assignments)

            accepted_this_pass = False
            for bucket_j in bad_buckets:
                if int(bucket_counts[bucket_j].item()) < config.local_swap_min_bucket_size:
                    continue

                positions_j = bucket_positions.get(int(bucket_j))
                if positions_j is None or positions_j.numel() == 0:
                    continue

                candidate_positions = self._select_candidate_positions(
                    positions_j=positions_j,
                    grad_ema=grad_ema,
                    bucket_grad_sum=bucket_grad_sum,
                    bucket_abs_grad_sum=bucket_abs_grad_sum,
                    bucket_id=int(bucket_j),
                    max_candidates=config.local_swap_candidates_per_bucket,
                )
                if not candidate_positions:
                    continue

                candidate_bucket_ids = torch.nonzero(
                    (bucket_counts >= config.local_swap_min_bucket_size) & (torch.arange(theta_d_length) != bucket_j),
                    as_tuple=False,
                ).reshape(-1)

                for pos_i in candidate_positions:
                    target_buckets = self._sample_target_buckets(
                        generator=generator,
                        candidate_buckets=candidate_bucket_ids,
                        num_samples=config.local_swap_target_bucket_samples,
                    )
                    if not target_buckets:
                        continue

                    grad_i = float(grad_ema[pos_i].item())
                    ratio_j_before = float(bucket_ratio[bucket_j].item())
                    best_pair = None
                    best_delta = float("-inf")

                    for bucket_k in target_buckets:
                        positions_k = bucket_positions.get(int(bucket_k))
                        if positions_k is None or positions_k.numel() == 0:
                            continue

                        ratio_k_before = float(bucket_ratio[bucket_k].item())
                        grad_k = grad_ema[positions_k]
                        g_j_new = bucket_grad_sum[bucket_j] - grad_i + grad_k
                        h_j_new = bucket_abs_grad_sum[bucket_j] - abs(grad_i) + grad_k.abs()
                        g_k_new = bucket_grad_sum[bucket_k] - grad_k + grad_i
                        h_k_new = bucket_abs_grad_sum[bucket_k] - grad_k.abs() + abs(grad_i)
                        valid = (h_j_new > 1e-12) & (h_k_new > 1e-12)
                        if not valid.any():
                            continue

                        ratio_j_after = torch.zeros_like(g_j_new)
                        ratio_k_after = torch.zeros_like(g_k_new)
                        ratio_j_after[valid] = g_j_new[valid].abs() / h_j_new[valid].clamp_min(1e-12)
                        ratio_k_after[valid] = g_k_new[valid].abs() / h_k_new[valid].clamp_min(1e-12)
                        total_delta = (ratio_j_after + ratio_k_after) - (ratio_j_before + ratio_k_before)
                        total_delta[~valid] = -1e9
                        ratio_k_drop = ratio_k_before - ratio_k_after
                        total_delta[ratio_k_drop > config.local_swap_max_target_drop] = -1e9

                        local_best_idx = int(torch.argmax(total_delta).item())
                        local_best_delta = float(total_delta[local_best_idx].item())
                        candidate_evaluations += int(positions_k.numel())
                        if local_best_delta > best_delta:
                            best_delta = local_best_delta
                            best_pair = (
                                int(positions_k[local_best_idx].item()),
                                int(bucket_k),
                                float(ratio_j_after[local_best_idx].item()),
                                float(ratio_k_after[local_best_idx].item()),
                            )

                    if best_pair is None or best_delta <= config.local_swap_min_delta:
                        continue

                    pos_p, bucket_k, ratio_j_after, ratio_k_after = best_pair
                    assignments[pos_i] = bucket_k
                    assignments[pos_p] = bucket_j

                    new_theta_j, new_theta_k = self._refit_bucket_values(theta_cpu, bucket_counts, int(bucket_j), int(bucket_k))
                    theta_cpu[bucket_j] = new_theta_j
                    theta_cpu[bucket_k] = new_theta_k
                    accepted_swaps.append(
                        {
                            "source_bucket": int(bucket_j),
                            "target_bucket": int(bucket_k),
                            "source_pos": int(pos_i),
                            "target_pos": int(pos_p),
                            "delta": float(best_delta),
                            "source_ratio_after": ratio_j_after,
                            "target_ratio_after": ratio_k_after,
                        }
                    )
                    accepted_this_pass = True
                    break

                if accepted_this_pass:
                    break

            if not accepted_this_pass:
                break

        if not accepted_swaps:
            return {"swapped": False, "reason": "no_accepted_swap", "evaluated_pairs": candidate_evaluations}

        self._apply_flat_assignments(adapter_name, assignments, layout)
        theta_d.data.copy_(theta_cpu.to(device=theta_d.device, dtype=theta_d.dtype))
        counts_after = self.refresh_unilora_scales(adapter_name, theta_d_length=theta_d_length)

        if optimizer is not None and config.local_swap_reset_optimizer_state:
            optimizer_state = optimizer.state.get(theta_d)
            if optimizer_state is not None:
                touched_buckets = {
                    swap_info["source_bucket"] for swap_info in accepted_swaps
                } | {
                    swap_info["target_bucket"] for swap_info in accepted_swaps
                }
                self._reset_optimizer_state(optimizer_state, touched_buckets)

        mean_delta = sum(swap_info["delta"] for swap_info in accepted_swaps) / len(accepted_swaps)
        changed_positions = 2 * len(accepted_swaps)
        return {
            "swapped": True,
            "num_swaps": len(accepted_swaps),
            "changed_positions": int(changed_positions),
            "changed_ratio": float(changed_positions / max(int(assignments.numel()), 1)),
            "max_swaps_this_round": int(max_swaps_this_round),
            "evaluated_pairs": int(candidate_evaluations),
            "mean_delta": float(mean_delta),
            "max_delta": float(max(swap_info["delta"] for swap_info in accepted_swaps)),
            "count_max_after": int(counts_after.max().item()) if counts_after.numel() > 0 else 0,
            "count_min_after": int(counts_after.min().item()) if counts_after.numel() > 0 else 0,
        }

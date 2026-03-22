from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

from .config import UniLoRAHessianAwareConfig
from .layer import Linear, UniLoRAHessianAwareLayer


class UniLoRAHessianAwareModel(BaseTuner):
    """
    Uni-LoRA with low-frequency Hessian-aware structure updates.
    """

    prefix: str = "unilora_hessian_aware_"
    tuner_layer_cls = UniLoRAHessianAwareLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

        lora_para_cnt = 0
        for _, module in model.named_modules():
            if isinstance(module, UniLoRAHessianAwareLayer):
                lora_para_cnt += module.unilora_indices_A[adapter_name].numel()
                lora_para_cnt += module.unilora_indices_B[adapter_name].numel()

        theta_d_length = config[adapter_name].theta_d_length
        proj_seed = config[adapter_name].proj_seed
        all_elements = self.generate_index(lora_para_cnt, theta_d_length, proj_seed)
        pointer = 0

        for _, module in model.named_modules():
            if isinstance(module, UniLoRAHessianAwareLayer):
                param_numel = module.unilora_indices_A[adapter_name].numel()
                chunk = all_elements[pointer : pointer + param_numel]
                target_device = module.get_base_layer().weight.device
                module.unilora_indices_A[adapter_name] = chunk.view_as(module.unilora_indices_A[adapter_name]).clone().to(
                    device=target_device, dtype=torch.long
                )
                pointer += param_numel

                param_numel = module.unilora_indices_B[adapter_name].numel()
                chunk = all_elements[pointer : pointer + param_numel]
                module.unilora_indices_B[adapter_name] = chunk.view_as(module.unilora_indices_B[adapter_name]).clone().to(
                    device=target_device, dtype=torch.long
                )
                pointer += param_numel

        assert pointer == len(all_elements)

        counts = torch.bincount(all_elements, minlength=theta_d_length)
        inv_sqrt_counts = torch.zeros(theta_d_length, dtype=torch.float32)
        non_zero = counts > 0
        inv_sqrt_counts[non_zero] = 1.0 / torch.sqrt(counts[non_zero].float())

        for _, module in model.named_modules():
            if isinstance(module, UniLoRAHessianAwareLayer):
                scale_a = inv_sqrt_counts[module.unilora_indices_A[adapter_name].detach().cpu().long()]
                scale_b = inv_sqrt_counts[module.unilora_indices_B[adapter_name].detach().cpu().long()]
                module.update_norm(adapter_name, scale_a, scale_b)

    @staticmethod
    def _bucket_candidate_ids(sorted_bucket_values, sorted_bucket_ids, target_value, candidate_pool_size, current_bucket):
        half_window = max(0, candidate_pool_size // 2)
        insert_pos = int(
            torch.searchsorted(sorted_bucket_values, torch.tensor([target_value], dtype=sorted_bucket_values.dtype)).item()
        )
        candidate_ids = [int(current_bucket)]

        for offset in range(-half_window, half_window + 1):
            candidate_pos = min(max(insert_pos + offset, 0), sorted_bucket_values.numel() - 1)
            candidate_ids.append(int(sorted_bucket_ids[candidate_pos].item()))

        deduped = []
        seen = set()
        for bucket_id in candidate_ids:
            if bucket_id in seen:
                continue
            seen.add(bucket_id)
            deduped.append(bucket_id)
        return deduped

    @staticmethod
    def _bucket_sse(weight_sum, value_sum, square_sum):
        if weight_sum <= 0.0:
            return 0.0
        return square_sum - (value_sum * value_sum) / max(weight_sum, 1e-12)

    def _iter_unilora_modules(self):
        return [module for module in self.model.modules() if isinstance(module, UniLoRAHessianAwareLayer)]

    def enable_curvature_capture(self, enabled: bool = True) -> None:
        for module in self._iter_unilora_modules():
            module.set_capture_curvature(enabled)

    def accumulate_curvature_statistics(self, adapter_name: str = "default", ema_momentum: float | None = None) -> dict[str, int]:
        if ema_momentum is None:
            ema_momentum = self.peft_config[adapter_name].curvature_ema_momentum

        updated_modules = 0
        updated_tensors = 0
        for module in self._iter_unilora_modules():
            updated = module.accumulate_curvature_statistics(adapter_name, ema_momentum)
            if updated > 0:
                updated_modules += 1
                updated_tensors += updated

        return {"updated_modules": updated_modules, "updated_tensors": updated_tensors}

    def get_structure_stats(self, adapter_name: str = "default") -> dict[str, float]:
        all_indices = []
        for module in self._iter_unilora_modules():
            if adapter_name not in module.unilora_indices_A:
                continue
            all_indices.append(module.unilora_indices_A[adapter_name].detach().cpu().reshape(-1).long())
            all_indices.append(module.unilora_indices_B[adapter_name].detach().cpu().reshape(-1).long())

        if not all_indices:
            return {"num_positions": 0, "num_buckets": 0}

        all_indices = torch.cat(all_indices, dim=0)
        theta_d = self.unilora_hessian_aware_theta_d[adapter_name].detach().cpu()
        bucket_loads = torch.bincount(all_indices, minlength=theta_d.numel()).float()
        target_load = float(all_indices.numel()) / float(max(theta_d.numel(), 1))

        return {
            "num_positions": int(all_indices.numel()),
            "num_buckets": int(theta_d.numel()),
            "target_load": float(target_load),
            "load_min": int(bucket_loads.min().item()),
            "load_max": int(bucket_loads.max().item()),
            "load_mean": float(bucket_loads.mean().item()),
            "load_std": float(bucket_loads.std(unbiased=False).item()),
        }

    @torch.no_grad()
    def update_structure(
        self,
        adapter_name: str = "default",
        candidate_pool_size: int | None = None,
        reassign_ratio: float | None = None,
        capacity_penalty: float | None = None,
        capacity_slack: float | None = None,
    ) -> dict[str, float]:
        config: UniLoRAHessianAwareConfig = self.peft_config[adapter_name]
        candidate_pool_size = candidate_pool_size or config.candidate_pool_size
        reassign_ratio = reassign_ratio if reassign_ratio is not None else config.structure_reassign_ratio
        capacity_penalty = capacity_penalty if capacity_penalty is not None else config.capacity_penalty
        capacity_slack = capacity_slack if capacity_slack is not None else config.capacity_slack

        modules = self._iter_unilora_modules()
        if not modules:
            return {"skipped": True, "reason": "no_unilora_modules"}

        values = []
        curvatures = []
        assignments = []
        shapes = []
        theta_d = self.unilora_hessian_aware_theta_d[adapter_name].detach().cpu().to(torch.float32)

        for module in modules:
            state = module.get_structure_state(adapter_name)
            values.append(state["values_A"])
            values.append(state["values_B"])
            curvatures.append(state["curvature_A"])
            curvatures.append(state["curvature_B"])
            assignments.append(state["indices_A"])
            assignments.append(state["indices_B"])
            shapes.append((module, state["shape_A"], state["shape_B"]))

        target_values = torch.cat(values, dim=0)
        curvature = torch.cat(curvatures, dim=0)
        old_assignments = torch.cat(assignments, dim=0)

        num_positions = target_values.numel()
        if num_positions == 0:
            return {"skipped": True, "reason": "empty_projection"}

        if reassign_ratio >= 1.0:
            selected_positions = torch.argsort(curvature, descending=True)
        else:
            selected_count = max(1, int(math.ceil(num_positions * reassign_ratio)))
            selected_positions = torch.topk(curvature, k=selected_count, largest=True).indices
            selected_positions = selected_positions[torch.argsort(curvature[selected_positions], descending=True)]

        selected_mask = torch.zeros(num_positions, dtype=torch.bool)
        selected_mask[selected_positions] = True
        fixed_mask = ~selected_mask

        bucket_counts = torch.bincount(old_assignments[fixed_mask], minlength=theta_d.numel()).to(torch.float32)
        bucket_weight_sum = torch.zeros(theta_d.numel(), dtype=torch.float32)
        bucket_value_sum = torch.zeros(theta_d.numel(), dtype=torch.float32)
        bucket_square_sum = torch.zeros(theta_d.numel(), dtype=torch.float32)

        if fixed_mask.any():
            fixed_assignments = old_assignments[fixed_mask]
            fixed_curvature = curvature[fixed_mask]
            fixed_values = target_values[fixed_mask]
            bucket_weight_sum.index_add_(0, fixed_assignments, fixed_curvature)
            bucket_value_sum.index_add_(0, fixed_assignments, fixed_curvature * fixed_values)
            bucket_square_sum.index_add_(0, fixed_assignments, fixed_curvature * fixed_values.square())

        current_counts = torch.bincount(old_assignments, minlength=theta_d.numel()).float()
        effective_bucket_values = theta_d.clone()
        nonempty_current = current_counts > 0
        effective_bucket_values[nonempty_current] = theta_d[nonempty_current] / torch.sqrt(current_counts[nonempty_current])

        target_load = float(num_positions) / float(max(theta_d.numel(), 1))
        hard_capacity = max(1, int(math.ceil(target_load * capacity_slack)))
        sorted_bucket_values, sorted_bucket_ids = torch.sort(effective_bucket_values)

        new_assignments = old_assignments.clone()
        for position in selected_positions.tolist():
            value = float(target_values[position].item())
            curv = float(curvature[position].item())
            current_bucket = int(old_assignments[position].item())
            candidate_ids = self._bucket_candidate_ids(
                sorted_bucket_values,
                sorted_bucket_ids,
                value,
                candidate_pool_size,
                current_bucket,
            )

            best_bucket = current_bucket
            best_cost = None
            for bucket_id in candidate_ids:
                current_count = int(bucket_counts[bucket_id].item())
                if current_count >= hard_capacity and bucket_id != current_bucket:
                    continue

                old_weight = float(bucket_weight_sum[bucket_id].item())
                old_value = float(bucket_value_sum[bucket_id].item())
                old_square = float(bucket_square_sum[bucket_id].item())
                old_sse = self._bucket_sse(old_weight, old_value, old_square)

                new_weight = old_weight + curv
                new_value = old_value + curv * value
                new_square = old_square + curv * value * value
                new_sse = self._bucket_sse(new_weight, new_value, new_square)
                approximation_cost = new_sse - old_sse

                overload = max(0.0, float(current_count + 1) - target_load)
                capacity_cost = capacity_penalty * (overload / max(target_load, 1.0)) ** 2
                total_cost = approximation_cost + capacity_cost

                if best_cost is None or total_cost < best_cost:
                    best_cost = total_cost
                    best_bucket = bucket_id

            new_assignments[position] = best_bucket
            bucket_counts[best_bucket] += 1.0
            bucket_weight_sum[best_bucket] += curv
            bucket_value_sum[best_bucket] += curv * value
            bucket_square_sum[best_bucket] += curv * value * value

        final_counts = torch.bincount(new_assignments, minlength=theta_d.numel()).to(torch.float32)
        final_weight_sum = torch.zeros(theta_d.numel(), dtype=torch.float32)
        final_value_sum = torch.zeros(theta_d.numel(), dtype=torch.float32)
        final_weight_sum.index_add_(0, new_assignments, curvature)
        final_value_sum.index_add_(0, new_assignments, curvature * target_values)

        effective_new = effective_bucket_values.clone()
        nonempty_final = final_weight_sum > 0
        effective_new[nonempty_final] = final_value_sum[nonempty_final] / final_weight_sum[nonempty_final].clamp_min(1e-12)

        theta_new = theta_d.clone()
        theta_new[nonempty_final] = effective_new[nonempty_final] * torch.sqrt(final_counts[nonempty_final])

        projected_values = effective_new[new_assignments]
        changed_mask = new_assignments != old_assignments
        changed_count = int(changed_mask.sum().item())
        approximation_mse = float((target_values - projected_values).square().mean().item())
        weighted_bias_surrogate = float(
            (curvature * (target_values - projected_values).square()).mean().item()
        )

        theta_param = self.unilora_hessian_aware_theta_d[adapter_name]
        theta_param.data.copy_(theta_new.to(device=theta_param.device, dtype=theta_param.dtype))

        inv_sqrt_counts = torch.zeros_like(final_counts, dtype=torch.float32)
        non_zero = final_counts > 0
        inv_sqrt_counts[non_zero] = 1.0 / torch.sqrt(final_counts[non_zero])

        pointer = 0
        for module, shape_a, shape_b in shapes:
            base_device = module.get_base_layer().weight.device

            num_a = math.prod(shape_a)
            idx_a_cpu = new_assignments[pointer : pointer + num_a].view(shape_a).clone()
            pointer += num_a

            num_b = math.prod(shape_b)
            idx_b_cpu = new_assignments[pointer : pointer + num_b].view(shape_b).clone()
            pointer += num_b

            module.unilora_indices_A[adapter_name] = idx_a_cpu.to(device=base_device, dtype=torch.long)
            module.unilora_indices_B[adapter_name] = idx_b_cpu.to(device=base_device, dtype=torch.long)
            module.update_norm(
                adapter_name,
                inv_sqrt_counts[idx_a_cpu.long()],
                inv_sqrt_counts[idx_b_cpu.long()],
            )

        return {
            "selected_positions": int(selected_positions.numel()),
            "changed_positions": changed_count,
            "changed_ratio": float(changed_count / float(num_positions)),
            "approximation_mse": approximation_mse,
            "weighted_bias_surrogate": weighted_bias_surrogate,
            "hard_capacity": int(hard_capacity),
            **self.get_structure_stats(adapter_name),
        }

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

    def _init_unilora_theta_d(self, config: UniLoRAHessianAwareConfig, adapter_name: str) -> None:
        unilora_theta_d = torch.zeros(config.theta_d_length)
        torch.nn.init.uniform_(unilora_theta_d, -config.init_theta_d_bound, config.init_theta_d_bound)
        self.unilora_hessian_aware_theta_d[adapter_name] = unilora_theta_d

    def _pre_injection_hook(self, model: nn.Module, config: UniLoRAHessianAwareConfig, adapter_name: str) -> None:
        self.unilora_hessian_aware_theta_d = nn.ParameterDict({})

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
                unilora_hessian_aware_theta_d=self.unilora_hessian_aware_theta_d,
                r=unilora_config.r,
                theta_d_length=unilora_config.theta_d_length,
                unilora_dropout=unilora_config.unilora_dropout,
            )
        else:
            new_module = self._create_new_module(
                unilora_config=unilora_config,
                unilora_hessian_aware_theta_d=self.unilora_hessian_aware_theta_d,
                adapter_name=adapter_name,
                target=target,
                **kwargs,
            )
            if adapter_name not in self.active_adapter:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(unilora_config, unilora_hessian_aware_theta_d, adapter_name, target, **kwargs):
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
            unilora_hessian_aware_theta_d=unilora_hessian_aware_theta_d,
            adapter_name=adapter_name,
            r=unilora_config.r,
            theta_d_length=unilora_config.theta_d_length,
            unilora_dropout=unilora_config.unilora_dropout,
            **kwargs,
        )
        return new_module

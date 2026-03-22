from __future__ import annotations

import hashlib
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

from .config import UniLoRATrajectoryInitialConfig
from .layer import Linear, UniLoRALayer


class UniLoRATrajectoryInitialModel(BaseTuner):
    """
    UniLoRA trajectory-initial model.

    The adapter keeps a global theta_d but clusters target layers into several
    buckets using deterministic block signatures. Each bucket receives a slice of
    theta_d and layers in the same cluster sample indices from the same slice.
    """

    prefix: str = "unilora_trajectory_initial_"
    tuner_layer_cls = UniLoRALayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

        unilora_config = config[adapter_name] if isinstance(config, dict) else config
        entries = self._collect_entries(model, adapter_name)
        if not entries:
            return

        num_clusters = min(unilora_config.num_buckets, len(entries), unilora_config.theta_d_length)
        if num_clusters < unilora_config.num_buckets:
            warnings.warn(
                f"Reducing num_buckets from {unilora_config.num_buckets} to {num_clusters} "
                "to match the number of target layers / theta_d budget."
            )

        features = self._build_feature_matrix(entries, unilora_config)
        assignments = self._run_kmeans(
            features,
            num_clusters=num_clusters,
            num_iters=unilora_config.kmeans_iters,
            seed=unilora_config.proj_seed,
        )

        bucket_param_counts = [0 for _ in range(num_clusters)]
        for entry, bucket_id in zip(entries, assignments.tolist()):
            entry["bucket_id"] = int(bucket_id)
            bucket_param_counts[bucket_id] += entry["num_params"]

        bucket_lengths = self._allocate_bucket_lengths(unilora_config.theta_d_length, bucket_param_counts)
        bucket_offsets = []
        running_offset = 0
        for length in bucket_lengths:
            bucket_offsets.append(running_offset)
            running_offset += length

        bucket_streams = {}
        for bucket_id, bucket_length in enumerate(bucket_lengths):
            local_cnt = bucket_param_counts[bucket_id]
            if local_cnt == 0:
                continue
            if bucket_length <= 0:
                raise ValueError(
                    f"Bucket {bucket_id} got theta_d_length=0 but still has {local_cnt} parameters to index. "
                    "Increase `theta_d_length` or reduce `num_buckets`."
                )
            local_indices = self.generate_index(local_cnt, bucket_length, unilora_config.proj_seed + 10007 * bucket_id)
            bucket_streams[bucket_id] = local_indices + bucket_offsets[bucket_id]

        bucket_pointers = {bucket_id: 0 for bucket_id in bucket_streams}
        for entry in entries:
            module = entry["module"]
            bucket_id = entry["bucket_id"]
            stream = bucket_streams[bucket_id]
            pointer = bucket_pointers[bucket_id]

            num_a = module.unilora_indices_A[adapter_name].numel()
            chunk_a = stream[pointer : pointer + num_a]
            module.unilora_indices_A[adapter_name] = chunk_a.view_as(module.unilora_indices_A[adapter_name]).clone()
            pointer += num_a

            num_b = module.unilora_indices_B[adapter_name].numel()
            chunk_b = stream[pointer : pointer + num_b]
            module.unilora_indices_B[adapter_name] = chunk_b.view_as(module.unilora_indices_B[adapter_name]).clone()
            pointer += num_b

            bucket_pointers[bucket_id] = pointer

        for bucket_id, stream in bucket_streams.items():
            if bucket_pointers[bucket_id] != stream.numel():
                raise RuntimeError(f"Bucket {bucket_id} index assignment is inconsistent.")

        all_indices = []
        for entry in entries:
            module = entry["module"]
            all_indices.append(module.unilora_indices_A[adapter_name].reshape(-1).long())
            all_indices.append(module.unilora_indices_B[adapter_name].reshape(-1).long())
        all_indices = torch.cat(all_indices, dim=0)

        counts = torch.bincount(all_indices, minlength=unilora_config.theta_d_length)
        inv_sqrt_counts = torch.zeros_like(counts, dtype=torch.float32)
        non_zero = counts > 0
        inv_sqrt_counts[non_zero] = 1.0 / torch.sqrt(counts[non_zero].float())

        for entry in entries:
            module = entry["module"]
            scale_a = inv_sqrt_counts[module.unilora_indices_A[adapter_name].long()]
            scale_b = inv_sqrt_counts[module.unilora_indices_B[adapter_name].long()]
            module.update_norm(adapter_name, scale_a, scale_b)

    def _init_unilora_theta_d(self, config: UniLoRATrajectoryInitialConfig, adapter_name: str) -> None:
        unilora_theta_d = torch.zeros(config.theta_d_length)
        torch.nn.init.uniform_(unilora_theta_d, -config.init_theta_d_bound, config.init_theta_d_bound)
        self.unilora_trajectory_initial_theta_d[adapter_name] = unilora_theta_d

    def _pre_injection_hook(self, model: nn.Module, config: UniLoRATrajectoryInitialConfig, adapter_name: str) -> None:
        self.unilora_trajectory_initial_theta_d = nn.ParameterDict({})

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
                unilora_theta_d=self.unilora_trajectory_initial_theta_d,
                r=unilora_config.r,
                theta_d_length=unilora_config.theta_d_length,
                unilora_dropout=unilora_config.unilora_dropout,
            )
        else:
            new_module = self._create_new_module(
                unilora_config=unilora_config,
                unilora_theta_d=self.unilora_trajectory_initial_theta_d,
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
            if "unilora_trajectory_initial_theta_d" in name:
                theta_d_params += param.numel()
            elif "unilora_indices" in name or "unilora_scales" in name:
                other_params += param.numel()
        return theta_d_params, other_params

    def print_savable_parameters(self) -> None:
        theta_d_params, other_params = self.get_nb_savable_parameters()
        print(
            f"UniLoRA-Trajectory-Initial params to-be-saved (float32-equivalent): {theta_d_params:,d} "
            f"|| total params to-be-saved: {(theta_d_params + other_params):,d}"
        )

    def _collect_entries(self, model, adapter_name: str) -> list[dict]:
        entries = []
        for name, module in model.named_modules():
            if not isinstance(module, UniLoRALayer):
                continue
            entries.append(
                {
                    "name": name,
                    "module": module,
                    "num_params": (
                        module.unilora_indices_A[adapter_name].numel() + module.unilora_indices_B[adapter_name].numel()
                    ),
                }
            )
        return entries

    def _build_feature_matrix(self, entries: list[dict], config: UniLoRATrajectoryInitialConfig) -> torch.Tensor:
        features = []
        total_layers = len(entries)
        for index, entry in enumerate(entries):
            module = entry["module"]
            name = entry["name"]
            base_layer = module.get_base_layer()
            weight = base_layer.weight.detach()
            if isinstance(base_layer, Conv1D):
                weight = weight.transpose(0, 1)

            seed = config.proj_seed + self._stable_int_seed(name)
            features.append(
                self._extract_layer_signature(
                    weight=weight,
                    block_rows=config.block_rows,
                    block_cols=config.block_cols,
                    hash_seed=seed,
                    layer_position=index,
                    total_layers=total_layers,
                )
            )
        return torch.stack(features, dim=0)

    @staticmethod
    def _stable_int_seed(text: str) -> int:
        digest = hashlib.md5(text.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], byteorder="little", signed=False)

    @staticmethod
    def _extract_layer_signature(
        weight: torch.Tensor,
        block_rows: int,
        block_cols: int,
        hash_seed: int,
        layer_position: int,
        total_layers: int,
    ) -> torch.Tensor:
        weight = weight.detach().to(dtype=torch.float32, device="cpu")
        out_features, in_features = weight.shape

        num_block_rows = math.ceil(out_features / block_rows)
        num_block_cols = math.ceil(in_features / block_cols)
        padded_rows = num_block_rows * block_rows
        padded_cols = num_block_cols * block_cols
        block_size = block_rows * block_cols

        generator = torch.Generator(device="cpu")
        generator.manual_seed(hash_seed)
        position_signs = torch.randint(
            0,
            2,
            (out_features * in_features,),
            generator=generator,
            dtype=torch.int64,
        )
        position_signs = position_signs.mul(2).sub(1).to(torch.float32).view(out_features, in_features)

        valid_mask = torch.ones(out_features, in_features, dtype=torch.float32)
        padded_weight = F.pad(weight, (0, padded_cols - in_features, 0, padded_rows - out_features))
        padded_sign = F.pad(position_signs, (0, padded_cols - in_features, 0, padded_rows - out_features), value=1.0)
        padded_valid = F.pad(valid_mask, (0, padded_cols - in_features, 0, padded_rows - out_features), value=0.0)

        blocks = (
            padded_weight.view(num_block_rows, block_rows, num_block_cols, block_cols)
            .permute(0, 2, 1, 3)
            .reshape(-1, block_size)
        )
        block_signs = (
            padded_sign.view(num_block_rows, block_rows, num_block_cols, block_cols)
            .permute(0, 2, 1, 3)
            .reshape(-1, block_size)
        )
        block_valid = (
            padded_valid.view(num_block_rows, block_rows, num_block_cols, block_cols)
            .permute(0, 2, 1, 3)
            .reshape(-1, block_size)
        )

        signed_blocks = blocks * block_signs * block_valid
        abs_blocks = signed_blocks.abs()
        block_rms = signed_blocks.square().mean(dim=1).sqrt()

        scalar_features = torch.tensor(
            [
                signed_blocks.mean().item(),
                signed_blocks.std(unbiased=False).item(),
                abs_blocks.mean().item(),
                abs_blocks.std(unbiased=False).item(),
                block_rms.mean().item(),
                block_rms.std(unbiased=False).item(),
                signed_blocks.sign().mean(dim=1).abs().mean().item(),
                math.log1p(float(out_features)),
                math.log1p(float(in_features)),
                float(layer_position) / max(1.0, float(total_layers - 1)),
            ],
            dtype=torch.float32,
        )

        return torch.cat([signed_blocks.mean(dim=0), abs_blocks.mean(dim=0), scalar_features], dim=0)

    @staticmethod
    def _standardize_features(features: torch.Tensor) -> torch.Tensor:
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        return (features - mean) / std

    @staticmethod
    def _compute_cluster_centers(data: torch.Tensor, assignments: torch.Tensor, num_clusters: int):
        centers = torch.zeros(num_clusters, data.size(1), device=data.device, dtype=data.dtype)
        counts = torch.zeros(num_clusters, device=data.device, dtype=data.dtype)
        centers.index_add_(0, assignments, data)
        counts.index_add_(0, assignments, torch.ones_like(assignments, dtype=data.dtype))
        centers = centers / counts.clamp_min(1.0).unsqueeze(1)
        return centers, counts

    @classmethod
    def _run_kmeans(cls, features: torch.Tensor, num_clusters: int, num_iters: int, seed: int) -> torch.Tensor:
        features = cls._standardize_features(features)
        num_points = features.size(0)
        if num_clusters >= num_points:
            return torch.arange(num_points, dtype=torch.long)

        data = features.to("cpu")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        permutation = torch.randperm(num_points, generator=generator)
        centers = data[permutation[:num_clusters]].clone()

        previous_assignments = None
        for _ in range(num_iters):
            center_norms = (centers * centers).sum(dim=1)
            distances = (
                (data * data).sum(dim=1, keepdim=True)
                - 2.0 * data @ centers.t()
                + center_norms.unsqueeze(0)
            )
            assignments = distances.argmin(dim=1)

            if previous_assignments is not None and torch.equal(assignments, previous_assignments):
                break
            previous_assignments = assignments.clone()

            new_centers, counts = cls._compute_cluster_centers(data, assignments, num_clusters)
            empty = counts == 0
            if empty.any():
                replacement = torch.randperm(num_points, generator=generator)[: int(empty.sum().item())]
                new_centers[empty] = data[replacement]
            centers = new_centers

        return assignments.cpu()

    @staticmethod
    def _allocate_bucket_lengths(total_d: int, bucket_param_counts: list[int]) -> list[int]:
        lengths = [0 for _ in bucket_param_counts]
        active_buckets = [bucket_id for bucket_id, count in enumerate(bucket_param_counts) if count > 0]
        if not active_buckets:
            return lengths

        total_params = sum(bucket_param_counts[bucket_id] for bucket_id in active_buckets)
        raw = [total_d * bucket_param_counts[bucket_id] / total_params for bucket_id in active_buckets]
        floors = [int(value) for value in raw]
        remain = total_d - sum(floors)

        frac_order = sorted(range(len(active_buckets)), key=lambda idx: raw[idx] - floors[idx], reverse=True)
        for idx in frac_order[:remain]:
            floors[idx] += 1

        if total_d >= len(active_buckets):
            for idx in range(len(floors)):
                if floors[idx] != 0:
                    continue
                donor = max(range(len(floors)), key=lambda donor_idx: floors[donor_idx])
                if floors[donor] > 1:
                    floors[donor] -= 1
                    floors[idx] = 1

        for local_idx, bucket_id in enumerate(active_buckets):
            lengths[bucket_id] = floors[local_idx]
        return lengths

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

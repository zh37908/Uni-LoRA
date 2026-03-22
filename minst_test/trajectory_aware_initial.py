import argparse
import json
import math
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms


def build_results_path(args):
    if args.results_path is not None:
        return args.results_path

    return "mnist_trajectory_aware_initial_br{}_bc{}_compress{}_seed{}.json".format(
        args.block_rows,
        args.block_cols,
        args.compress,
        args.seed,
    )


def save_results(
    args,
    warmup_parameter_count,
    parameter_count,
    warmup_history,
    finetune_history,
    clustering,
    test_loss,
    test_acc,
):
    results_path = build_results_path(args)
    results_dir = os.path.dirname(results_path)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)

    payload = {
        "args": vars(args),
        "warmup_parameter_count": warmup_parameter_count,
        "parameter_count": parameter_count,
        "warmup_history": warmup_history,
        "finetune_history": finetune_history,
        "clustering": clustering,
        "final_test": {
            "loss": test_loss,
            "accuracy": test_acc,
        },
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Saved results to {}".format(results_path))


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="PyTorch Trajectory-Aware Initial HashNet on MNIST",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--nhLayers", type=int, default=1, help="# hidden layers, excluding input/output layers")
    parser.add_argument("--nhu", type=int, default=1000, help="Number of hidden units")
    parser.add_argument("--compress", type=float, default=0.03125, help="Compression rate")
    parser.add_argument("--block-rows", type=int, default=4, help="Block height used for clustering")
    parser.add_argument("--block-cols", type=int, default=4, help="Block width used for clustering")
    parser.add_argument("--hash-bias", default=False, action="store_true", help="Learn dense bias terms")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate at t=0")
    parser.add_argument("--decay-factor", type=float, default=0.1, help="Learning rate decay factor")
    parser.add_argument("--batch-size", type=int, default=50, help="Mini-batch size")
    parser.add_argument(
        "--validation-percent",
        type=float,
        default=0.1,
        help="Percent of training data used for validation",
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum (SGD only)")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout rate")
    parser.add_argument("--l2reg", type=float, default=0.0, help="l2 regularisation")
    parser.add_argument("--epochs", type=int, default=50, help="Total number of epochs including warmup")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Dense warmup epochs used to collect signatures")
    parser.add_argument("--patience", type=int, default=2, help="Number of epochs to wait before scaling lr.")
    parser.add_argument("--hash-seed", type=int, default=2, help="Seed for sign hash functions")
    parser.add_argument("--kmeans-iters", type=int, default=15, help="Number of k-means refinement iterations")
    parser.add_argument(
        "--kmeans-device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used for k-means",
    )
    parser.add_argument(
        "--weight-feature-scale",
        type=float,
        default=1.0,
        help="Scale of current-weight block features in clustering",
    )
    parser.add_argument(
        "--grad-feature-scale",
        type=float,
        default=0.5,
        help="Scale of gradient block features in clustering",
    )
    parser.add_argument(
        "--pred-feature-scale",
        type=float,
        default=1.0,
        help="Scale of one-step predicted block features in clustering",
    )
    parser.add_argument(
        "--trajectory-step-scale",
        type=float,
        default=1.0,
        help="Multiplier for lr when forming one-step predicted weights",
    )
    parser.add_argument("--results-path", type=str, default=None, help="Path to save training metrics as JSON")
    parser.add_argument(
        "--save-model-path",
        type=str,
        default="mnist_trajectory_aware_initial.pt",
        help="Path to save the final model checkpoint",
    )
    parser.add_argument("--save-model", action="store_true", default=False, help="Save the final model checkpoint")
    args = parser.parse_args()

    if not 0.0 < args.validation_percent < 1.0:
        parser.error("--validation-percent must be in (0, 1)")
    if args.compress <= 0.0:
        parser.error("--compress must be > 0")
    if args.block_rows < 1 or args.block_cols < 1:
        parser.error("--block-rows and --block-cols must be >= 1")
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if args.warmup_epochs < 1:
        parser.error("--warmup-epochs must be >= 1")
    if args.warmup_epochs >= args.epochs:
        parser.error("--warmup-epochs must be < --epochs")
    if args.kmeans_iters < 1:
        parser.error("--kmeans-iters must be >= 1")
    if args.weight_feature_scale < 0.0:
        parser.error("--weight-feature-scale must be >= 0")
    if args.grad_feature_scale < 0.0:
        parser.error("--grad-feature-scale must be >= 0")
    if args.pred_feature_scale < 0.0:
        parser.error("--pred-feature-scale must be >= 0")
    if (
        args.weight_feature_scale == 0.0
        and args.grad_feature_scale == 0.0
        and args.pred_feature_scale == 0.0
    ):
        parser.error("At least one feature scale must be > 0")
    if args.trajectory_step_scale <= 0.0:
        parser.error("--trajectory-step-scale must be > 0")

    print(args)
    return args


def build_plateau_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.decay_factor,
        patience=args.patience,
        verbose=True,
    )


def load_data(batch_size, validation_percent, kwargs):
    train_dataset = datasets.MNIST(
        "../data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    random.shuffle(indices)
    split = int(math.floor(validation_percent * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        **kwargs,
    )
    valid_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        **kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )

    return train_loader, valid_loader, test_loader


def get_kmeans_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def get_linear_layers(model):
    layers = [("linear1", model.linear1)]
    for layer in range(2, model.nhLayers + 1):
        layers.append(("linear" + str(layer), getattr(model, "linear" + str(layer))))
    layers.append(("linear_out", model.linear_out))
    return layers


def make_block_collectors(model, args):
    collectors = []
    for index, (_, layer) in enumerate(get_linear_layers(model)):
        collectors.append(
            BlockTrajectoryCollector(
                layer.out_features,
                layer.in_features,
                args.block_rows,
                args.block_cols,
                args.hash_seed + index,
                args.weight_feature_scale,
                args.grad_feature_scale,
                args.pred_feature_scale,
                args.trajectory_step_scale,
            )
        )
    return collectors


def standardize_features(features):
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return (features - mean) / std


def compute_cluster_centers(data, assignments, num_clusters):
    centers = torch.zeros(num_clusters, data.size(1), device=data.device, dtype=data.dtype)
    counts = torch.zeros(num_clusters, device=data.device, dtype=data.dtype)
    centers.index_add_(0, assignments, data)
    counts.index_add_(0, assignments, torch.ones_like(assignments, dtype=data.dtype))
    centers = centers / counts.clamp_min(1.0).unsqueeze(1)
    return centers, counts


def run_kmeans(features, num_clusters, num_iters=15, device=None, seed=0, chunk_size=4096):
    features = standardize_features(features)
    num_points = features.size(0)

    if num_clusters >= num_points:
        assignments = torch.arange(num_points, dtype=torch.long)
        return assignments, features

    kmeans_device = device if device is not None else torch.device("cpu")
    data = features.to(kmeans_device)

    generator = torch.Generator(device=kmeans_device)
    generator.manual_seed(seed)
    permutation = torch.randperm(num_points, generator=generator, device=kmeans_device)
    centers = data[permutation[:num_clusters]].clone()

    previous_assignments = None
    for _ in range(num_iters):
        assignments = []
        center_norms = (centers * centers).sum(dim=1)

        for start in range(0, num_points, chunk_size):
            chunk = data[start:start + chunk_size]
            distances = (chunk * chunk).sum(dim=1, keepdim=True) - 2.0 * chunk @ centers.t() + center_norms.unsqueeze(0)
            assignments.append(distances.argmin(dim=1))

        assignments = torch.cat(assignments, dim=0)

        if previous_assignments is not None and torch.equal(assignments, previous_assignments):
            break
        previous_assignments = assignments.clone()

        new_centers, counts = compute_cluster_centers(data, assignments, num_clusters)
        empty = counts == 0
        if empty.any():
            replacement_indices = torch.randperm(num_points, generator=generator, device=kmeans_device)[: int(empty.sum().item())]
            new_centers[empty] = data[replacement_indices]

        centers = new_centers

    return assignments.cpu(), centers.cpu()


class BlockTrajectoryCollector:
    """
    收集 block 级的轨迹签名：
    1. 当前权重块
    2. 当前梯度块
    3. 一步预测后的权重块：w - lr * g
    再加少量 norm/sign 统计，作为聚类特征。
    """

    def __init__(
        self,
        out_features,
        in_features,
        block_rows,
        block_cols,
        hash_seed,
        weight_feature_scale,
        grad_feature_scale,
        pred_feature_scale,
        trajectory_step_scale,
    ):
        self.out_features = out_features
        self.in_features = in_features
        self.block_rows = block_rows
        self.block_cols = block_cols
        self.weight_feature_scale = weight_feature_scale
        self.grad_feature_scale = grad_feature_scale
        self.pred_feature_scale = pred_feature_scale
        self.trajectory_step_scale = trajectory_step_scale

        self.num_block_rows = math.ceil(out_features / block_rows)
        self.num_block_cols = math.ceil(in_features / block_cols)
        self.num_blocks = self.num_block_rows * self.num_block_cols
        self.padded_rows = self.num_block_rows * block_rows
        self.padded_cols = self.num_block_cols * block_cols
        self.block_size = block_rows * block_cols

        self.block_signs, self.block_valid_mask = self._build_block_signs(hash_seed)

        self.sum_weight_blocks = torch.zeros(self.num_blocks, self.block_size, dtype=torch.float32)
        self.sum_grad_blocks = torch.zeros(self.num_blocks, self.block_size, dtype=torch.float32)
        self.sum_pred_blocks = torch.zeros(self.num_blocks, self.block_size, dtype=torch.float32)
        self.sum_weight_norm = torch.zeros(self.num_blocks, dtype=torch.float32)
        self.sum_grad_norm = torch.zeros(self.num_blocks, dtype=torch.float32)
        self.sum_pred_norm = torch.zeros(self.num_blocks, dtype=torch.float32)
        self.sum_grad_sign_consistency = torch.zeros(self.num_blocks, dtype=torch.float32)
        self.num_updates = 0

    def _build_block_signs(self, hash_seed):
        generator = torch.Generator()
        generator.manual_seed(hash_seed)

        position_signs = torch.randint(
            0,
            2,
            (self.out_features * self.in_features,),
            generator=generator,
            dtype=torch.int64,
        )
        position_signs = position_signs.mul(2).sub(1).to(torch.float32)
        sign_matrix = position_signs.view(self.out_features, self.in_features)
        valid_matrix = torch.ones(self.out_features, self.in_features, dtype=torch.float32)

        padded_sign = F.pad(
            sign_matrix,
            (0, self.padded_cols - self.in_features, 0, self.padded_rows - self.out_features),
            value=1.0,
        )
        padded_valid = F.pad(
            valid_matrix,
            (0, self.padded_cols - self.in_features, 0, self.padded_rows - self.out_features),
            value=0.0,
        )

        block_signs = (
            padded_sign.view(self.num_block_rows, self.block_rows, self.num_block_cols, self.block_cols)
            .permute(0, 2, 1, 3)
            .reshape(self.num_blocks, self.block_size)
        )
        block_valid_mask = (
            padded_valid.view(self.num_block_rows, self.block_rows, self.num_block_cols, self.block_cols)
            .permute(0, 2, 1, 3)
            .reshape(self.num_blocks, self.block_size)
        )
        return block_signs, block_valid_mask

    def _tensor_to_blocks(self, tensor):
        padded = F.pad(
            tensor.detach().to(dtype=torch.float32, device="cpu"),
            (0, self.padded_cols - self.in_features, 0, self.padded_rows - self.out_features),
        )
        blocks = (
            padded.view(self.num_block_rows, self.block_rows, self.num_block_cols, self.block_cols)
            .permute(0, 2, 1, 3)
            .reshape(self.num_blocks, self.block_size)
        )
        return blocks * self.block_signs * self.block_valid_mask

    def update(self, weight, grad, step_size):
        if weight is None or grad is None:
            return

        weight_blocks = self._tensor_to_blocks(weight)
        grad_blocks = self._tensor_to_blocks(grad)
        pred_blocks = weight_blocks - (step_size * self.trajectory_step_scale) * grad_blocks

        self.sum_weight_blocks += weight_blocks
        self.sum_grad_blocks += grad_blocks
        self.sum_pred_blocks += pred_blocks
        self.sum_weight_norm += weight_blocks.square().mean(dim=1).sqrt()
        self.sum_grad_norm += grad_blocks.square().mean(dim=1).sqrt()
        self.sum_pred_norm += pred_blocks.square().mean(dim=1).sqrt()
        self.sum_grad_sign_consistency += grad_blocks.sign().mean(dim=1).abs()
        self.num_updates += 1

    def get_features(self):
        if self.num_updates == 0:
            raise ValueError("No trajectory signatures were collected")

        denom = float(self.num_updates)
        avg_weight_blocks = self.sum_weight_blocks / denom
        avg_grad_blocks = self.sum_grad_blocks / denom
        avg_pred_blocks = self.sum_pred_blocks / denom
        scalar_features = torch.stack(
            [
                self.weight_feature_scale * (self.sum_weight_norm / denom),
                self.grad_feature_scale * (self.sum_grad_norm / denom),
                self.pred_feature_scale * (self.sum_pred_norm / denom),
                self.sum_grad_sign_consistency / denom,
            ],
            dim=1,
        )

        return torch.cat(
            [
                self.weight_feature_scale * avg_weight_blocks,
                self.grad_feature_scale * avg_grad_blocks,
                self.pred_feature_scale * avg_pred_blocks,
                scalar_features,
            ],
            dim=1,
        )

    def summary(self):
        return {
            "num_blocks": self.num_blocks,
            "num_updates": self.num_updates,
            "block_size": self.block_size,
            "feature_dim": 3 * self.block_size + 4,
            "weight_feature_scale": self.weight_feature_scale,
            "grad_feature_scale": self.grad_feature_scale,
            "pred_feature_scale": self.pred_feature_scale,
            "trajectory_step_scale": self.trajectory_step_scale,
        }


class TrajectoryAwareHashLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        compress=0.03125,
        block_rows=4,
        block_cols=4,
        hash_seed=2,
        hash_bias=False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compress = compress
        self.block_rows = block_rows
        self.block_cols = block_cols
        self.hash_seed = hash_seed

        self.original_weight_size = out_features * in_features
        self.num_block_rows = math.ceil(out_features / block_rows)
        self.num_block_cols = math.ceil(in_features / block_cols)
        self.num_blocks = self.num_block_rows * self.num_block_cols
        self.block_size = block_rows * block_cols
        self.template_count = max(
            1,
            min(
                self.num_blocks,
                int(self.original_weight_size * compress / self.block_size),
            ),
        )

        self.shared_templates = nn.Parameter(torch.empty(self.template_count, self.block_size))

        if hash_bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        (
            position_block_ids,
            position_local_offsets,
            position_signs,
            block_signs,
            block_valid_mask,
        ) = self._build_mappings(hash_seed)

        self.register_buffer("position_block_ids", position_block_ids)
        self.register_buffer("position_local_offsets", position_local_offsets)
        self.register_buffer("position_signs", position_signs)
        self.register_buffer("block_signs", block_signs)
        self.register_buffer("block_valid_mask", block_valid_mask)
        self.register_buffer("block_assignments", torch.zeros(self.num_blocks, dtype=torch.long))

        self.collect_weight_grad = False
        self._last_weight = None

        self.reset_parameters()

    def _build_mappings(self, hash_seed):
        row_ids = torch.arange(self.out_features, dtype=torch.int64).unsqueeze(1)
        col_ids = torch.arange(self.in_features, dtype=torch.int64).unsqueeze(0)

        row_grid = row_ids.repeat(1, self.in_features).reshape(-1)
        col_grid = col_ids.repeat(self.out_features, 1).reshape(-1)

        block_row_ids = torch.div(row_grid, self.block_rows, rounding_mode="floor")
        block_col_ids = torch.div(col_grid, self.block_cols, rounding_mode="floor")
        block_ids = block_row_ids * self.num_block_cols + block_col_ids

        local_row_ids = row_grid % self.block_rows
        local_col_ids = col_grid % self.block_cols
        local_offsets = local_row_ids * self.block_cols + local_col_ids

        generator = torch.Generator()
        generator.manual_seed(hash_seed)
        position_signs = torch.randint(
            0,
            2,
            (self.original_weight_size,),
            generator=generator,
            dtype=torch.int64,
        )
        position_signs = position_signs.mul(2).sub(1).to(torch.float32)

        padded_rows = self.num_block_rows * self.block_rows
        padded_cols = self.num_block_cols * self.block_cols

        sign_matrix = position_signs.view(self.out_features, self.in_features)
        valid_matrix = torch.ones(self.out_features, self.in_features, dtype=torch.float32)

        padded_sign = F.pad(sign_matrix, (0, padded_cols - self.in_features, 0, padded_rows - self.out_features), value=1.0)
        padded_valid = F.pad(valid_matrix, (0, padded_cols - self.in_features, 0, padded_rows - self.out_features), value=0.0)

        block_signs = (
            padded_sign.view(self.num_block_rows, self.block_rows, self.num_block_cols, self.block_cols)
            .permute(0, 2, 1, 3)
            .reshape(self.num_blocks, self.block_size)
        )
        block_valid_mask = (
            padded_valid.view(self.num_block_rows, self.block_rows, self.num_block_cols, self.block_cols)
            .permute(0, 2, 1, 3)
            .reshape(self.num_blocks, self.block_size)
        )

        return block_ids, local_offsets, position_signs, block_signs, block_valid_mask

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.shared_templates, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features) if self.in_features > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def set_collect_weight_grad(self, enabled):
        self.collect_weight_grad = enabled
        if not enabled:
            self._last_weight = None

    def clear_cached_weight(self):
        self._last_weight = None

    def get_last_weight(self):
        return self._last_weight

    def get_last_weight_grad(self):
        if self._last_weight is None:
            return None
        return self._last_weight.grad

    def _dense_weight_to_blocks(self, dense_weight):
        padded_rows = self.num_block_rows * self.block_rows
        padded_cols = self.num_block_cols * self.block_cols
        padded_weight = F.pad(
            dense_weight.detach(),
            (0, padded_cols - self.in_features, 0, padded_rows - self.out_features),
        )
        blocks = (
            padded_weight.view(self.num_block_rows, self.block_rows, self.num_block_cols, self.block_cols)
            .permute(0, 2, 1, 3)
            .reshape(self.num_blocks, self.block_size)
        )
        return blocks

    def _build_template_stats(self, assignments, dense_weight):
        assignments = assignments.to(self.shared_templates.device, dtype=torch.long)
        dense_blocks = self._dense_weight_to_blocks(dense_weight.to(self.shared_templates.device, dtype=self.shared_templates.dtype))
        signed_blocks = dense_blocks * self.block_signs.to(self.shared_templates.dtype)
        masked_blocks = signed_blocks * self.block_valid_mask.to(self.shared_templates.dtype)

        template_sums = torch.zeros(
            self.template_count,
            self.block_size,
            device=self.shared_templates.device,
            dtype=self.shared_templates.dtype,
        )
        template_counts = torch.zeros_like(template_sums)
        template_sums.index_add_(0, assignments, masked_blocks)
        template_counts.index_add_(0, assignments, self.block_valid_mask.to(self.shared_templates.dtype))

        template_means = template_sums / template_counts.clamp_min(1.0)
        nonempty = template_counts.sum(dim=1) > 0
        return assignments, template_means, nonempty

    def apply_assignments_from_dense_weight(self, dense_weight, assignments):
        assignments, template_means, nonempty = self._build_template_stats(assignments, dense_weight)
        updated_templates = self.shared_templates.data.clone()
        updated_templates[nonempty] = template_means[nonempty]
        self.shared_templates.data.copy_(updated_templates)
        self.block_assignments.copy_(assignments)

        cluster_counts = torch.bincount(assignments.cpu(), minlength=self.template_count)
        return {
            "num_blocks": self.num_blocks,
            "template_count": self.template_count,
            "block_size": self.block_size,
            "cluster_min_size": int(cluster_counts.min().item()),
            "cluster_max_size": int(cluster_counts.max().item()),
            "cluster_mean_size": float(cluster_counts.float().mean().item()),
        }

    def initialize_from_dense_weight(self, dense_weight, trajectory_features, kmeans_iters, kmeans_device, seed):
        assignments, _ = run_kmeans(
            trajectory_features,
            self.template_count,
            num_iters=kmeans_iters,
            device=kmeans_device,
            seed=seed,
        )
        return self.apply_assignments_from_dense_weight(dense_weight, assignments)

    def forward(self, input_tensor):
        assigned_template_ids = self.block_assignments[self.position_block_ids]
        reconstructed_weight_flat = self.shared_templates[assigned_template_ids, self.position_local_offsets]
        reconstructed_weight_flat = reconstructed_weight_flat * self.position_signs.to(reconstructed_weight_flat.dtype)
        weight = reconstructed_weight_flat.view(self.out_features, self.in_features)
        if self.training and self.collect_weight_grad:
            weight.retain_grad()
            self._last_weight = weight
        else:
            self._last_weight = None
        return F.linear(input_tensor, weight, self.bias)


class WarmupDenseNet(nn.Module):
    def __init__(self, input_dim, output_dim, nhLayers=1, nhu=1000, dropout=0.25):
        super().__init__()
        self.nhLayers = nhLayers
        self.input_dim = input_dim

        self.dropout0 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(input_dim, nhu)
        self.dropout1 = nn.Dropout(dropout)

        for layer in range(2, nhLayers + 1):
            setattr(self, "linear" + str(layer), nn.Linear(nhu, nhu))
            setattr(self, "dropout" + str(layer), nn.Dropout(dropout))

        self.linear_out = nn.Linear(nhu, output_dim)

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        x = self.dropout0(x)
        x = F.relu(self.linear1(x))
        x = self.dropout1(x)

        for layer in range(2, self.nhLayers + 1):
            x = F.relu(getattr(self, "linear" + str(layer))(x))
            x = getattr(self, "dropout" + str(layer))(x)

        x = self.linear_out(x)
        return F.log_softmax(x, dim=1)


class TrajectoryAwareHashNet(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        nhLayers=1,
        nhu=1000,
        compress=0.03125,
        dropout=0.25,
        hash_seed=2,
        block_rows=4,
        block_cols=4,
        hash_bias=False,
    ):
        super().__init__()
        self.nhLayers = nhLayers
        self.input_dim = input_dim

        self.dropout0 = nn.Dropout(dropout)
        self.linear1 = TrajectoryAwareHashLinear(
            input_dim,
            nhu,
            compress=compress,
            block_rows=block_rows,
            block_cols=block_cols,
            hash_seed=hash_seed,
            hash_bias=hash_bias,
        )
        self.dropout1 = nn.Dropout(dropout)

        for layer in range(2, nhLayers + 1):
            setattr(
                self,
                "linear" + str(layer),
                TrajectoryAwareHashLinear(
                    nhu,
                    nhu,
                    compress=compress,
                    block_rows=block_rows,
                    block_cols=block_cols,
                    hash_seed=hash_seed + layer - 1,
                    hash_bias=hash_bias,
                ),
            )
            setattr(self, "dropout" + str(layer), nn.Dropout(dropout))

        self.linear_out = TrajectoryAwareHashLinear(
            nhu,
            output_dim,
            compress=compress,
            block_rows=block_rows,
            block_cols=block_cols,
            hash_seed=hash_seed + nhLayers,
            hash_bias=hash_bias,
        )

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        x = self.dropout0(x)
        x = F.relu(self.linear1(x))
        x = self.dropout1(x)

        for layer in range(2, self.nhLayers + 1):
            x = F.relu(getattr(self, "linear" + str(layer))(x))
            x = getattr(self, "dropout" + str(layer))(x)

        x = self.linear_out(x)
        return F.log_softmax(x, dim=1)


def set_gradient_collection(model, enabled):
    for _, layer in get_linear_layers(model):
        if isinstance(layer, TrajectoryAwareHashLinear):
            layer.set_collect_weight_grad(enabled)


def clear_gradient_collection_cache(model):
    for _, layer in get_linear_layers(model):
        if isinstance(layer, TrajectoryAwareHashLinear):
            layer.clear_cached_weight()


def get_layer_weight_and_grad(layer):
    if isinstance(layer, nn.Linear):
        return layer.weight, layer.weight.grad
    if isinstance(layer, TrajectoryAwareHashLinear):
        return layer.get_last_weight(), layer.get_last_weight_grad()
    raise TypeError("Unsupported layer type: {}".format(type(layer)))


def train_epoch(model, device, train_loader, optimizer, epoch, collectors=None, log_interval=5, stage_name="Train"):
    model.train()
    train_loss = 0.0
    linear_layers = get_linear_layers(model) if collectors is not None else None
    set_gradient_collection(model, collectors is not None)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()

        if collectors is not None:
            step_size = optimizer.param_groups[0]["lr"]
            for (_, layer), collector in zip(linear_layers, collectors):
                weight, grad = get_layer_weight_and_grad(layer)
                collector.update(weight, grad, step_size)
            clear_gradient_collection_cache(model)

        optimizer.step()

        if batch_idx % log_interval == 0:
            print(
                "{} Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}".format(
                    stage_name,
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.sampler),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                ),
                end="\r",
            )

        train_loss += loss.item() * data.size(0)

    set_gradient_collection(model, False)
    clear_gradient_collection_cache(model)
    return train_loss / len(train_loader.sampler)


def evaluate(model, device, loader):
    model.eval()
    loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    loss /= len(loader.sampler)
    accuracy = 100.0 * correct / len(loader.sampler)
    return loss, accuracy


def initialize_clustered_model(clustered_model, warmup_model, collectors, args, device):
    warmup_layers = get_linear_layers(warmup_model)
    clustered_layers = get_linear_layers(clustered_model)
    clustering = {}
    kmeans_device = get_kmeans_device(args.kmeans_device)

    for index, (((warmup_name, warmup_layer), (clustered_name, clustered_layer)), collector) in enumerate(
        zip(zip(warmup_layers, clustered_layers), collectors)
    ):
        if warmup_name != clustered_name:
            raise ValueError("Layer mismatch: {} vs {}".format(warmup_name, clustered_name))

        features = collector.get_features()
        info = clustered_layer.initialize_from_dense_weight(
            warmup_layer.weight.detach(),
            features,
            args.kmeans_iters,
            kmeans_device,
            seed=args.seed + index,
        )

        if clustered_layer.bias is not None and warmup_layer.bias is not None:
            clustered_layer.bias.data.copy_(
                warmup_layer.bias.detach().to(device=device, dtype=clustered_layer.bias.dtype)
            )

        info["trajectory_signature"] = collector.summary()
        clustering[warmup_name] = info
        print(
            "Clustered {} into {} templates from {} blocks".format(
                warmup_name,
                info["template_count"],
                info["num_blocks"],
            )
        )

    return clustering


def save_model_checkpoint(model, path):
    checkpoint_dir = os.path.dirname(path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(model.state_dict(), path)
    print("Saved model checkpoint to {}".format(path))


def main():
    args = parse_arguments()

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    tr_loader, val_loader, test_loader = load_data(args.batch_size, args.validation_percent, kwargs)
    input_dim = 784
    output_dim = 10

    warmup_model = WarmupDenseNet(
        input_dim=input_dim,
        output_dim=output_dim,
        nhLayers=args.nhLayers,
        nhu=args.nhu,
        dropout=args.dropout,
    ).to(device)

    collectors = make_block_collectors(warmup_model, args)

    warmup_optimizer = optim.SGD(
        warmup_model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.l2reg,
    )
    warmup_scheduler = build_plateau_scheduler(warmup_optimizer, args)

    warmup_history = []
    warmup_parameter_count = sum(p.numel() for p in warmup_model.parameters() if p.requires_grad)
    print("Warmup dense parameter count: {}".format(warmup_parameter_count))

    for epoch in range(1, args.warmup_epochs + 1):
        train_loss = train_epoch(
            warmup_model,
            device,
            tr_loader,
            warmup_optimizer,
            epoch,
            collectors=collectors,
            stage_name="Warmup",
        )
        val_loss, val_acc = evaluate(warmup_model, device, val_loader)
        warmup_scheduler.step(val_loss)
        warmup_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "learning_rate": warmup_optimizer.param_groups[0]["lr"],
            }
        )
        print(
            "\nWarmup Epoch {} Train loss: {:.3f} Val loss: {:.3f} Val acc: {:.2f}%".format(
                epoch,
                train_loss,
                val_loss,
                val_acc,
            )
        )

    clustered_model = TrajectoryAwareHashNet(
        input_dim=input_dim,
        output_dim=output_dim,
        nhLayers=args.nhLayers,
        nhu=args.nhu,
        compress=args.compress,
        dropout=args.dropout,
        hash_seed=args.hash_seed,
        block_rows=args.block_rows,
        block_cols=args.block_cols,
        hash_bias=args.hash_bias,
    ).to(device)

    clustering = initialize_clustered_model(clustered_model, warmup_model, collectors, args, device)
    del warmup_model

    finetune_optimizer = optim.SGD(
        clustered_model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.l2reg,
    )
    finetune_scheduler = build_plateau_scheduler(finetune_optimizer, args)

    finetune_history = []
    parameter_count = sum(p.numel() for p in clustered_model.parameters() if p.requires_grad)
    print("Trajectory-aware initial parameter count: {}".format(parameter_count))

    finetune_epochs = args.epochs - args.warmup_epochs
    for offset in range(1, finetune_epochs + 1):
        epoch = args.warmup_epochs + offset
        train_loss = train_epoch(
            clustered_model,
            device,
            tr_loader,
            finetune_optimizer,
            offset,
            collectors=None,
            stage_name="Finetune",
        )
        val_loss, val_acc = evaluate(clustered_model, device, val_loader)
        finetune_scheduler.step(val_loss)
        finetune_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "learning_rate": finetune_optimizer.param_groups[0]["lr"],
            }
        )
        print(
            "\nFinetune Epoch {} Train loss: {:.3f} Val loss: {:.3f} Val acc: {:.2f}%".format(
                epoch,
                train_loss,
                val_loss,
                val_acc,
            )
        )

    test_loss, test_acc = evaluate(clustered_model, device, test_loader)
    print("Test loss: {:.3f} Test acc: {:.2f}%".format(test_loss, test_acc))
    save_results(
        args,
        warmup_parameter_count,
        parameter_count,
        warmup_history,
        finetune_history,
        clustering,
        test_loss,
        test_acc,
    )

    if args.save_model:
        save_model_checkpoint(clustered_model, args.save_model_path)


if __name__ == "__main__":
    main()

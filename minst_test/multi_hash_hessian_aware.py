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


def get_equivalent_compression(input_dim, output_dim, nhu, nh_layers, compress):
    return compress


def build_results_path(args):
    if args.results_path is not None:
        return args.results_path

    model_name = "hessian_aware_hashed" if args.hashed else "dense"
    return "mnist_{}_compress{}_seed{}.json".format(model_name, args.compress, args.seed)


def save_results(args, parameter_count, history, structure_history, hessian_history, test_loss, test_acc):
    results_path = build_results_path(args)
    results_dir = os.path.dirname(results_path)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)

    payload = {
        "args": vars(args),
        "parameter_count": parameter_count,
        "history": history,
        "structure_history": structure_history,
        "hessian_history": hessian_history,
        "final_test": {
            "loss": test_loss,
            "accuracy": test_acc,
        },
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Saved results to {}".format(results_path))


def save_model_checkpoint(model, path):
    checkpoint_dir = os.path.dirname(path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(model.state_dict(), path)
    print("Saved model checkpoint to {}".format(path))


def build_plateau_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.decay_factor,
        patience=args.patience,
        verbose=True,
    )


def snapshot_module_state(module):
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def should_accept_structure_update(baseline_val_loss, baseline_val_acc, trial_val_loss, trial_val_acc):
    loss_eps = 1e-6
    acc_eps = 1e-6
    if trial_val_loss < baseline_val_loss - loss_eps:
        return True
    return trial_val_loss <= baseline_val_loss + loss_eps and trial_val_acc >= baseline_val_acc - acc_eps


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


class HessianAwareHashLinear(nn.Module):
    def __init__(self, in_features, out_features, compress=0.03125, hash_seed=2, hash_bias=False):
        super(HessianAwareHashLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compress = compress

        self.original_weight_size = out_features * in_features
        self.compressed_size = max(1, int(self.original_weight_size * compress))

        self.shared_weights = nn.Parameter(torch.empty(self.compressed_size))
        if hash_bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        generator = torch.Generator()
        generator.manual_seed(hash_seed)

        hash_indices = torch.randint(
            0,
            self.compressed_size,
            (self.original_weight_size,),
            generator=generator,
        )
        hash_signs = torch.randint(
            0,
            2,
            (self.original_weight_size,),
            generator=generator,
        )
        hash_signs = hash_signs.mul(2).sub(1).to(torch.float32)

        self.register_buffer("hash_indices", hash_indices)
        self.register_buffer("hash_signs", hash_signs)
        self.register_buffer("curvature_ema", torch.ones(self.original_weight_size, dtype=torch.float32))

        self._last_dense_weight = None
        self.capture_weight_for_analysis = False
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.in_features) if self.in_features > 0 else 0.0
        nn.init.uniform_(self.shared_weights, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def export_dense_weight(self):
        gathered = self.shared_weights[self.hash_indices]
        signed = gathered * self.hash_signs.to(gathered.dtype)
        return signed.view(self.out_features, self.in_features)

    def forward(self, input_tensor):
        weight = self.export_dense_weight()
        if weight.requires_grad and (self.training or self.capture_weight_for_analysis):
            weight.retain_grad()
            self._last_dense_weight = weight
        else:
            self._last_dense_weight = None
        return F.linear(input_tensor, weight, self.bias)

    def accumulate_curvature_statistics(self, ema_momentum):
        if self._last_dense_weight is None or self._last_dense_weight.grad is None:
            return False

        grad = self._last_dense_weight.grad.detach().reshape(-1).to(self.curvature_ema.dtype)
        self.curvature_ema.mul_(ema_momentum).add_(grad.square(), alpha=1.0 - ema_momentum)
        self._last_dense_weight = None
        return True

    def get_structure_stats(self):
        bucket_loads = torch.bincount(self.hash_indices.detach().cpu(), minlength=self.compressed_size).float()
        target_load = float(self.original_weight_size) / float(max(self.compressed_size, 1))
        return {
            "compressed_size": int(self.compressed_size),
            "target_load": float(target_load),
            "load_min": int(bucket_loads.min().item()),
            "load_max": int(bucket_loads.max().item()),
            "load_mean": float(bucket_loads.mean().item()),
            "load_std": float(bucket_loads.std(unbiased=False).item()),
            "avg_curvature": float(self.curvature_ema.mean().item()),
            "max_curvature": float(self.curvature_ema.max().item()),
        }

    def _project_shared_weights(self, signed_target_cpu, curvature_cpu, assignments_cpu):
        numerator = torch.zeros(self.compressed_size, dtype=torch.float32)
        denominator = torch.zeros(self.compressed_size, dtype=torch.float32)
        numerator.index_add_(0, assignments_cpu, signed_target_cpu * curvature_cpu)
        denominator.index_add_(0, assignments_cpu, curvature_cpu)

        updated = self.shared_weights.detach().cpu().to(torch.float32)
        nonempty = denominator > 0
        updated[nonempty] = numerator[nonempty] / denominator[nonempty].clamp_min(1e-12)
        return updated

    def _bucket_candidate_ids(self, sorted_bucket_values, sorted_bucket_ids, target_value, candidate_pool_size, current_bucket):
        half_window = max(0, candidate_pool_size // 2)
        insert_pos = int(torch.searchsorted(sorted_bucket_values, torch.tensor([target_value], dtype=sorted_bucket_values.dtype)).item())
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

    @torch.no_grad()
    def update_structure(self, candidate_pool_size, reassign_ratio, capacity_penalty, capacity_slack):
        dense_target_cpu = self.export_dense_weight().detach().reshape(-1).cpu().to(torch.float32)
        sign_cpu = self.hash_signs.detach().cpu().to(torch.float32)
        signed_target_cpu = dense_target_cpu * sign_cpu
        curvature_cpu = self.curvature_ema.detach().cpu().clamp_min(1e-8).to(torch.float32)
        old_assignments_cpu = self.hash_indices.detach().cpu()
        shared_cpu = self.shared_weights.detach().cpu().to(torch.float32)

        num_positions = dense_target_cpu.numel()
        if reassign_ratio >= 1.0:
            selected_positions = torch.argsort(curvature_cpu, descending=True)
        else:
            selected_count = max(1, int(math.ceil(num_positions * reassign_ratio)))
            selected_positions = torch.topk(curvature_cpu, k=selected_count, largest=True).indices
            selected_positions = selected_positions[torch.argsort(curvature_cpu[selected_positions], descending=True)]

        selected_mask = torch.zeros(num_positions, dtype=torch.bool)
        selected_mask[selected_positions] = True
        fixed_mask = ~selected_mask

        bucket_counts = torch.bincount(old_assignments_cpu[fixed_mask], minlength=self.compressed_size).to(torch.float32)
        bucket_weight_sum = torch.zeros(self.compressed_size, dtype=torch.float32)
        bucket_value_sum = torch.zeros(self.compressed_size, dtype=torch.float32)
        bucket_square_sum = torch.zeros(self.compressed_size, dtype=torch.float32)

        if fixed_mask.any():
            fixed_assignments = old_assignments_cpu[fixed_mask]
            fixed_curvature = curvature_cpu[fixed_mask]
            fixed_values = signed_target_cpu[fixed_mask]
            bucket_weight_sum.index_add_(0, fixed_assignments, fixed_curvature)
            bucket_value_sum.index_add_(0, fixed_assignments, fixed_curvature * fixed_values)
            bucket_square_sum.index_add_(0, fixed_assignments, fixed_curvature * fixed_values.square())

        target_load = float(num_positions) / float(max(self.compressed_size, 1))
        hard_capacity = max(1, int(math.ceil(target_load * capacity_slack)))
        sorted_bucket_values, sorted_bucket_ids = torch.sort(shared_cpu)

        new_assignments_cpu = old_assignments_cpu.clone()

        for position in selected_positions.tolist():
            value = float(signed_target_cpu[position].item())
            curvature = float(curvature_cpu[position].item())
            current_bucket = int(old_assignments_cpu[position].item())

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

                new_weight = old_weight + curvature
                new_value = old_value + curvature * value
                new_square = old_square + curvature * value * value
                new_sse = self._bucket_sse(new_weight, new_value, new_square)
                approximation_cost = new_sse - old_sse

                overload = max(0.0, float(current_count + 1) - target_load)
                capacity_cost = capacity_penalty * (overload / max(target_load, 1.0)) ** 2
                total_cost = approximation_cost + capacity_cost

                if best_cost is None or total_cost < best_cost:
                    best_cost = total_cost
                    best_bucket = bucket_id

            new_assignments_cpu[position] = best_bucket
            bucket_counts[best_bucket] += 1.0
            bucket_weight_sum[best_bucket] += curvature
            bucket_value_sum[best_bucket] += curvature * value
            bucket_square_sum[best_bucket] += curvature * value * value

        projected_shared_cpu = self._project_shared_weights(signed_target_cpu, curvature_cpu, new_assignments_cpu)
        projected_dense_cpu = projected_shared_cpu[new_assignments_cpu] * sign_cpu

        changed_mask = new_assignments_cpu != old_assignments_cpu
        changed_count = int(changed_mask.sum().item())
        approximation_mse = float((dense_target_cpu - projected_dense_cpu).square().mean().item())
        weighted_bias_surrogate = float(
            (curvature_cpu * (signed_target_cpu - projected_shared_cpu[new_assignments_cpu]).square()).mean().item()
        )

        self.hash_indices.copy_(new_assignments_cpu.to(self.hash_indices.device))
        self.shared_weights.data.copy_(projected_shared_cpu.to(self.shared_weights.device, dtype=self.shared_weights.dtype))

        stats = self.get_structure_stats()
        stats.update(
            {
                "selected_positions": int(selected_positions.numel()),
                "changed_positions": changed_count,
                "changed_ratio": float(changed_count / float(num_positions)),
                "approximation_mse": approximation_mse,
                "weighted_bias_surrogate": weighted_bias_surrogate,
                "hard_capacity": int(hard_capacity),
            }
        )
        return stats


def get_hashed_layers(model):
    layers = [("linear1", model.linear1)]
    for layer_idx in range(2, model.nhLayers + 1):
        layers.append(("linear" + str(layer_idx), getattr(model, "linear" + str(layer_idx))))
    layers.append(("linear_out", model.linear_out))
    return [(name, module) for name, module in layers if isinstance(module, HessianAwareHashLinear)]


def get_hashed_layer_map(model):
    return dict(get_hashed_layers(model))


class Net(nn.Module):
    def __init__(self, input_dim, output_dim, nh_layers=1, nhu=1000, compress=1.0, dropout=0.25):
        super(Net, self).__init__()
        self.nhLayers = nh_layers
        self.input_dim = input_dim
        compressed_nhu = round(nhu * compress)

        self.dropout0 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(input_dim, compressed_nhu)
        self.dropout1 = nn.Dropout(dropout)

        for layer_idx in range(2, nh_layers + 1):
            setattr(self, "linear" + str(layer_idx), nn.Linear(compressed_nhu, compressed_nhu))
            setattr(self, "dropout" + str(layer_idx), nn.Dropout(dropout))

        self.linear_out = nn.Linear(compressed_nhu, output_dim)

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        x = self.dropout0(x)
        x = F.relu(self.linear1(x))
        x = self.dropout1(x)

        for layer_idx in range(2, self.nhLayers + 1):
            x = F.relu(getattr(self, "linear" + str(layer_idx))(x))
            x = getattr(self, "dropout" + str(layer_idx))(x)

        x = self.linear_out(x)
        return F.log_softmax(x, dim=1)


class HessianAwareHashNet(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        nh_layers=1,
        nhu=1000,
        compress=1.0,
        dropout=0.25,
        hash_seed=2,
        hash_bias=False,
    ):
        super(HessianAwareHashNet, self).__init__()
        self.nhLayers = nh_layers
        self.input_dim = input_dim

        self.dropout0 = nn.Dropout(dropout)
        self.linear1 = HessianAwareHashLinear(
            input_dim,
            nhu,
            compress=compress,
            hash_seed=hash_seed,
            hash_bias=hash_bias,
        )
        self.dropout1 = nn.Dropout(dropout)

        for layer_idx in range(2, nh_layers + 1):
            setattr(
                self,
                "linear" + str(layer_idx),
                HessianAwareHashLinear(
                    nhu,
                    nhu,
                    compress=compress,
                    hash_seed=hash_seed + layer_idx - 1,
                    hash_bias=hash_bias,
                ),
            )
            setattr(self, "dropout" + str(layer_idx), nn.Dropout(dropout))

        self.linear_out = HessianAwareHashLinear(
            nhu,
            output_dim,
            compress=compress,
            hash_seed=hash_seed + nh_layers,
            hash_bias=hash_bias,
        )

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        x = self.dropout0(x)
        x = F.relu(self.linear1(x))
        x = self.dropout1(x)

        for layer_idx in range(2, self.nhLayers + 1):
            x = F.relu(getattr(self, "linear" + str(layer_idx))(x))
            x = getattr(self, "dropout" + str(layer_idx))(x)

        x = self.linear_out(x)
        return F.log_softmax(x, dim=1)


def train_epoch(model, device, train_loader, optimizer, epoch, log_interval=5):
    model.train()
    train_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.sampler),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                ),
                end="\r",
            )
        train_loss += loss.item() * data.size(0)

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


def collect_curvature_statistics(model, device, train_loader, args):
    hashed_layers = get_hashed_layers(model)
    if not hashed_layers:
        return {"num_batches": 0}

    model.train()
    batches_used = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx >= args.stats_batches:
            break

        data, target = data.to(device), target.to(device)
        model.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()

        for _, layer in hashed_layers:
            layer.accumulate_curvature_statistics(args.stats_ema_momentum)

        model.zero_grad()
        batches_used += 1

    return {"num_batches": batches_used}


def compute_hessian_snapshot(model, device, loader, args, epoch):
    if not args.record_hessian_snapshots:
        return None

    layer_map = get_hashed_layer_map(model)
    if args.hessian_snapshot_layer not in layer_map:
        raise ValueError("Unknown Hessian snapshot layer: {}".format(args.hessian_snapshot_layer))

    target_layer = layer_map[args.hessian_snapshot_layer]
    was_training = model.training
    model.eval()
    target_layer.capture_weight_for_analysis = True

    try:
        data, target = next(iter(loader))
        data, target = data.to(device), target.to(device)
        model.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)

        dense_weight = target_layer._last_dense_weight
        if dense_weight is None:
            raise RuntimeError("Failed to capture dense weight for Hessian analysis")

        dense_grad = torch.autograd.grad(loss, dense_weight, create_graph=True, retain_graph=True)[0].reshape(-1)
        curvature = target_layer.curvature_ema.detach().to(device=dense_grad.device, dtype=dense_grad.dtype)
        sample_size = min(args.hessian_snapshot_size, dense_grad.numel())

        if args.hessian_snapshot_selection == "top_curvature":
            selected_indices = torch.topk(curvature, k=sample_size, largest=True).indices
        else:
            generator = torch.Generator(device=dense_grad.device)
            generator.manual_seed(args.seed + epoch * 1009)
            selected_indices = torch.randperm(dense_grad.numel(), generator=generator, device=dense_grad.device)[:sample_size]

        hessian_rows = []
        for index in selected_indices.tolist():
            second_grad = torch.autograd.grad(dense_grad[index], dense_weight, retain_graph=True)[0].reshape(-1)
            hessian_rows.append(second_grad[selected_indices].detach().cpu())

        hessian_matrix = torch.stack(hessian_rows, dim=0)
        diagonal = torch.diagonal(hessian_matrix)
        off_diagonal = hessian_matrix - torch.diag(diagonal)
        abs_diag = diagonal.abs()
        abs_off_diag = off_diagonal.abs()
        diag_norm = float(torch.norm(diagonal).item())
        off_diag_norm = float(torch.norm(off_diagonal).item())
        symmetry_gap = float((hessian_matrix - hessian_matrix.t()).abs().max().item())

        sample_indices_cpu = selected_indices.detach().cpu()
        snapshot = {
            "epoch": int(epoch),
            "layer": args.hessian_snapshot_layer,
            "source": args.hessian_snapshot_source,
            "batch_size": int(data.size(0)),
            "sample_size": int(sample_size),
            "selection": args.hessian_snapshot_selection,
            "loss": float(loss.item()),
            "sample_indices": [int(index) for index in sample_indices_cpu.tolist()],
            "sample_coordinates": [
                {
                    "row": int(index // target_layer.in_features),
                    "col": int(index % target_layer.in_features),
                }
                for index in sample_indices_cpu.tolist()
            ],
            "curvature_values": [float(curvature[index].item()) for index in selected_indices.tolist()],
            "hessian_matrix": hessian_matrix.tolist(),
            "metrics": {
                "mean_abs_diag": float(abs_diag.mean().item()),
                "mean_abs_off_diag": float(abs_off_diag.mean().item()),
                "max_abs_diag": float(abs_diag.max().item()),
                "max_abs_off_diag": float(abs_off_diag.max().item()),
                "diag_to_offdiag_norm_ratio": float(diag_norm / max(off_diag_norm, 1e-12)),
                "diag_mass_ratio": float(abs_diag.sum().item() / max(abs_diag.sum().item() + abs_off_diag.sum().item(), 1e-12)),
                "symmetry_max_abs_error": symmetry_gap,
            },
        }
    finally:
        target_layer.capture_weight_for_analysis = False
        model.zero_grad()
        if was_training:
            model.train()
        else:
            model.eval()

    return snapshot


def run_structure_update(model, args):
    layer_updates = {}
    for layer_name, layer in get_hashed_layers(model):
        if layer_name == "linear_out" and not args.update_output_layer:
            layer_updates[layer_name] = {
                "skipped": True,
                "reason": "output layer frozen during structure update",
            }
            continue

        update_info = layer.update_structure(
            candidate_pool_size=args.candidate_pool_size,
            reassign_ratio=args.reassign_ratio,
            capacity_penalty=args.capacity_penalty,
            capacity_slack=args.capacity_slack,
        )
        layer_updates[layer_name] = update_info
        print(
            "Updated structure for {}: changed {:.2f}% of positions".format(
                layer_name,
                100.0 * update_info["changed_ratio"],
            )
        )
    return layer_updates


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="PyTorch Hessian-aware HashNet on MNIST",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--nhLayers", type=int, default=1, help="# hidden layers, excluding input/output layers")
    parser.add_argument("--nhu", type=int, default=1000, help="Number of hidden units")
    parser.add_argument("--hashed", default=False, action="store_true", help="Enable Hessian-aware hashing")
    parser.add_argument("--compress", type=float, default=0.03125, help="Compression rate")
    parser.add_argument("--hash-bias", default=False, action="store_true", help="Learn dense bias terms")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate at t=0")
    parser.add_argument("--decay-factor", type=float, default=0.1, help="Learning rate decay factor")
    parser.add_argument("--batch-size", type=int, default=50, help="Mini-batch size")
    parser.add_argument("--validation-percent", type=float, default=0.1, help="Percent of training data used for validation")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum (SGD only)")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout rate")
    parser.add_argument("--l2reg", type=float, default=0.0, help="l2 regularisation")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum number of epochs")
    parser.add_argument("--patience", type=int, default=2, help="Number of epochs to wait before scaling lr")
    parser.add_argument("--hash-seed", type=int, default=2, help="Seed for hash functions")
    parser.add_argument(
        "--structure-update-interval",
        type=int,
        default=5,
        help="If > 0, run one structure update every N epochs",
    )
    parser.add_argument(
        "--reassign-ratio",
        type=float,
        default=0.1,
        help="Top curvature fraction greedily reassigned in each structure update",
    )
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=8,
        help="Number of value-near candidate buckets considered per parameter",
    )
    parser.add_argument(
        "--capacity-penalty",
        type=float,
        default=0.1,
        help="Penalty coefficient for overloaded buckets",
    )
    parser.add_argument(
        "--capacity-slack",
        type=float,
        default=2.0,
        help="Hard capacity multiplier relative to average target load",
    )
    parser.add_argument(
        "--stats-batches",
        type=int,
        default=10,
        help="Mini-batches used to estimate Hessian/Fisher diagonal surrogates",
    )
    parser.add_argument(
        "--stats-ema-momentum",
        type=float,
        default=0.9,
        help="EMA momentum for curvature statistics",
    )
    parser.add_argument(
        "--structure-lr-drop",
        type=float,
        default=0.5,
        help="Multiply current learning rate by this factor after an accepted structure update",
    )
    parser.add_argument(
        "--update-output-layer",
        action="store_true",
        default=False,
        help="Also update the output layer structure",
    )
    parser.add_argument(
        "--record-hessian-snapshots",
        action="store_true",
        default=False,
        help="Record sampled Hessian submatrices during training for diagonality inspection",
    )
    parser.add_argument(
        "--hessian-snapshot-interval",
        type=int,
        default=5,
        help="If > 0, record one Hessian snapshot every N epochs",
    )
    parser.add_argument(
        "--hessian-snapshot-layer",
        type=str,
        default="linear1",
        help="Layer name whose dense reconstructed weight Hessian is sampled",
    )
    parser.add_argument(
        "--hessian-snapshot-size",
        type=int,
        default=16,
        help="Number of parameter positions used to build each Hessian submatrix",
    )
    parser.add_argument(
        "--hessian-snapshot-source",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Dataset split used to build the Hessian snapshot batch",
    )
    parser.add_argument(
        "--hessian-snapshot-selection",
        type=str,
        default="top_curvature",
        choices=["top_curvature", "random"],
        help="How sampled parameter indices are chosen for Hessian snapshots",
    )
    parser.add_argument("--results-path", type=str, default=None, help="Path to save training metrics as JSON")
    parser.add_argument(
        "--save-model-path",
        type=str,
        default="mnist_hessian_aware.pt",
        help="Path to save the final model checkpoint",
    )
    parser.add_argument("--save-model", action="store_true", default=False, help="Save the final model checkpoint")
    args = parser.parse_args()

    if not 0.0 < args.validation_percent < 1.0:
        parser.error("--validation-percent must be in (0, 1)")
    if args.compress <= 0.0:
        parser.error("--compress must be > 0")
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if args.structure_update_interval < 0:
        parser.error("--structure-update-interval must be >= 0")
    if not 0.0 < args.reassign_ratio <= 1.0:
        parser.error("--reassign-ratio must be in (0, 1]")
    if args.candidate_pool_size < 1:
        parser.error("--candidate-pool-size must be >= 1")
    if args.capacity_penalty < 0.0:
        parser.error("--capacity-penalty must be >= 0")
    if args.capacity_slack < 1.0:
        parser.error("--capacity-slack must be >= 1")
    if args.stats_batches < 1:
        parser.error("--stats-batches must be >= 1")
    if not 0.0 <= args.stats_ema_momentum < 1.0:
        parser.error("--stats-ema-momentum must be in [0, 1)")
    if not 0.0 < args.structure_lr_drop <= 1.0:
        parser.error("--structure-lr-drop must be in (0, 1]")
    if args.hessian_snapshot_interval < 0:
        parser.error("--hessian-snapshot-interval must be >= 0")
    if args.hessian_snapshot_size < 1:
        parser.error("--hessian-snapshot-size must be >= 1")

    print(args)
    return args


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

    if args.hashed:
        model = HessianAwareHashNet(
            input_dim=input_dim,
            output_dim=output_dim,
            nh_layers=args.nhLayers,
            nhu=args.nhu,
            compress=args.compress,
            dropout=args.dropout,
            hash_seed=args.hash_seed,
            hash_bias=args.hash_bias,
        ).to(device)
    else:
        eq_compress = get_equivalent_compression(input_dim, output_dim, args.nhu, args.nhLayers, args.compress)
        model = Net(
            input_dim=input_dim,
            output_dim=output_dim,
            nh_layers=args.nhLayers,
            nhu=args.nhu,
            compress=eq_compress,
            dropout=args.dropout,
        ).to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.l2reg,
    )
    scheduler = build_plateau_scheduler(optimizer, args)

    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("The number of parameters is: {}".format(parameter_count))

    history = []
    structure_history = []
    hessian_history = []
    structure_event_index = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, device, tr_loader, optimizer, epoch)
        val_loss, val_acc = evaluate(model, device, val_loader)
        scheduler.step(val_loss)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "structure_update_performed": False,
        }

        should_record_hessian = (
            args.hashed
            and args.record_hessian_snapshots
            and args.hessian_snapshot_interval > 0
            and epoch % args.hessian_snapshot_interval == 0
        )
        should_update = (
            args.hashed
            and args.structure_update_interval > 0
            and epoch % args.structure_update_interval == 0
            and epoch < args.epochs
        )
        stats_info = None
        hessian_snapshot = None
        if should_update or should_record_hessian:
            print("\nCollecting curvature surrogates before analysis/update...")
            stats_info = collect_curvature_statistics(model, device, tr_loader, args)

        if should_record_hessian:
            snapshot_loader = val_loader if args.hessian_snapshot_source == "val" else tr_loader
            hessian_snapshot = compute_hessian_snapshot(model, device, snapshot_loader, args, epoch)
            hessian_history.append(hessian_snapshot)
            metrics = hessian_snapshot["metrics"]
            print(
                "Recorded Hessian snapshot for {} at epoch {}: diag/offdiag norm ratio {:.4f}, diag mass {:.4f}".format(
                    hessian_snapshot["layer"],
                    epoch,
                    metrics["diag_to_offdiag_norm_ratio"],
                    metrics["diag_mass_ratio"],
                )
            )
            epoch_record["hessian_snapshot_recorded"] = True
            epoch_record["hessian_snapshot_metrics"] = metrics
        else:
            epoch_record["hessian_snapshot_recorded"] = False

        if should_update:
            structure_event_index += 1

            print("Running Hessian-aware structure update after epoch {}...".format(epoch))
            pre_update_state = snapshot_module_state(model)
            update_details = run_structure_update(model, args)
            trial_val_loss, trial_val_acc = evaluate(model, device, val_loader)
            accepted = should_accept_structure_update(val_loss, val_acc, trial_val_loss, trial_val_acc)

            if accepted:
                optimizer.state.clear()
                for group in optimizer.param_groups:
                    group["lr"] *= args.structure_lr_drop
                scheduler = build_plateau_scheduler(optimizer, args)
                print(
                    "Accepted structure update: val loss {:.4f} -> {:.4f}, val acc {:.2f}% -> {:.2f}%".format(
                        val_loss,
                        trial_val_loss,
                        val_acc,
                        trial_val_acc,
                    )
                )
            else:
                model.load_state_dict(pre_update_state)
                print(
                    "Rolled back structure update: val loss {:.4f} -> {:.4f}, val acc {:.2f}% -> {:.2f}%".format(
                        val_loss,
                        trial_val_loss,
                        val_acc,
                        trial_val_acc,
                    )
                )

            structure_record = {
                "after_epoch": epoch,
                "structure_update_index": structure_event_index,
                "stats_collection": stats_info if stats_info is not None else {"num_batches": 0},
                "update": update_details,
                "accepted": accepted,
                "rolled_back": not accepted,
                "baseline_val_loss": val_loss,
                "baseline_val_accuracy": val_acc,
                "trial_val_loss": trial_val_loss,
                "trial_val_accuracy": trial_val_acc,
                "learning_rate_after_update": optimizer.param_groups[0]["lr"],
                "hessian_snapshot": hessian_snapshot,
            }
            structure_history.append(structure_record)

            epoch_record["structure_update_performed"] = True
            epoch_record["structure_update"] = structure_record
            epoch_record["learning_rate_after_update"] = optimizer.param_groups[0]["lr"]

        history.append(epoch_record)
        print(
            "\nEpoch {} Train loss: {:.3f} Val loss: {:.3f} Val acc: {:.2f}%".format(
                epoch,
                train_loss,
                val_loss,
                val_acc,
            )
        )

    test_loss, test_acc = evaluate(model, device, test_loader)
    print("Test loss: {:.3f} Test acc: {:.2f}%".format(test_loss, test_acc))
    save_results(args, parameter_count, history, structure_history, hessian_history, test_loss, test_acc)

    if args.save_model:
        save_model_checkpoint(model, args.save_model_path)


if __name__ == "__main__":
    main()

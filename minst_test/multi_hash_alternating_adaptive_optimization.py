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

    model_name = "alternating_adaptive_hashed" if args.hashed else "dense"
    return "mnist_{}_compress{}_seed{}.json".format(model_name, args.compress, args.seed)


def save_results(args, parameter_count, history, structure_history, test_loss, test_acc):
    results_path = build_results_path(args)
    results_dir = os.path.dirname(results_path)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)

    payload = {
        "args": vars(args),
        "parameter_count": parameter_count,
        "history": history,
        "structure_history": structure_history,
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


class AlternatingAdaptiveHashLinear(nn.Module):
    def __init__(self, in_features, out_features, compress=0.03125, hash_seed=2, hash_bias=False):
        super(AlternatingAdaptiveHashLinear, self).__init__()
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
        self.register_buffer("hessian_ema", torch.ones(self.original_weight_size, dtype=torch.float32))
        self.register_buffer("noise_ema", torch.zeros(self.original_weight_size, dtype=torch.float32))
        self.register_buffer("grad_mean_ema", torch.zeros(self.original_weight_size, dtype=torch.float32))

        self._last_dense_weight = None
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
        if self.training and weight.requires_grad:
            weight.retain_grad()
            self._last_dense_weight = weight
        else:
            self._last_dense_weight = None
        return F.linear(input_tensor, weight, self.bias)

    def accumulate_curvature_statistics(self, ema_momentum):
        if self._last_dense_weight is None or self._last_dense_weight.grad is None:
            return False

        grad = self._last_dense_weight.grad.detach().reshape(-1).to(self.hessian_ema.dtype)
        grad_delta = grad - self.grad_mean_ema

        self.hessian_ema.mul_(ema_momentum).add_(grad.square(), alpha=1.0 - ema_momentum)
        self.grad_mean_ema.mul_(ema_momentum).add_(grad, alpha=1.0 - ema_momentum)
        self.noise_ema.mul_(ema_momentum).add_(grad_delta.square(), alpha=1.0 - ema_momentum)

        self._last_dense_weight = None
        return True

    def _candidate_buckets(self, sorted_values, sorted_ids, targets, candidate_pool_size):
        half_window = max(0, candidate_pool_size // 2)
        offsets = torch.arange(-half_window, half_window + 1, device=targets.device)
        insertion = torch.searchsorted(sorted_values, targets)
        candidate_positions = insertion.unsqueeze(1) + offsets.unsqueeze(0)
        candidate_positions = candidate_positions.clamp_(0, sorted_values.numel() - 1)
        return sorted_ids[candidate_positions]

    def _project_shared_weights(self, dense_target, curvature, hash_indices, hash_signs):
        signed_target = dense_target * hash_signs.to(dense_target.dtype)
        weighted_target = curvature * signed_target

        numerator = torch.zeros(
            self.compressed_size,
            device=dense_target.device,
            dtype=dense_target.dtype,
        )
        denominator = torch.zeros_like(numerator)
        numerator.index_add_(0, hash_indices, weighted_target)
        denominator.index_add_(0, hash_indices, curvature)

        updated = self.shared_weights.detach().clone()
        nonempty = denominator > 0
        updated[nonempty] = numerator[nonempty] / denominator[nonempty].clamp_min(1e-12)
        return updated

    def get_structure_stats(self):
        bucket_loads = torch.bincount(self.hash_indices.detach().cpu(), minlength=self.compressed_size).float()
        target_load = float(self.original_weight_size) / float(max(self.compressed_size, 1))
        sign_positive_ratio = float((self.hash_signs > 0).float().mean().item())
        return {
            "compressed_size": int(self.compressed_size),
            "target_load": float(target_load),
            "load_min": int(bucket_loads.min().item()),
            "load_max": int(bucket_loads.max().item()),
            "load_mean": float(bucket_loads.mean().item()),
            "load_std": float(bucket_loads.std(unbiased=False).item()),
            "positive_sign_ratio": sign_positive_ratio,
            "avg_hessian": float(self.hessian_ema.mean().item()),
            "avg_noise": float(self.noise_ema.mean().item()),
        }

    @torch.no_grad()
    def update_structure(self, mu, lam, candidate_pool_size, update_ratio):
        dense_target = self.export_dense_weight().detach().reshape(-1)
        curvature = self.hessian_ema.detach().clamp_min(1e-8).to(dense_target.dtype)
        noise = self.noise_ema.detach().clamp_min(0.0).to(dense_target.dtype)

        num_positions = dense_target.numel()
        if update_ratio >= 1.0:
            selected_positions = torch.arange(num_positions, device=dense_target.device)
        else:
            max_updates = max(1, int(math.ceil(num_positions * update_ratio)))
            importance = curvature * dense_target.square() + mu * noise
            selected_positions = torch.topk(importance, k=max_updates, largest=True).indices

        old_indices = self.hash_indices.detach().clone()
        old_signs = self.hash_signs.detach().clone()

        shared = self.shared_weights.detach()
        sorted_values, sorted_order = torch.sort(shared)
        selected_targets = dense_target[selected_positions]
        selected_curvature = curvature[selected_positions].unsqueeze(1)
        selected_noise = noise[selected_positions].unsqueeze(1)

        pos_buckets = self._candidate_buckets(sorted_values, sorted_order, selected_targets, candidate_pool_size)
        neg_buckets = self._candidate_buckets(sorted_values, sorted_order, -selected_targets, candidate_pool_size)

        base_loads = torch.bincount(old_indices, minlength=self.compressed_size).to(shared.dtype)
        removed_loads = torch.bincount(old_indices[selected_positions], minlength=self.compressed_size).to(shared.dtype)
        remaining_loads = (base_loads - removed_loads).clamp_min(0.0)
        target_load = float(num_positions) / float(max(self.compressed_size, 1))
        target_load = max(target_load, 1.0)

        pos_values = shared[pos_buckets]
        neg_values = -shared[neg_buckets]

        pos_load = (remaining_loads[pos_buckets] + 1.0) / target_load
        neg_load = (remaining_loads[neg_buckets] + 1.0) / target_load
        pos_balance = ((remaining_loads[pos_buckets] + 1.0 - target_load) / target_load).square()
        neg_balance = ((remaining_loads[neg_buckets] + 1.0 - target_load) / target_load).square()

        target_matrix = selected_targets.unsqueeze(1)
        pos_scores = selected_curvature * (target_matrix - pos_values).square()
        pos_scores = pos_scores + mu * selected_noise * pos_load + lam * pos_balance

        neg_scores = selected_curvature * (target_matrix - neg_values).square()
        neg_scores = neg_scores + mu * selected_noise * neg_load + lam * neg_balance

        all_scores = torch.cat([pos_scores, neg_scores], dim=1)
        all_bucket_ids = torch.cat([pos_buckets, neg_buckets], dim=1)
        all_signs = torch.cat(
            [
                torch.ones_like(pos_buckets, dtype=self.hash_signs.dtype),
                -torch.ones_like(neg_buckets, dtype=self.hash_signs.dtype),
            ],
            dim=1,
        )

        best_choice = all_scores.argmin(dim=1)
        new_bucket_ids = all_bucket_ids.gather(1, best_choice.unsqueeze(1)).squeeze(1)
        new_signs = all_signs.gather(1, best_choice.unsqueeze(1)).squeeze(1)

        updated_indices = old_indices.clone()
        updated_signs = old_signs.clone()
        updated_indices[selected_positions] = new_bucket_ids
        updated_signs[selected_positions] = new_signs

        projected_shared = self._project_shared_weights(
            dense_target,
            curvature,
            updated_indices,
            updated_signs,
        )
        projected_dense = projected_shared[updated_indices] * updated_signs.to(projected_shared.dtype)

        changed_mask = (updated_indices != old_indices) | (updated_signs != old_signs)
        changed_count = int(changed_mask.sum().item())
        sign_flip_count = int((updated_signs != old_signs).sum().item())
        approximation_error = float(((dense_target - projected_dense).square().mean()).item())
        weighted_bias_surrogate = float((curvature * (dense_target - projected_dense).square()).mean().item())

        self.hash_indices.copy_(updated_indices)
        self.hash_signs.copy_(updated_signs)
        self.shared_weights.data.copy_(projected_shared.to(self.shared_weights.dtype))

        stats = self.get_structure_stats()
        stats.update(
            {
                "selected_positions": int(selected_positions.numel()),
                "changed_positions": changed_count,
                "changed_ratio": float(changed_count / float(num_positions)),
                "sign_flip_ratio": float(sign_flip_count / float(num_positions)),
                "approximation_mse": approximation_error,
                "weighted_bias_surrogate": weighted_bias_surrogate,
            }
        )
        return stats


def get_hashed_layers(model):
    layers = [("linear1", model.linear1)]
    for layer_idx in range(2, model.nhLayers + 1):
        layers.append(("linear" + str(layer_idx), getattr(model, "linear" + str(layer_idx))))
    layers.append(("linear_out", model.linear_out))
    return [(name, module) for name, module in layers if isinstance(module, AlternatingAdaptiveHashLinear)]


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


class AlternatingAdaptiveHashNet(nn.Module):
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
        super(AlternatingAdaptiveHashNet, self).__init__()
        self.nhLayers = nh_layers
        self.input_dim = input_dim

        self.dropout0 = nn.Dropout(dropout)
        self.linear1 = AlternatingAdaptiveHashLinear(
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
                AlternatingAdaptiveHashLinear(
                    nhu,
                    nhu,
                    compress=compress,
                    hash_seed=hash_seed + layer_idx - 1,
                    hash_bias=hash_bias,
                ),
            )
            setattr(self, "dropout" + str(layer_idx), nn.Dropout(dropout))

        self.linear_out = AlternatingAdaptiveHashLinear(
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


def collect_structure_statistics(model, device, train_loader, args):
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
            mu=args.surrogate_mu,
            lam=args.surrogate_lambda,
            candidate_pool_size=args.candidate_pool_size,
            update_ratio=args.structure_update_ratio,
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
        description="PyTorch alternating adaptive optimization HashNet on MNIST",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--nhLayers", type=int, default=1, help="# hidden layers, excluding input/output layers")
    parser.add_argument("--nhu", type=int, default=1000, help="Number of hidden units")
    parser.add_argument("--hashed", default=False, action="store_true", help="Enable alternating adaptive hashing")
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
        "--structure-update-ratio",
        type=float,
        default=0.1,
        help="Fraction of weight positions considered in each structure update",
    )
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=8,
        help="Neighborhood size used to search candidate buckets per sign",
    )
    parser.add_argument(
        "--stats-batches",
        type=int,
        default=10,
        help="Mini-batches used to estimate curvature and gradient noise before each structure update",
    )
    parser.add_argument(
        "--stats-ema-momentum",
        type=float,
        default=0.9,
        help="EMA momentum for Hessian and noise surrogates",
    )
    parser.add_argument("--surrogate-mu", type=float, default=0.1, help="Weight for variance surrogate")
    parser.add_argument("--surrogate-lambda", type=float, default=0.1, help="Weight for load-balance regulariser")
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
    parser.add_argument("--results-path", type=str, default=None, help="Path to save training metrics as JSON")
    parser.add_argument(
        "--save-model-path",
        type=str,
        default="mnist_alternating_adaptive_optimization.pt",
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
    if not 0.0 < args.structure_update_ratio <= 1.0:
        parser.error("--structure-update-ratio must be in (0, 1]")
    if args.candidate_pool_size < 1:
        parser.error("--candidate-pool-size must be >= 1")
    if args.stats_batches < 1:
        parser.error("--stats-batches must be >= 1")
    if not 0.0 <= args.stats_ema_momentum < 1.0:
        parser.error("--stats-ema-momentum must be in [0, 1)")
    if not 0.0 <= args.surrogate_mu:
        parser.error("--surrogate-mu must be >= 0")
    if not 0.0 <= args.surrogate_lambda:
        parser.error("--surrogate-lambda must be >= 0")
    if not 0.0 < args.structure_lr_drop <= 1.0:
        parser.error("--structure-lr-drop must be in (0, 1]")

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
        model = AlternatingAdaptiveHashNet(
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

        should_update = (
            args.hashed
            and args.structure_update_interval > 0
            and epoch % args.structure_update_interval == 0
            and epoch < args.epochs
        )
        if should_update:
            structure_event_index += 1
            print("\nCollecting curvature and noise surrogates before structure update...")
            stats_info = collect_structure_statistics(model, device, tr_loader, args)

            print("Running alternating structure update after epoch {}...".format(epoch))
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
                "stats_collection": stats_info,
                "update": update_details,
                "accepted": accepted,
                "rolled_back": not accepted,
                "baseline_val_loss": val_loss,
                "baseline_val_accuracy": val_acc,
                "trial_val_loss": trial_val_loss,
                "trial_val_accuracy": trial_val_acc,
                "learning_rate_after_update": optimizer.param_groups[0]["lr"],
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
    save_results(args, parameter_count, history, structure_history, test_loss, test_acc)

    if args.save_model:
        save_model_checkpoint(model, args.save_model_path)


if __name__ == "__main__":
    main()

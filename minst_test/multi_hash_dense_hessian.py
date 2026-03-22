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
    return "mnist_dense_hessian_seed{}.json".format(args.seed)


def save_results(args, parameter_count, history, hessian_history, test_loss, test_acc):
    results_path = build_results_path(args)
    results_dir = os.path.dirname(results_path)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)

    payload = {
        "args": vars(args),
        "parameter_count": parameter_count,
        "history": history,
        "hessian_history": hessian_history,
        "final_test": {
            "loss": test_loss,
            "accuracy": test_acc,
        },
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Saved results to {}".format(results_path))
    return results_path


def build_plateau_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.decay_factor,
        patience=args.patience,
        verbose=True,
    )


def save_model_checkpoint(model, path):
    checkpoint_dir = os.path.dirname(path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(model.state_dict(), path)
    print("Saved model checkpoint to {}".format(path))


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


class DenseNet(nn.Module):
    def __init__(self, input_dim, output_dim, nh_layers=1, nhu=1000, dropout=0.25):
        super(DenseNet, self).__init__()
        self.nhLayers = nh_layers
        self.input_dim = input_dim

        self.dropout0 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(input_dim, nhu)
        self.dropout1 = nn.Dropout(dropout)

        for layer_idx in range(2, nh_layers + 1):
            setattr(self, "linear" + str(layer_idx), nn.Linear(nhu, nhu))
            setattr(self, "dropout" + str(layer_idx), nn.Dropout(dropout))

        self.linear_out = nn.Linear(nhu, output_dim)

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


def get_linear_layers(model):
    layers = [("linear1", model.linear1)]
    for layer_idx in range(2, model.nhLayers + 1):
        layers.append(("linear" + str(layer_idx), getattr(model, "linear" + str(layer_idx))))
    layers.append(("linear_out", model.linear_out))
    return layers


def initialize_curvature_trackers(model):
    trackers = {}
    for layer_name, layer in get_linear_layers(model):
        trackers[layer_name] = torch.ones_like(layer.weight.detach(), dtype=torch.float32, device="cpu")
    return trackers


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


def collect_curvature_statistics(model, device, train_loader, args, curvature_trackers):
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

        for layer_name, layer in get_linear_layers(model):
            grad = layer.weight.grad.detach().cpu().to(torch.float32)
            curvature_trackers[layer_name].mul_(args.stats_ema_momentum).add_(grad.square(), alpha=1.0 - args.stats_ema_momentum)

        model.zero_grad()
        batches_used += 1

    return {"num_batches": batches_used}


def compute_hessian_snapshot(model, device, loader, args, epoch, curvature_trackers):
    layer_map = dict(get_linear_layers(model))
    if args.hessian_snapshot_layer not in layer_map:
        raise ValueError("Unknown Hessian snapshot layer: {}".format(args.hessian_snapshot_layer))

    target_layer = layer_map[args.hessian_snapshot_layer]
    was_training = model.training
    model.eval()

    try:
        data, target = next(iter(loader))
        data, target = data.to(device), target.to(device)
        model.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)

        weight_param = target_layer.weight
        dense_grad = torch.autograd.grad(loss, weight_param, create_graph=True, retain_graph=True)[0].reshape(-1)
        curvature = curvature_trackers[args.hessian_snapshot_layer].to(device=dense_grad.device, dtype=dense_grad.dtype).reshape(-1)
        sample_size = min(args.hessian_snapshot_size, dense_grad.numel())

        if args.hessian_snapshot_selection == "top_curvature":
            selected_indices = torch.topk(curvature, k=sample_size, largest=True).indices
        else:
            generator = torch.Generator(device=dense_grad.device)
            generator.manual_seed(args.seed + epoch * 1009)
            selected_indices = torch.randperm(dense_grad.numel(), generator=generator, device=dense_grad.device)[:sample_size]

        hessian_rows = []
        for index in selected_indices.tolist():
            second_grad = torch.autograd.grad(dense_grad[index], weight_param, retain_graph=True)[0].reshape(-1)
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
        model.zero_grad()
        if was_training:
            model.train()
        else:
            model.eval()

    return snapshot


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="PyTorch dense MNIST training with Hessian snapshots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--nhLayers", type=int, default=1, help="# hidden layers, excluding input/output layers")
    parser.add_argument("--nhu", type=int, default=1000, help="Number of hidden units")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate at t=0")
    parser.add_argument("--decay-factor", type=float, default=0.1, help="Learning rate decay factor")
    parser.add_argument("--batch-size", type=int, default=50, help="Mini-batch size")
    parser.add_argument("--validation-percent", type=float, default=0.1, help="Percent of training data used for validation")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum (SGD only)")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout rate")
    parser.add_argument("--l2reg", type=float, default=0.0, help="l2 regularisation")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum number of epochs")
    parser.add_argument("--patience", type=int, default=2, help="Number of epochs to wait before scaling lr")
    parser.add_argument("--stats-batches", type=int, default=10, help="Mini-batches used to estimate diagonal curvature surrogates")
    parser.add_argument("--stats-ema-momentum", type=float, default=0.9, help="EMA momentum for curvature statistics")
    parser.add_argument("--hessian-snapshot-interval", type=int, default=5, help="If > 0, record one Hessian snapshot every N epochs")
    parser.add_argument("--hessian-snapshot-layer", type=str, default="linear1", help="Layer name whose weight Hessian is sampled")
    parser.add_argument("--hessian-snapshot-size", type=int, default=16, help="Number of parameter positions used to build each Hessian submatrix")
    parser.add_argument("--hessian-snapshot-source", type=str, default="val", choices=["train", "val"], help="Dataset split used to build the Hessian snapshot batch")
    parser.add_argument("--hessian-snapshot-selection", type=str, default="top_curvature", choices=["top_curvature", "random"], help="How sampled parameter indices are chosen for Hessian snapshots")
    parser.add_argument("--results-path", type=str, default=None, help="Path to save training metrics as JSON")
    parser.add_argument("--save-model-path", type=str, default="mnist_dense_hessian.pt", help="Path to save the final model checkpoint")
    parser.add_argument("--save-model", action="store_true", default=False, help="Save the final model checkpoint")
    args = parser.parse_args()

    if not 0.0 < args.validation_percent < 1.0:
        parser.error("--validation-percent must be in (0, 1)")
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if args.stats_batches < 1:
        parser.error("--stats-batches must be >= 1")
    if not 0.0 <= args.stats_ema_momentum < 1.0:
        parser.error("--stats-ema-momentum must be in [0, 1)")
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
    model = DenseNet(input_dim=784, output_dim=10, nh_layers=args.nhLayers, nhu=args.nhu, dropout=args.dropout).to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.l2reg,
    )
    scheduler = build_plateau_scheduler(optimizer, args)

    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("The number of parameters is: {}".format(parameter_count))

    curvature_trackers = initialize_curvature_trackers(model)
    history = []
    hessian_history = []

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
            "hessian_snapshot_recorded": False,
        }

        should_record_hessian = args.hessian_snapshot_interval > 0 and epoch % args.hessian_snapshot_interval == 0
        if should_record_hessian:
            print("\nCollecting curvature surrogates before Hessian snapshot...")
            stats_info = collect_curvature_statistics(model, device, tr_loader, args, curvature_trackers)
            snapshot_loader = val_loader if args.hessian_snapshot_source == "val" else tr_loader
            hessian_snapshot = compute_hessian_snapshot(model, device, snapshot_loader, args, epoch, curvature_trackers)
            hessian_snapshot["stats_collection"] = stats_info
            hessian_history.append(hessian_snapshot)
            metrics = hessian_snapshot["metrics"]
            epoch_record["hessian_snapshot_recorded"] = True
            epoch_record["hessian_snapshot_metrics"] = metrics
            print(
                "Recorded Hessian snapshot for {} at epoch {}: diag/offdiag norm ratio {:.4f}, diag mass {:.4f}".format(
                    hessian_snapshot["layer"],
                    epoch,
                    metrics["diag_to_offdiag_norm_ratio"],
                    metrics["diag_mass_ratio"],
                )
            )

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
    save_results(args, parameter_count, history, hessian_history, test_loss, test_acc)

    if args.save_model:
        save_model_checkpoint(model, args.save_model_path)


if __name__ == "__main__":
    main()

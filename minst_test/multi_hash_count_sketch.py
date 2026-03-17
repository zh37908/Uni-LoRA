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


class CountSketchLinear(nn.Module):
    """
    经典 Count Sketch 风格线性层。

    - 每一行都有独立的 bucket hash 和 sign hash
    - 聚合方式可分别为 train/eval 配置
    - 默认 train 使用 mean，eval/test 使用 median
    """

    def __init__(
        self,
        in_features,
        out_features,
        compress=0.03125,
        num_rows=3,
        hash_seed=2,
        hash_bias=False,
        train_aggregation="mean",
        eval_aggregation="median",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compress = compress
        self.num_rows = num_rows
        self.train_aggregation = train_aggregation
        self.eval_aggregation = eval_aggregation

        self.original_weight_size = out_features * in_features
        self.compressed_size = max(1, int(self.original_weight_size * compress))

        self.sketch_states = nn.Parameter(torch.empty(num_rows, self.compressed_size))

        if hash_bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        generator = torch.Generator()
        generator.manual_seed(hash_seed)

        hash_indices = torch.randint(
            0,
            self.compressed_size,
            (num_rows, self.original_weight_size),
            generator=generator,
        )
        hash_signs = torch.randint(
            0,
            2,
            (num_rows, self.original_weight_size),
            generator=generator,
            dtype=torch.int64,
        )
        hash_signs = hash_signs.mul(2).sub(1).to(torch.float32)

        self.register_buffer("hash_indices", hash_indices)
        self.register_buffer("hash_signs", hash_signs)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.sketch_states, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def _reduce_rows(self, values, aggregation):
        if aggregation == "mean":
            return values.mean(dim=0)
        if aggregation == "median":
            return torch.median(values, dim=0).values
        if aggregation == "max":
            return torch.max(values, dim=0).values
        raise ValueError("Unsupported aggregation: {}".format(aggregation))

    def _aggregate_signed_rows(self, signed_values):
        aggregation = self.train_aggregation if self.training else self.eval_aggregation
        return self._reduce_rows(signed_values, aggregation)

    def forward(self, input_tensor):
        gathered = torch.gather(self.sketch_states, 1, self.hash_indices)
        signed_values = gathered * self.hash_signs.to(gathered.dtype)
        reconstructed_weight_flat = self._aggregate_signed_rows(signed_values)
        weight = reconstructed_weight_flat.view(self.out_features, self.in_features)
        return F.linear(input_tensor, weight, self.bias)


def build_results_path(args):
    if args.results_path is not None:
        return args.results_path
    return f"mnist_count_sketch_rows{args.num_rows}_compress{args.compress}_seed{args.seed}.json"


def save_results(args, parameter_count, history, test_loss, test_acc):
    results_path = build_results_path(args)
    results_dir = os.path.dirname(results_path)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)

    payload = {
        "args": vars(args),
        "aggregation": {
            "train": args.train_aggregation,
            "eval": args.eval_aggregation,
        },
        "parameter_count": parameter_count,
        "history": history,
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
        description="PyTorch Count Sketch MNIST classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--nhLayers", type=int, default=1, help="# hidden layers, excluding input/output layers")
    parser.add_argument("--nhu", type=int, default=1000, help="Number of hidden units")
    parser.add_argument("--num-rows", type=int, default=3, help="Number of Count Sketch rows")
    parser.add_argument("--compress", type=float, default=0.03125, help="Compression rate")
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
    parser.add_argument("--epochs", type=int, default=50, help="Maximum # of epochs")
    parser.add_argument("--patience", type=int, default=2, help="Number of epochs to wait before scaling lr.")
    parser.add_argument("--hash-seed", type=int, default=2, help="Seed for hash functions")
    parser.add_argument(
        "--train-aggregation",
        type=str,
        default="mean",
        choices=["mean", "median", "max"],
        help="Aggregation used across sketch rows during training",
    )
    parser.add_argument(
        "--eval-aggregation",
        type=str,
        default="median",
        choices=["mean", "median", "max"],
        help="Aggregation used across sketch rows during validation and test",
    )
    parser.add_argument("--results-path", type=str, default=None, help="Path to save training metrics as JSON")
    parser.add_argument("--save-model", action="store_true", default=False, help="For saving the current model")
    args = parser.parse_args()

    if args.num_rows < 1:
        parser.error("--num-rows must be >= 1")
    if not 0.0 < args.validation_percent < 1.0:
        parser.error("--validation-percent must be in (0, 1)")
    if args.compress <= 0.0:
        parser.error("--compress must be > 0")

    print(args)
    return args


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
        train_dataset, batch_size=batch_size, sampler=train_sampler, **kwargs
    )
    valid_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=valid_sampler, **kwargs
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


def train(model, device, train_loader, optimizer, epoch, log_interval=5):
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


class CountSketchNet(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        nhLayers=1,
        nhu=1000,
        compress=1.0,
        dropout=0.25,
        hash_seed=2,
        num_rows=3,
        hash_bias=False,
        train_aggregation="mean",
        eval_aggregation="median",
    ):
        super().__init__()
        self.nhLayers = nhLayers
        self.input_dim = input_dim

        self.dropout0 = nn.Dropout(dropout)
        self.linear1 = CountSketchLinear(
            input_dim,
            nhu,
            compress=compress,
            num_rows=num_rows,
            hash_seed=hash_seed,
            hash_bias=hash_bias,
            train_aggregation=train_aggregation,
            eval_aggregation=eval_aggregation,
        )
        self.dropout1 = nn.Dropout(dropout)

        for layer in range(2, nhLayers + 1):
            setattr(
                self,
                "linear" + str(layer),
                CountSketchLinear(
                    nhu,
                    nhu,
                    compress=compress,
                    num_rows=num_rows,
                    hash_seed=hash_seed + layer - 1,
                    hash_bias=hash_bias,
                    train_aggregation=train_aggregation,
                    eval_aggregation=eval_aggregation,
                ),
            )
            setattr(self, "dropout" + str(layer), nn.Dropout(dropout))

        self.linear_out = CountSketchLinear(
            nhu,
            output_dim,
            compress=compress,
            num_rows=num_rows,
            hash_seed=hash_seed + nhLayers,
            hash_bias=hash_bias,
            train_aggregation=train_aggregation,
            eval_aggregation=eval_aggregation,
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


def main():
    args = parse_arguments()

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    tr_loader, val_loader, test_loader = load_data(args.batch_size, args.validation_percent, kwargs)

    model = CountSketchNet(
        input_dim=784,
        output_dim=10,
        nhLayers=args.nhLayers,
        nhu=args.nhu,
        compress=args.compress,
        dropout=args.dropout,
        hash_seed=args.hash_seed,
        num_rows=args.num_rows,
        hash_bias=args.hash_bias,
        train_aggregation=args.train_aggregation,
        eval_aggregation=args.eval_aggregation,
    ).to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.l2reg,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.decay_factor,
        patience=args.patience,
        verbose=True,
    )

    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("The number of parameters is: {}".format(parameter_count))

    history = []

    for epoch in range(1, args.epochs + 1):
        tr_loss = train(model, device, tr_loader, optimizer, epoch)
        val_loss, val_acc = evaluate(model, device, val_loader)
        scheduler.step(val_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": tr_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            "\nEpoch {} Train loss: {:.3f} Val loss: {:.3f} Val acc: {:.2f}%".format(
                epoch, tr_loss, val_loss, val_acc
            )
        )

    test_loss, test_acc = evaluate(model, device, test_loader)
    print("Test loss: {:.3f} Test acc: {:.2f}%".format(test_loss, test_acc))
    save_results(args, parameter_count, history, test_loss, test_acc)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_count_sketch.pt")


if __name__ == "__main__":
    main()

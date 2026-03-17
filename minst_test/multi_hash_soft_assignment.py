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


class SoftAssignmentHashLinear(nn.Module):
    """
    Soft Assignment HashNet 线性层。

    为了让方案在当前 MNIST 配置下可训练，这里不对全部 K 个桶直接做 softmax，
    而是先为每条连接采样少量候选桶，再在候选桶集合上做可微软分配：

        p_ij = softmax(g_theta(feature(i, j)) / tau)
        V_ij = sign(i, j) * sum_c p_ij[c] * w[candidate_ij[c]]

    assignment 支持两种训练模式：
    - softmax: 确定性的 soft assignment
    - gumbel: Gumbel-Softmax 采样；可选 hard straight-through

    评估/测试阶段默认使用确定性的 argmax 硬化；若启用 soft eval，
    则使用 logits 对应的确定性 softmax 分配。
    """

    def __init__(
        self,
        in_features,
        out_features,
        compress=0.03125,
        num_candidates=4,
        gate_hidden=32,
        hash_seed=2,
        layer_index=0,
        hash_bias=False,
        temperature=1.0,
        hard_eval=True,
        assignment_mode="softmax",
        gumbel_hard=False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compress = compress
        self.num_candidates = num_candidates
        self.gate_hidden = gate_hidden
        self.layer_index = layer_index
        self.hard_eval = hard_eval
        self.assignment_mode = assignment_mode
        self.gumbel_hard = gumbel_hard

        self.original_weight_size = out_features * in_features
        self.compressed_size = max(1, int(self.original_weight_size * compress))

        self.shared_weights = nn.Parameter(torch.empty(self.compressed_size))

        if hash_bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        generator = torch.Generator()
        generator.manual_seed(hash_seed)

        candidate_indices = torch.randint(
            0,
            self.compressed_size,
            (self.original_weight_size, num_candidates),
            generator=generator,
            dtype=torch.int64,
        )
        hash_signs = torch.randint(
            0,
            2,
            (self.original_weight_size,),
            generator=generator,
            dtype=torch.int64,
        )
        hash_signs = hash_signs.mul(2).sub(1).to(torch.float32)

        self.register_buffer("candidate_indices", candidate_indices)
        self.register_buffer("hash_signs", hash_signs)
        self.register_buffer("connection_features", self._build_connection_features())
        self.register_buffer("temperature", torch.tensor(float(temperature)))

        self.gate = nn.Sequential(
            nn.Linear(self.connection_features.size(1), gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, num_candidates),
        )

        self.last_entropy = None
        self.last_balance = None
        self.last_assignment_stats = None
        self._cached_eval_weight = None

        self.reset_parameters()

    def _build_connection_features(self):
        row_ids = torch.arange(self.out_features, dtype=torch.float32).unsqueeze(1)
        col_ids = torch.arange(self.in_features, dtype=torch.float32).unsqueeze(0)

        row_grid = row_ids.repeat(1, self.in_features).reshape(-1)
        col_grid = col_ids.repeat(self.out_features, 1).reshape(-1)

        row_norm = row_grid / max(self.out_features - 1, 1)
        col_norm = col_grid / max(self.in_features - 1, 1)

        row_block_size = max(1, math.ceil(self.out_features / 16))
        col_block_size = max(1, math.ceil(self.in_features / 16))
        num_row_blocks = math.ceil(self.out_features / row_block_size)
        num_col_blocks = math.ceil(self.in_features / col_block_size)

        row_block = torch.floor(row_grid / row_block_size)
        col_block = torch.floor(col_grid / col_block_size)
        row_block_norm = row_block / max(num_row_blocks - 1, 1)
        col_block_norm = col_block / max(num_col_blocks - 1, 1)

        interaction = row_norm * col_norm
        layer_feature = torch.full_like(row_norm, float(self.layer_index))

        return torch.stack(
            [row_norm, col_norm, row_block_norm, col_block_norm, interaction, layer_feature],
            dim=1,
        )

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.shared_weights.unsqueeze(0), a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.gate[0].weight, a=math.sqrt(5))
        nn.init.zeros_(self.gate[0].bias)
        nn.init.zeros_(self.gate[2].weight)
        nn.init.zeros_(self.gate[2].bias)
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features) if self.in_features > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def set_temperature(self, temperature):
        self.temperature.fill_(float(temperature))
        self._cached_eval_weight = None

    def _compute_logits(self):
        return self.gate(self.connection_features)

    def _compute_softmax_assignments(self, logits):
        return torch.softmax(logits / self.temperature.clamp_min(1e-6), dim=1)

    def _sample_gumbel_noise(self, logits):
        uniform = torch.rand_like(logits).clamp_(1e-6, 1.0 - 1e-6)
        return -torch.log(-torch.log(uniform))

    def _compute_train_assignments(self, logits):
        if self.assignment_mode == "softmax":
            soft_probs = self._compute_softmax_assignments(logits)
            return soft_probs, soft_probs

        if self.assignment_mode != "gumbel":
            raise ValueError("Unsupported assignment mode: {}".format(self.assignment_mode))

        gumbel_noise = self._sample_gumbel_noise(logits)
        gumbel_logits = (logits + gumbel_noise) / self.temperature.clamp_min(1e-6)
        soft_probs = torch.softmax(gumbel_logits, dim=1)

        if not self.gumbel_hard:
            return soft_probs, soft_probs

        hard_choice = soft_probs.argmax(dim=1, keepdim=True)
        hard_probs = torch.zeros_like(soft_probs).scatter_(1, hard_choice, 1.0)
        st_probs = hard_probs - soft_probs.detach() + soft_probs
        return st_probs, soft_probs

    def _compute_eval_soft_assignments(self, logits):
        return self._compute_softmax_assignments(logits)

    def _compute_regularizers(self, probs):
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1).mean()

        usage = torch.zeros(
            self.compressed_size,
            device=probs.device,
            dtype=probs.dtype,
        )
        usage.scatter_add_(0, self.candidate_indices.reshape(-1), probs.reshape(-1))
        usage = usage / float(self.original_weight_size)

        target = 1.0 / float(self.compressed_size)
        balance = torch.sum((usage - target) ** 2)
        return entropy, balance

    def _compute_assignment_stats(self, probs):
        max_prob = probs.max(dim=1).values.mean()
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1).mean()
        top1_choice = probs.argmax(dim=1)
        top1_counts = torch.bincount(top1_choice, minlength=self.num_candidates).to(probs.dtype)
        top1_ratio = top1_counts.max() / max(float(probs.size(0)), 1.0)

        return {
            "avg_max_p": max_prob.detach(),
            "avg_entropy": entropy.detach(),
            "top1_ratio": top1_ratio.detach(),
        }

    def _reconstruct_weight_from_probs(self, probs):
        candidate_weights = self.shared_weights[self.candidate_indices]
        reconstructed_weight_flat = torch.sum(probs * candidate_weights, dim=1)
        reconstructed_weight_flat = reconstructed_weight_flat * self.hash_signs.to(candidate_weights.dtype)
        return reconstructed_weight_flat.view(self.out_features, self.in_features)

    def _reconstruct_train_weight(self):
        logits = self._compute_logits()
        forward_probs, report_probs = self._compute_train_assignments(logits)
        weight = self._reconstruct_weight_from_probs(forward_probs)
        return weight, report_probs

    def _reconstruct_soft_weight(self):
        logits = self._compute_logits()
        probs = self._compute_eval_soft_assignments(logits)
        weight = self._reconstruct_weight_from_probs(probs)
        return weight, probs

    def _reconstruct_hard_weight(self):
        logits = self._compute_logits()
        hard_choice = logits.argmax(dim=1, keepdim=True)
        probs = torch.zeros(
            logits.size(0),
            logits.size(1),
            device=logits.device,
            dtype=self.shared_weights.dtype,
        ).scatter_(1, hard_choice, 1.0)
        return self._reconstruct_weight_from_probs(probs)

    def forward(self, input_tensor):
        if self.training:
            self._cached_eval_weight = None
            weight, report_probs = self._reconstruct_train_weight()
            self.last_entropy, self.last_balance = self._compute_regularizers(report_probs)
            self.last_assignment_stats = self._compute_assignment_stats(report_probs)
            return F.linear(input_tensor, weight, self.bias)

        if self._cached_eval_weight is None or self._cached_eval_weight.device != input_tensor.device:
            if self.hard_eval:
                self._cached_eval_weight = self._reconstruct_hard_weight().detach()
            else:
                weight, _ = self._reconstruct_soft_weight()
                self._cached_eval_weight = weight.detach()

        zero = self.shared_weights.new_zeros(())
        self.last_entropy = zero
        self.last_balance = zero
        self.last_assignment_stats = None
        return F.linear(input_tensor, self._cached_eval_weight, self.bias)


def get_equivalent_compression(input_dim, output_dim, nhu, nhLayers, compress):
    return compress


def build_results_path(args):
    if args.results_path is not None:
        return args.results_path

    if args.hashed:
        model_name = "soft_assignment_{}".format(args.assignment_mode)
        if args.assignment_mode == "gumbel" and args.gumbel_hard:
            model_name += "_hardst"
    else:
        model_name = "dense"
    return (
        "mnist_{}_cand{}_compress{}_seed{}.json".format(
            model_name,
            args.num_candidates,
            args.compress,
            args.seed,
        )
    )


def save_results(args, parameter_count, history, test_loss, test_acc):
    results_path = build_results_path(args)
    results_dir = os.path.dirname(results_path)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)

    payload = {
        "args": vars(args),
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
        description="PyTorch Soft Assignment HashNet on MNIST",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--nhLayers", type=int, default=1, help="# hidden layers, excluding input/output layers")
    parser.add_argument("--nhu", type=int, default=1000, help="Number of hidden units")
    parser.add_argument("--hashed", default=False, action="store_true", help="Enable soft assignment hashing")
    parser.add_argument("--compress", type=float, default=0.03125, help="Compression rate")
    parser.add_argument("--num-candidates", type=int, default=4, help="Candidate buckets per connection")
    parser.add_argument("--gate-hidden", type=int, default=32, help="Hidden width of the gate network")
    parser.add_argument(
        "--assignment-mode",
        type=str,
        default="softmax",
        choices=["softmax", "gumbel"],
        help="Assignment distribution used during training",
    )
    parser.add_argument(
        "--gumbel-hard",
        action="store_true",
        default=False,
        help="Use hard Gumbel-Softmax straight-through during training",
    )
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
    parser.add_argument("--temperature-start", type=float, default=1.0, help="Initial softmax temperature")
    parser.add_argument("--temperature-end", type=float, default=0.1, help="Final softmax temperature")
    parser.add_argument("--lambda-ent", type=float, default=1e-3, help="Entropy regularization weight")
    parser.add_argument("--lambda-bal", type=float, default=1e-2, help="Load balance regularization weight")
    parser.add_argument(
        "--soft-eval",
        action="store_true",
        default=False,
        help="Use deterministic soft assignment during validation/test instead of argmax hardening",
    )
    parser.add_argument("--results-path", type=str, default=None, help="Path to save training metrics as JSON")
    parser.add_argument("--save-model-path", type=str, default="mnist_soft_assignment.pt", help="Path to save model")
    parser.add_argument("--save-model", action="store_true", default=False, help="Save the final model checkpoint")
    args = parser.parse_args()

    if not 0.0 < args.validation_percent < 1.0:
        parser.error("--validation-percent must be in (0, 1)")
    if args.compress <= 0.0:
        parser.error("--compress must be > 0")
    if args.num_candidates < 1:
        parser.error("--num-candidates must be >= 1")
    if args.gate_hidden < 1:
        parser.error("--gate-hidden must be >= 1")
    if args.temperature_start <= 0.0 or args.temperature_end <= 0.0:
        parser.error("--temperature-start and --temperature-end must be > 0")
    if args.gumbel_hard and args.assignment_mode != "gumbel":
        parser.error("--gumbel-hard requires --assignment-mode gumbel")

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


def set_model_temperature(model, temperature):
    for module in model.modules():
        if isinstance(module, SoftAssignmentHashLinear):
            module.set_temperature(temperature)


def collect_regularization_terms(model, device):
    entropy = torch.zeros((), device=device)
    balance = torch.zeros((), device=device)

    for module in model.modules():
        if isinstance(module, SoftAssignmentHashLinear):
            if module.last_entropy is not None:
                entropy = entropy + module.last_entropy
            if module.last_balance is not None:
                balance = balance + module.last_balance

    return entropy, balance


def collect_assignment_stats(model):
    stats = {}

    for name, module in model.named_modules():
        if not isinstance(module, SoftAssignmentHashLinear):
            continue
        if module.last_assignment_stats is None:
            continue

        stats[name] = {
            "avg_max_p": float(module.last_assignment_stats["avg_max_p"].item()),
            "avg_entropy": float(module.last_assignment_stats["avg_entropy"].item()),
            "top1_ratio": float(module.last_assignment_stats["top1_ratio"].item()),
        }

    return stats


def format_assignment_stats(stats):
    if not stats:
        return "No assignment stats"

    parts = []
    for layer_name, layer_stats in stats.items():
        parts.append(
            "{}[max(p)={:.4f}, H={:.4f}, top1={:.4f}]".format(
                layer_name,
                layer_stats["avg_max_p"],
                layer_stats["avg_entropy"],
                layer_stats["top1_ratio"],
            )
        )
    return " | ".join(parts)


def compute_temperature(step, total_steps, tau_start, tau_end):
    if total_steps <= 1:
        return tau_end

    progress = min(max(step / float(total_steps - 1), 0.0), 1.0)
    return tau_start * ((tau_end / tau_start) ** progress)


def train(model, device, train_loader, optimizer, epoch, global_step, total_steps, args, log_interval=5):
    model.train()
    total_loss = 0.0
    task_loss_sum = 0.0
    entropy_loss_sum = 0.0
    balance_loss_sum = 0.0
    last_temperature = args.temperature_end
    last_assignment_stats = {}

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        temperature = compute_temperature(
            global_step,
            total_steps,
            args.temperature_start,
            args.temperature_end,
        )
        set_model_temperature(model, temperature)
        last_temperature = temperature

        optimizer.zero_grad()
        output = model(data)
        task_loss = F.nll_loss(output, target)
        entropy_reg, balance_reg = collect_regularization_terms(model, device)
        assignment_stats = collect_assignment_stats(model)
        last_assignment_stats = assignment_stats
        entropy_loss = args.lambda_ent * entropy_reg
        balance_loss = args.lambda_bal * balance_reg
        loss = task_loss + entropy_loss + balance_loss
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}\tTask: {:.6f}\tEnt: {:.6f}\tBal: {:.6f}\tTau: {:.4f}\tMode: {}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.sampler),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                    task_loss.item(),
                    entropy_loss.item(),
                    balance_loss.item(),
                    temperature,
                    args.assignment_mode + ("-hard" if args.assignment_mode == "gumbel" and args.gumbel_hard else ""),
                ),
                "\t{}".format(format_assignment_stats(assignment_stats)),
                end="\r",
            )

        batch_size = data.size(0)
        total_loss += loss.item() * batch_size
        task_loss_sum += task_loss.item() * batch_size
        entropy_loss_sum += entropy_loss.item() * batch_size
        balance_loss_sum += balance_loss.item() * batch_size
        global_step += 1

    denom = len(train_loader.sampler)
    return (
        {
            "train_loss": total_loss / denom,
            "train_task_loss": task_loss_sum / denom,
            "train_entropy_loss": entropy_loss_sum / denom,
            "train_balance_loss": balance_loss_sum / denom,
            "temperature": last_temperature,
            "assignment_stats": last_assignment_stats,
        },
        global_step,
    )


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


class Net(nn.Module):
    def __init__(self, input_dim, output_dim, nhLayers=1, nhu=1000, compress=1.0, dropout=0.25):
        super().__init__()
        self.nhLayers = nhLayers
        self.input_dim = input_dim
        c_nhu = round(nhu * compress)

        self.dropout0 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(input_dim, c_nhu)
        self.dropout1 = nn.Dropout(dropout)

        for layer in range(2, nhLayers + 1):
            setattr(self, "linear" + str(layer), nn.Linear(c_nhu, c_nhu))
            setattr(self, "dropout" + str(layer), nn.Dropout(dropout))

        self.linear_out = nn.Linear(c_nhu, output_dim)

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


class SoftAssignmentHashNet(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        nhLayers=1,
        nhu=1000,
        compress=1.0,
        dropout=0.25,
        hash_seed=2,
        num_candidates=4,
        gate_hidden=32,
        hash_bias=False,
        temperature=1.0,
        hard_eval=True,
        assignment_mode="softmax",
        gumbel_hard=False,
    ):
        super().__init__()
        self.nhLayers = nhLayers
        self.input_dim = input_dim

        self.dropout0 = nn.Dropout(dropout)
        self.linear1 = SoftAssignmentHashLinear(
            input_dim,
            nhu,
            compress=compress,
            num_candidates=num_candidates,
            gate_hidden=gate_hidden,
            hash_seed=hash_seed,
            layer_index=0,
            hash_bias=hash_bias,
            temperature=temperature,
            hard_eval=hard_eval,
            assignment_mode=assignment_mode,
            gumbel_hard=gumbel_hard,
        )
        self.dropout1 = nn.Dropout(dropout)

        for layer in range(2, nhLayers + 1):
            setattr(
                self,
                "linear" + str(layer),
                SoftAssignmentHashLinear(
                    nhu,
                    nhu,
                    compress=compress,
                    num_candidates=num_candidates,
                    gate_hidden=gate_hidden,
                    hash_seed=hash_seed + layer - 1,
                    layer_index=layer - 1,
                    hash_bias=hash_bias,
                    temperature=temperature,
                    hard_eval=hard_eval,
                    assignment_mode=assignment_mode,
                    gumbel_hard=gumbel_hard,
                ),
            )
            setattr(self, "dropout" + str(layer), nn.Dropout(dropout))

        self.linear_out = SoftAssignmentHashLinear(
            nhu,
            output_dim,
            compress=compress,
            num_candidates=num_candidates,
            gate_hidden=gate_hidden,
            hash_seed=hash_seed + nhLayers,
            layer_index=nhLayers,
            hash_bias=hash_bias,
            temperature=temperature,
            hard_eval=hard_eval,
            assignment_mode=assignment_mode,
            gumbel_hard=gumbel_hard,
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

    if args.hashed:
        model = SoftAssignmentHashNet(
            input_dim=input_dim,
            output_dim=output_dim,
            nhLayers=args.nhLayers,
            nhu=args.nhu,
            compress=args.compress,
            dropout=args.dropout,
            hash_seed=args.hash_seed,
            num_candidates=args.num_candidates,
            gate_hidden=args.gate_hidden,
            hash_bias=args.hash_bias,
            temperature=args.temperature_start,
            hard_eval=not args.soft_eval,
            assignment_mode=args.assignment_mode,
            gumbel_hard=args.gumbel_hard,
        ).to(device)
    else:
        eq_compress = get_equivalent_compression(
            input_dim,
            output_dim,
            args.nhu,
            args.nhLayers,
            args.compress,
        )
        model = Net(
            input_dim,
            output_dim,
            args.nhLayers,
            args.nhu,
            eq_compress,
            args.dropout,
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
    print("The number of trainable parameters is: {}".format(parameter_count))

    history = []
    global_step = 0
    total_steps = args.epochs * len(tr_loader)

    for epoch in range(1, args.epochs + 1):
        train_metrics, global_step = train(
            model,
            device,
            tr_loader,
            optimizer,
            epoch,
            global_step,
            total_steps,
            args,
        )
        if args.hashed:
            set_model_temperature(model, args.temperature_end)
        val_loss, val_acc = evaluate(model, device, val_loader)
        scheduler.step(val_loss)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["train_loss"],
                "train_task_loss": train_metrics["train_task_loss"],
                "train_entropy_loss": train_metrics["train_entropy_loss"],
                "train_balance_loss": train_metrics["train_balance_loss"],
                "temperature": train_metrics["temperature"],
                "assignment_stats": train_metrics["assignment_stats"],
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            "\nEpoch {} Train loss: {:.3f} Task: {:.3f} Ent: {:.3f} Bal: {:.3f} Tau: {:.4f} Val loss: {:.3f} Val acc: {:.2f}%".format(
                epoch,
                train_metrics["train_loss"],
                train_metrics["train_task_loss"],
                train_metrics["train_entropy_loss"],
                train_metrics["train_balance_loss"],
                train_metrics["temperature"],
                val_loss,
                val_acc,
            )
        )
        if train_metrics["assignment_stats"]:
            print("Assignment stats: {}".format(format_assignment_stats(train_metrics["assignment_stats"])))

    if args.hashed:
        set_model_temperature(model, args.temperature_end)
    test_loss, test_acc = evaluate(model, device, test_loader)
    print("Test loss: {:.3f} Test acc: {:.2f}%".format(test_loss, test_acc))
    save_results(args, parameter_count, history, test_loss, test_acc)

    if args.save_model:
        save_model_checkpoint(model, args.save_model_path)


if __name__ == "__main__":
    main()

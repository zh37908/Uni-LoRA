#!/usr/bin/env python
# coding: utf-8

"""
Uni-LoRA Variants GLUE script (robust and adaptive)

Supports:
- unilora
- unilora_nonorm
"""

import os
import json
import argparse
import random
import math
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import evaluate
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from peft import get_peft_model
from peft import (
    UniLoRAConfig,
    UniLoRACountSketchConfig,
    UniLoRANonormConfig,
    UniLoRAFastFoodConfig,
    UniLoRALearnableConfig,
    UniLoRALearnableColumnConfig,
    UniLoRAIsometricControlConfig,
    PeftType,
)


# =========================
# Reproducibility
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# GLUE config
# =========================
GLUE_TASKS = ["cola", "sst2", "mrpc", "qnli", "rte", "stsb"]

TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "stsb": ("sentence1", "sentence2"),
}

MAX_LENGTH = {
    "roberta-base": 512,
    "roberta-large": 128,
}

EPOCHS = {
    "roberta-base": {
        "sst2": 60, "mrpc": 30, "cola": 80, "qnli": 25, "rte": 160, "stsb": 80,
    },
    "roberta-large": {
        "sst2": 20, "mrpc": 40, "cola": 40, "qnli": 20, "rte": 40, "stsb": 40,
    },
}

TASK_TO_METRIC = {
    "cola": "matthews_correlation",
    "sst2": "accuracy",
    "mrpc": "accuracy",
    "qnli": "accuracy",
    "rte": "accuracy",
    "stsb": "pearson",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=["roberta-base", "roberta-large"])
    parser.add_argument("--task", type=str, required=True, choices=GLUE_TASKS)
    parser.add_argument(
        "--variant",
        type=str,
        default="unilora",
        choices=[
            "unilora",
            "unilora_count_sketch",
            "unilora_nonorm",
            "unilora_fastfood",
            "unilora_learnable",
            "unilora_learnable_column",
            "unilora_isometric_control",
        ],
    )
    parser.add_argument("--isometry_alpha", type=float, default=0.0, help="Control parameter for unilora_isometric_control (0.0: isometric, 1.0: non-isometric)")
    parser.add_argument("--head_lr", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out_dir", "--output_dir", dest="out_dir", type=str, default="results_variants")

    # UniLoRA common hyperparams
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--theta_d_length", type=int, default=23040)
    parser.add_argument("--v", type=int, default=3, help="Number of Count-Sketch rows (num_sketches).")
    parser.add_argument("--init_theta_d_bound", type=float, default=0.02)
    parser.add_argument("--unilora_dropout", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int, default=None, help="Override default number of epochs")
    # Monitoring controls
    parser.add_argument("--monitor_every", type=int, default=50, help="Log grad/variance stats every N steps (0 to disable)")
    parser.add_argument("--topk_energy_k", type=int, default=128, help="Top-k for energy ratio on theta_d/grad")
    parser.add_argument("--delta_weight_every", type=int, default=1, help="Log ||Delta W||_F per epoch (0 to disable)")
    parser.add_argument("--lanczos_every", type=int, default=1, help="Run Lanczos every N epochs (0 to disable)")
    parser.add_argument("--lanczos_k", type=int, default=3, help="Top-k eigenvalues to report")
    parser.add_argument("--lanczos_iters", type=int, default=5, help="Lanczos iterations per run")

    return parser.parse_args()


def _get_active_adapters(module):
    if hasattr(module, "active_adapters"):
        return list(module.active_adapters)
    if hasattr(module, "_active_adapter"):
        return [module._active_adapter]
    return []


def _get_theta_d_params(model):
    params = []
    names = []
    for name, p in model.named_parameters():
        if "theta_d" in name:
            params.append(p)
            names.append(name)
    return params, names


def _flatten_tensors(tensors):
    if not tensors:
        return None
    return torch.cat([t.reshape(-1) for t in tensors])


def _grad_stats(params, topk):
    grads = [p.grad.detach() for p in params if p.grad is not None]
    if not grads:
        return None
    g = _flatten_tensors([g.float() for g in grads])
    if g is None or g.numel() == 0:
        return None
    g_norm = torch.norm(g)
    g_var = g.var(unbiased=False)
    energy = torch.sum(g * g)
    k = min(int(topk), g.numel())
    if energy.item() > 0 and k > 0:
        topk_energy = torch.topk(g * g, k).values.sum()
        topk_ratio = (topk_energy / energy).item()
    else:
        topk_ratio = 0.0
    return g_norm.item(), g_var.item(), topk_ratio


def _theta_d_stats(params, topk):
    if not params:
        return None
    vec = _flatten_tensors([p.detach().float() for p in params])
    if vec is None or vec.numel() == 0:
        return None
    var = vec.var(unbiased=False).item()
    energy = torch.sum(vec * vec)
    k = min(int(topk), vec.numel())
    if energy.item() > 0 and k > 0:
        topk_energy = torch.topk(vec * vec, k).values.sum()
        topk_ratio = (topk_energy / energy).item()
    else:
        topk_ratio = 0.0
    return var, topk_ratio


def _scales_stats(model):
    scales = []
    for module in model.modules():
        if hasattr(module, "unilora_scales_A") and hasattr(module, "unilora_scales_B"):
            for adapter in module.unilora_scales_A.keys():
                scales.append(module.unilora_scales_A[adapter].detach().float().reshape(-1))
                scales.append(module.unilora_scales_B[adapter].detach().float().reshape(-1))
    vec = _flatten_tensors(scales)
    if vec is None or vec.numel() == 0:
        return None
    return vec.var(unbiased=False).item()


def _compute_load_balance_stats(model):
    diag = None
    counts = None
    device = None
    for module in model.modules():
        if hasattr(module, "unilora_indices_A") and hasattr(module, "unilora_indices_B") and \
           hasattr(module, "unilora_scales_A") and hasattr(module, "unilora_scales_B"):
            adapters = module.unilora_indices_A.keys()
            for adapter in adapters:
                if adapter not in module.unilora_scales_A.keys():
                    continue
                idx_a = module.unilora_indices_A[adapter].reshape(-1)
                idx_b = module.unilora_indices_B[adapter].reshape(-1)
                s_a = module.unilora_scales_A[adapter].reshape(-1).float()
                s_b = module.unilora_scales_B[adapter].reshape(-1).float()
                if device is None:
                    device = idx_a.device
                    theta_len = module.unilora_theta_d[adapter].numel()
                    diag = torch.zeros(theta_len, device=device, dtype=torch.float32)
                    counts = torch.zeros(theta_len, device=device, dtype=torch.float32)
                diag.scatter_add_(0, idx_a, s_a * s_a)
                diag.scatter_add_(0, idx_b, s_b * s_b)
                counts.scatter_add_(0, idx_a, torch.ones_like(idx_a, dtype=torch.float32))
                counts.scatter_add_(0, idx_b, torch.ones_like(idx_b, dtype=torch.float32))
    if diag is None or counts is None:
        return None
    ptp_minus_i = torch.norm(diag - 1.0).item()
    return ptp_minus_i, counts.detach()


def _compute_delta_weight_norms(model):
    local = {}
    for name, module in model.named_modules():
        if not hasattr(module, "get_delta_weight"):
            continue
        adapters = _get_active_adapters(module)
        for adapter in adapters:
            try:
                delta = module.get_delta_weight(adapter)
            except Exception:
                continue
            if delta is None:
                continue
            norm = torch.norm(delta.detach().float()).item()
            key = f"{name}/{adapter}" if adapter is not None else name
            local[key] = norm
    global_norm = math.sqrt(sum(v * v for v in local.values())) if local else 0.0
    return local, global_norm


def _estimate_top_eigs_lanczos(model, batch, params, k, iters):
    if not params or k <= 0 or iters <= 0:
        return []
    device = params[0].device
    n = sum(p.numel() for p in params)
    if n == 0:
        return []

    def hvp_fn(v):
        model.zero_grad(set_to_none=True)
        out = model(**batch)
        loss = out.loss
        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
        flat_grads = torch.cat([g.reshape(-1) for g in grads])
        dot = torch.dot(flat_grads, v)
        hv = torch.autograd.grad(dot, params, retain_graph=False)
        hv_vec = torch.cat([h.reshape(-1) for h in hv]).detach()
        return hv_vec

    v = torch.randn(n, device=device)
    v = v / (v.norm() + 1e-12)
    v_prev = None
    alphas = []
    betas = []
    for i in range(iters):
        w = hvp_fn(v)
        if v_prev is not None and betas:
            w = w - betas[-1] * v_prev
        alpha = torch.dot(v, w)
        w = w - alpha * v
        beta = w.norm()
        alphas.append(alpha)
        if i < iters - 1:
            betas.append(beta)
        if beta.item() < 1e-10:
            break
        v_prev = v
        v = w / (beta + 1e-12)

    m = len(alphas)
    if m == 0:
        return []
    T = torch.zeros((m, m), device=device)
    for i in range(m):
        T[i, i] = alphas[i]
        if i < m - 1:
            T[i, i + 1] = betas[i]
            T[i + 1, i] = betas[i]
    eigs = torch.linalg.eigvalsh(T)
    topk = torch.topk(eigs, min(k, eigs.numel())).values
    return [v.item() for v in topk.flip(0)]


def _force_math_sdp():
    if not torch.cuda.is_available():
        return
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        return


def main():
    args = parse_args()
    set_seed(args.seed)

    model_name = args.model_name
    task = args.task
    variant = args.variant

    batch_size = 32
    max_length = MAX_LENGTH[model_name]
    num_epochs = args.num_epochs if args.num_epochs is not None else EPOCHS[model_name][task]
    warmup_ratio = 0.06

    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric_name = TASK_TO_METRIC[task]

    # Variant-specific defaults for stability
    current_init_bound = args.init_theta_d_bound
    theta_d_lr = 5e-3

    if variant == "unilora_nonorm":
        # Since nonorm increases the scale of Delta weights significantly,
        # we lower the default initialization if not specified.
        if current_init_bound == 0.02:
            current_init_bound = 0.005
            print(f">>> Variant {variant} detected. Lowering init_theta_d_bound to {current_init_bound} for stability.")
        print(f">>> Variant {variant} detected. Using theta_d_lr = {theta_d_lr}.")

    print("=" * 80)
    print(f"Run Variant: {variant.upper()}")
    print(f"  model_name = {model_name} | task = {task} | seed = {args.seed}")
    print(f"  head_lr = {args.head_lr} | theta_d_lr = {theta_d_lr}")
    print("=" * 80)

    # Data
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    datasets = load_dataset("nyu-mll/glue", task)
    s1_key, s2_key = TASK_TO_KEYS[task]

    def tokenize_fn(examples):
        if s2_key is None:
            return tokenizer(examples[s1_key], truncation=True, padding="max_length", max_length=max_length)
        return tokenizer(examples[s1_key], examples[s2_key], truncation=True, padding="max_length", max_length=max_length)

    remove_cols = [col for col in ["idx", s1_key, s2_key] if col and col in datasets["train"].column_names]
    datasets = datasets.map(tokenize_fn, batched=True, remove_columns=remove_cols)
    if "label" in datasets["train"].column_names:
        datasets = datasets.rename_column("label", "labels")

    def collate_fn(examples):
        return tokenizer.pad(examples, return_tensors="pt")

    train_loader = DataLoader(datasets["train"], shuffle=True, batch_size=batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(datasets["validation"], shuffle=False, batch_size=batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    # Model
    num_labels = 1 if task == "stsb" else 2
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, return_dict=True)

    if variant == "unilora_count_sketch":
        peft_config = UniLoRACountSketchConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_COUNT_SKETCH,
            r=args.rank, theta_d_length=args.theta_d_length,
            num_sketches=args.v,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_nonorm":
        peft_config = UniLoRANonormConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_NONORM,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_fastfood":
        peft_config = UniLoRAFastFoodConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_FASTFOOD,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_learnable":
        peft_config = UniLoRALearnableConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_LEARNABLE,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_learnable_column":
        peft_config = UniLoRALearnableColumnConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_LEARNABLE_COLUMN,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_isometric_control":
        peft_config = UniLoRAIsometricControlConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_ISOMETRIC_CONTROL,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            isometry_alpha=args.isometry_alpha,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    else:
        peft_config = UniLoRAConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )

    model = get_peft_model(base_model, peft_config)
    model.to(device)

    # Adaptive Parameter Grouping
    head_params, theta_d_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            # Force requires_grad if it's a known UniLoRA variant parameter
            if any(term in n for term in ["theta_d", "unilora_indices", "unilora_scales"]):
                p.requires_grad = True
            else:
                continue
        
        # Identify shared vector bank parameters (ending with theta_d or variant-specific theta_d)
        if (
            n.endswith("theta_d")
            or "theta_d." in n
            or "unilora_scales" in n
        ):
            theta_d_params.append(p)
        else:
            head_params.append(p)

    print(f"Detected {len(theta_d_params)} shared vector bank parameters.")
    print(f"Detected {len(head_params)} other trainable parameters (head/classifier).")

    optimizer = AdamW([
        {"params": head_params, "lr": args.head_lr},
        {"params": theta_d_params, "lr": theta_d_lr},
    ], weight_decay=0.01)

    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(warmup_ratio * total_steps), num_training_steps=total_steps)

    # TensorBoard initialization
    log_dir = os.path.join(args.out_dir, "runs", f"{variant}_{task}_{model_name}_lr{args.head_lr}_seed{args.seed}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging to: {log_dir}")

    if args.lanczos_every > 0:
        _force_math_sdp()

    theta_d_monitor_params, _ = _get_theta_d_params(model)
    if not theta_d_monitor_params:
        theta_d_monitor_params = theta_d_params

    # Static load-balance stats (P^T P - I and n_j distribution)
    with torch.no_grad():
        load_stats = _compute_load_balance_stats(model)
        if load_stats is not None:
            ptp_minus_i, counts = load_stats
            writer.add_scalar("LoadBalance/PTP_minus_I_fro", ptp_minus_i, 0)
            writer.add_scalar("LoadBalance/n_j_mean", counts.mean().item(), 0)
            writer.add_scalar("LoadBalance/n_j_std", counts.std(unbiased=False).item(), 0)
            writer.add_scalar("LoadBalance/n_j_max", counts.max().item(), 0)
            writer.add_scalar("LoadBalance/n_j_min", counts.min().item(), 0)
            writer.add_histogram("LoadBalance/n_j_hist", counts.cpu().numpy(), 0)

    # Train / Eval
    best_score = -1e18
    best_metric = None
    history = []
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        epoch_loss = 0
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            if args.monitor_every > 0 and global_step % args.monitor_every == 0:
                grad_stats = _grad_stats(theta_d_monitor_params, args.topk_energy_k)
                if grad_stats is not None:
                    g_norm, g_var, g_topk = grad_stats
                    writer.add_scalar("Grad/theta_d_norm", g_norm, global_step)
                    writer.add_scalar("Grad/theta_d_var", g_var, global_step)
                    writer.add_scalar("Grad/theta_d_topk_energy", g_topk, global_step)
                theta_stats = _theta_d_stats(theta_d_monitor_params, args.topk_energy_k)
                if theta_stats is not None:
                    t_var, t_topk = theta_stats
                    writer.add_scalar("ThetaD/var", t_var, global_step)
                    writer.add_scalar("ThetaD/topk_energy", t_topk, global_step)
                scales_var = _scales_stats(model)
                if scales_var is not None:
                    writer.add_scalar("Scales/var", scales_var, global_step)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            writer.add_scalar("Train/Loss", loss.item(), global_step)
            epoch_loss += loss.item()
            global_step += 1
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        writer.add_scalar("Train/Epoch_Loss", avg_epoch_loss, epoch)

        model.eval()
        metric = evaluate.load("glue", task)
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                logits = model(**batch).logits
            if task == "stsb":
                metric.add_batch(predictions=logits.squeeze(-1).cpu().numpy(), references=batch["labels"].cpu().numpy())
            else:
                metric.add_batch(predictions=logits.argmax(dim=-1).cpu().numpy(), references=batch["labels"].cpu().numpy())

        eval_results = metric.compute()
        score = eval_results[metric_name]
        print(f"Epoch {epoch} | {metric_name}: {score:.4f} | {eval_results}")

        # Log metrics to TensorBoard
        for k, v in eval_results.items():
            writer.add_scalar(f"Eval/{k}", v, epoch)

        history.append({
            "epoch": epoch,
            "score": score,
            "metrics": eval_results
        })

        if score > best_score:
            best_score = score
            best_metric = eval_results

        if args.delta_weight_every > 0 and (epoch % args.delta_weight_every == 0):
            with torch.no_grad():
                local_norms, global_norm = _compute_delta_weight_norms(model)
            writer.add_scalar("DeltaW/Global", global_norm, epoch)
            if local_norms:
                local_mean = sum(local_norms.values()) / max(1, len(local_norms))
                writer.add_scalar("DeltaW/LocalMean", local_mean, epoch)
                for name, norm in local_norms.items():
                    safe_name = name.replace(".", "/")
                    writer.add_scalar(f"DeltaW/Local/{safe_name}", norm, epoch)

        if args.lanczos_every > 0 and (epoch % args.lanczos_every == 0):
            if theta_d_monitor_params:
                was_training = model.training
                model.eval()
                lanczos_batch = next(iter(train_loader))
                lanczos_batch = {k: v.to(device) for k, v in lanczos_batch.items()}
                try:
                    eigs = _estimate_top_eigs_lanczos(
                        model=model,
                        batch=lanczos_batch,
                        params=theta_d_monitor_params,
                        k=args.lanczos_k,
                        iters=args.lanczos_iters,
                    )
                    for i, val in enumerate(eigs):
                        writer.add_scalar(f"Curvature/Lanczos/eig_{i}", val, epoch)
                except RuntimeError as exc:
                    print(f"[Lanczos] skipped due to RuntimeError: {exc}")
                if was_training:
                    model.train()

    writer.close()

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{variant}_{task}_{model_name}_lr{args.head_lr}_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump({
            "variant": variant, 
            "best_score": best_score, 
            "best_metric": best_metric, 
            "history": history,
            "args": vars(args)
        }, f, indent=2)
    print(f"Best score: {best_score} saved to {out_path}")

if __name__ == "__main__":
    main()

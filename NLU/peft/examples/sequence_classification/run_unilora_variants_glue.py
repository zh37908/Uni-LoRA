#!/usr/bin/env python
# coding: utf-8

"""
Uni-LoRA Variants GLUE script (robust and adaptive)

Supports:
- lora
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
    LoraConfig,
    UniLoRAConfig,
    UniLoRACountSketchConfig,
    UniLoRASignConfig,
    UniLoRANonormConfig,
    UniLoRAFastFoodConfig,
    UniLoRAGSConfig,
    UniLoRABlockRoutingConfig,
    UniLoRAStageRatioConfig,
    UniLoRALearnableConfig,
    UniLoRALearnableColumnConfig,
    UniLoRAIsometricControlConfig,
    DirectUniLoRAConfig,
    UniLoRALayerWiseConfig,
    UniLoRALearnableLayerConfig,
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
            "lora",
            "unilora",
            "unilora_count_sketch",
            "unilora_sign",
            "unilora_nonorm",
            "unilora_fastfood",
            "unilora_gs",
            "unilora_block_routing",
            "unilora_stage_ratio",
            "unilora_learnable",
            "unilora_learnable_column",
            "unilora_isometric_control",
            "direct_unilora",
            "unilora_layer_wise",
            "unilora_learnable_layer",
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
    parser.add_argument("--theta_d_lr", type=float, default=5e-3)
    parser.add_argument("--alpha_lr", type=float, default=None, help="LR for unilora_layer_alpha_* parameters; defaults to theta_d_lr / 50.")
    parser.add_argument("--alpha_freeze_ratio", type=float, default=0.1, help="Fraction of total steps to freeze alpha params at the start.")
    parser.add_argument("--alpha_init", type=float, default=1.0, help="Initial bounded alpha value for unilora_learnable_layer.")
    parser.add_argument("--alpha_min", type=float, default=0.5, help="Lower bound for alpha in unilora_learnable_layer.")
    parser.add_argument("--alpha_max", type=float, default=1.5, help="Upper bound for alpha in unilora_learnable_layer.")
    parser.add_argument("--unilora_dropout", type=float, default=0.0)
    parser.add_argument("--init_logits_std", type=float, default=0.1)
    parser.add_argument("--init_logits_bias", type=float, default=2.0)
    parser.add_argument("--gumbel_tau", type=float, default=1.0)
    parser.add_argument("--num_blocks", type=int, default=8, help="Number of blocks for unilora_block_routing")
    parser.add_argument(
        "--stage_theta_d_ratios",
        type=float,
        nargs=3,
        default=[0.2, 0.3, 0.5],
        metavar=("FRONT", "MIDDLE", "BACK"),
        help="Theta_d ratios for front/middle/back stages in unilora_stage_ratio.",
    )
    parser.add_argument("--num_epochs", type=int, default=None, help="Override default number of epochs")

    return parser.parse_args()


def estimate_nonorm_scale(model_name: str, rank: int, theta_d_length: int):
    # Assumes target_modules are: query/key/value/output.dense/intermediate.dense
    # Per layer LoRA indices ≈ r*(h+h) * 4 + r*(h+4h) = 13*h*r
    model_specs = {
        "roberta-base": (12, 768),
        "roberta-large": (24, 1024),
    }
    if model_name not in model_specs:
        return None, 1.0
    num_layers, hidden_size = model_specs[model_name]
    lora_para_cnt = num_layers * 13 * hidden_size * rank
    if theta_d_length <= 0:
        return None, 1.0
    count = lora_para_cnt / float(theta_d_length)
    if count <= 0:
        return count, 1.0
    scale = 1.0 / math.sqrt(count)
    return count, scale


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
    theta_d_lr = args.theta_d_lr
    alpha_lr = args.alpha_lr if args.alpha_lr is not None else (theta_d_lr / 50.0)

    if variant == "unilora_nonorm":
        # Since nonorm increases the scale of Delta weights significantly,
        # scale init bound and lr by 1/sqrt(count) to match unilora.
        count, scale = estimate_nonorm_scale(model_name, args.rank, args.theta_d_length)
        if count is not None:
            current_init_bound = current_init_bound * scale
            theta_d_lr = theta_d_lr * scale
            print(
                f">>> Variant {variant} detected. "
                f"Estimated count={count:.2f}, scale={scale:.4f}. "
                f"Scaled init_theta_d_bound={current_init_bound:.6f}, theta_d_lr={theta_d_lr:.6f}."
            )
        else:
            if current_init_bound == 0.02:
                current_init_bound = 0.005
                print(f">>> Variant {variant} detected. Lowering init_theta_d_bound to {current_init_bound} for stability.")
            print(f">>> Variant {variant} detected. Using theta_d_lr = {theta_d_lr}.")

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

    if variant == "lora":
        peft_config = LoraConfig(
            task_type="SEQ_CLS",
            r=args.rank,
            lora_alpha=args.rank,
            lora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_count_sketch":
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
    elif variant == "unilora_sign":
        peft_config = UniLoRASignConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_SIGN,
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
    elif variant == "unilora_gs":
        peft_config = UniLoRAGSConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_GS,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            init_logits_std=args.init_logits_std,
            init_logits_bias=args.init_logits_bias,
            gumbel_tau=args.gumbel_tau,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_block_routing":
        peft_config = UniLoRABlockRoutingConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_BLOCK_ROUTING,
            r=args.rank, theta_d_length=args.theta_d_length,
            num_blocks=args.num_blocks,
            router_tau=args.gumbel_tau,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_stage_ratio":
        peft_config = UniLoRAStageRatioConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_STAGE_RATIO,
            r=args.rank, theta_d_length=args.theta_d_length,
            stage_theta_d_ratios=args.stage_theta_d_ratios,
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
    elif variant == "direct_unilora":
        peft_config = DirectUniLoRAConfig(
            task_type="SEQ_CLS", peft_type=PeftType.DIRECT_UNILORA,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_layer_wise":
        peft_config = UniLoRALayerWiseConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_LAYER_WISE,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_learnable_layer":
        peft_config = UniLoRALearnableLayerConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_LEARNABLE_LAYER,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            alpha_init=args.alpha_init, alpha_min=args.alpha_min, alpha_max=args.alpha_max,
            unilora_dropout=args.unilora_dropout,
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
    head_params, theta_d_params, alpha_params = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            # Force requires_grad if it's a known UniLoRA variant parameter
            if any(term in n for term in ["theta_d", "unilora_layer_alpha"]):
                p.requires_grad = True
            else:
                continue
        
        if "unilora_layer_alpha" in n:
            alpha_params.append(p)
        elif n.endswith("theta_d") or "theta_d." in n:
            theta_d_params.append(p)
        else:
            head_params.append(p)

    theta_d_lr_display = f"{theta_d_lr}" if theta_d_params else "N/A"
    alpha_lr_display = f"{alpha_lr}" if alpha_params else "N/A"
    print("=" * 80)
    print(f"Run Variant: {variant.upper()}")
    print(f"  model_name = {model_name} | task = {task} | seed = {args.seed}")
    print(f"  head_lr = {args.head_lr} | theta_d_lr = {theta_d_lr_display} | alpha_lr = {alpha_lr_display}")
    print("=" * 80)

    print(f"Detected {len(theta_d_params)} shared vector bank parameters.")
    print(f"Detected {len(alpha_params)} layer alpha parameters.")
    print(f"Detected {len(head_params)} other trainable parameters (classifier/adapters).")

    optimizer_groups = []
    if head_params:
        optimizer_groups.append({"params": head_params, "lr": args.head_lr, "weight_decay": 0.01})
    if theta_d_params:
        optimizer_groups.append({"params": theta_d_params, "lr": theta_d_lr, "weight_decay": 0.01})
    if alpha_params:
        optimizer_groups.append({"params": alpha_params, "lr": alpha_lr, "weight_decay": 0.0})

    optimizer = AdamW(optimizer_groups)

    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(warmup_ratio * total_steps), num_training_steps=total_steps)
    alpha_freeze_steps = int(args.alpha_freeze_ratio * total_steps) if alpha_params else 0
    if alpha_params and alpha_freeze_steps > 0:
        for p in alpha_params:
            p.requires_grad = False
        print(f"Freezing alpha parameters for first {alpha_freeze_steps}/{total_steps} steps.")

    # TensorBoard initialization
    log_dir = os.path.join(args.out_dir, "runs", f"{variant}_{task}_{model_name}_lr{args.head_lr}_seed{args.seed}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging to: {log_dir}")

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
            if alpha_params and alpha_freeze_steps > 0 and global_step == alpha_freeze_steps:
                for p in alpha_params:
                    p.requires_grad = True
                print(f"Unfroze alpha parameters at step {global_step}.")

            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
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
        eval_loss = 0
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                logits = outputs.logits
                eval_loss += outputs.loss.item()
            if task == "stsb":
                metric.add_batch(predictions=logits.squeeze(-1).cpu().numpy(), references=batch["labels"].cpu().numpy())
            else:
                metric.add_batch(predictions=logits.argmax(dim=-1).cpu().numpy(), references=batch["labels"].cpu().numpy())

        avg_eval_loss = eval_loss / len(eval_loader)
        eval_results = metric.compute()
        score = eval_results[metric_name]
        print(f"Epoch {epoch} | train_loss: {avg_epoch_loss:.4f} | val_loss: {avg_eval_loss:.4f} | {metric_name}: {score:.4f} | {eval_results}")

        # Log metrics to TensorBoard
        writer.add_scalar("Eval/Loss", avg_eval_loss, epoch)
        for k, v in eval_results.items():
            writer.add_scalar(f"Eval/{k}", v, epoch)

        history.append({
            "epoch": epoch,
            "train_loss": avg_epoch_loss,
            "val_loss": avg_eval_loss,
            "score": score,
            "metrics": eval_results
        })

        if score > best_score:
            best_score = score
            best_metric = eval_results

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

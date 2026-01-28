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
from peft import UniLoRAConfig, UniLoRANonormConfig, UniLoRAFastFoodConfig, PeftType


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
    parser.add_argument("--variant", type=str, default="unilora", choices=["unilora", "unilora_nonorm", "unilora_fastfood"])
    parser.add_argument("--head_lr", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out_dir", "--output_dir", dest="out_dir", type=str, default="results_variants")

    # UniLoRA common hyperparams
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--theta_d_length", type=int, default=23040)
    parser.add_argument("--init_theta_d_bound", type=float, default=0.02)
    parser.add_argument("--unilora_dropout", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int, default=None, help="Override default number of epochs")

    return parser.parse_args()


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
    theta_d_lr = 1e-4

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

    train_loader = DataLoader(datasets["train"], shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
    eval_loader = DataLoader(datasets["validation"], shuffle=False, batch_size=batch_size, collate_fn=collate_fn)

    # Model
    num_labels = 1 if task == "stsb" else 2
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, return_dict=True)

    if variant == "unilora_nonorm":
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
        if n.endswith("theta_d") or "theta_d." in n:
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

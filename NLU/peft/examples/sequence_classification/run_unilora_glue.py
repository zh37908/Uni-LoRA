#!/usr/bin/env python
# coding: utf-8

"""
Uni-LoRA GLUE head_lr sweep script (reproducible, dataloader-style)

Features:
- model_name from command line: roberta-base / roberta-large
- task from command line: cola / sst2 / mrpc / qnli / rte / stsb
- max_length:
    - roberta-base  -> 512
    - roberta-large -> 128
- num_epochs follows the table (base/large × task)
- theta_d_lr fixed to 5e-3
- head_lr from command line
- warmup_ratio fixed to 0.06, linear schedule
- track best validation metric (task-dependent) across epochs
- save best result to JSON
"""

import os
import json
import argparse
import random
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

import evaluate
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from peft import get_peft_model
from peft import UniLoRAConfig, PeftType


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

# max seq length rule
MAX_LENGTH = {
    "roberta-base": 512,
    "roberta-large": 128,
}

# epochs from your table
EPOCHS = {
    "roberta-base": {
        "sst2": 60,
        "mrpc": 30,
        "cola": 80,
        "qnli": 25,
        "rte": 160,
        "stsb": 80,
    },
    "roberta-large": {
        "sst2": 20,
        "mrpc": 40,
        "cola": 40,
        "qnli": 20,
        "rte": 40,
        "stsb": 40,
    },
}

# best-tracking metric per task (as you used in table logic)
TASK_TO_METRIC = {
    "cola": "matthews_correlation",
    "sst2": "accuracy",
    "mrpc": "accuracy",   # table通常用accuracy（如你要f1可改这里）
    "qnli": "accuracy",
    "rte": "accuracy",
    "stsb": "pearson",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=["roberta-base", "roberta-large"])
    parser.add_argument("--task", type=str, required=True, choices=GLUE_TASKS)
    parser.add_argument("--head_lr", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)

    # keep compatibility with your bash usage
    parser.add_argument("--out_dir", "--output_dir", dest="out_dir", type=str, default="results_glue")

    # UniLoRA hyperparams (keep same defaults as your CoLA script)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--theta_d_length", type=int, default=23040)   # if you have per-model/table values, change here or pass by args
    parser.add_argument("--init_theta_d_bound", type=float, default=0.02)
    parser.add_argument("--unilora_dropout", type=float, default=0.0)

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)

    model_name = args.model_name
    task = args.task

    if model_name not in MAX_LENGTH:
        raise ValueError(f"Unsupported model_name: {model_name}")
    if task not in GLUE_TASKS:
        raise ValueError(f"Unsupported task: {task}")

    batch_size = 32
    max_length = MAX_LENGTH[model_name]
    num_epochs = EPOCHS[model_name][task]

    theta_d_lr = 5e-3
    warmup_ratio = 0.06

    # UniLoRA params
    rank = args.rank
    theta_d_length = args.theta_d_length
    proj_seed = args.seed
    init_theta_d_bound = args.init_theta_d_bound

    device = "cuda" if torch.cuda.is_available() else "cpu"

    metric_name = TASK_TO_METRIC[task]

    print("=" * 80)
    print("Run config:")
    print(f"  model_name        = {model_name}")
    print(f"  task              = {task}")
    print(f"  seed              = {args.seed}")
    print(f"  head_lr           = {args.head_lr}")
    print(f"  theta_d_lr        = {theta_d_lr} (fixed)")
    print(f"  batch_size        = {batch_size}")
    print(f"  max_length        = {max_length}")
    print(f"  num_epochs        = {num_epochs}")
    print(f"  warmup_ratio      = {warmup_ratio}")
    print(f"  metric_for_best   = {metric_name}")
    print(f"  out_dir           = {args.out_dir}")
    print("=" * 80)

    # =========================
    # Data
    # =========================
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        # RoBERTa usually has pad_token_id, but keep safe like your script
        tokenizer.pad_token_id = tokenizer.eos_token_id

    datasets = load_dataset("glue", task)

    s1_key, s2_key = TASK_TO_KEYS[task]

    def tokenize_fn(examples):
        if s2_key is None:
            return tokenizer(
                examples[s1_key],
                truncation=True,
                padding="max_length",   # ✅ 强制 pad
                max_length=max_length,
            )
        return tokenizer(
            examples[s1_key],
            examples[s2_key],
            truncation=True,
            padding="max_length",       # ✅ 强制 pad
            max_length=max_length,
        )

    # remove ONLY text columns + idx (keep label)
    remove_cols = []
    for col in ["idx", s1_key, s2_key]:
        if col is not None and col in datasets["train"].column_names:
            remove_cols.append(col)

    datasets = datasets.map(
        tokenize_fn,
        batched=True,
        remove_columns=remove_cols,
    )

    # HF model expects "labels"
    if "label" in datasets["train"].column_names:
        datasets = datasets.rename_column("label", "labels")

    def collate_fn(examples):
        return tokenizer.pad(examples, return_tensors="pt")

    train_loader = DataLoader(
        datasets["train"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
    eval_loader = DataLoader(
        datasets["validation"],
        shuffle=False,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )

    # =========================
    # Model + UniLoRA
    # =========================
    num_labels = 1 if task == "stsb" else 2
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        return_dict=True,
    )

    # NOTE: Keep the same target_modules/modules_to_save pattern as your working CoLA script.
    peft_config = UniLoRAConfig(
        task_type="SEQ_CLS",
        peft_type=PeftType.UNILORA,
        r=rank,
        theta_d_length=theta_d_length,
        proj_seed=proj_seed,
        init_theta_d_bound=init_theta_d_bound,
        unilora_dropout=args.unilora_dropout,
        target_modules=[
            "query", "key", "value",
            "output.dense", "intermediate.dense",
        ],
        modules_to_save=["classifier"],
    )

    model = get_peft_model(base_model, peft_config)
    model.to(device)

    # =========================
    # Optimizer & Scheduler
    # =========================
    head_params, theta_d_params = [], []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # keep EXACT naming rule as your CoLA script
        if "unilora_theta_d" in n:
            theta_d_params.append(p)
        else:
            head_params.append(p)

    if len(theta_d_params) == 0:
        raise RuntimeError(
            "No theta_d params found (name contains 'unilora_theta_d'). "
            "Please check UniLoRA parameter naming in your PEFT version."
        )
    if len(head_params) == 0:
        raise RuntimeError("No head params found. Check modules_to_save/classifier naming.")

    optimizer = AdamW(
        [
            {"params": head_params, "lr": args.head_lr},
            {"params": theta_d_params, "lr": theta_d_lr},
        ],
        weight_decay=0.01,
    )

    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    # =========================
    # Train / Eval (best tracking)
    # =========================
    best_score = -1e18
    best_epoch = -1
    best_metric = None

    for epoch in range(num_epochs):
        # ---- train ----
        model.train()
        for batch in tqdm(train_loader, desc=f"Train epoch {epoch}", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # ---- eval ----
        model.eval()
        metric = evaluate.load("glue", task)

        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                logits = model(**batch).logits

            if task == "stsb":
                # regression: predictions must be float
                preds = logits.squeeze(-1).detach().cpu().numpy().astype(float)
                refs = batch["labels"].detach().cpu().numpy().astype(float)
                metric.add_batch(predictions=preds, references=refs)
            else:
                preds = logits.argmax(dim=-1)
                metric.add_batch(predictions=preds, references=batch["labels"])

        eval_metric = metric.compute()
        score = eval_metric[metric_name]

        print(
            f"[model={model_name} task={task} lr={args.head_lr:.0e} seed={args.seed}] "
            f"epoch {epoch}: {eval_metric}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_metric = eval_metric

    # =========================
    # Save best result
    # =========================
    os.makedirs(args.out_dir, exist_ok=True)

    result = {
        "task": task,
        "model": model_name,
        "head_lr": args.head_lr,
        "theta_d_lr": theta_d_lr,
        "seed": args.seed,
        "proj_seed": proj_seed,
        "batch_size": batch_size,
        "max_length": max_length,
        "num_epochs": num_epochs,
        "warmup_ratio": warmup_ratio,
        "metric_for_best": metric_name,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "rank": rank,
        "theta_d_length": theta_d_length,
        "init_theta_d_bound": init_theta_d_bound,
        "unilora_dropout": args.unilora_dropout,
    }

    out_path = os.path.join(
        args.out_dir,
        f"{task}_{model_name}_lr_{args.head_lr:.0e}_seed_{args.seed}.json",
    )

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("Saved best result to:", out_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# coding: utf-8

"""
Uni-LoRA CoLA head_lr sweep script (fully reproducible)

Features:
- head_lr from command line
- seed from command line (controls ALL randomness, incl. UniLoRA proj_seed)
- track best validation Matthews correlation across epochs
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


def main():
    # =========================
    # Args
    # =========================
    parser = argparse.ArgumentParser()
    parser.add_argument("--head_lr", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out_dir", type=str, default="results_cola")
    args = parser.parse_args()

    # Set seeds (includes UniLoRA proj_seed)
    set_seed(args.seed)

    # =========================
    # Fixed config (Table 8 – LARGE)
    # =========================
    model_name = "roberta-large"
    task = "cola"

    batch_size = 32
    max_length = 128
    num_epochs = 40

    theta_d_lr = 5e-3
    warmup_ratio = 0.06

    rank = 4
    theta_d_length = 23040
    proj_seed = args.seed
    init_theta_d_bound = 0.02

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================
    # Data
    # =========================
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    datasets = load_dataset("glue", task)

    def tokenize_fn(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            max_length=max_length,
        )

    datasets = datasets.map(
        tokenize_fn,
        batched=True,
        remove_columns=["idx", "sentence"],
    )
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
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        return_dict=True,
    )

    peft_config = UniLoRAConfig(
        task_type="SEQ_CLS",
        peft_type=PeftType.UNILORA,
        r=rank,
        theta_d_length=theta_d_length,
        proj_seed=proj_seed,
        init_theta_d_bound=init_theta_d_bound,
        unilora_dropout=0.0,
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
        if "unilora_theta_d" in n:
            theta_d_params.append(p)
        else:
            head_params.append(p)

    optimizer = AdamW(
        [
            {"params": head_params, "lr": args.head_lr},
            {"params": theta_d_params, "lr": theta_d_lr},
        ]
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
    metric_name = "matthews_correlation"
    best_score = -1e9
    best_epoch = -1
    best_metric = None

    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Train epoch {epoch}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        model.eval()
        metric = evaluate.load("glue", task)

        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                logits = model(**batch).logits
            preds = logits.argmax(dim=-1)
            metric.add_batch(predictions=preds, references=batch["labels"])

        eval_metric = metric.compute()
        score = eval_metric[metric_name]

        print(
            f"[lr={args.head_lr:.0e} | seed={args.seed}] "
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
        "seed": args.seed,
        "proj_seed": proj_seed,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
    }

    out_path = os.path.join(
        args.out_dir,
        f"cola_lr_{args.head_lr:.0e}_seed_{args.seed}.json",
    )

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("Saved best result to:", out_path)


if __name__ == "__main__":
    main()

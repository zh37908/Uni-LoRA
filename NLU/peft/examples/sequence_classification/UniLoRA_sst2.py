#!/usr/bin/env python
# coding: utf-8

"""
Uni-LoRA on GLUE SST-2 (RoBERTa-large)
Hyperparameters strictly follow Table 8 (LARGE setting).
"""

import os
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

from peft import get_peft_model, PeftModel, PeftConfig
from peft import UniLoRAConfig, PeftType


def main():
    # =========================
    # Table-8 Hyperparameters
    # =========================
    model_name = "roberta-large"
    task = "sst2"

    batch_size = 32
    max_length = 128
    num_epochs = 20

    head_lr = 2e-4
    theta_d_lr = 5e-3
    warmup_ratio = 0.06

    rank = 4
    theta_d_length = 23040
    proj_seed = 0
    init_theta_d_bound = 0.02

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    save_dir = "./roberta-large-unilora-sst2"

    # =========================
    # Tokenizer & Dataset
    # =========================
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    datasets = load_dataset("glue", task)
    metric = evaluate.load("glue", task)

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
    # Model + Uni-LoRA
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
            "output.dense", "intermediate.dense"
        ],
        modules_to_save=["classifier"],
    )

    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    model.print_savable_parameters()

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
            {"params": head_params, "lr": head_lr},
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
    # Training
    # =========================
    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        model.eval()
        metric = evaluate.load("glue", task)

        for batch in tqdm(eval_loader, desc=f"Eval Epoch {epoch}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                logits = model(**batch).logits
            preds = logits.argmax(dim=-1)
            metric.add_batch(predictions=preds, references=batch["labels"])

        print(f"Epoch {epoch}:", metric.compute())

    # =========================
    # Save & Reload Adapter
    # =========================
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)

    cfg = PeftConfig.from_pretrained(save_dir)
    base2 = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model_name_or_path
    )
    model2 = PeftModel.from_pretrained(base2, save_dir)
    model2.to(device).eval()

    metric = evaluate.load("glue", task)
    for batch in eval_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            logits = model2(**batch).logits
        preds = logits.argmax(dim=-1)
        metric.add_batch(predictions=preds, references=batch["labels"])

    print("Loaded Adapter:", metric.compute())


if __name__ == "__main__":
    main()

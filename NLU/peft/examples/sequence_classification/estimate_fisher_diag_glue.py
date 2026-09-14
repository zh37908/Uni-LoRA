#!/usr/bin/env python3
"""Estimate the diagonal empirical Fisher H_ii in LoRA coordinate space.

Rebuilds the fine-tuned model from a plain-LoRA `*_theory.pt` artifact
(delta_theta = flattened per-module [A, B] weights, plus classifier head),
then accumulates H_ii = (1/B) sum_b (d loss_b / d theta_i)^2 over B mini-batches
of downstream data. The output vector is aligned with the artifact's
delta_theta flattening order (module order, A then B), so it can be passed to
analyze_theory_e4_e5.py via --fisher task=fisher.pt for curvature-weighted
E4/E5 statistics (the Level-2 synthetic lesson: bare parameter-space energy
does not predict functional bias, Eq. (4) is H-weighted).

Example:
  python estimate_fisher_diag_glue.py \
    --artifact results_theory_e3_bias_variance/roberta-large/mrpc/lora_r4/seed_0/lora_mrpc_roberta-large_lr0.0002_seed0_theory.pt \
    --output fisher_diag/mrpc_fisher.pt --num_batches 200
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model

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


def load_artifact(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_model(model_name, task, rank):
    num_labels = 1 if task == "stsb" else 2
    base = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, return_dict=True)
    config = LoraConfig(
        task_type="SEQ_CLS",
        r=rank,
        lora_alpha=rank,
        lora_dropout=0.0,
        target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
        modules_to_save=["classifier"],
    )
    return get_peft_model(base, config)


def inject_lora_weights(model, artifact):
    """Slice the flat delta_theta back into per-module A/B weights."""
    module_names = artifact["module_names"]
    flat = artifact["delta_theta"].to(torch.float32)
    modules = dict(model.named_modules())
    offset = 0
    lora_params = []
    for name in module_names:
        module = modules.get(name)
        if module is None:
            raise KeyError(f"Module {name} from the artifact not found in the rebuilt model.")
        w_a = module.lora_A["default"].weight
        w_b = module.lora_B["default"].weight
        n_a, n_b = w_a.numel(), w_b.numel()
        with torch.no_grad():
            w_a.copy_(flat[offset : offset + n_a].view_as(w_a))
            w_b.copy_(flat[offset + n_a : offset + n_a + n_b].view_as(w_b))
        offset += n_a + n_b
        lora_params.append(w_a)
        lora_params.append(w_b)
    if offset != flat.numel():
        raise ValueError(f"Consumed {offset} values but artifact has {flat.numel()}.")

    classifier_state = artifact.get("classifier_state") or {}
    if classifier_state:
        missing = model.load_state_dict(classifier_state, strict=False)
        n_loaded = len(classifier_state) - len(missing.unexpected_keys)
        print(f"Loaded {n_loaded}/{len(classifier_state)} classifier tensors "
              f"({len(missing.unexpected_keys)} unexpected).")
    else:
        msg = ("artifact has no classifier_state; Fisher would use a random head "
               "and be unreliable. Prefer artifacts saved after 2026-08-31.")
        if os.environ.get("FISHER_REQUIRE_CLASSIFIER", "1") == "1":
            raise RuntimeError(msg)
        print("WARNING: " + msg)
    return lora_params


def build_loader(task, model_name, tokenizer, split, batch_size, seed):
    raw = load_dataset("nyu-mll/glue", task)[split]
    s1_key, s2_key = TASK_TO_KEYS[task]
    max_length = MAX_LENGTH.get(model_name, 128)

    def tokenize_fn(examples):
        if s2_key is None:
            return tokenizer(examples[s1_key], truncation=True, padding="max_length", max_length=max_length)
        return tokenizer(examples[s1_key], examples[s2_key], truncation=True, padding="max_length", max_length=max_length)

    remove_cols = [c for c in ["idx", s1_key, s2_key] if c and c in raw.column_names]
    ds = raw.map(tokenize_fn, batched=True, remove_columns=remove_cols)
    if "label" in ds.column_names:
        ds = ds.rename_column("label", "labels")
    ds = ds.shuffle(seed=seed)

    def collate_fn(examples):
        return tokenizer.pad(examples, return_tensors="pt")

    return DataLoader(ds, shuffle=False, batch_size=batch_size, collate_fn=collate_fn, num_workers=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, help="Plain-LoRA *_theory.pt file.")
    parser.add_argument("--output", required=True, help="Output .pt path for the Fisher diagonal.")
    parser.add_argument("--model_name", default=None, help="Defaults to the artifact's model_name.")
    parser.add_argument("--task", default=None, help="Defaults to the artifact's task.")
    parser.add_argument("--rank", type=int, default=None, help="Defaults to the artifact's rank.")
    parser.add_argument("--split", default="train", choices=["train", "validation"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_batches", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    artifact = load_artifact(args.artifact)
    if artifact.get("variant") not in (None, "lora"):
        print(f"WARNING: artifact variant is {artifact.get('variant')}; "
              "Fisher rebuild assumes plain-LoRA parametrization.")
    model_name = args.model_name or artifact.get("model_name") or "roberta-large"
    task = args.task or artifact["task"]
    rank = args.rank or int(artifact.get("rank") or 4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = build_model(model_name, task, rank)
    lora_params = inject_lora_weights(model, artifact)
    model.to(device)
    model.eval()  # deterministic point: no dropout

    loader = build_loader(task, model_name, tokenizer, args.split, args.batch_size, args.seed)

    fisher = [torch.zeros_like(p, dtype=torch.float64, device=device) for p in lora_params]
    n_batches = 0
    for batch in loader:
        if n_batches >= args.num_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch).loss
        grads = torch.autograd.grad(loss, lora_params, retain_graph=False)
        for acc, g in zip(fisher, grads):
            acc += g.detach().double() ** 2
        n_batches += 1
        if n_batches % 50 == 0:
            print(f"  processed {n_batches}/{args.num_batches} batches")
    if n_batches == 0:
        raise RuntimeError("No batches processed.")

    fisher_flat = torch.cat([f.reshape(-1) / n_batches for f in fisher]).cpu()
    assert fisher_flat.numel() == artifact["delta_theta"].numel(), "Fisher/delta size mismatch"

    Path(os.path.dirname(os.path.abspath(args.output))).mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "fisher_diag": fisher_flat,
            "task": task,
            "model_name": model_name,
            "rank": rank,
            "split": args.split,
            "batch_size": args.batch_size,
            "num_batches": n_batches,
            "artifact": str(args.artifact),
            "module_names": artifact["module_names"],
        },
        args.output,
    )
    stats = fisher_flat.numpy()
    print(f"Saved Fisher diagonal ({stats.size} coords) to {args.output}")
    print(f"  mean={stats.mean():.3e} median={np.median(stats):.3e} max={stats.max():.3e}")


if __name__ == "__main__":
    main()

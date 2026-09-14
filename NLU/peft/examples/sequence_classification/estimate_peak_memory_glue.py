#!/usr/bin/env python
# coding: utf-8
"""Fast peak-GPU-memory estimate for GLUE LoRA / Uni-LoRA (few train steps only).

This does NOT run full training. It builds the same adapter setup as the
efficiency profiler, runs a handful of AdamW train steps on max-length pads,
and reports torch + nvidia-smi peak memory. Enough for a same-batch-size
memory estimate within minutes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

from peft import LoraConfig, PeftType, UniLoRAConfig, get_peft_model


TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "stsb": ("sentence1", "sentence2"),
}
MAX_LENGTH = {"roberta-base": 512, "roberta-large": 128}
TARGET_MODULES = ["query", "key", "value", "output.dense", "intermediate.dense"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=["lora", "unilora"], default="lora")
    p.add_argument("--model_name", choices=["roberta-base", "roberta-large"], default="roberta-large")
    p.add_argument("--task", choices=sorted(TASK_TO_KEYS), required=True)
    p.add_argument("--head_lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--theta_d_length", type=int, default=23040)
    p.add_argument("--unilora_dropout", type=float, default=0.0)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--train_steps", type=int, default=8, help="Optimizer steps used to force peak memory.")
    p.add_argument("--eval_steps", type=int, default=1, help="Eval forward passes after train steps.")
    p.add_argument("--memory_sample_interval", type=float, default=0.25)
    p.add_argument("--gpu_setup", type=str, default=os.environ.get("GPU_SETUP", "1xNVIDIA_L20"))
    p.add_argument("--out_json", type=str, required=True)
    return p.parse_args()


def parse_int_token(value: str) -> int | None:
    match = re.search(r"(\d+)", value)
    return int(match.group(1)) if match else None


def resolve_gpu_query_ids() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible not in {"NoDevFiles", "-1"}:
        return [item.strip() for item in visible.split(",") if item.strip()]
    return ["0"]


class SmiPeakMonitor:
    def __init__(self, interval: float, gpu_query_ids: list[str] | None = None) -> None:
        self.interval = max(0.1, float(interval))
        self.gpu_query_ids = gpu_query_ids or resolve_gpu_query_ids()
        self.peak_mb = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.interval + 2.0)

    def _run(self) -> None:
        cmd = [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "-i",
            ",".join(self.gpu_query_ids),
        ]
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=5)
                vals = [parse_int_token(line) for line in out.splitlines()]
                vals = [v for v in vals if v is not None]
                if vals:
                    self.samples += 1
                    self.peak_mb = max(self.peak_mb, max(vals))
            except (subprocess.SubprocessError, FileNotFoundError):
                pass
            self._stop.wait(self.interval)


def build_tiny_loader(task: str, model_name: str, batch_size: int, n_examples: int, seed: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    ds = load_dataset("nyu-mll/glue", task, split="train")
    n = min(n_examples, len(ds))
    ds = ds.shuffle(seed=seed).select(range(n))

    s1_key, s2_key = TASK_TO_KEYS[task]
    max_length = MAX_LENGTH[model_name]

    def tokenize_fn(examples):
        if s2_key is None:
            return tokenizer(examples[s1_key], truncation=True, padding="max_length", max_length=max_length)
        return tokenizer(
            examples[s1_key],
            examples[s2_key],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    remove_cols = [c for c in ["idx", s1_key, s2_key] if c and c in ds.column_names]
    ds = ds.map(tokenize_fn, batched=True, remove_columns=remove_cols)
    if "label" in ds.column_names:
        ds = ds.rename_column("label", "labels")

    def collate_fn(examples):
        return tokenizer.pad(examples, return_tensors="pt")

    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)


def build_model(args: argparse.Namespace):
    num_labels = 1 if args.task == "stsb" else 2
    base = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=num_labels, return_dict=True
    )
    if args.method == "lora":
        cfg = LoraConfig(
            task_type="SEQ_CLS",
            r=args.rank,
            lora_alpha=args.rank,
            lora_dropout=args.unilora_dropout,
            target_modules=TARGET_MODULES,
            modules_to_save=["classifier"],
        )
    else:
        cfg = UniLoRAConfig(
            task_type="SEQ_CLS",
            peft_type=PeftType.UNILORA,
            r=args.rank,
            theta_d_length=args.theta_d_length,
            proj_seed=args.seed,
            init_theta_d_bound=0.02,
            unilora_dropout=args.unilora_dropout,
            target_modules=TARGET_MODULES,
            modules_to_save=["classifier"],
        )
    model = get_peft_model(base, cfg)
    return model.cuda()


def split_trainable_params(model):
    head, lora, theta = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = name.lower()
        if "classifier" in lname or "score" in lname:
            head.append(p)
        elif "lora_" in lname:
            lora.append(p)
        elif "theta_d" in lname or "theta_d" in name:
            theta.append(p)
        else:
            lora.append(p)
    return head, lora, theta


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for peak-memory estimation.")

    device = torch.device("cuda")
    need = args.batch_size * (args.train_steps + max(args.eval_steps, 1) + 1)
    loader = build_tiny_loader(args.task, args.model_name, args.batch_size, need, args.seed)

    smi = SmiPeakMonitor(args.memory_sample_interval, resolve_gpu_query_ids())
    smi.start()
    t0 = time.perf_counter()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build_model(args)
    model.train()

    head, lora, theta = split_trainable_params(model)
    groups = []
    if head:
        groups.append({"params": head, "lr": args.head_lr, "weight_decay": args.weight_decay})
    if lora:
        groups.append({"params": lora, "lr": args.head_lr, "weight_decay": args.weight_decay})
    if theta:
        groups.append({"params": theta, "lr": 5e-3, "weight_decay": args.weight_decay})
    optimizer = AdamW(groups)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    it = iter(loader)
    trained = 0
    for _ in range(args.train_steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch)
        outputs.loss.backward()
        optimizer.step()
        trained += 1

    evaled = 0
    model.eval()
    with torch.no_grad():
        for _ in range(args.eval_steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            batch = move_batch(batch, device)
            _ = model(**batch)
            evaled += 1

    torch.cuda.synchronize()
    alloc_peak_mb = int(torch.cuda.max_memory_allocated() / (1024 * 1024))
    reserved_peak_mb = int(torch.cuda.max_memory_reserved() / (1024 * 1024))
    # Give nvidia-smi one last sample window.
    time.sleep(max(0.3, args.memory_sample_interval))
    smi.stop()
    wall_s = time.perf_counter() - t0

    # Prefer nvidia-smi when available (matches full profiler); fall back to allocator.
    peak_mb = smi.peak_mb if smi.samples > 0 else alloc_peak_mb
    profile = {
        "status": "ok",
        "estimate_only": True,
        "method": args.method,
        "model_name": args.model_name,
        "task": args.task,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "rank": args.rank,
        "max_length": MAX_LENGTH[args.model_name],
        "train_steps": trained,
        "eval_steps": evaled,
        "trainable_params": trainable,
        "adapter_reported_params": trainable,
        "gpu_setup": args.gpu_setup,
        "wall_clock_total_s": wall_s,
        "peak_gpu_memory_mb": peak_mb,
        "peak_gpu_memory_gb": round(peak_mb / 1024.0, 2),
        "torch_max_memory_allocated_mb": alloc_peak_mb,
        "torch_max_memory_reserved_mb": reserved_peak_mb,
        "nvidia_smi_peak_memory_mb": smi.peak_mb,
        "nvidia_smi_samples": smi.samples,
        "memory_monitor_method": (
            "nvidia-smi device memory (fast estimate)"
            if smi.samples > 0
            else "torch.cuda.max_memory_allocated"
        ),
        "note": (
            "Fast estimate: tiny data slice + few train/eval steps at full max_length padding. "
            "Peak should be comparable to full training for the same batch size / sequence length."
        ),
    }
    out_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(profile, indent=2))
    print(
        f"[estimate] method={args.method} task={args.task} "
        f"peak={peak_mb} MB ({peak_mb/1024:.2f} GB) wall={wall_s:.1f}s -> {out_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

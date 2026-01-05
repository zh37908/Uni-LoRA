#!/usr/bin/env python
# coding: utf-8

"""
Collect best validation accuracy from UniLoRA GLUE logs
and compute mean ± std across seeds for each learning rate,
for multiple models and tasks.
"""

import os
import re
import json
import numpy as np
from collections import defaultdict

# =========================
# Config
# =========================
ROOT_DIR = "/home/kli/unilora_peft_submit/peft/examples/sequence_classification/results_glue"

MODELS = ["roberta-base", "roberta-large"]
TASKS = ["mrpc", "qnli", "rte", "sst2", "stsb"]
SEEDS = ["seed_0", "seed_1", "seed_2", "seed_3", "seed_4"]

LRS = [
    "1e-4", "2e-4", "5e-4",
    "1e-3", "2e-3", "5e-3",
    "1e-2", "2e-2",
]

# Regex to extract accuracy
ACC_PATTERN = re.compile(r"'accuracy':\s*([0-9.]+)")

# =========================
# Main loop
# =========================
summary = {}

for model in MODELS:
    summary[model] = {}

    for task in TASKS:
        print(f"\n=== {model} | {task.upper()} ===")

        task_dir = os.path.join(ROOT_DIR, model, task)
        if not os.path.isdir(task_dir):
            print(f"[WARN] Missing {task_dir}, skipping")
            continue

        lr_to_accs = defaultdict(list)

        # -------------------------
        # Collect best acc per seed
        # -------------------------
        for seed in SEEDS:
            seed_dir = os.path.join(task_dir, seed)
            if not os.path.isdir(seed_dir):
                print(f"[WARN] Missing {seed_dir}, skipping")
                continue

            for lr in LRS:
                log_file = os.path.join(seed_dir, f"log_lr_{lr}.txt")
                if not os.path.isfile(log_file):
                    continue

                accs = []
                with open(log_file, "r") as f:
                    for line in f:
                        match = ACC_PATTERN.search(line)
                        if match:
                            accs.append(float(match.group(1)))

                if accs:
                    lr_to_accs[lr].append(max(accs))

        # -------------------------
        # Compute mean ± std
        # -------------------------
        results = {}

        for lr, accs in lr_to_accs.items():
            mean = float(np.mean(accs))
            std = float(np.std(accs))

            results[lr] = {
                "mean": mean,
                "std": std,
                "values": accs,
            }

        if not results:
            print("[WARN] No valid results found")
            continue

        # -------------------------
        # Find best LR (by mean)
        # -------------------------
        best_lr, best_stats = max(
            results.items(),
            key=lambda x: x[1]["mean"]
        )

        print(
            f"Best LR = {best_lr} | "
            f"mean={best_stats['mean']:.4f} ± {best_stats['std']:.4f}"
        )

        summary[model][task] = {
            "all_lrs": results,
            "best_lr": {
                "lr": best_lr,
                **best_stats
            }
        }

# =========================
# Save summary
# =========================
out_path = os.path.join(ROOT_DIR, "summary_best_accuracy_all.json")
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved full summary to {out_path}")

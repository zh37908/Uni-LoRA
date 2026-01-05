#!/usr/bin/env python
# coding: utf-8

"""
Collect CoLA Matthews Correlation from UniLoRA JSON results
and compute mean ± std across seeds for each learning rate.
"""

import os
import json
import numpy as np
from collections import defaultdict

# =========================
# Config
# =========================
ROOT_DIR = "results_cola"

SEEDS = ["seed_0", "seed_1", "seed_2", "seed_3", "seed_4"]

LRS = [
    "1e-04", "2e-04", "5e-04",
    "1e-03", "2e-03", "5e-03",
    "1e-02", "2e-02",
]

# =========================
# Collect results
# =========================
lr_to_mcc = defaultdict(list)

for seed in SEEDS:
    seed_dir = os.path.join(ROOT_DIR, seed)
    if not os.path.isdir(seed_dir):
        print(f"[WARN] Missing {seed_dir}, skipping")
        continue

    for lr in LRS:
        json_file = os.path.join(seed_dir, f"cola_lr_{lr}_seed_{seed.split('_')[-1]}.json")

        if not os.path.isfile(json_file):
            print(f"[WARN] Missing {json_file}, skipping")
            continue

        with open(json_file, "r") as f:
            data = json.load(f)

        mcc = data["best_metric"]["matthews_correlation"]
        lr_to_mcc[lr].append(mcc)

# =========================
# Compute mean ± std
# =========================
print("\n=== CoLA Validation Matthews Correlation (Best per seed) ===\n")

results = {}

for lr in LRS:
    mccs = lr_to_mcc.get(lr, [])
    if len(mccs) == 0:
        continue

    mean = np.mean(mccs)
    std = np.std(mccs)

    results[lr] = {
        "mean": float(mean),
        "std": float(std),
        "values": mccs,
    }

    print(
        f"lr={lr:>5} | "
        f"mean={mean:.4f} ± {std:.4f} | "
        f"values={['%.4f' % v for v in mccs]}"
    )

# =========================
# Save summary
# =========================
out_path = os.path.join(ROOT_DIR, "summary_cola_mcc.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved summary to {out_path}")


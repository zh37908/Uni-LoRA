import os
import json
import re
import numpy as np
from collections import defaultdict

ROOT_DIR = "/home/kli/unilora_peft_submit/peft/examples/sequence_classification/results_glue"

TASK2METRIC = {
    "cola": "matthews_correlation",
    "sst2": "accuracy",
    "mrpc": "accuracy",
    "qnli": "accuracy",
    "rte":  "accuracy",
    "stsb": "pearson",
}

lr_pattern = re.compile(r"_lr_([^_]+)_seed_")

def summarize_one_task(task_dir, metric_name):
    """
    task_dir: e.g. results_glue/roberta-base/cola
    """
    lr2scores = defaultdict(list)

    for seed_dir in sorted(os.listdir(task_dir)):
        seed_path = os.path.join(task_dir, seed_dir)
        if not os.path.isdir(seed_path):
            continue

        for fname in os.listdir(seed_path):
            if not fname.endswith(".json"):
                continue

            match = lr_pattern.search(fname)
            if match is None:
                continue

            lr = float(match.group(1))

            with open(os.path.join(seed_path, fname), "r") as f:
                data = json.load(f)

            score = data["best_metric"][metric_name]
            lr2scores[lr].append(score)

    results = []
    for lr, scores in lr2scores.items():
        scores = np.array(scores)
        mean = scores.mean()
        std = scores.std(ddof=1) if len(scores) > 1 else 0.0
        results.append((lr, mean, std, len(scores)))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ======================
# main
# ======================
for model in sorted(os.listdir(ROOT_DIR)):
    model_dir = os.path.join(ROOT_DIR, model)
    if not os.path.isdir(model_dir):
        continue

    print("\n" + "=" * 80)
    print(f"Model: {model}")
    print("=" * 80)

    for task, metric in TASK2METRIC.items():
        task_dir = os.path.join(model_dir, task)
        if not os.path.isdir(task_dir):
            continue

        results = summarize_one_task(task_dir, metric)
        if len(results) == 0:
            continue

        best_lr, best_mean, best_std, _ = results[0]

        print(f"\nTask: {task}  (metric: {metric})")
        print(f"{'LR':>10} | {'Mean':>8} | {'Std':>8} | {'#Seeds':>7}")
        print("-" * 45)

        for lr, mean, std, n in results:
            print(f"{lr:>10.2e} | {mean:8.4f} | {std:8.4f} | {n:7d}")

        print(f"--> Best LR: {best_lr:.2e} | {best_mean:.4f} ± {best_std:.4f}")

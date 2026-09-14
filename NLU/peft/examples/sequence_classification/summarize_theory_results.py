#!/usr/bin/env python3
"""Summarize E1/E2/E3 GLUE theory-validation JSON results.

Revised per bias_variance_glue_theory_experiment_summary.md: the theory's main
metric is DEV LOSS (the official GLUE metric stays as the practical metric).
For E1-style roots (multiple train_subset_ratio values) we additionally fit

    Delta_L(n) = L_LoRA(n) - L_Comp(n) ~= a/n - b

by least squares on (1/n, Delta_L) and report the predicted crossover n* = a/b.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path

TASK_METRIC = {
    "cola": "matthews_correlation",
    "sst2": "accuracy",
    "mrpc": "accuracy",
    "qnli": "accuracy",
    "rte": "accuracy",
    "stsb": "pearson",
}

COMP_VARIANTS = ("unilora", "unilora_rosa_snip")


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    mid = n // 2
    return xs[mid] if n % 2 else (xs[mid - 1] + xs[mid]) / 2.0


def extract_losses(data: dict):
    """best_val_loss = dev loss at the best-score eval; min_val_loss = lowest dev loss.
    Falls back to history for runs saved before these fields existed."""
    best_val_loss = data.get("best_val_loss")
    min_val_loss = data.get("min_val_loss")
    if best_val_loss is None or min_val_loss is None:
        history = data.get("history") or []
        losses = [h.get("val_loss") for h in history if h.get("val_loss") is not None]
        if losses:
            if min_val_loss is None:
                min_val_loss = float(min(losses))
            if best_val_loss is None:
                best_h = max(history, key=lambda h: h.get("score", -1e18))
                best_val_loss = best_h.get("val_loss")
    return best_val_loss, min_val_loss


def collect(root: Path):
    rows = []
    for path in sorted(root.rglob("*.json")):
        if path.name.startswith("profile_") or path.name.endswith("_theory.json"):
            continue
        try:
            data = json.load(path.open())
        except Exception:
            continue
        if not isinstance(data, dict) or "best_score" not in data:
            continue
        args = data.get("args") or {}
        task = args.get("task") or (path.parts[-5] if len(path.parts) >= 5 else None)
        variant = args.get("variant") or data.get("variant")
        if task not in TASK_METRIC:
            continue
        # Plain LoRA with a dedicated adapter lr (theory_common_lora_lr.sh) is labelled
        # separately so it never mixes with legacy runs that used the shared head lr.
        if variant == "lora":
            rank = int(args.get("rank", 4) or 4)
            if rank != 4:
                variant = f"lora_r{rank}"
            lora_lr = args.get("lora_lr")
            if lora_lr is not None and float(lora_lr) != float(args.get("head_lr", lora_lr)):
                variant = f"{variant}(lr={float(lora_lr):g})"
        best_val_loss, min_val_loss = extract_losses(data)
        rows.append(
            {
                "path": str(path),
                "task": task,
                "variant": variant,
                "seed": args.get("seed"),
                "head_lr": args.get("head_lr"),
                "subset_ratio": float(args.get("train_subset_ratio", 1.0) or 1.0),
                "subset_seed": args.get("subset_seed"),
                "label_noise_ratio": float(args.get("label_noise_ratio", 0.0) or 0.0),
                "label_noise_std": float(args.get("label_noise_std", 0.0) or 0.0),
                "score": data["best_score"],
                "best_val_loss": best_val_loss,
                "min_val_loss": min_val_loss,
                "last_train_loss": data.get("last_train_loss"),
                "actual_train_size": data.get("actual_train_size"),
                "original_train_size": data.get("original_train_size"),
            }
        )
    return rows


def fit_crossover(groups, task):
    """Least-squares fit of Delta_L = a * (1/n) - b for each compressed variant
    against LoRA, using per-p mean min_val_loss. Returns markdown lines."""
    lines = []
    # gather per (variant, p): mean loss and n
    per_vp = {}
    for (t, variant, p, eta, tau), rows in groups.items():
        if t != task or eta != 0.0 or tau != 0.0:
            continue
        # Median across seeds: robust to occasional diverged runs (LoRA at large lr
        # can diverge on some seeds, which would otherwise dominate the fit).
        loss_med = median([r["min_val_loss"] for r in rows])
        ns = [r["actual_train_size"] for r in rows if r["actual_train_size"]]
        if loss_med is None or not ns:
            continue
        per_vp[(variant, p)] = (loss_med, sum(ns) / len(ns))

    # Prefer the LoRA baseline trained with its dedicated adapter lr when available.
    lora_variants = {variant for (variant, _p) in per_vp if variant == "lora" or variant.startswith("lora(lr")}
    baseline = next((v for v in sorted(lora_variants) if v.startswith("lora(lr")), "lora")
    lora_pts = {p: v for (variant, p), v in per_vp.items() if variant == baseline}
    if len(lora_pts) < 3:
        return lines

    for comp in COMP_VARIANTS:
        pts = []
        for (variant, p), (loss, n) in per_vp.items():
            if variant != comp or p not in lora_pts:
                continue
            lora_loss, _ = lora_pts[p]
            pts.append((1.0 / n, lora_loss - loss))  # Delta_L = L_LoRA - L_Comp
        if len(pts) < 3:
            continue
        xs = [x for x, _ in pts]
        ys = [y for _, y in pts]
        n_pts = len(pts)
        mx = sum(xs) / n_pts
        my = sum(ys) / n_pts
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx <= 0:
            continue
        a = sum((x - mx) * (y - my) for x, y in pts) / sxx
        b = -(my - a * mx)  # Delta_L = a/n - b
        n_star = a / b if b > 0 and a > 0 else None
        ss_res = sum((y - (a * x - b)) ** 2 for x, y in pts)
        ss_tot = sum((y - my) ** 2 for y in ys)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        n_star_text = f"{n_star:.0f}" if n_star else "none (no positive crossover)"
        lines.append(
            f"- `{task}` / `{comp}` vs `{baseline}`: Delta_L = a/n - b with "
            f"a={a:.4g}, b={b:.4g}, R^2={r2:.3f}, predicted n* = {n_star_text} "
            f"({n_pts} p-points)"
        )
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_md", default=None)
    args = parser.parse_args()
    root = Path(args.input_root)
    rows = collect(root)
    groups = defaultdict(list)
    for row in rows:
        key = (
            row["task"],
            row["variant"],
            row["subset_ratio"],
            row["label_noise_ratio"],
            row["label_noise_std"],
        )
        groups[key].append(row)

    lines = [f"# Theory results: `{root}`", ""]
    lines.append("Main theory metric: dev loss (min over evals); score is the practical GLUE metric.")
    lines.append("")
    lines.append("| task | variant | p | eta | tau | n_runs | score mean | score median | score std | dev loss mean | dev loss median | dev loss std |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key in sorted(groups):
        task, variant, p, eta, tau = key
        rows_g = groups[key]
        sm, ss = mean_std([r["score"] for r in rows_g])
        smed = median([r["score"] for r in rows_g])
        lm, ls = mean_std([r["min_val_loss"] for r in rows_g])
        lmed = median([r["min_val_loss"] for r in rows_g])

        def f(v):
            return f"{v:.4f}" if v is not None else "-"

        lines.append(
            f"| {task} | {variant} | {p:g} | {eta:g} | {tau:g} | {len(rows_g)} | "
            f"{f(sm)} | {f(smed)} | {f(ss)} | {f(lm)} | {f(lmed)} | {f(ls)} |"
        )

    # E1 crossover fit: only meaningful when several subset ratios are present.
    tasks_with_ratios = defaultdict(set)
    for task, variant, p, eta, tau in groups:
        if eta == 0.0 and tau == 0.0:
            tasks_with_ratios[task].add(p)
    fit_lines = []
    for task, ps in sorted(tasks_with_ratios.items()):
        if len(ps) >= 3:
            fit_lines.extend(fit_crossover(groups, task))
    if fit_lines:
        lines.append("")
        lines.append("## E1 crossover fit (Delta_L_dev = a/n - b)")
        lines.append("")
        lines.extend(fit_lines)

    text = "\n".join(lines) + "\n"
    print(text)
    if args.output_md:
        os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)
        Path(args.output_md).write_text(text)


if __name__ == "__main__":
    main()

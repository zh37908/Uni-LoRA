#!/usr/bin/env python
# coding: utf-8

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize UniLoRA-DeepK GLUE JSON results.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results_variants",
        help="Directory that contains per-run json results.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="Optional CSV output path.",
    )
    return parser.parse_args()


def _safe_get(dct, *keys, default=None):
    cur = dct
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def load_rows(results_dir: Path):
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if data.get("variant") != "unilora_deepk":
            continue
        args = data.get("args", {})
        deepk_stats = data.get("deepk_stats") or {}
        deepk_finalize = data.get("deepk_finalize_info") or {}
        rows.append(
            {
                "file": path.name,
                "task": args.get("task"),
                "model_name": args.get("model_name"),
                "seed": args.get("seed"),
                "head_lr": args.get("head_lr"),
                "best_score": data.get("best_score"),
                "deepk_num_clusters_a": args.get("deepk_num_clusters_a"),
                "deepk_num_clusters_b": args.get("deepk_num_clusters_b"),
                "deepk_tau": args.get("deepk_tau"),
                "deepk_f_update_interval": args.get("deepk_f_update_interval"),
                "deepk_reg_total": _safe_get(deepk_stats, "reg_total"),
                "deepk_reg_a": _safe_get(deepk_stats, "reg_a"),
                "deepk_reg_b": _safe_get(deepk_stats, "reg_b"),
                "deepk_num_terms": _safe_get(deepk_stats, "num_terms"),
                "deepk_finalize_changed_ratio": _safe_get(deepk_finalize, "changed_ratio"),
                "deepk_finalize_export_entries": _safe_get(deepk_finalize, "export_entries"),
            }
        )
    return rows


def print_summary(rows):
    if not rows:
        print("No unilora_deepk results found.")
        return

    print(f"Found {len(rows)} unilora_deepk runs.")
    print("-" * 100)
    for row in rows:
        print(
            f"{row['file']}: task={row['task']} model={row['model_name']} seed={row['seed']} "
            f"best_score={row['best_score']:.6f} "
            f"K_A={row['deepk_num_clusters_a']} K_B={row['deepk_num_clusters_b']} tau={row['deepk_tau']}"
        )

    grouped = {}
    for row in rows:
        key = (row["task"], row["model_name"], row["head_lr"], row["deepk_num_clusters_a"], row["deepk_num_clusters_b"], row["deepk_tau"])
        grouped.setdefault(key, []).append(row["best_score"])

    print("\nAggregated (grouped by task/model/lr/KA/KB/tau):")
    for key, scores in sorted(grouped.items()):
        task, model_name, head_lr, ka, kb, tau = key
        avg = mean(scores)
        std = pstdev(scores) if len(scores) > 1 else 0.0
        print(
            f"task={task} model={model_name} lr={head_lr} KA={ka} KB={kb} tau={tau} | "
            f"mean={avg:.6f} std={std:.6f} n={len(scores)}"
        )


def write_csv(rows, output_csv: Path):
    headers = [
        "file",
        "task",
        "model_name",
        "seed",
        "head_lr",
        "best_score",
        "deepk_num_clusters_a",
        "deepk_num_clusters_b",
        "deepk_tau",
        "deepk_f_update_interval",
        "deepk_reg_total",
        "deepk_reg_a",
        "deepk_reg_b",
        "deepk_num_terms",
        "deepk_finalize_changed_ratio",
        "deepk_finalize_export_entries",
    ]
    lines = [",".join(headers)]
    for row in rows:
        values = [str(row.get(h, "")) for h in headers]
        lines.append(",".join(values))
    output_csv.write_text("\n".join(lines) + "\n")
    print(f"\nWrote CSV to: {output_csv}")


def main():
    args = parse_args()
    rows = load_rows(Path(args.results_dir))
    print_summary(rows)
    if args.output_csv:
        write_csv(rows, Path(args.output_csv))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# coding: utf-8
"""
Scan results_glue_variants_unilora_rosa_acf (or a custom root) for UniLoRA-RoSA JSON summaries,
parse method folder names (density / warmup / mask_steps / sparse_lr_mult / reset_optimizer),
and print a ranked leaderboard plus optional CSV.

Expected layout (from acf_submit_unilora_variant_unilora_rosa.sh):

    <root>/<model>/<task>/<method_dir>/seed_<k>/unilora_rosa_<task>_<model>_lr<...>_seed<k>.json

method_dir example: unilora_rosa_d0.01_w256_m8_slrm0p2_rst1
  (refined sweep): unilora_rosa_refined_d0.01_w256_m8_slrm0p2_rst1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

# Folder immediately under <task>/ must match this tail (slrm uses "p" for decimal point).
# Optional "_refined" matches acf_submit_unilora_variant_unilora_rosa_refined.sh output dirs.
METHOD_DIR_RE = re.compile(
    r"^unilora_rosa(?:_refined)?_d(?P<rosa_density>[\d.]+)_w(?P<rosa_warmup>\d+)_m(?P<rosa_mask>\d+)"
    r"_slrm(?P<slrm>[\dp]+)_rst(?P<rst>[01])$"
)

LR_SEED_RE = re.compile(r"_lr(?P<head_lr>[\d.eE+-]+)_seed(?P<seed>\d+)\.json$")


def parse_sparse_lr_mult(slrm: str) -> float:
    return float(slrm.replace("p", "."))


def parse_method_dir(name: str) -> dict[str, str | float | int] | None:
    m = METHOD_DIR_RE.match(name)
    if not m:
        return None
    d = m.groupdict()
    return {
        "rosa_density": float(d["rosa_density"]),
        "rosa_warmup_steps": int(d["rosa_warmup"]),
        "rosa_mask_steps": int(d["rosa_mask"]),
        "sparse_lr_mult": parse_sparse_lr_mult(d["slrm"]),
        "rosa_reset_optimizer_on_mask": int(d["rst"]),
    }


def parse_json_basename(basename: str) -> dict[str, str | float] | None:
    m = LR_SEED_RE.search(basename)
    if not m:
        return None
    return {"head_lr": float(m.group("head_lr")), "seed_from_file": int(m.group("seed"))}


def discover_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*.json")):
        rel = path.relative_to(root)
        parts = rel.parts
        if len(parts) < 5:
            continue
        model, task, method_dir, seed_dir, fname = parts[0], parts[1], parts[2], parts[3], parts[4]
        if not seed_dir.startswith("seed_"):
            continue
        meta = parse_method_dir(method_dir)
        if meta is None:
            continue
        file_meta = parse_json_basename(fname)
        if file_meta is None:
            continue
        try:
            seed = int(seed_dir.split("_", 1)[1])
        except (IndexError, ValueError):
            continue

        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        best_metric = data.get("best_metric") or {}
        row: dict[str, object] = {
            "model": model,
            "task": task,
            "seed": seed,
            "method_dir": method_dir,
            "json_path": str(path),
            "head_lr": file_meta["head_lr"],
            **meta,
            "best_score": data.get("best_score"),
            "best_accuracy": best_metric.get("accuracy"),
            "best_f1": best_metric.get("f1"),
            "best_epoch": data.get("best_epoch"),
            "variant": data.get("variant"),
        }
        rows.append(row)
    return rows


def print_leaderboard(rows: list[dict[str, object]], top: int | None) -> None:
    valid = [r for r in rows if r.get("best_score") is not None]
    valid.sort(key=lambda r: float(r["best_score"]), reverse=True)
    if top is not None:
        valid = valid[: top]

    headers = [
        "rank",
        "best_score",
        "acc",
        "f1",
        "task",
        "seed",
        "head_lr",
        "d",
        "w",
        "m",
        "slr_mult",
        "rst",
        "method_dir",
    ]
    lines = []
    lines.append("\t".join(headers))
    for i, r in enumerate(valid, start=1):
        lines.append(
            "\t".join(
                [
                    str(i),
                    f'{float(r["best_score"]):.4f}',
                    f'{float(r["best_accuracy"]):.4f}' if r.get("best_accuracy") is not None else "",
                    f'{float(r["best_f1"]):.4f}' if r.get("best_f1") is not None else "",
                    str(r["task"]),
                    str(r["seed"]),
                    str(r["head_lr"]),
                    str(r["rosa_density"]),
                    str(r["rosa_warmup_steps"]),
                    str(r["rosa_mask_steps"]),
                    str(r["sparse_lr_mult"]),
                    str(r["rosa_reset_optimizer_on_mask"]),
                    str(r["method_dir"]),
                ]
            )
        )
    print("\n".join(lines))


def write_csv(rows: list[dict[str, object]], csv_path: Path) -> None:
    fieldnames = [
        "rank_by_score",
        "model",
        "task",
        "seed",
        "head_lr",
        "rosa_density",
        "rosa_warmup_steps",
        "rosa_mask_steps",
        "sparse_lr_mult",
        "rosa_reset_optimizer_on_mask",
        "best_score",
        "best_accuracy",
        "best_f1",
        "best_epoch",
        "method_dir",
        "json_path",
    ]
    sorted_rows = sorted(
        (r for r in rows if r.get("best_score") is not None),
        key=lambda r: float(r["best_score"]),
        reverse=True,
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for i, r in enumerate(sorted_rows, start=1):
            out = {k: r.get(k) for k in fieldnames if k != "rank_by_score"}
            out["rank_by_score"] = i
            w.writerow(out)


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_root = script_dir / "results_glue_variants_unilora_rosa_acf"

    p = argparse.ArgumentParser(description="Summarize UniLoRA-RoSA GLUE JSON results under a results root.")
    p.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help=f"Results root (default: {default_root})",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Write ranked CSV to this path (default: <root>/unilora_rosa_summary.csv if --write-default-csv)",
    )
    p.add_argument(
        "--write-default-csv",
        action="store_true",
        help=f"Also write {default_root.name}/unilora_rosa_summary.csv under --root",
    )
    p.add_argument("--top", type=int, default=None, help="Only show top N rows in the printed leaderboard")
    p.add_argument("--quiet", action="store_true", help="Do not print leaderboard (only write CSV)")
    args = p.parse_args()

    root: Path = args.root.resolve()
    if not root.is_dir():
        print(f"Error: root is not a directory: {root}", file=sys.stderr)
        return 1

    rows = discover_rows(root)
    if not rows:
        print(f"No matching JSON files under {root}", file=sys.stderr)
        return 2

    if not args.quiet:
        print(f"Found {len(rows)} result JSON file(s) under {root}\n")
        print_leaderboard(rows, args.top)

    csv_path = args.csv
    if args.write_default_csv and csv_path is None:
        csv_path = root / "unilora_rosa_summary.csv"
    if csv_path is not None:
        write_csv(rows, csv_path)
        if not args.quiet:
            print(f"\nWrote CSV: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

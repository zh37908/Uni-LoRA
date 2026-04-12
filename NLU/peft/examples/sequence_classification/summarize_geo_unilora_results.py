#!/usr/bin/env python3
# coding: utf-8
"""
Scan results_glue_variants_geo_unilora_acf (or custom --root) for Geo-UniLoRA JSON summaries.

Expected layout (from acf_submit_unilora_variant_geo_unilora.sh):

    <root>/<model>/<task>/geo_g<...>/seed_<k>/geo_unilora_<task>_<model>_lr<...>_seed<k>_g....json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

METHOD_DIR_RE = re.compile(r"^geo_g\d+_sr[\d.pneg]+_(?P<id>prank|erank)$")
LR_SEED_RE = re.compile(r"_lr(?P<head_lr>[\d.eE+-]+)_seed(?P<seed>\d+)\.json$")


def parse_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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
        if not METHOD_DIR_RE.match(method_dir):
            continue
        m = LR_SEED_RE.search(fname)
        if not m:
            continue
        try:
            seed = int(seed_dir.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        try:
            data = parse_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        best = data.get("best_score")
        geo = data.get("geo_plan_stats") or {}
        rows.append(
            {
                "path": str(path),
                "model": model,
                "task": task,
                "method_dir": method_dir,
                "seed": seed,
                "head_lr": float(m.group("head_lr")),
                "best_score": best,
                "B_sh_actual": geo.get("B_sh_actual"),
                "B_res_actual": geo.get("B_res_actual"),
                "num_modules": geo.get("num_modules"),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Summarize Geo-UniLoRA GLUE JSON results.")
    p.add_argument(
        "--root",
        type=Path,
        default=Path("results_glue_variants_geo_unilora_acf"),
        help="Results root (default: results_glue_variants_geo_unilora_acf).",
    )
    p.add_argument("--csv", type=Path, default=None, help="Optional CSV output path.")
    args = p.parse_args(argv)

    if not args.root.is_dir():
        print(f"Root not found: {args.root}", file=sys.stderr)
        return 1

    rows = discover_rows(args.root)
    if not rows:
        print("No matching JSON files found.")
        return 0

    rows.sort(key=lambda r: (str(r["task"]), str(r["model"]), -float(r["best_score"] or 0.0)))

    for r in rows:
        print(
            f"{r['task']:6} {r['model']:16} seed={r['seed']} head_lr={r['head_lr']} "
            f"best={r['best_score']} B_sh={r['B_sh_actual']} B_res={r['B_res_actual']} "
            f"mods={r['num_modules']}"
        )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
# coding: utf-8

"""Compact paper figures for the Transformer teacher--student experiment.

Reads conditional summaries for separate fixed teachers from
``results_transformer_lab_level2_v2_full/summary.csv`` and produces:

  transformer_risk_vs_n.pdf   population excess risk vs. n (both profiles)
  transformer_biasvar.pdf     function-space bias--variance decomposition
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METHODS = ("lora", "unilora_matched", "prolosa", "prolosa_random", "oracle")
LABELS = {
    "full_ft": "Full FT",
    "lora": "LoRA",
    "unilora": "Compressed ($d$)",
    "unilora_matched": "Compressed ($B=d+K$)",
    "prolosa": "ProLoSA",
    "prolosa_random": "Random support",
    "oracle": "Oracle support",
}
# Match the method palette in synthetic_matched_budget_baselines.py (Fig. 2).
COLORS = {
    "full_ft": "#6b7280",
    "lora": "#2563eb",
    "unilora": "#f97316",
    "unilora_matched": "#b45309",
    "prolosa": "#16a34a",
    "prolosa_random": "#9333ea",
    "oracle": "#0f766e",
}
# Match the decomposition palette in the sequence-model figures (Figs. 3–4).
COMPONENT_COLORS = {"bias": "#2f6fbb", "variance": "#f2b84b"}
LEGEND_FONTSIZE = 8
GRID_ALPHA = 0.25

STYLES = {
    "full_ft": ":",
    "lora": "-",
    "unilora": ":",
    "unilora_matched": "-",
    "prolosa": "-",
    "prolosa_random": "--",
    "oracle": "-.",
}
PROFILE_TITLES = {
    "sparse_heavytail": "Concentrated target",
    "dense_gaussian": "Diffuse target",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def fnum(row: dict[str, str], key: str) -> float:
    return float(row[key])


def risk_figure(rows: list[dict[str, str]], noise: float, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
    for panel, ax, profile in zip("AB", axes, ("sparse_heavytail", "dense_gaussian")):
        for method in METHODS:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["profile"] == profile
                    and row["method"] == method
                    and abs(fnum(row, "noise_std") - noise) < 1e-9
                    and row["risk_mean"] not in ("", "nan")
                ),
                key=lambda row: int(row["n"]),
            )
            ns = [int(row["n"]) for row in selected]
            risks = [fnum(row, "risk_mean") for row in selected]
            errs = [fnum(row, "risk_std") / np.sqrt(fnum(row, "n_seeds")) for row in selected]
            ax.errorbar(
                ns,
                risks,
                yerr=errs,
                marker="o",
                markersize=3.5,
                linewidth=1.8,
                capsize=2.5,
                color=COLORS[method],
                linestyle=STYLES[method],
                label=LABELS[method],
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("training-set size $n$")
        ax.set_title(
            rf"{panel}. {PROFILE_TITLES[profile]}  ($\sigma_y={noise:g}$)",
            loc="left",
            fontweight="bold",
        )
        ax.grid(alpha=GRID_ALPHA)
    axes[0].set_ylabel("Excess prediction MSE")
    axes[0].legend(frameon=False, fontsize=LEGEND_FONTSIZE)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def biasvar_figure(rows: list[dict[str, str]], noise: float, out: Path) -> None:
    cells = [
        ("sparse_heavytail", 64),
        ("sparse_heavytail", 4096),
        ("dense_gaussian", 64),
        ("dense_gaussian", 4096),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.4))
    methods = ("lora",) + METHODS[1:]
    for panel, ax, (profile, n) in zip("ABCD", axes.flat, cells):
        bias, variance = [], []
        for method in methods:
            row = next(
                row
                for row in rows
                if row["profile"] == profile
                and row["method"] == method
                and int(row["n"]) == n
                and abs(fnum(row, "noise_std") - noise) < 1e-9
            )
            bias.append(fnum(row, "bias_fn"))
            variance.append(fnum(row, "variance_fn"))
        x = np.arange(len(methods))
        ax.bar(x, bias, color=COMPONENT_COLORS["bias"], label="Conditional squared bias")
        ax.bar(x, variance, bottom=bias, color=COMPONENT_COLORS["variance"], label="Within-teacher variance")
        for xi, (b, v) in enumerate(zip(bias, variance)):
            ax.annotate(
                f"{b + v:.3g}",
                xy=(xi, b + v),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[m] for m in methods], rotation=20, ha="right", fontsize=8)
        ax.set_title(
            rf"{panel}. {PROFILE_TITLES[profile]}, $n={n}$, $\sigma_y={noise:g}$",
            loc="left",
            fontweight="bold",
            fontsize=10,
        )
        ax.set_ylim(0.0, max(float(np.max(np.array(bias) + variance)) * 1.12, 1e-12))
        ax.grid(alpha=GRID_ALPHA, axis="y")
    axes[0, 0].set_ylabel("Prediction-MSE components")
    axes[1, 0].set_ylabel("Prediction-MSE components")
    axes[0, 0].legend(frameon=False, fontsize=LEGEND_FONTSIZE)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results_transformer_lab_level2_v2_full/summary.csv"),
    )
    parser.add_argument("--risk-noise", type=float, default=0.5)
    parser.add_argument("--biasvar-noise", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("figure_summaries"))
    args = parser.parse_args()
    rows = read_rows(args.results)
    # Never pool profiles, n/noise cells, or the separate calibration experiment.
    keys = [(r["profile"], int(r["n"]), float(r["noise_std"]), r["method"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate conditional cells: do not merge bias/variance across targets or shards")
    for r in rows:
        if not np.isclose(float(r["bias_fn"])+float(r["variance_fn"]), float(r["risk_mean"]), rtol=1e-9, atol=1e-12):
            raise ValueError("Conditional bias + variance differs from cell risk")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    risk_figure(rows, args.risk_noise, args.output_dir / "transformer_risk_vs_n")
    biasvar_figure(rows, args.biasvar_noise, args.output_dir / "transformer_biasvar")
    print(f"Saved transformer paper figures to {args.output_dir}")


if __name__ == "__main__":
    main()

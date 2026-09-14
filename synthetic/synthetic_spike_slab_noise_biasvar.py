#!/usr/bin/env python
# coding: utf-8

"""
Compare LoRA, Uni-LoRA, and ProLoSA under spike-and-slab source distributions.

    theta_j ~ (1 - pi) N(0, tau_0^2) + pi t_nu(0, s^2),  tau_0 << s.

The Student-t slab is rescaled so ``s`` is its actual standard deviation.
Several default (tau_0, pi, s) groups vary sparsity, spike width, and slab
strength. Custom groups use ``NAME:TAU0:PI:SLAB_STD``.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from synthetic_contaminated_gaussian_noise_biasvar import (
    Config as MixtureConfig,
    run_experiment,
)
from synthetic_powerlaw_noise_biasvar import method_label


METHODS = ("lora", "unilora", "prolosa")
COLORS = {"lora": "#2563eb", "unilora": "#f97316", "prolosa": "#16a34a"}
DEFAULT_GROUPS = [
    "very_sparse:0.005:0.005:0.5",
    "budget_matched:0.005:0.01:0.5",
    "moderately_sparse:0.005:0.02:0.5",
    "wider_spike:0.02:0.01:0.5",
    "strong_slab:0.005:0.01:1.0",
    "dense_slab:0.005:0.05:0.5",
]


@dataclass(frozen=True)
class Group:
    name: str
    spike_std: float
    slab_probability: float
    slab_std: float


@dataclass(frozen=True)
class SweepConfig:
    D: int
    d: int
    n_eff: int
    sparse_budget: int
    slab_df: float
    noise_stds: list[float]
    trials: int
    support: str
    ridge: float
    seed: int
    groups: list[Group]
    output_dir: Path


def parse_group(value: str) -> Group:
    fields = value.split(":")
    if len(fields) != 4:
        raise ValueError(
            f"Invalid group {value!r}; expected NAME:TAU0:PI:SLAB_STD."
        )
    name, spike_std, slab_probability, slab_std = fields
    group = Group(name, float(spike_std), float(slab_probability), float(slab_std))
    if not name:
        raise ValueError("Group name cannot be empty.")
    if group.spike_std <= 0 or group.slab_std <= 0:
        raise ValueError(f"Group standard deviations must be positive: {group}.")
    if not 0 <= group.slab_probability <= 1:
        raise ValueError(f"Slab probability must lie in [0, 1]: {group}.")
    if group.spike_std >= group.slab_std:
        raise ValueError(f"Require spike_std < slab_std: {group}.")
    return group


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_sweep(
    config: SweepConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []
    for group_index, group in enumerate(config.groups):
        mixture_config = MixtureConfig(
            D=config.D,
            d=config.d,
            n_eff=config.n_eff,
            sparse_budget=config.sparse_budget,
            source_std=group.spike_std,
            contamination_rates=[group.slab_probability],
            outlier_std_multiplier=group.slab_std / group.spike_std,
            outlier_df=config.slab_df,
            noise_stds=config.noise_stds,
            trials=config.trials,
            support=config.support,
            ridge=config.ridge,
            seed=config.seed + 1009 * group_index,
            output_dir=config.output_dir,
        )
        group_rows, group_metadata = run_experiment(mixture_config)
        for row in group_rows:
            rows.append(
                {
                    "group": group.name,
                    "spike_std": group.spike_std,
                    "slab_probability": group.slab_probability,
                    "slab_std": group.slab_std,
                    "slab_df": config.slab_df,
                    "noise_std": float(row["noise_std"]),
                    "method": str(row["method"]),
                    "risk_mean": float(row["risk_mean"]),
                    "risk_std": float(row["risk_std"]),
                    "bias": float(row["bias"]),
                    "variance": float(row["variance"]),
                    "outlier_recall": float(row["outlier_recall"]),
                    "realized_slab_count": int(row["realized_outlier_count"]),
                    "realized_source_std": float(row["realized_source_std"]),
                }
            )
        item = group_metadata[0]
        metadata_rows.append(
            {
                "group": group.name,
                "spike_std": group.spike_std,
                "slab_probability": group.slab_probability,
                "slab_std": group.slab_std,
                "realized_slab_count": int(item["realized_outlier_count"]),
                "realized_slab_probability": float(
                    item["realized_contamination_rate"]
                ),
                "realized_source_std": float(item["realized_source_std"]),
                "theta_norm": float(item["theta_norm"]),
                "top_k_energy_fraction": float(item["top_k_energy_fraction"]),
                "top_k_slab_fraction": float(item["top_k_outlier_fraction"]),
            }
        )
    return rows, metadata_rows


def plot_results(config: SweepConfig, rows: list[dict[str, object]]) -> None:
    plt = importlib.import_module("matplotlib.pyplot")
    column_count = 3
    row_count = int(np.ceil(len(config.groups) / column_count))
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(15, 4.4 * row_count),
        squeeze=False,
    )
    for ax, group in zip(axes.flat, config.groups):
        group_rows = [row for row in rows if row["group"] == group.name]
        for method in METHODS:
            method_rows = sorted(
                (row for row in group_rows if row["method"] == method),
                key=lambda row: float(row["noise_std"]),
            )
            ax.errorbar(
                [float(row["noise_std"]) for row in method_rows],
                [float(row["risk_mean"]) for row in method_rows],
                yerr=[float(row["risk_std"]) for row in method_rows],
                marker="o",
                linewidth=2,
                capsize=3,
                color=COLORS[method],
                label=method_label(method),
            )
        ax.set_title(
            (
                f"{group.name}\n"
                rf"$\tau_0={group.spike_std:g},\ \pi={group.slab_probability:g},"
                rf"\ s={group.slab_std:g}$"
            ),
            fontweight="bold",
        )
        ax.set_xlabel(r"regression noise $\sigma_y$")
        ax.set_ylabel("Population excess risk")
        ax.grid(alpha=0.25)
    for ax in axes.flat[len(config.groups) :]:
        ax.set_visible(False)
    axes.flat[min(len(config.groups), len(axes.flat)) - 1].legend(frameon=False)
    fig.suptitle("Spike-and-slab three-method comparison", fontweight="bold")
    fig.tight_layout()
    fig.savefig(config.output_dir / "spike_slab_comparison.png", dpi=240)
    fig.savefig(config.output_dir / "spike_slab_comparison.pdf")
    plt.close(fig)


def write_summary(
    path: Path,
    config: SweepConfig,
    rows: list[dict[str, object]],
    metadata_rows: list[dict[str, object]],
) -> None:
    lines = [
        "# Spike-and-slab synthetic validation",
        "",
        r"$\theta_j\sim(1-\pi)N(0,\tau_0^2)+\pi t_\nu(0,s^2)$.",
        "",
        (
            f"Common settings: D={config.D}, d={config.d}, K={config.sparse_budget}, "
            f"n_eff={config.n_eff}, slab df={config.slab_df:g}, "
            f"trials={config.trials}, support={config.support}."
        ),
        "",
        "## Realized parameter groups",
        "",
        "| group | tau0 | pi | s | slabs | realized pi | realized std | top-K energy | top-K slab fraction |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in metadata_rows:
        lines.append(
            f"| {item['group']} | {float(item['spike_std']):g} | "
            f"{float(item['slab_probability']):g} | {float(item['slab_std']):g} | "
            f"{int(item['realized_slab_count'])} | "
            f"{float(item['realized_slab_probability']):.4f} | "
            f"{float(item['realized_source_std']):.6g} | "
            f"{float(item['top_k_energy_fraction']):.4f} | "
            f"{float(item['top_k_slab_fraction']):.4f} |"
        )

    win_counts = {method: 0 for method in METHODS}
    for group in config.groups:
        lines.extend(
            [
                "",
                f"## {group.name}",
                "",
                "| noise | LoRA risk | Uni-LoRA risk | ProLoSA risk | winner | ProLoSA bias | ProLoSA variance | slab recall |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        group_rows = [row for row in rows if row["group"] == group.name]
        for noise_std in config.noise_stds:
            by_method = {
                str(row["method"]): row
                for row in group_rows
                if float(row["noise_std"]) == noise_std
            }
            winner = min(
                METHODS,
                key=lambda method: float(by_method[method]["risk_mean"]),
            )
            win_counts[winner] += 1
            pro = by_method["prolosa"]
            lines.append(
                f"| {noise_std:g} | "
                f"{float(by_method['lora']['risk_mean']):.6g} | "
                f"{float(by_method['unilora']['risk_mean']):.6g} | "
                f"{float(pro['risk_mean']):.6g} | {method_label(winner)} | "
                f"{float(pro['bias']):.6g} | {float(pro['variance']):.6g} | "
                f"{float(pro['outlier_recall']):.4f} |"
            )
    lines.extend(
        [
            "",
            "## Win counts",
            "",
            *[
                f"- {method_label(method)}: {win_counts[method]}/{len(config.groups) * len(config.noise_stds)}"
                for method in METHODS
            ],
            "",
            "Full bias, variance, and error bars are in `spike_slab_results.csv`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--D", type=int, default=4096)
    parser.add_argument("--d", type=int, default=16)
    parser.add_argument("--n-eff", type=int, default=512)
    parser.add_argument("--sparse-budget", type=int, default=32)
    parser.add_argument("--slab-df", type=float, default=3.0)
    parser.add_argument(
        "--noise-stds",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 1.5, 2.0],
    )
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument(
        "--support",
        choices=["oracle", "noisy_residual", "random"],
        default="noisy_residual",
    )
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--groups",
        nargs="+",
        default=DEFAULT_GROUPS,
        metavar="NAME:TAU0:PI:SLAB_STD",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results_spike_slab_synthetic_theory"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    groups = [parse_group(value) for value in args.groups]
    if len({group.name for group in groups}) != len(groups):
        raise ValueError("Spike-and-slab group names must be unique.")
    config = SweepConfig(
        D=args.D,
        d=args.d,
        n_eff=args.n_eff,
        sparse_budget=args.sparse_budget,
        slab_df=args.slab_df,
        noise_stds=args.noise_stds,
        trials=args.trials,
        support=args.support,
        ridge=args.ridge,
        seed=args.seed,
        groups=groups,
        output_dir=args.output_dir,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows, metadata_rows = run_sweep(config)
    write_csv(config.output_dir / "spike_slab_results.csv", rows)
    write_csv(config.output_dir / "spike_slab_parameter_groups.csv", metadata_rows)
    with (config.output_dir / "config.json").open("w") as handle:
        json.dump(
            {
                **asdict(config),
                "output_dir": str(config.output_dir),
            },
            handle,
            indent=2,
        )
    plot_results(config, rows)
    write_summary(config.output_dir / "summary.md", config, rows, metadata_rows)
    print(f"Saved spike-and-slab results to {config.output_dir}")


if __name__ == "__main__":
    main()

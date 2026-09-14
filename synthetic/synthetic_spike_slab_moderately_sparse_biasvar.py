#!/usr/bin/env python
# coding: utf-8

"""Paper-style bias/variance figure for one moderately sparse spike-and-slab model."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from synthetic_powerlaw_noise_biasvar import method_label
from synthetic_spike_slab_noise_biasvar import Group, SweepConfig, run_sweep


METHODS = ("lora", "unilora", "prolosa")


def select_noise_rows(
    rows: list[dict[str, object]],
    target_noise: float,
) -> tuple[float, dict[str, dict[str, object]]]:
    noise_stds = sorted({float(row["noise_std"]) for row in rows})
    selected_noise = min(noise_stds, key=lambda value: abs(value - target_noise))
    selected = {
        str(row["method"]): row
        for row in rows
        if math.isclose(
            float(row["noise_std"]),
            selected_noise,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    }
    return selected_noise, selected


def plot_paper_style(
    config: SweepConfig,
    rows: list[dict[str, object]],
    middle_noise: float,
    high_noise: float,
) -> None:
    plt = importlib.import_module("matplotlib.pyplot")
    colors = {"lora": "#2563eb", "unilora": "#f97316", "prolosa": "#16a34a"}
    component_colors = {"bias": "#2f6fbb", "variance": "#f2b84b"}
    middle_selected, middle_rows = select_noise_rows(rows, middle_noise)
    high_selected, high_rows = select_noise_rows(rows, high_noise)

    def totals(by_method: dict[str, dict[str, object]]) -> np.ndarray:
        return np.array(
            [
                float(by_method[method]["bias"])
                + float(by_method[method]["variance"])
                for method in METHODS
            ]
        )

    max_total = float(max(np.max(totals(middle_rows)), np.max(totals(high_rows))))
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(14.2, 4.6))
    for method in METHODS:
        method_rows = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: float(row["noise_std"]),
        )
        ax_a.errorbar(
            [float(row["noise_std"]) for row in method_rows],
            [float(row["risk_mean"]) for row in method_rows],
            yerr=[float(row["risk_std"]) for row in method_rows],
            color=colors[method],
            marker="o",
            markersize=4,
            linewidth=2.2,
            capsize=2,
            label=method_label(method),
        )
    group = config.groups[0]
    ax_a.set_title("A. Risk vs. Noise Level", loc="left", fontweight="bold")
    display_name = group.name.replace("_", " ")
    fig.suptitle(
        (
            rf"{display_name.title()} spike-and-slab: "
            rf"$\tau_0={group.spike_std:g},\ \pi={group.slab_probability:g},"
            rf"\ s={group.slab_std:g}$"
        ),
        fontweight="bold",
    )
    ax_a.set_xlabel(r"noise standard deviation $\sigma_y$")
    ax_a.set_ylabel("Population excess risk")
    ax_a.grid(alpha=0.25)
    ax_a.legend(frameon=False)

    def plot_decomposition(
        ax: Any,
        by_method: dict[str, dict[str, object]],
        selected_noise: float,
        prefix: str,
        show_ylabel: bool,
        show_legend: bool,
    ) -> None:
        x = np.arange(len(METHODS))
        bias = [float(by_method[method]["bias"]) for method in METHODS]
        variance = [float(by_method[method]["variance"]) for method in METHODS]
        component_totals = np.array(bias) + np.array(variance)
        ax.bar(x, bias, color=component_colors["bias"], label="Bias")
        ax.bar(
            x,
            variance,
            bottom=bias,
            color=component_colors["variance"],
            label="Variance",
        )
        for index, total in enumerate(component_totals):
            ax.text(index, total, f"{total:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(
            rf"{prefix}. Decomposition ($\sigma_y={selected_noise:g}$)",
            loc="left",
            fontweight="bold",
        )
        ax.set_xticks(x, [method_label(method) for method in METHODS], rotation=25)
        if show_ylabel:
            ax.set_ylabel("Excess risk components")
        ax.set_ylim(0.0, max(max_total * 1.12, 1e-12))
        ax.grid(axis="y", alpha=0.25)
        if show_legend:
            ax.legend(frameon=False)

    plot_decomposition(ax_b, middle_rows, middle_selected, "B", True, True)
    plot_decomposition(ax_c, high_rows, high_selected, "C", False, False)
    fig.tight_layout()
    fig.savefig(config.output_dir / "spike_slab_noise_biasvar.png", dpi=240)
    fig.savefig(config.output_dir / "spike_slab_noise_biasvar.pdf")
    plt.close(fig)


def write_outputs(
    config: SweepConfig,
    rows: list[dict[str, object]],
    metadata: list[dict[str, object]],
    middle_noise: float,
    high_noise: float,
) -> None:
    with (config.output_dir / "spike_slab_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (config.output_dir / "config.json").open("w") as handle:
        json.dump(
            {
                **asdict(config),
                "output_dir": str(config.output_dir),
                "realized_distribution": metadata[0],
            },
            handle,
            indent=2,
        )

    group = config.groups[0]
    distribution = metadata[0]
    lines = [
        f"# {group.name.replace('_', ' ').title()} spike-and-slab validation",
        "",
        (
            rf"$\theta_j\sim(1-\pi)N(0,\tau_0^2)+\pi t_\nu(0,s^2)$, "
            rf"$\tau_0={group.spike_std:g}$, "
            rf"$\pi={group.slab_probability:g}$, $s={group.slab_std:g}$."
        ),
        "",
        f"- realized slab count: {int(distribution['realized_slab_count'])}",
        f"- realized slab probability: {float(distribution['realized_slab_probability']):.4f}",
        f"- top-K energy fraction: {float(distribution['top_k_energy_fraction']):.4f}",
        f"- trials: {config.trials}",
    ]
    for title, target in (("Middle noise", middle_noise), ("High noise", high_noise)):
        selected_noise, by_method = select_noise_rows(rows, target)
        lines.extend(
            [
                "",
                f"## {title}: sigma_y={selected_noise:g}",
                "",
                "| method | risk | bias | variance |",
                "|:---|---:|---:|---:|",
            ]
        )
        for method in METHODS:
            row = by_method[method]
            lines.append(
                f"| {method_label(method)} | {float(row['risk_mean']):.6g} | "
                f"{float(row['bias']):.6g} | {float(row['variance']):.6g} |"
            )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `spike_slab_results.csv`",
            "- `spike_slab_noise_biasvar.png`",
            "- `spike_slab_noise_biasvar.pdf`",
        ]
    )
    (config.output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-name", default="moderately_sparse")
    parser.add_argument("--spike-std", type=float, default=0.005)
    parser.add_argument("--slab-probability", type=float, default=0.02)
    parser.add_argument("--slab-std", type=float, default=0.5)
    parser.add_argument("--slab-df", type=float, default=3.0)
    parser.add_argument("--middle-noise", type=float, default=0.5)
    parser.add_argument("--high-noise", type=float, default=1.5)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results_spike_slab_moderately_sparse_theory"),
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config = SweepConfig(
        D=4096,
        d=16,
        n_eff=512,
        sparse_budget=32,
        slab_df=args.slab_df,
        noise_stds=[round(float(value), 10) for value in np.linspace(0.1, 2.0, 20)],
        trials=args.trials,
        support="noisy_residual",
        ridge=0.0,
        seed=args.seed,
        groups=[
            Group(
                args.group_name,
                args.spike_std,
                args.slab_probability,
                args.slab_std,
            )
        ],
        output_dir=output_dir,
    )
    rows, metadata = run_sweep(config)
    plot_paper_style(config, rows, args.middle_noise, args.high_noise)
    write_outputs(config, rows, metadata, args.middle_noise, args.high_noise)
    print(f"Saved paper-style spike-and-slab results to {output_dir}")


if __name__ == "__main__":
    main()

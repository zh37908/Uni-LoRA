#!/usr/bin/env python
"""Compare completed Gaussian experiments with different source standard deviations."""

from __future__ import annotations

import argparse
import csv
import importlib
from pathlib import Path

import numpy as np


METHODS = ("lora", "unilora", "prolosa")
LABELS = {"lora": "LoRA", "unilora": "Uni-LoRA", "prolosa": "ProLoSA"}
COLORS = {"lora": "#2563eb", "unilora": "#f97316", "prolosa": "#16a34a"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-stds",
        nargs="+",
        type=float,
        default=[0.03125, 0.0625, 0.125],
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results_gaussian_source_std_comparison"),
    )
    args = parser.parse_args()

    all_rows: dict[float, list[dict[str, str]]] = {}
    summaries: list[dict[str, object]] = []
    for source_std in args.source_stds:
        csv_path = (
            args.root
            / f"results_gaussian_synthetic_theory_std_{source_std:g}"
            / "gaussian_results.csv"
        )
        rows = read_rows(csv_path)
        all_rows[source_std] = rows

        prolosa_rows = [row for row in rows if row["method"] == "prolosa"]
        risks_by_noise = {
            float(row["noise_std"]): {
                method: float(
                    next(
                        candidate["risk_mean"]
                        for candidate in rows
                        if candidate["method"] == method
                        and candidate["noise_std"] == row["noise_std"]
                    )
                )
                for method in METHODS
            }
            for row in prolosa_rows
        }
        win_count = sum(
            values["prolosa"] == min(values.values())
            for values in risks_by_noise.values()
        )
        gaps = {
            noise: values["prolosa"] - min(values["lora"], values["unilora"])
            for noise, values in risks_by_noise.items()
        }
        closest_noise = min(gaps, key=gaps.get)
        summaries.append(
            {
                "source_std": source_std,
                "realized_source_std": float(rows[0]["realized_source_std"]),
                "mean_prolosa_risk": float(
                    np.mean([float(row["risk_mean"]) for row in prolosa_rows])
                ),
                "prolosa_win_count": win_count,
                "noise_point_count": len(prolosa_rows),
                "closest_noise_std": closest_noise,
                "closest_gap_to_best_baseline": gaps[closest_noise],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "source_std_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    plt = importlib.import_module("matplotlib.pyplot")
    fig, axes = plt.subplots(1, len(args.source_stds), figsize=(5 * len(args.source_stds), 4.5))
    axes = np.atleast_1d(axes)
    for ax, source_std in zip(axes, args.source_stds):
        rows = all_rows[source_std]
        for method in METHODS:
            method_rows = sorted(
                (row for row in rows if row["method"] == method),
                key=lambda row: float(row["noise_std"]),
            )
            ax.plot(
                [float(row["noise_std"]) for row in method_rows],
                [float(row["risk_mean"]) for row in method_rows],
                marker="o",
                markersize=3,
                linewidth=2,
                color=COLORS[method],
                label=LABELS[method],
            )
        ax.set_title(rf"$\sigma_\theta={source_std:g}$", fontweight="bold")
        ax.set_xlabel(r"regression noise $\sigma_y$")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Population excess risk")
    axes[-1].legend(frameon=False)
    fig.suptitle("Gaussian source-standard-deviation comparison", fontweight="bold")
    fig.tight_layout()
    fig.savefig(args.output_dir / "gaussian_source_std_comparison.png", dpi=240)
    fig.savefig(args.output_dir / "gaussian_source_std_comparison.pdf")
    plt.close(fig)

    best_absolute = min(summaries, key=lambda row: float(row["mean_prolosa_risk"]))
    best_competitive = min(
        summaries,
        key=lambda row: float(row["closest_gap_to_best_baseline"]),
    )
    lines = [
        "# Gaussian source-standard-deviation comparison",
        "",
        "| source std | realized std | mean ProLoSA risk | ProLoSA wins | closest noise | closest gap |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {float(row['source_std']):g} | {float(row['realized_source_std']):.6g} | "
            f"{float(row['mean_prolosa_risk']):.6g} | "
            f"{int(row['prolosa_win_count'])}/{int(row['noise_point_count'])} | "
            f"{float(row['closest_noise_std']):g} | "
            f"{float(row['closest_gap_to_best_baseline']):.6g} |"
        )
    lines.extend(
        [
            "",
            f"- Lowest absolute mean ProLoSA risk: source_std={float(best_absolute['source_std']):g}.",
            (
                "- Best competitive point: "
                f"source_std={float(best_competitive['source_std']):g}, "
                f"sigma_y={float(best_competitive['closest_noise_std']):g}, "
                f"gap={float(best_competitive['closest_gap_to_best_baseline']):.6g}."
            ),
            "- A negative closest gap means ProLoSA beats both baselines at that noise level.",
        ]
    )
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Saved source-std comparison to {args.output_dir}")


if __name__ == "__main__":
    main()

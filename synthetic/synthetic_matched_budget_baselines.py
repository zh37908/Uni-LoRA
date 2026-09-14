#!/usr/bin/env python
# coding: utf-8

"""
Matched-budget baselines for the projection-blind synthetic validation.

Reproduces the exact spike-and-slab target, projection matrix, and per-trial
observation noise of the ``budget_matched`` run in
``results_spike_slab_budget_matched_theory`` (seed 13), and adds the
matched-budget comparisons required by Proposition 2:

    lora              full-space estimator (D parameters)
    unilora           compressed-only, d = 16
    unilora_matched   compressed-only with the full budget B = d + K = 48
    prolosa           hybrid d + K, support = top-K of |y - PP^T y| (SNIP analog)
    prolosa_random    hybrid d + K, support drawn uniformly at random
    prolosa_oracle    hybrid d + K, support = top-K of |(I - PP^T) theta*|

The target and projections are fixed across all noise levels and trials.
The random-hybrid equality in Proposition 2 additionally averages over targets;
a fixed target need not give an exact tie. Each decomposition below is
conditional on the single target, and includes support randomness when present.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from synthetic_contaminated_gaussian_noise_biasvar import (
    make_contaminated_gaussian_theta,
)
from synthetic_powerlaw_noise_biasvar import (
    exact_excess_risk,
    orthonormal_projection,
    project_onto_basis,
    project_prolosa,
    stable_seed,
    summarize_estimates,
)

FloatArray = NDArray[np.float64]

METHODS = (
    "lora",
    "unilora",
    "unilora_matched",
    "prolosa",
    "prolosa_random",
    "prolosa_oracle",
)
METHOD_LABELS = {
    "lora": "LoRA (full space)",
    "unilora": "Compressed-only ($d=16$)",
    "unilora_matched": "Compressed-only ($B=48$)",
    "prolosa": "Hybrid, SNIP support",
    "prolosa_random": "Hybrid, random support",
    "prolosa_oracle": "Hybrid, oracle support",
}
METHOD_COLORS = {
    "lora": "#2563eb",
    "unilora": "#f97316",
    "unilora_matched": "#b45309",
    "prolosa": "#16a34a",
    "prolosa_random": "#9333ea",
    "prolosa_oracle": "#0f766e",
}
METHOD_STYLES = {
    "lora": "-",
    "unilora": ":",
    "unilora_matched": "-",
    "prolosa": "-",
    "prolosa_random": "--",
    "prolosa_oracle": "-.",
}


@dataclass(frozen=True)
class Config:
    D: int
    d: int
    sparse_budget: int
    n_eff: int
    spike_std: float
    slab_probability: float
    slab_std: float
    slab_df: float
    noise_stds: list[float]
    trials: int
    seed: int
    output_dir: Path


def run(config: Config) -> tuple[list[dict[str, object]], dict[str, object]]:
    # Reproduce the projection and target of the original budget_matched run.
    projection_rng = np.random.default_rng(
        stable_seed(config.seed, "contaminated_projection")
    )
    P = orthonormal_projection(config.D, config.d, projection_rng)
    problem_rng = np.random.default_rng(
        stable_seed(config.seed, "contaminated_problem", config.slab_probability)
    )
    theta_star, outlier_mask = make_contaminated_gaussian_theta(
        config.D,
        config.spike_std,
        config.slab_probability,
        config.slab_std / config.spike_std,
        config.slab_df,
        problem_rng,
    )

    budget = config.d + config.sparse_budget
    matched_rng = np.random.default_rng(
        stable_seed(config.seed, "matched_projection", budget)
    )
    P_matched = orthonormal_projection(config.D, budget, matched_rng)

    residual = theta_star - P @ (P.T @ theta_star)
    oracle_support = np.sort(
        np.argpartition(np.abs(residual), -config.sparse_budget)[
            -config.sparse_budget :
        ]
    )

    v_hat = float(theta_star @ theta_star) / config.D
    metadata: dict[str, object] = {
        "realized_slab_count": int(np.sum(outlier_mask)),
        "theta_norm": float(np.linalg.norm(theta_star)),
        "v_hat": v_hat,
        "off_subspace_energy": float(residual @ residual),
        "oracle_topk_energy_fraction": float(
            np.sum(residual[oracle_support] ** 2) / (residual @ residual)
        ),
        "budget": budget,
    }

    rows: list[dict[str, object]] = []
    for noise_std in config.noise_stds:
        estimates: dict[str, list[FloatArray]] = {m: [] for m in METHODS}
        risks: dict[str, list[float]] = {m: [] for m in METHODS}
        for trial in range(config.trials):
            trial_rng = np.random.default_rng(
                stable_seed(
                    config.seed,
                    "contaminated_trial",
                    config.slab_probability,
                    noise_std,
                    trial,
                )
            )
            noise = (noise_std / math.sqrt(config.n_eff)) * trial_rng.normal(
                size=config.D
            )
            y = theta_star + noise

            theta_unilora = project_onto_basis(y, P, 0.0)
            snip_support = np.sort(
                np.argpartition(np.abs(y - theta_unilora), -config.sparse_budget)[
                    -config.sparse_budget :
                ]
            )
            random_rng = np.random.default_rng(
                stable_seed(config.seed, "random_support", noise_std, trial)
            )
            random_support = np.sort(
                random_rng.choice(config.D, size=config.sparse_budget, replace=False)
            )

            trial_estimates = {
                "lora": y,
                "unilora": theta_unilora,
                "unilora_matched": project_onto_basis(y, P_matched, 0.0),
                "prolosa": project_prolosa(y, P, snip_support, 0.0),
                "prolosa_random": project_prolosa(y, P, random_support, 0.0),
                "prolosa_oracle": project_prolosa(y, P, oracle_support, 0.0),
            }
            for method, theta_hat in trial_estimates.items():
                estimates[method].append(theta_hat)
                risks[method].append(exact_excess_risk(theta_hat, theta_star))

        for method in METHODS:
            risk_mean, risk_std, bias, variance = summarize_estimates(
                estimates[method], risks[method], theta_star
            )
            rows.append(
                {
                    "noise_std": noise_std,
                    "method": method,
                    "risk_mean": risk_mean,
                    "risk_std": risk_std,
                    "bias": bias,
                    "variance": variance,
                }
            )
    return rows, metadata


def plot_results(
    config: Config, rows: list[dict[str, object]], v_hat: float
) -> None:
    plt = importlib.import_module("matplotlib.pyplot")
    fig, (ax, ax_zoom) = plt.subplots(1, 2, figsize=(10.2, 4.0))
    for panel, methods in (
        (ax, METHODS),
        (ax_zoom, ("unilora_matched", "prolosa", "prolosa_random", "prolosa_oracle")),
    ):
        for method in methods:
            method_rows = sorted(
                (row for row in rows if row["method"] == method),
                key=lambda row: float(row["noise_std"]),
            )
            panel.errorbar(
                [float(row["noise_std"]) for row in method_rows],
                [float(row["risk_mean"]) for row in method_rows],
                yerr=[float(row["risk_std"]) for row in method_rows],
                marker="o",
                markersize=3.5,
                linewidth=1.8,
                capsize=2.5,
                color=METHOD_COLORS[method],
                linestyle=METHOD_STYLES[method],
                label=METHOD_LABELS[method],
            )
        panel.set_xlabel(r"observation noise $\sigma_y$")
        panel.grid(alpha=0.25)
    ax.set_ylabel("Population excess risk")
    sigma_star = math.sqrt(config.n_eff * v_hat)
    ax.axvline(sigma_star, color="#6b7280", linestyle="--", linewidth=1.2)
    ax.annotate(
        rf"predicted crossover $\sigma_y^\ast\approx{sigma_star:.2f}$",
        xy=(sigma_star, ax.get_ylim()[1] * 0.55),
        xytext=(4, 0),
        textcoords="offset points",
        rotation=90,
        fontsize=8,
        color="#6b7280",
        va="center",
    )
    ax.set_title("A. All parameterizations", loc="left", fontweight="bold")
    ax_zoom.set_title(
        r"B. Matched budget $B=d+K=48$", loc="left", fontweight="bold"
    )
    ax_zoom.set_yscale("log")
    ax.legend(frameon=False, fontsize=8)
    ax_zoom.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(config.output_dir / "matched_budget_baselines.png", dpi=300)
    fig.savefig(config.output_dir / "matched_budget_baselines.pdf")
    plt.close(fig)


def write_summary(
    path: Path,
    config: Config,
    rows: list[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    lines = [
        "# Matched-budget baselines (Proposition 2 check)",
        "",
        f"D={config.D}, d={config.d}, K={config.sparse_budget}, "
        f"B={config.d + config.sparse_budget}, n_eff={config.n_eff}, "
        f"trials={config.trials}, seed={config.seed}.",
        "",
        f"- realized slab count: {metadata['realized_slab_count']}",
        f"- v_hat (per-coordinate energy): {float(metadata['v_hat']):.6g}",
        f"- oracle top-K off-subspace energy fraction: "
        f"{float(metadata['oracle_topk_energy_fraction']):.4f}",
        "",
        "| noise | "
        + " | ".join(METHOD_LABELS[m] for m in METHODS)
        + " |",
        "|---:|" + "---:|" * len(METHODS),
    ]
    by_noise: dict[float, dict[str, dict[str, object]]] = {}
    for row in rows:
        by_noise.setdefault(float(row["noise_std"]), {})[str(row["method"])] = row
    for noise_std in sorted(by_noise):
        by_method = by_noise[noise_std]
        lines.append(
            f"| {noise_std:g} | "
            + " | ".join(
                f"{float(by_method[m]['risk_mean']):.4g}" for m in METHODS
            )
            + " |"
        )
    lines.append("")
    lines.append("Full bias/variance values are in `matched_budget_results.csv`.")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--D", type=int, default=4096)
    parser.add_argument("--d", type=int, default=16)
    parser.add_argument("--sparse-budget", type=int, default=32)
    parser.add_argument("--n-eff", type=int, default=512)
    parser.add_argument("--spike-std", type=float, default=0.005)
    parser.add_argument("--slab-probability", type=float, default=0.01)
    parser.add_argument("--slab-std", type=float, default=0.5)
    parser.add_argument("--slab-df", type=float, default=3.0)
    parser.add_argument(
        "--noise-stds",
        nargs="+",
        type=float,
        default=[round(0.1 * i, 1) for i in range(1, 21)],
    )
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results_matched_budget_baselines"),
    )
    args = parser.parse_args()
    config = Config(
        D=args.D,
        d=args.d,
        sparse_budget=args.sparse_budget,
        n_eff=args.n_eff,
        spike_std=args.spike_std,
        slab_probability=args.slab_probability,
        slab_std=args.slab_std,
        slab_df=args.slab_df,
        noise_stds=list(args.noise_stds),
        trials=args.trials,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows, metadata = run(config)
    with (config.output_dir / "matched_budget_results.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (config.output_dir / "config.json").open("w") as handle:
        json.dump(
            {**asdict(config), "output_dir": str(config.output_dir), **metadata},
            handle,
            indent=2,
        )
    plot_results(config, rows, float(metadata["v_hat"]))
    write_summary(
        config.output_dir / "summary.md", config, rows, metadata
    )
    print(f"Saved matched-budget baselines to {config.output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# coding: utf-8

"""
Gaussian synthetic validation for the ProLoSA sparse residual branch.

This is the Gaussian counterpart of synthetic_powerlaw_noise_biasvar.py.
The ground-truth update is sampled directly in D dimensions:

    theta_star[j] ~ N(0, sigma_theta^2),

where ``--source-std`` controls sigma_theta.  Independently, the local linear
regression observation model is

    theta_noisy = theta_star + (sigma_y / sqrt(n_eff)) * epsilon,
    epsilon[j] ~ N(0, 1),

where ``--noise-stds`` controls sigma_y.  Thus source-distribution spread and
regression-noise spread can be varied independently.

Outputs:
  - gaussian_noise_biasvar.png / .pdf: three-panel risk/decomposition figure.
  - gaussian_results.csv: risk, bias, and variance for each noise level.
  - config.json and summary.md: configuration and reproducibility summary.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

from synthetic_powerlaw_noise_biasvar import (
    choose_sparse_support,
    exact_excess_risk,
    method_label,
    orthonormal_projection,
    project_onto_basis,
    project_prolosa,
    stable_seed,
    summarize_estimates,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
METHODS = ("lora", "unilora", "prolosa")


@dataclass(frozen=True)
class Config:
    D: int
    d: int
    n_eff: int
    sparse_budget: int
    source_std: float
    noise_stds: list[float]
    middle_noise: float
    high_noise: float
    trials: int
    support: str
    ridge: float
    seed: int
    output_dir: Path


def parse_float_list(values: Iterable[str]) -> list[float]:
    return [float(value) for value in values]


def validate_config(config: Config) -> None:
    if config.D <= 0:
        raise ValueError(f"D must be positive, got {config.D}.")
    if not 0 < config.d <= config.D:
        raise ValueError(f"d must lie in [1, D], got d={config.d}, D={config.D}.")
    if config.n_eff <= 0:
        raise ValueError(f"n_eff must be positive, got {config.n_eff}.")
    if not 0 <= config.sparse_budget <= config.D:
        raise ValueError(
            f"sparse_budget must lie in [0, D], got K={config.sparse_budget}, D={config.D}."
        )
    if config.source_std <= 0:
        raise ValueError(f"source_std must be positive, got {config.source_std}.")
    if not config.noise_stds:
        raise ValueError("noise_stds cannot be empty.")
    if any(value < 0 for value in config.noise_stds):
        raise ValueError(f"noise_stds must be nonnegative, got {config.noise_stds}.")
    if config.trials <= 0:
        raise ValueError(f"trials must be positive, got {config.trials}.")
    if config.ridge < 0:
        raise ValueError(f"ridge must be nonnegative, got {config.ridge}.")


def make_gaussian_theta(
    D: int,
    source_std: float,
    rng: np.random.Generator,
) -> tuple[FloatArray, IntArray]:
    """Sample i.i.d. Gaussian coordinates and return descending magnitude order."""
    theta_star = rng.normal(loc=0.0, scale=source_std, size=D)
    rank_to_coordinate = np.argsort(-np.abs(theta_star)).astype(np.int64)
    return theta_star, rank_to_coordinate


def run_experiment(config: Config) -> tuple[list[dict[str, object]], dict[str, object]]:
    validate_config(config)
    problem_rng = np.random.default_rng(stable_seed(config.seed, "gaussian_problem"))
    P = orthonormal_projection(config.D, config.d, problem_rng)
    theta_star, rank_to_coordinate = make_gaussian_theta(
        config.D,
        source_std=config.source_std,
        rng=problem_rng,
    )

    theta_energy = float(theta_star @ theta_star)
    top_coordinates = rank_to_coordinate[: config.sparse_budget]
    top_k_energy = (
        float(np.sum(theta_star[top_coordinates] ** 2) / theta_energy)
        if theta_energy > 0
        else 0.0
    )
    projected_energy = (
        float(np.sum((P.T @ theta_star) ** 2) / theta_energy)
        if theta_energy > 0
        else 0.0
    )

    fixed_oracle_support: IntArray | None = None
    if config.support == "oracle":
        fixed_oracle_support = choose_sparse_support(
            mode="oracle",
            theta_star=theta_star,
            noisy_theta=theta_star,
            theta_unilora=project_onto_basis(theta_star, P, config.ridge),
            k=config.sparse_budget,
            rng=problem_rng,
        )

    rows: list[dict[str, object]] = []
    for noise_std in config.noise_stds:
        estimates: dict[str, list[FloatArray]] = {method: [] for method in METHODS}
        risks: dict[str, list[float]] = {method: [] for method in METHODS}

        for trial in range(config.trials):
            trial_rng = np.random.default_rng(
                stable_seed(config.seed, "gaussian_trial", trial, noise_std)
            )
            regression_noise = (
                noise_std / math.sqrt(config.n_eff)
            ) * trial_rng.normal(size=config.D)
            noisy_theta = theta_star + regression_noise

            theta_lora = noisy_theta
            theta_unilora = project_onto_basis(noisy_theta, P, config.ridge)
            support = (
                fixed_oracle_support
                if fixed_oracle_support is not None
                else choose_sparse_support(
                    mode=config.support,
                    theta_star=theta_star,
                    noisy_theta=noisy_theta,
                    theta_unilora=theta_unilora,
                    k=config.sparse_budget,
                    rng=trial_rng,
                )
            )
            theta_prolosa = project_prolosa(noisy_theta, P, support, config.ridge)

            trial_estimates = {
                "lora": theta_lora,
                "unilora": theta_unilora,
                "prolosa": theta_prolosa,
            }
            for method, theta_hat in trial_estimates.items():
                estimates[method].append(theta_hat)
                risks[method].append(exact_excess_risk(theta_hat, theta_star))

        for method in METHODS:
            risk_mean, risk_std, bias, variance = summarize_estimates(
                estimates[method],
                risks[method],
                theta_star,
            )
            rows.append(
                {
                    "D": config.D,
                    "d": config.d,
                    "n_eff": config.n_eff,
                    "sparse_budget": config.sparse_budget,
                    "source_distribution": "gaussian",
                    "source_std": config.source_std,
                    "realized_source_std": float(np.std(theta_star, ddof=1)),
                    "noise_std": noise_std,
                    "method": method,
                    "risk_mean": risk_mean,
                    "risk_std": risk_std,
                    "bias": bias,
                    "variance": variance,
                    "bias_plus_variance": bias + variance,
                    "support": config.support,
                }
            )

    metadata: dict[str, object] = {
        "realized_source_mean": float(np.mean(theta_star)),
        "realized_source_std": float(np.std(theta_star, ddof=1)),
        "realized_theta_norm": float(np.linalg.norm(theta_star)),
        "top_k_energy_fraction": top_k_energy,
        "learner_projection_energy_fraction": projected_energy,
        "top_k_coordinates": top_coordinates.tolist(),
    }
    return rows, metadata


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def select_noise_rows(
    rows: list[dict[str, object]],
    target_noise: float,
) -> tuple[float, dict[str, dict[str, object]]]:
    noise_stds = sorted({float(row["noise_std"]) for row in rows})
    selected_noise = min(noise_stds, key=lambda value: abs(value - target_noise))
    selected = [
        row
        for row in rows
        if math.isclose(float(row["noise_std"]), selected_noise, rel_tol=1e-9, abs_tol=1e-9)
    ]
    return selected_noise, {str(row["method"]): row for row in selected}


def plot_results(
    output_dir: Path,
    rows: list[dict[str, object]],
    source_std: float,
    middle_noise: float,
    high_noise: float,
) -> None:
    try:
        plt = importlib.import_module("matplotlib.pyplot")
    except ImportError:
        print("matplotlib is not installed; skipped PNG/PDF figure generation.")
        return

    colors = {"lora": "#2563eb", "unilora": "#f97316", "prolosa": "#16a34a"}
    component_colors = {"bias": "#2f6fbb", "variance": "#f2b84b"}
    middle_selected, middle_by_method = select_noise_rows(rows, middle_noise)
    high_selected, high_by_method = select_noise_rows(rows, high_noise)

    def component_total(rows_by_method: dict[str, dict[str, object]]) -> FloatArray:
        return np.array(
            [
                float(rows_by_method[method]["bias"])
                + float(rows_by_method[method]["variance"])
                for method in METHODS
            ]
        )

    max_total = float(
        max(np.max(component_total(middle_by_method)), np.max(component_total(high_by_method)))
    )
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(14.2, 4.6))

    for method in METHODS:
        method_rows = sorted(
            [row for row in rows if row["method"] == method],
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
    ax_a.set_title(
        rf"A. Risk ($\sigma_\theta={source_std:g}$)",
        loc="left",
        fontweight="bold",
    )
    ax_a.set_xlabel(r"regression-noise standard deviation $\sigma_y$")
    ax_a.set_ylabel("Population excess risk")
    ax_a.grid(alpha=0.25)
    ax_a.legend(frameon=False)

    def plot_decomposition(
        ax: Any,
        rows_by_method: dict[str, dict[str, object]],
        selected_noise: float,
        title_prefix: str,
        show_ylabel: bool,
        show_legend: bool,
    ) -> None:
        x = np.arange(len(METHODS))
        bias = [float(rows_by_method[method]["bias"]) for method in METHODS]
        variance = [float(rows_by_method[method]["variance"]) for method in METHODS]
        totals = np.array(bias) + np.array(variance)
        ax.bar(x, bias, color=component_colors["bias"], label="Bias")
        ax.bar(
            x,
            variance,
            bottom=bias,
            color=component_colors["variance"],
            label="Variance",
        )
        for index, total in enumerate(totals):
            ax.text(index, total, f"{total:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(
            rf"{title_prefix}. Decomposition ($\sigma_y={selected_noise:g}$)",
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

    plot_decomposition(ax_b, middle_by_method, middle_selected, "B", True, True)
    plot_decomposition(ax_c, high_by_method, high_selected, "C", False, False)
    fig.tight_layout()
    fig.savefig(output_dir / "gaussian_noise_biasvar.png", dpi=240)
    fig.savefig(output_dir / "gaussian_noise_biasvar.pdf")
    plt.close(fig)


def write_summary(
    path: Path,
    config: Config,
    rows: list[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    middle_noise, middle_by_method = select_noise_rows(rows, config.middle_noise)
    high_noise, high_by_method = select_noise_rows(rows, config.high_noise)

    def table_lines(by_method: dict[str, dict[str, object]]) -> list[str]:
        lines = ["| method | risk | bias | variance |", "|---|---:|---:|---:|"]
        for method in METHODS:
            row = by_method[method]
            lines.append(
                f"| {method_label(method)} | {float(row['risk_mean']):.6g} | "
                f"{float(row['bias']):.6g} | {float(row['variance']):.6g} |"
            )
        return lines

    lines = [
        "# Gaussian synthetic validation",
        "",
        r"Ground-truth coordinates are sampled i.i.d. from N(0, sigma_theta^2).",
        r"Regression noise is controlled independently by sigma_y.",
        "",
        "## Configuration",
        "",
        f"- D: {config.D}",
        f"- d: {config.d}",
        f"- n_eff: {config.n_eff}",
        f"- sparse_budget K: {config.sparse_budget}",
        f"- source_std sigma_theta: {config.source_std:g}",
        f"- realized source std: {float(metadata['realized_source_std']):.6g}",
        f"- realized theta norm: {float(metadata['realized_theta_norm']):.6g}",
        f"- trials: {config.trials}",
        f"- support: {config.support}",
        f"- top-K energy fraction: {float(metadata['top_k_energy_fraction']):.4f}",
        f"- random P energy fraction: {float(metadata['learner_projection_energy_fraction']):.4f}",
        "",
        f"## Bias-variance at sigma_y={middle_noise:g}",
        "",
        *table_lines(middle_by_method),
        "",
        f"## Bias-variance at sigma_y={high_noise:g}",
        "",
        *table_lines(high_by_method),
        "",
        "## Files",
        "",
        "- `gaussian_results.csv`",
        "- `gaussian_noise_biasvar.png`",
        "- `gaussian_noise_biasvar.pdf`",
    ]
    path.write_text("\n".join(lines) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--D", type=int, default=4096)
    parser.add_argument("--d", type=int, default=16, help="Compressed Uni-LoRA dimension.")
    parser.add_argument(
        "--n-eff",
        type=int,
        default=512,
        help="Effective sample size in sigma_y / sqrt(n_eff).",
    )
    parser.add_argument(
        "--sparse-budget",
        type=int,
        default=32,
        help="K sparse coordinates opened by the ProLoSA residual branch.",
    )
    parser.add_argument(
        "--source-std",
        type=float,
        default=0.0625,
        help=(
            "Standard deviation sigma_theta of Gaussian ground-truth coordinates. "
            "The default gives E[||theta_star||_2] approximately 4 when D=4096."
        ),
    )
    parser.add_argument(
        "--noise-stds",
        nargs="+",
        default=None,
        help="Regression-noise std sweep sigma_y. Defaults to 0.1, 0.2, ..., 2.0.",
    )
    parser.add_argument(
        "--middle-noise",
        type=float,
        default=0.5,
        help="Noise point used for the middle bias-variance decomposition.",
    )
    parser.add_argument(
        "--high-noise",
        type=float,
        default=1.5,
        help="Noise point used for the right bias-variance decomposition.",
    )
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument(
        "--support",
        choices=["oracle", "noisy_residual", "random"],
        default="noisy_residual",
        help="Sparse coordinate selection strategy.",
    )
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results_gaussian_synthetic_theory"),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    noise_stds = (
        parse_float_list(args.noise_stds)
        if args.noise_stds is not None
        else [round(float(value), 10) for value in np.linspace(0.1, 2.0, 20)]
    )
    config = Config(
        D=args.D,
        d=args.d,
        n_eff=args.n_eff,
        sparse_budget=args.sparse_budget,
        source_std=args.source_std,
        noise_stds=noise_stds,
        middle_noise=args.middle_noise,
        high_noise=args.high_noise,
        trials=args.trials,
        support=args.support,
        ridge=args.ridge,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows, metadata = run_experiment(config)
    write_csv(config.output_dir / "gaussian_results.csv", rows)
    with (config.output_dir / "config.json").open("w") as handle:
        json.dump(
            {**asdict(config), "output_dir": str(config.output_dir), **metadata},
            handle,
            indent=2,
        )
    plot_results(
        config.output_dir,
        rows,
        source_std=config.source_std,
        middle_noise=config.middle_noise,
        high_noise=config.high_noise,
    )
    write_summary(config.output_dir / "summary.md", config, rows, metadata)
    print(f"Saved Gaussian synthetic results to {config.output_dir}")


if __name__ == "__main__":
    main()

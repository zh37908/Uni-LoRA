#!/usr/bin/env python
# coding: utf-8

"""
Synthetic validation under a contaminated Gaussian source distribution.

Each ground-truth coordinate is sampled independently from

    p(theta_j) = (1 - epsilon) N(0, sigma_theta^2)
                 + epsilon p_outlier(theta_j),

where p_outlier is a scaled Student-t distribution.  ``--source-std`` controls
the Gaussian-core standard deviation, ``--contamination-rates`` controls
epsilon, and ``--outlier-std-multiplier`` controls the outlier standard
deviation relative to the Gaussian core.  Linear-regression noise is controlled
independently by ``--noise-stds``.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

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
    contamination_rates: list[float]
    outlier_std_multiplier: float
    outlier_df: float
    noise_stds: list[float]
    trials: int
    support: str
    ridge: float
    seed: int
    output_dir: Path


def parse_float_list(values: Iterable[str]) -> list[float]:
    return [float(value) for value in values]


def validate_config(config: Config) -> None:
    if config.D <= 0 or not 0 < config.d <= config.D:
        raise ValueError(f"Require D > 0 and 0 < d <= D; got D={config.D}, d={config.d}.")
    if config.n_eff <= 0 or config.trials <= 0:
        raise ValueError("n_eff and trials must be positive.")
    if not 0 <= config.sparse_budget <= config.D:
        raise ValueError("sparse_budget must lie in [0, D].")
    if config.source_std <= 0:
        raise ValueError("source_std must be positive.")
    if not config.contamination_rates or any(
        not 0 <= value <= 1 for value in config.contamination_rates
    ):
        raise ValueError("contamination_rates must be nonempty and lie in [0, 1].")
    if config.outlier_std_multiplier <= 0:
        raise ValueError("outlier_std_multiplier must be positive.")
    if config.outlier_df <= 2:
        raise ValueError("outlier_df must exceed 2 so its variance is finite.")
    if not config.noise_stds or any(value < 0 for value in config.noise_stds):
        raise ValueError("noise_stds must be nonempty and nonnegative.")
    if config.ridge < 0:
        raise ValueError("ridge must be nonnegative.")


def make_contaminated_gaussian_theta(
    D: int,
    source_std: float,
    contamination_rate: float,
    outlier_std_multiplier: float,
    outlier_df: float,
    rng: np.random.Generator,
) -> tuple[FloatArray, NDArray[np.bool_]]:
    """Sample Gaussian-core coordinates and replace a Bernoulli subset by thick tails."""
    theta_star = rng.normal(0.0, source_std, size=D)
    outlier_mask = rng.random(D) < contamination_rate
    outlier_count = int(np.sum(outlier_mask))
    if outlier_count:
        # Standard Student-t(df) has std sqrt(df / (df - 2)).  Rescale it so
        # outlier_std_multiplier denotes the actual target standard deviation.
        unit_std_scale = math.sqrt((outlier_df - 2.0) / outlier_df)
        theta_star[outlier_mask] = (
            rng.standard_t(outlier_df, size=outlier_count)
            * unit_std_scale
            * outlier_std_multiplier
            * source_std
        )
    return theta_star, outlier_mask


def run_experiment(config: Config) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    validate_config(config)
    projection_rng = np.random.default_rng(stable_seed(config.seed, "contaminated_projection"))
    P = orthonormal_projection(config.D, config.d, projection_rng)
    rows: list[dict[str, object]] = []
    distribution_metadata: list[dict[str, object]] = []

    for contamination_rate in config.contamination_rates:
        problem_rng = np.random.default_rng(
            stable_seed(config.seed, "contaminated_problem", contamination_rate)
        )
        theta_star, outlier_mask = make_contaminated_gaussian_theta(
            config.D,
            config.source_std,
            contamination_rate,
            config.outlier_std_multiplier,
            config.outlier_df,
            problem_rng,
        )
        outlier_coordinates = np.flatnonzero(outlier_mask).astype(np.int64)
        theta_energy = float(theta_star @ theta_star)
        ranked_coordinates = np.argsort(-np.abs(theta_star)).astype(np.int64)
        top_coordinates = ranked_coordinates[: config.sparse_budget]
        top_k_energy = float(np.sum(theta_star[top_coordinates] ** 2) / theta_energy)
        top_k_outlier_fraction = (
            float(np.mean(outlier_mask[top_coordinates]))
            if config.sparse_budget > 0
            else 0.0
        )
        distribution_metadata.append(
            {
                "contamination_rate": contamination_rate,
                "realized_outlier_count": int(np.sum(outlier_mask)),
                "realized_contamination_rate": float(np.mean(outlier_mask)),
                "realized_source_std": float(np.std(theta_star, ddof=1)),
                "theta_norm": float(np.linalg.norm(theta_star)),
                "top_k_energy_fraction": top_k_energy,
                "top_k_outlier_fraction": top_k_outlier_fraction,
                "outlier_coordinates": outlier_coordinates.tolist(),
            }
        )

        fixed_oracle_support: IntArray | None = None
        if config.support == "oracle":
            fixed_oracle_support = choose_sparse_support(
                "oracle",
                theta_star,
                theta_star,
                project_onto_basis(theta_star, P, config.ridge),
                config.sparse_budget,
                problem_rng,
            )

        for noise_std in config.noise_stds:
            estimates: dict[str, list[FloatArray]] = {method: [] for method in METHODS}
            risks: dict[str, list[float]] = {method: [] for method in METHODS}
            selected_outlier_recalls: list[float] = []

            for trial in range(config.trials):
                trial_rng = np.random.default_rng(
                    stable_seed(
                        config.seed,
                        "contaminated_trial",
                        contamination_rate,
                        noise_std,
                        trial,
                    )
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
                        config.support,
                        theta_star,
                        noisy_theta,
                        theta_unilora,
                        config.sparse_budget,
                        trial_rng,
                    )
                )
                theta_prolosa = project_prolosa(noisy_theta, P, support, config.ridge)
                if len(outlier_coordinates):
                    selected_outlier_recalls.append(
                        float(np.sum(outlier_mask[support]) / len(outlier_coordinates))
                    )

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
                        "contamination_rate": contamination_rate,
                        "realized_contamination_rate": float(np.mean(outlier_mask)),
                        "realized_outlier_count": int(np.sum(outlier_mask)),
                        "source_std": config.source_std,
                        "realized_source_std": float(np.std(theta_star, ddof=1)),
                        "outlier_std_multiplier": config.outlier_std_multiplier,
                        "outlier_df": config.outlier_df,
                        "noise_std": noise_std,
                        "method": method,
                        "risk_mean": risk_mean,
                        "risk_std": risk_std,
                        "bias": bias,
                        "variance": variance,
                        "outlier_recall": (
                            float(np.mean(selected_outlier_recalls))
                            if selected_outlier_recalls
                            else 0.0
                        ),
                        "support": config.support,
                    }
                )
    return rows, distribution_metadata


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def rows_by_case(
    rows: list[dict[str, object]],
) -> dict[tuple[float, float], dict[str, dict[str, object]]]:
    cases: dict[tuple[float, float], dict[str, dict[str, object]]] = {}
    for row in rows:
        key = (float(row["contamination_rate"]), float(row["noise_std"]))
        cases.setdefault(key, {})[str(row["method"])] = row
    return cases


def plot_results(config: Config, rows: list[dict[str, object]]) -> None:
    try:
        plt = importlib.import_module("matplotlib.pyplot")
    except ImportError:
        print("matplotlib is not installed; skipped figure generation.")
        return

    colors = {"lora": "#2563eb", "unilora": "#f97316", "prolosa": "#16a34a"}
    fig, axes = plt.subplots(
        1,
        len(config.contamination_rates),
        figsize=(5 * len(config.contamination_rates), 4.5),
        squeeze=False,
    )
    for ax, contamination_rate in zip(axes[0], config.contamination_rates):
        for method in METHODS:
            method_rows = sorted(
                (
                    row
                    for row in rows
                    if row["method"] == method
                    and math.isclose(
                        float(row["contamination_rate"]),
                        contamination_rate,
                    )
                ),
                key=lambda row: float(row["noise_std"]),
            )
            ax.errorbar(
                [float(row["noise_std"]) for row in method_rows],
                [float(row["risk_mean"]) for row in method_rows],
                yerr=[float(row["risk_std"]) for row in method_rows],
                marker="o",
                linewidth=2,
                capsize=3,
                color=colors[method],
                label=method_label(method),
            )
        ax.set_title(rf"$\epsilon={contamination_rate:g}$", fontweight="bold")
        ax.set_xlabel(r"regression noise $\sigma_y$")
        ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel("Population excess risk")
    axes[0, -1].legend(frameon=False)
    fig.suptitle(
        rf"Contaminated Gaussian: $\sigma_\theta={config.source_std:g}$, "
        rf"outlier std $={config.outlier_std_multiplier:g}\sigma_\theta$",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(config.output_dir / "contaminated_gaussian_comparison.png", dpi=240)
    fig.savefig(config.output_dir / "contaminated_gaussian_comparison.pdf")
    plt.close(fig)


def write_summary(
    path: Path,
    config: Config,
    rows: list[dict[str, object]],
    metadata: list[dict[str, object]],
) -> None:
    cases = rows_by_case(rows)
    lines = [
        "# Contaminated Gaussian validation",
        "",
        (
            r"$p(\theta)=(1-\epsilon)N(0,\sigma_\theta^2)"
            r"+\epsilon\,p_{\mathrm{outlier}}(\theta)$."
        ),
        "",
        f"- Gaussian source std: {config.source_std:g}",
        f"- outlier distribution: Student-t(df={config.outlier_df:g})",
        f"- outlier std multiplier: {config.outlier_std_multiplier:g}",
        f"- D / d / K: {config.D} / {config.d} / {config.sparse_budget}",
        f"- trials: {config.trials}",
        f"- support: {config.support}",
        "",
        "## Realized distributions",
        "",
        "| epsilon | outliers | realized fraction | realized std | top-K energy | top-K outliers |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in metadata:
        lines.append(
            f"| {float(item['contamination_rate']):g} | "
            f"{int(item['realized_outlier_count'])} | "
            f"{float(item['realized_contamination_rate']):.4f} | "
            f"{float(item['realized_source_std']):.6g} | "
            f"{float(item['top_k_energy_fraction']):.4f} | "
            f"{float(item['top_k_outlier_fraction']):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Three-method risk table",
            "",
            "| epsilon | noise std | LoRA | Uni-LoRA | ProLoSA | winner | ProLoSA outlier recall |",
            "|---:|---:|---:|---:|---:|:---|---:|",
        ]
    )
    prolosa_wins = 0
    for (contamination_rate, noise_std), by_method in sorted(cases.items()):
        winner = min(METHODS, key=lambda method: float(by_method[method]["risk_mean"]))
        prolosa_wins += winner == "prolosa"
        lines.append(
            f"| {contamination_rate:g} | {noise_std:g} | "
            f"{float(by_method['lora']['risk_mean']):.6g} | "
            f"{float(by_method['unilora']['risk_mean']):.6g} | "
            f"{float(by_method['prolosa']['risk_mean']):.6g} | "
            f"{method_label(winner)} | "
            f"{float(by_method['prolosa']['outlier_recall']):.4f} |"
        )
    lines.extend(
        [
            "",
            f"ProLoSA has the lowest risk in {prolosa_wins}/{len(cases)} tested cases.",
            "",
            "Full bias/variance/error-bar values are in `contaminated_gaussian_results.csv`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--D", type=int, default=4096)
    parser.add_argument("--d", type=int, default=16)
    parser.add_argument("--n-eff", type=int, default=512)
    parser.add_argument("--sparse-budget", type=int, default=32)
    parser.add_argument("--source-std", type=float, default=0.0625)
    parser.add_argument(
        "--contamination-rates",
        nargs="+",
        default=["0", "0.01", "0.05"],
        help="Mixture probabilities epsilon.",
    )
    parser.add_argument(
        "--outlier-std-multiplier",
        type=float,
        default=10.0,
        help="Outlier standard deviation divided by Gaussian-core std.",
    )
    parser.add_argument("--outlier-df", type=float, default=3.0)
    parser.add_argument(
        "--noise-stds",
        nargs="+",
        default=["0.5", "1.0", "1.5"],
        help="Linear-regression noise standard deviations.",
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
        "--output-dir",
        type=Path,
        default=Path("results_contaminated_gaussian_synthetic_theory"),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = Config(
        D=args.D,
        d=args.d,
        n_eff=args.n_eff,
        sparse_budget=args.sparse_budget,
        source_std=args.source_std,
        contamination_rates=parse_float_list(args.contamination_rates),
        outlier_std_multiplier=args.outlier_std_multiplier,
        outlier_df=args.outlier_df,
        noise_stds=parse_float_list(args.noise_stds),
        trials=args.trials,
        support=args.support,
        ridge=args.ridge,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows, metadata = run_experiment(config)
    write_csv(config.output_dir / "contaminated_gaussian_results.csv", rows)
    with (config.output_dir / "config.json").open("w") as handle:
        json.dump(
            {
                **asdict(config),
                "output_dir": str(config.output_dir),
                "distributions": metadata,
            },
            handle,
            indent=2,
        )
    plot_results(config, rows)
    write_summary(config.output_dir / "summary.md", config, rows, metadata)
    print(f"Saved contaminated-Gaussian results to {config.output_dir}")


if __name__ == "__main__":
    main()

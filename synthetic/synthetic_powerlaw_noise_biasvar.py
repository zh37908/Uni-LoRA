#!/usr/bin/env python
# coding: utf-8

"""
Power-law synthetic validation for the ProLoSA sparse residual branch.

Unlike synthetic_validate_theory.py, the ground-truth update is not generated
from the learner projection P.  The target is sampled directly in D dimensions:

    |theta_star|_(i) = C / i^a

after sorting by magnitude, followed by random signs and a random coordinate
permutation.  This mimics a heavy-tailed update where a few coordinates carry
large residual mass.  The learner projection P is sampled independently and is
used only by Uni-LoRA / ProLoSA estimators.

The local quadratic observation model is

    theta_noisy = theta_star + (sigma_y / sqrt(n_eff)) * epsilon,

so population excess risk remains exactly

    0.5 * ||theta_hat - theta_star||_2^2.

Outputs:
  - powerlaw_noise_biasvar.png / .pdf: requested three-panel figure.
  - powerlaw_results.csv: risk, bias, and variance for each noise level.
  - summary.md: short reproducibility summary.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray


FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
Method: TypeAlias = Literal["lora", "unilora", "prolosa"]


@dataclass(frozen=True)
class Config:
    D: int
    d: int
    n_eff: int
    sparse_budget: int
    powerlaw_exponent: float
    theta_norm: float
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


def stable_seed(*items: object) -> int:
    """Create a deterministic 32-bit seed without relying on Python hash()."""
    text = "::".join(str(item) for item in items)
    value = 2166136261
    for char in text.encode("utf-8"):
        value ^= char
        value = (value * 16777619) % (2**32)
    return int(value)


def orthonormal_projection(D: int, d: int, rng: np.random.Generator) -> FloatArray:
    if d > D:
        raise ValueError(f"Compressed dimension d={d} cannot exceed D={D}.")
    gaussian = rng.normal(size=(D, d))
    q, r = np.linalg.qr(gaussian, mode="reduced")
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs


def make_powerlaw_theta(
    D: int,
    exponent: float,
    theta_norm: float,
    rng: np.random.Generator,
) -> tuple[FloatArray, IntArray, FloatArray]:
    if exponent <= 0:
        raise ValueError(f"powerlaw_exponent must be positive, got {exponent}.")
    if theta_norm <= 0:
        raise ValueError(f"theta_norm must be positive, got {theta_norm}.")

    ranks = np.arange(1, D + 1, dtype=np.float64)
    magnitudes = ranks ** (-exponent)
    magnitudes *= theta_norm / np.linalg.norm(magnitudes)

    signs = rng.choice(np.array([-1.0, 1.0]), size=D)
    sorted_theta = signs * magnitudes
    permutation = rng.permutation(D)

    theta = np.empty(D, dtype=np.float64)
    theta[permutation] = sorted_theta
    # permutation[:k] are the largest-magnitude coordinates after scrambling.
    return theta, permutation, magnitudes


def project_onto_basis(vector: FloatArray, basis: FloatArray, ridge: float) -> FloatArray:
    if ridge == 0:
        return basis @ (basis.T @ vector)
    gram = basis.T @ basis
    coef = np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), basis.T @ vector)
    return basis @ coef


def project_prolosa(vector: FloatArray, P: FloatArray, support: IntArray, ridge: float) -> FloatArray:
    """Project onto span(P, selected coordinate axes) without a large QR factorization."""
    if len(support) == 0:
        return project_onto_basis(vector, P, ridge)

    d = P.shape[1]
    k = len(support)
    P_support = P[support, :]
    gram = np.block(
        [
            [np.eye(d), P_support.T],
            [P_support, np.eye(k)],
        ]
    )
    if ridge > 0:
        gram = gram + ridge * np.eye(d + k)

    rhs = np.concatenate([P.T @ vector, vector[support]])
    coef = np.linalg.solve(gram, rhs)
    projected = P @ coef[:d]
    projected[support] += coef[d:]
    return projected


def choose_sparse_support(
    mode: str,
    theta_star: FloatArray,
    noisy_theta: FloatArray,
    theta_unilora: FloatArray,
    k: int,
    rng: np.random.Generator,
) -> IntArray:
    D = theta_star.shape[0]
    if not 0 <= k <= D:
        raise ValueError(f"sparse_budget K={k} must lie in [0, D={D}].")
    if k == 0:
        return np.array([], dtype=np.int64)

    if mode == "oracle":
        scores = np.abs(theta_star)
    elif mode == "noisy_residual":
        scores = np.abs(noisy_theta - theta_unilora)
    elif mode == "random":
        return np.sort(rng.choice(D, size=k, replace=False))
    else:
        raise ValueError(f"Unknown support mode: {mode}")

    return np.sort(np.argpartition(scores, -k)[-k:])


def exact_excess_risk(theta_hat: FloatArray, theta_star: FloatArray) -> float:
    diff = theta_hat - theta_star
    return 0.5 * float(diff @ diff)


def summarize_estimates(
    estimates: list[FloatArray],
    risks: list[float],
    theta_star: FloatArray,
) -> tuple[float, float, float, float]:
    from fixed_target import fixed_target_decomposition
    risk_mean, bias, variance = fixed_target_decomposition(estimates, theta_star, risks)
    risk_values = np.asarray(risks, dtype=np.float64)
    risk_std = float(risk_values.std(ddof=1)) if len(risk_values) > 1 else 0.0
    return risk_mean, risk_std, bias, variance


def run_experiment(config: Config) -> tuple[list[dict[str, object]], dict[str, object]]:
    problem_rng = np.random.default_rng(stable_seed(config.seed, "problem"))
    P = orthonormal_projection(config.D, config.d, problem_rng)
    theta_star, rank_to_coordinate, magnitudes = make_powerlaw_theta(
        config.D,
        exponent=config.powerlaw_exponent,
        theta_norm=config.theta_norm,
        rng=problem_rng,
    )

    rows: list[dict[str, object]] = []
    methods: tuple[Method, ...] = ("lora", "unilora", "prolosa")

    top_k_energy = float(np.sum(magnitudes[: config.sparse_budget] ** 2) / np.sum(magnitudes**2))
    projected_energy = float(np.sum((P.T @ theta_star) ** 2) / np.sum(theta_star**2))
    fixed_oracle_support = None
    if config.support == "oracle":
        fixed_oracle_support = choose_sparse_support(
            mode="oracle",
            theta_star=theta_star,
            noisy_theta=theta_star,
            theta_unilora=project_onto_basis(theta_star, P, config.ridge),
            k=config.sparse_budget,
            rng=problem_rng,
        )

    for noise_std in config.noise_stds:
        estimates: dict[Method, list[FloatArray]] = {method: [] for method in methods}
        risks: dict[Method, list[float]] = {method: [] for method in methods}

        for trial in range(config.trials):
            trial_rng = np.random.default_rng(stable_seed(config.seed, "trial", trial, noise_std))
            noise = (noise_std / math.sqrt(config.n_eff)) * trial_rng.normal(size=config.D)
            noisy_theta = theta_star + noise

            theta_lora = noisy_theta
            theta_unilora = project_onto_basis(noisy_theta, P, config.ridge)
            if fixed_oracle_support is None:
                support = choose_sparse_support(
                    mode=config.support,
                    theta_star=theta_star,
                    noisy_theta=noisy_theta,
                    theta_unilora=theta_unilora,
                    k=config.sparse_budget,
                    rng=trial_rng,
                )
            else:
                support = fixed_oracle_support
            theta_prolosa = project_prolosa(noisy_theta, P, support, config.ridge)

            trial_estimates: dict[Method, FloatArray] = {
                "lora": theta_lora,
                "unilora": theta_unilora,
                "prolosa": theta_prolosa,
            }
            for method, theta_hat in trial_estimates.items():
                estimates[method].append(theta_hat)
                risks[method].append(exact_excess_risk(theta_hat, theta_star))

        for method in methods:
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
                    "powerlaw_exponent": config.powerlaw_exponent,
                    "theta_norm": config.theta_norm,
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

    metadata = {
        "top_k_energy_fraction": top_k_energy,
        "learner_projection_energy_fraction": projected_energy,
        "top_k_coordinates": rank_to_coordinate[: config.sparse_budget].tolist(),
    }
    return rows, metadata


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def method_label(method: str) -> str:
    return {"lora": "LoRA", "unilora": "Uni-LoRA", "prolosa": "ProLoSA"}.get(method, method)


def plot_results(
    output_dir: Path,
    rows: list[dict[str, object]],
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
    methods = ["lora", "unilora", "prolosa"]
    noise_stds = sorted({float(row["noise_std"]) for row in rows})

    def select_noise_rows(target_noise: float) -> tuple[float, dict[str, dict[str, object]]]:
        selected_noise = min(noise_stds, key=lambda value: abs(value - target_noise))
        selected = [
            row
            for row in rows
            if math.isclose(float(row["noise_std"]), selected_noise, rel_tol=1e-9, abs_tol=1e-9)
        ]
        return selected_noise, {str(row["method"]): row for row in selected}

    middle_selected_noise, middle_by_method = select_noise_rows(middle_noise)
    high_selected_noise, high_by_method = select_noise_rows(high_noise)

    def component_totals(rows_by_method: dict[str, dict[str, object]]) -> FloatArray:
        return np.array(
            [
                float(rows_by_method[method]["bias"]) + float(rows_by_method[method]["variance"])
                for method in methods
            ]
        )

    max_component_total = float(
        max(np.max(component_totals(middle_by_method)), np.max(component_totals(high_by_method)))
    )

    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1,
        3,
        figsize=(14.2, 4.6),
        gridspec_kw={"width_ratios": [1, 1, 1]},
    )

    for method in methods:
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
    ax_a.set_title("A. Risk vs. Noise Level", loc="left", fontweight="bold")
    ax_a.set_xlabel(r"noise standard deviation $\sigma_y$")
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
        x = np.arange(len(methods))
        bias = [float(rows_by_method[method]["bias"]) for method in methods]
        variance = [float(rows_by_method[method]["variance"]) for method in methods]
        totals = np.array(bias) + np.array(variance)

        ax.bar(x, bias, color=component_colors["bias"], label="Bias")
        ax.bar(
            x,
            variance,
            bottom=bias,
            color=component_colors["variance"],
            edgecolor=component_colors["variance"],
            label="Variance",
        )
        for i, total in enumerate(totals):
            ax.text(i, total, f"{total:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(
            rf"{title_prefix}. Decomposition ($\sigma_y={selected_noise:g}$)",
            loc="left",
            fontweight="bold",
        )
        ax.set_xticks(x, [method_label(method) for method in methods], rotation=25)
        if show_ylabel:
            ax.set_ylabel("Excess risk components")
        ax.set_ylim(0.0, max_component_total * 1.12)
        ax.grid(axis="y", alpha=0.25)
        if show_legend:
            ax.legend(frameon=False)

    plot_decomposition(
        ax_b,
        middle_by_method,
        middle_selected_noise,
        title_prefix="B",
        show_ylabel=True,
        show_legend=True,
    )
    plot_decomposition(
        ax_c,
        high_by_method,
        high_selected_noise,
        title_prefix="C",
        show_ylabel=False,
        show_legend=False,
    )

    fig.tight_layout()
    fig.savefig(output_dir / "powerlaw_noise_biasvar.png", dpi=240)
    fig.savefig(output_dir / "powerlaw_noise_biasvar.pdf")
    plt.close(fig)


def write_summary(path: Path, config: Config, rows: list[dict[str, object]], metadata: dict[str, object]) -> None:
    noise_stds = sorted({float(row["noise_std"]) for row in rows})

    def selected_rows(target_noise: float) -> tuple[float, list[dict[str, object]]]:
        selected_noise = min(noise_stds, key=lambda value: abs(value - target_noise))
        selected = [
            row
            for row in rows
            if math.isclose(float(row["noise_std"]), selected_noise, rel_tol=1e-9, abs_tol=1e-9)
        ]
        selected = sorted(selected, key=lambda row: ["lora", "unilora", "prolosa"].index(str(row["method"])))
        return selected_noise, selected

    middle_noise, middle_selected = selected_rows(config.middle_noise)
    high_noise, high_selected = selected_rows(config.high_noise)

    def table_lines(selected: list[dict[str, object]]) -> list[str]:
        lines = [
            "| method | risk | bias | variance |",
            "|---|---:|---:|---:|",
        ]
        for row in selected:
            lines.append(
                f"| {method_label(str(row['method']))} | {float(row['risk_mean']):.6g} | "
                f"{float(row['bias']):.6g} | {float(row['variance']):.6g} |"
            )
        return lines

    lines = [
        "# Power-law synthetic validation",
        "",
        "Ground truth is generated directly from a shuffled heavy-tailed coordinate update, not from P.",
        "",
        "## Configuration",
        "",
        f"- D: {config.D}",
        f"- d: {config.d}",
        f"- n_eff: {config.n_eff}",
        f"- sparse_budget K: {config.sparse_budget}",
        f"- powerlaw_exponent: {config.powerlaw_exponent:g}",
        f"- theta_norm: {config.theta_norm:g}",
        f"- trials: {config.trials}",
        f"- support: {config.support}",
        f"- top-K energy fraction: {float(metadata['top_k_energy_fraction']):.4f}",
        f"- random P energy fraction: {float(metadata['learner_projection_energy_fraction']):.4f}",
        "",
        f"## Bias-variance at sigma_y={middle_noise:g}",
        "",
        *table_lines(middle_selected),
        "",
        f"## Bias-variance at sigma_y={high_noise:g}",
        "",
        *table_lines(high_selected),
        "",
        "## Files",
        "",
        "- `powerlaw_results.csv`",
        "- `powerlaw_noise_biasvar.png`",
        "- `powerlaw_noise_biasvar.pdf`",
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
        help="Effective sample size controlling estimator-noise variance.",
    )
    parser.add_argument(
        "--sparse-budget",
        type=int,
        default=32,
        help="K sparse coordinates opened by the ProLoSA residual branch.",
    )
    parser.add_argument(
        "--powerlaw-exponent",
        type=float,
        default=0.85,
        help="Exponent a in |theta|_(i) = C / i^a.",
    )
    parser.add_argument(
        "--theta-norm",
        type=float,
        default=4.0,
        help="L2 norm of the generated heavy-tailed theta_star.",
    )
    parser.add_argument(
        "--noise-stds",
        nargs="+",
        default=None,
        help="Noise std sweep. Defaults to 0.1, 0.2, ..., 2.0.",
    )
    parser.add_argument(
        "--middle-noise",
        type=float,
        default=0.5,
        help="Noise point used for the middle bias-variance bar decomposition.",
    )
    parser.add_argument(
        "--high-noise",
        type=float,
        default=1.5,
        help="Noise point used for the bias-variance bar decomposition.",
    )
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument(
        "--support",
        choices=["oracle", "noisy_residual", "random"],
        default="noisy_residual",
        help=(
            "Sparse coordinate selection. oracle isolates the best-case sparse "
            "bias-correction effect; noisy_residual uses the observed noisy residual."
        ),
    )
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results_powerlaw_synthetic_theory"),
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
        powerlaw_exponent=args.powerlaw_exponent,
        theta_norm=args.theta_norm,
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
    write_csv(config.output_dir / "powerlaw_results.csv", rows)
    with (config.output_dir / "config.json").open("w") as handle:
        json.dump({**asdict(config), "output_dir": str(config.output_dir), **metadata}, handle, indent=2)
    plot_results(config.output_dir, rows, middle_noise=config.middle_noise, high_noise=config.high_noise)
    write_summary(config.output_dir / "summary.md", config, rows, metadata)
    print(f"Saved power-law synthetic results to {config.output_dir}")


if __name__ == "__main__":
    main()

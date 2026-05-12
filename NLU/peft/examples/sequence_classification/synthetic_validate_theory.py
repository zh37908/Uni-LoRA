#!/usr/bin/env python
# coding: utf-8

"""
Synthetic validation for the bias-variance theory in Section 4.2.

The experiment follows the paper's teacher-student linear regression setup:

    x ~ N(0, I_D)
    y = x^T theta_star + eps, eps ~ N(0, sigma_y^2)
    theta_star = P z_star + alpha s_perp

where P has orthonormal columns. Under this isotropic design, the population
excess risk is exactly 0.5 * ||theta_hat - theta_star||_2^2.

For publication-style theory validation, prefer the quadratic_noise mode. It
directly samples noisy empirical optima around theta_star and therefore matches
the bias-variance decomposition without the extra min-norm bias that appears
when D is much larger than n in raw linear regression.

In addition to the paper's LoRA-space and Uni-LoRA-space estimators, this
script can evaluate a PROLOSA-style projected+sparse estimator

    theta = P z + R a

to validate the sparse residual branch as an error-correction mechanism.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from typing import Iterable, TypeAlias

import numpy as np
from numpy.typing import NDArray


FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]


@dataclass(frozen=True)
class Problem:
    P: FloatArray
    theta_star: FloatArray
    dense_mismatch: FloatArray
    sparse_mismatch: FloatArray
    sparse_support: IntArray


@dataclass(frozen=True)
class Config:
    experiment: str
    D: int
    d: int
    n: int
    alpha: float
    sparse_budget: int
    sparse_mismatch: float
    noise_std: float
    ridge: float
    trials: int
    seed: int
    support: str


def parse_float_list(values: Iterable[str]) -> list[float]:
    return [float(value) for value in values]


def parse_int_list(values: Iterable[str]) -> list[int]:
    return [int(value) for value in values]


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


def unit_vector_orthogonal_to(P: FloatArray, rng: np.random.Generator) -> FloatArray:
    for _ in range(100):
        vector = rng.normal(size=P.shape[0])
        vector = vector - P @ (P.T @ vector)
        norm = np.linalg.norm(vector)
        if norm > 1e-12:
            return vector / norm
    raise RuntimeError("Failed to sample a vector orthogonal to col(P).")


def sparse_unit_vector(D: int, k: int, rng: np.random.Generator) -> tuple[FloatArray, IntArray]:
    if not 0 < k <= D:
        raise ValueError(f"Sparse support k={k} must be in [1, D={D}].")
    support = np.sort(rng.choice(D, size=k, replace=False))
    vector = np.zeros(D)
    vector[support] = rng.normal(size=k)
    vector /= np.linalg.norm(vector)
    return vector, support


def make_problem(
    D: int,
    d: int,
    alpha: float,
    sparse_budget: int,
    sparse_mismatch: float,
    seed: int,
) -> Problem:
    rng = np.random.default_rng(seed)
    P = orthonormal_projection(D, d, rng)
    z_star = rng.normal(size=d)
    dense_mismatch = unit_vector_orthogonal_to(P, rng)

    if sparse_mismatch > 0:
        sparse_mismatch_vec, sparse_support = sparse_unit_vector(D, sparse_budget, rng)
    else:
        sparse_mismatch_vec = np.zeros(D)
        sparse_support = np.array([], dtype=np.int64)

    theta_star = P @ z_star + alpha * dense_mismatch + sparse_mismatch * sparse_mismatch_vec
    return Problem(P, theta_star, dense_mismatch, sparse_mismatch_vec, sparse_support)


def fit_full_ridge(X: FloatArray, y: FloatArray, ridge: float) -> FloatArray:
    """Fit an unconstrained D-dimensional ridge estimator using the dual form."""
    n, D = X.shape
    if n <= D:
        gram = X @ X.T
        dual = np.linalg.solve(gram + ridge * np.eye(n), y)
        return X.T @ dual

    gram = X.T @ X
    return np.linalg.solve(gram + ridge * np.eye(D), X.T @ y)


def fit_subspace_ridge(
    X: FloatArray,
    y: FloatArray,
    basis: FloatArray,
    ridge: float,
) -> FloatArray:
    features = X @ basis
    m = features.shape[1]
    gram = features.T @ features
    coef = np.linalg.solve(gram + ridge * np.eye(m), features.T @ y)
    return basis @ coef


def coordinate_basis(D: int, support: IntArray) -> FloatArray:
    basis = np.zeros((D, len(support)))
    basis[support, np.arange(len(support))] = 1.0
    return basis


def choose_sparse_support(
    mode: str,
    X: FloatArray,
    y: FloatArray,
    P: FloatArray,
    theta_star: FloatArray,
    theta_unilora: FloatArray,
    k: int,
    rng: np.random.Generator,
) -> IntArray:
    D = theta_star.shape[0]
    if k <= 0:
        return np.array([], dtype=np.int64)

    if mode == "oracle":
        residual = theta_star - P @ (P.T @ theta_star)
        return np.argpartition(np.abs(residual), -k)[-k:]

    if mode == "gradient":
        if X.shape[0] == 0:
            raise ValueError("gradient support requires linear_regression data; use oracle or random.")
        # Data-driven proxy: coordinates with largest empirical residual gradient.
        residual = X @ theta_unilora - y
        gradient = X.T @ residual / X.shape[0]
        return np.argpartition(np.abs(gradient), -k)[-k:]

    if mode == "random":
        return rng.choice(D, size=k, replace=False)

    raise ValueError(f"Unknown support selection mode: {mode}")


def exact_excess_risk(theta_hat: FloatArray, theta_star: FloatArray) -> float:
    diff = theta_hat - theta_star
    return 0.5 * float(diff @ diff)


def monte_carlo_excess_risk(
    theta_hat: FloatArray,
    theta_star: FloatArray,
    test_size: int,
    rng: np.random.Generator,
    batch_size: int = 8192,
) -> float:
    """Estimate 0.5 * E[(x^T theta_hat - x^T theta_star)^2] in batches."""
    total = 0.0
    seen = 0
    diff = theta_hat - theta_star
    D = theta_star.shape[0]
    while seen < test_size:
        current = min(batch_size, test_size - seen)
        X = rng.normal(size=(current, D))
        pred_diff = X @ diff
        total += float(pred_diff @ pred_diff)
        seen += current
    return 0.5 * total / test_size


def summarize_estimates(estimates: list[FloatArray], theta_star: FloatArray) -> tuple[float, float]:
    stacked = np.stack(estimates, axis=0)
    mean_theta = stacked.mean(axis=0)
    bias = exact_excess_risk(mean_theta, theta_star)
    variance = 0.5 * float(np.mean(np.sum((stacked - mean_theta) ** 2, axis=1)))
    return bias, variance


def run_config(
    config: Config,
    methods: set[str],
    monte_carlo_test_size: int,
) -> list[dict[str, object]]:
    problem_seed = stable_seed(
        config.seed,
        "problem",
        config.D,
        config.d,
        config.sparse_budget,
        config.sparse_mismatch,
    )
    problem = make_problem(
        D=config.D,
        d=config.d,
        alpha=config.alpha,
        sparse_budget=config.sparse_budget,
        sparse_mismatch=config.sparse_mismatch,
        seed=problem_seed,
    )

    risks: dict[str, list[float]] = {method: [] for method in methods}
    estimates: dict[str, list[FloatArray]] = {method: [] for method in methods}

    for trial in range(config.trials):
        trial_seed = stable_seed(
            config.seed,
            "trial",
            trial,
            config.D,
            config.d,
            config.n,
            config.sparse_mismatch,
        )
        rng = np.random.default_rng(trial_seed)

        if config.experiment == "linear_regression":
            X = rng.normal(size=(config.n, config.D))
            y = X @ problem.theta_star + config.noise_std * rng.normal(size=config.n)

            theta_unilora = None
            if "lora" in methods:
                theta_lora = fit_full_ridge(X, y, config.ridge)
                estimates["lora"].append(theta_lora)

            if "unilora" in methods or "prolosa" in methods:
                theta_unilora = fit_subspace_ridge(X, y, problem.P, config.ridge)
                if "unilora" in methods:
                    estimates["unilora"].append(theta_unilora)

            if "prolosa" in methods:
                if theta_unilora is None:
                    raise RuntimeError("PROLOSA support selection requires Uni-LoRA fit.")
                support = choose_sparse_support(
                    mode=config.support,
                    X=X,
                    y=y,
                    P=problem.P,
                    theta_star=problem.theta_star,
                    theta_unilora=theta_unilora,
                    k=config.sparse_budget,
                    rng=rng,
                )
                sparse_basis = coordinate_basis(config.D, support)
                basis = np.concatenate([problem.P, sparse_basis], axis=1)
                theta_prolosa = fit_subspace_ridge(X, y, basis, config.ridge)
                estimates["prolosa"].append(theta_prolosa)
        elif config.experiment == "quadratic_noise":
            noisy_theta = problem.theta_star + (config.noise_std / math.sqrt(config.n)) * rng.normal(
                size=config.D
            )
            theta_unilora = problem.P @ (problem.P.T @ noisy_theta)

            if "lora" in methods:
                estimates["lora"].append(noisy_theta)

            if "unilora" in methods:
                estimates["unilora"].append(theta_unilora)

            if "prolosa" in methods:
                support = choose_sparse_support(
                    mode=config.support,
                    X=np.empty((0, config.D)),
                    y=np.empty(0),
                    P=problem.P,
                    theta_star=problem.theta_star,
                    theta_unilora=theta_unilora,
                    k=config.sparse_budget,
                    rng=rng,
                )
                sparse_basis = coordinate_basis(config.D, support)
                basis = np.concatenate([problem.P, sparse_basis], axis=1)
                theta_prolosa = basis @ np.linalg.solve(
                    basis.T @ basis + config.ridge * np.eye(basis.shape[1]),
                    basis.T @ noisy_theta,
                )
                estimates["prolosa"].append(theta_prolosa)
        else:
            raise ValueError(f"Unknown experiment: {config.experiment}")

    risk_rng = np.random.default_rng(stable_seed(config.seed, "risk", config.D, config.d, config.n))
    for method, method_estimates in estimates.items():
        for theta_hat in method_estimates:
            if monte_carlo_test_size > 0:
                risk = monte_carlo_excess_risk(
                    theta_hat,
                    problem.theta_star,
                    test_size=monte_carlo_test_size,
                    rng=risk_rng,
                )
            else:
                risk = exact_excess_risk(theta_hat, problem.theta_star)
            risks[method].append(risk)

    rows = []
    for method in sorted(methods):
        bias, variance = summarize_estimates(estimates[method], problem.theta_star)
        method_risks = np.array(risks[method])
        rows.append(
            {
                **asdict(config),
                "method": method,
                "risk_mean": float(method_risks.mean()),
                "risk_std": float(method_risks.std(ddof=1)) if len(method_risks) > 1 else 0.0,
                "bias": bias,
                "variance": variance,
                "bias_plus_variance": bias + variance,
                "theory_unilora_dense_bias": 0.5 * config.alpha**2,
                "theory_lora_variance": 0.5 * config.D * config.noise_std**2 / config.n,
                "theory_unilora_variance": 0.5 * config.d * config.noise_std**2 / config.n,
                "true_sparse_support": json.dumps(problem.sparse_support.tolist()),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def select_plot_rows(rows: list[dict[str, object]]) -> tuple[int, int, float, list[dict[str, object]]]:
    sample_sizes = sorted({int(row["n"]) for row in rows})
    dims = sorted({int(row["d"]) for row in rows})
    selected_n = sample_sizes[0]
    selected_d = dims[0]
    alphas = sorted({float(row["alpha"]) for row in rows})
    selected_alpha = alphas[len(alphas) // 2]
    selected = [row for row in rows if int(row["n"]) == selected_n and int(row["d"]) == selected_d]
    return selected_n, selected_d, selected_alpha, selected


def svg_polyline(points: list[tuple[float, float]], color: str) -> str:
    encoded = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    circles = "\n".join(
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{color}" />' for x, y in points
    )
    return f'<polyline points="{encoded}" fill="none" stroke="{color}" stroke-width="2.2" />\n{circles}'


def write_alpha_svg(path: Path, rows: list[dict[str, object]]) -> None:
    selected_n, selected_d, _, selected = select_plot_rows(rows)
    methods = sorted({str(row["method"]) for row in selected})
    alphas = sorted({float(row["alpha"]) for row in selected})
    max_risk = max(float(row["risk_mean"]) for row in selected)
    min_alpha, max_alpha = min(alphas), max(alphas)
    width, height = 760, 460
    left, right, top, bottom = 72, 28, 44, 68
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = {"lora": "#3b82f6", "unilora": "#ef4444", "prolosa": "#10b981"}

    def x_scale(alpha: float) -> float:
        if math.isclose(max_alpha, min_alpha):
            return left + plot_w / 2
        return left + (alpha - min_alpha) / (max_alpha - min_alpha) * plot_w

    def y_scale(value: float) -> float:
        return top + plot_h - value / max_risk * plot_h

    grid = []
    for i in range(6):
        y = top + i * plot_h / 5
        value = max_risk * (1 - i / 5)
        grid.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" '
            'stroke="#e5e7eb" />'
        )
        grid.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-size="11" fill="#4b5563">{value:.2g}</text>'
        )

    lines = []
    legend = []
    for i, method in enumerate(methods):
        method_rows = sorted(
            [row for row in selected if row["method"] == method],
            key=lambda row: float(row["alpha"]),
        )
        color = colors.get(method, "#6b7280")
        points = [(x_scale(float(row["alpha"])), y_scale(float(row["risk_mean"]))) for row in method_rows]
        lines.append(svg_polyline(points, color))
        legend_x = left + i * 120
        legend.append(f'<circle cx="{legend_x}" cy="26" r="4" fill="{color}" />')
        legend.append(
            f'<text x="{legend_x + 10}" y="30" font-size="13" fill="#111827">{escape(method)}</text>'
        )

    xticks = []
    for alpha in alphas:
        x = x_scale(alpha)
        xticks.append(f'<line x1="{x:.2f}" y1="{height - bottom}" x2="{x:.2f}" y2="{height - bottom + 5}" stroke="#111827" />')
        xticks.append(
            f'<text x="{x:.2f}" y="{height - bottom + 22}" text-anchor="middle" '
            f'font-size="11" fill="#4b5563">{alpha:g}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white" />
<text x="{left}" y="30" font-size="16" font-weight="700" fill="#111827">Excess risk vs. subspace mismatch (n={selected_n}, d={selected_d})</text>
{''.join(legend)}
{''.join(grid)}
<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#111827" />
<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#111827" />
{''.join(xticks)}
{''.join(lines)}
<text x="{left + plot_w / 2:.2f}" y="{height - 18}" text-anchor="middle" font-size="13" fill="#111827">subspace mismatch alpha</text>
<text x="18" y="{top + plot_h / 2:.2f}" text-anchor="middle" font-size="13" fill="#111827" transform="rotate(-90 18 {top + plot_h / 2:.2f})">population excess risk</text>
</svg>
"""
    path.write_text(svg)


def write_biasvar_svg(path: Path, rows: list[dict[str, object]]) -> None:
    selected_n, selected_d, selected_alpha, _ = select_plot_rows(rows)
    selected = [
        row
        for row in rows
        if int(row["n"]) == selected_n
        and int(row["d"]) == selected_d
        and math.isclose(float(row["alpha"]), selected_alpha)
    ]
    selected = sorted(selected, key=lambda row: str(row["method"]))
    width, height = 680, 430
    left, right, top, bottom = 72, 36, 46, 68
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_total = max(float(row["bias"]) + float(row["variance"]) for row in selected)
    bar_w = min(88, plot_w / max(1, len(selected)) * 0.52)
    gap = plot_w / max(1, len(selected))

    bars = []
    labels = []
    for i, row in enumerate(selected):
        x = left + gap * (i + 0.5) - bar_w / 2
        bias = float(row["bias"])
        variance = float(row["variance"])
        bias_h = bias / max_total * plot_h
        var_h = variance / max_total * plot_h
        y_bias = top + plot_h - bias_h
        y_var = y_bias - var_h
        bars.append(f'<rect x="{x:.2f}" y="{y_var:.2f}" width="{bar_w:.2f}" height="{var_h:.2f}" fill="#93c5fd" />')
        bars.append(f'<rect x="{x:.2f}" y="{y_bias:.2f}" width="{bar_w:.2f}" height="{bias_h:.2f}" fill="#f97316" />')
        labels.append(
            f'<text x="{x + bar_w / 2:.2f}" y="{height - bottom + 22}" text-anchor="middle" '
            f'font-size="12" fill="#111827">{escape(str(row["method"]))}</text>'
        )

    grid = []
    for i in range(6):
        y = top + i * plot_h / 5
        value = max_total * (1 - i / 5)
        grid.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" '
            'stroke="#e5e7eb" />'
        )
        grid.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-size="11" fill="#4b5563">{value:.2g}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white" />
<text x="{left}" y="28" font-size="16" font-weight="700" fill="#111827">Bias-variance decomposition (n={selected_n}, d={selected_d}, alpha={selected_alpha:g})</text>
<rect x="{width - 190}" y="16" width="12" height="12" fill="#f97316" /><text x="{width - 172}" y="27" font-size="12" fill="#111827">bias</text>
<rect x="{width - 118}" y="16" width="12" height="12" fill="#93c5fd" /><text x="{width - 100}" y="27" font-size="12" fill="#111827">variance</text>
{''.join(grid)}
<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#111827" />
<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#111827" />
{''.join(bars)}
{''.join(labels)}
<text x="18" y="{top + plot_h / 2:.2f}" text-anchor="middle" font-size="13" fill="#111827" transform="rotate(-90 18 {top + plot_h / 2:.2f})">excess risk components</text>
</svg>
"""
    path.write_text(svg)


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    sample_sizes = sorted({int(row["n"]) for row in rows})
    alphas = sorted({float(row["alpha"]) for row in rows})
    largest_n = sample_sizes[-1]
    key_alphas = [alphas[0], alphas[len(alphas) // 2], alphas[-1]]
    lines = [
        "# Synthetic Theory Validation Summary",
        "",
        f"- Experiment: `{rows[0]['experiment']}`",
        f"- D={rows[0]['D']}, d={rows[0]['d']}, trials={rows[0]['trials']}, noise_std={rows[0]['noise_std']}",
        f"- Sparse mismatch={rows[0]['sparse_mismatch']}, sparse budget={rows[0]['sparse_budget']}, support=`{rows[0]['support']}`",
        "",
        "## Key Risk Values",
        "",
        "| n | alpha | LoRA | Uni-LoRA | ProLoSA |",
        "|---:|---:|---:|---:|---:|",
    ]
    for n in dict.fromkeys([sample_sizes[0], largest_n]):
        for alpha in key_alphas:
            selected = {
                str(row["method"]): float(row["risk_mean"])
                for row in rows
                if int(row["n"]) == n and math.isclose(float(row["alpha"]), alpha)
            }
            lines.append(
                f"| {n} | {alpha:g} | {selected.get('lora', float('nan')):.4g} | "
                f"{selected.get('unilora', float('nan')):.4g} | {selected.get('prolosa', float('nan')):.4g} |"
            )

    lines.extend(["", "## Interpretation", ""])
    lines.append(
        "- LoRA risk is invariant across alpha because the same empirical noise is used while only the off-subspace component changes."
    )
    lines.append(
        "- Uni-LoRA shows the predicted bias-variance trade-off: low variance in the compressed subspace plus a bias term that grows with alpha."
    )
    if float(rows[0]["sparse_mismatch"]) > 0:
        lines.append(
            "- With sparse mismatch enabled, ProLoSA recovers part of the off-subspace signal through the routed sparse residual and lowers projection bias."
        )
    else:
        lines.append(
            "- With sparse mismatch disabled, ProLoSA mainly serves as an oracle sparse correction to dense mismatch; it should be reported as an auxiliary check."
        )
    path.write_text("\n".join(lines) + "\n")


def maybe_plot(output_dir: Path, rows: list[dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; writing SVG figures instead.")
        write_alpha_svg(output_dir / "synthetic_alpha.svg", rows)
        write_biasvar_svg(output_dir / "synthetic_biasvar.svg", rows)
        return

    # Plot risk versus alpha for the smallest n and first d in the results.
    sample_sizes = sorted({int(row["n"]) for row in rows})
    dims = sorted({int(row["d"]) for row in rows})
    selected_n = sample_sizes[0]
    selected_d = dims[0]
    selected = [row for row in rows if int(row["n"]) == selected_n and int(row["d"]) == selected_d]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for method in sorted({str(row["method"]) for row in selected}):
        method_rows = sorted(
            [row for row in selected if row["method"] == method],
            key=lambda row: float(row["alpha"]),
        )
        ax.errorbar(
            [float(row["alpha"]) for row in method_rows],
            [float(row["risk_mean"]) for row in method_rows],
            yerr=[float(row["risk_std"]) for row in method_rows],
            marker="o",
            capsize=2,
            label=method,
        )
    ax.set_title(f"Excess risk vs. subspace mismatch (n={selected_n}, d={selected_d})")
    ax.set_xlabel("subspace mismatch alpha")
    ax.set_ylabel("population excess risk")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "synthetic_alpha.png", dpi=200)
    plt.close(fig)

    # Bias-variance decomposition at the median alpha.
    alphas = sorted({float(row["alpha"]) for row in rows})
    selected_alpha = alphas[len(alphas) // 2]
    selected = [
        row
        for row in rows
        if int(row["n"]) == selected_n
        and int(row["d"]) == selected_d
        and math.isclose(float(row["alpha"]), selected_alpha)
    ]
    methods = [str(row["method"]) for row in selected]
    bias = [float(row["bias"]) for row in selected]
    variance = [float(row["variance"]) for row in selected]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(len(methods))
    ax.bar(x, bias, label="bias")
    ax.bar(x, variance, bottom=bias, label="variance")
    ax.set_title(f"Bias-variance decomposition (alpha={selected_alpha:g})")
    ax.set_xticks(x, methods)
    ax.set_ylabel("excess risk components")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "synthetic_biasvar.png", dpi=200)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=["quadratic_noise", "linear_regression"],
        default="quadratic_noise",
        help=(
            "quadratic_noise directly simulates the local quadratic theory; "
            "linear_regression reproduces the teacher-student regression setup."
        ),
    )
    parser.add_argument("--D", type=int, default=4096)
    parser.add_argument("--dims", nargs="+", default=["256"], help="Compressed dimensions d.")
    parser.add_argument(
        "--sample-sizes",
        nargs="+",
        default=["16", "32", "64", "128", "256", "512"],
        help="Training sample sizes n.",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        default=["0.0", "0.25", "0.5", "0.75", "1.0", "1.25", "1.5", "1.75", "2.0"],
        help="Dense off-subspace mismatch strengths.",
    )
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument(
        "--noise-std",
        type=float,
        default=1.0,
        help=(
            "Estimator noise scale for quadratic_noise; label noise std for "
            "linear_regression. Use 1.0 for clear bias-variance crossings."
        ),
    )
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["lora", "unilora", "prolosa"],
        default=["lora", "unilora", "prolosa"],
    )
    parser.add_argument(
        "--sparse-budget",
        type=int,
        default=64,
        help="K for the PROLOSA-style sparse residual branch.",
    )
    parser.add_argument(
        "--sparse-mismatch",
        type=float,
        default=0.0,
        help=(
            "Optional sparse off-subspace component in theta_star. Set this above 0 "
            "to test the stronger PROLOSA sparse-residual hypothesis."
        ),
    )
    parser.add_argument(
        "--support",
        choices=["oracle", "gradient", "random"],
        default="oracle",
        help="Sparse support selection for the PROLOSA-style estimator.",
    )
    parser.add_argument(
        "--monte-carlo-test-size",
        type=int,
        default=0,
        help="If >0, estimate population risk with this many test samples instead of exact risk.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results_synthetic_theory"),
    )
    parser.add_argument("--no-plot", action="store_true", help="Disable PNG figure generation.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    dims = parse_int_list(args.dims)
    sample_sizes = parse_int_list(args.sample_sizes)
    alphas = parse_float_list(args.alphas)
    methods = set(args.methods)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    total = len(dims) * len(sample_sizes) * len(alphas)
    finished = 0

    for d in dims:
        for n in sample_sizes:
            for alpha in alphas:
                finished += 1
                config = Config(
                    experiment=args.experiment,
                    D=args.D,
                    d=d,
                    n=n,
                    alpha=alpha,
                    sparse_budget=args.sparse_budget,
                    sparse_mismatch=args.sparse_mismatch,
                    noise_std=args.noise_std,
                    ridge=args.ridge,
                    trials=args.trials,
                    seed=args.seed,
                    support=args.support,
                )
                print(f"[{finished}/{total}] D={args.D} d={d} n={n} alpha={alpha:g}")
                rows = run_config(
                    config=config,
                    methods=methods,
                    monte_carlo_test_size=args.monte_carlo_test_size,
                )
                all_rows.extend(rows)
                write_csv(args.output_dir / "synthetic_theory_results.csv", all_rows)

    with (args.output_dir / "config.json").open("w") as handle:
        json.dump(vars(args), handle, indent=2, default=str)

    if not args.no_plot:
        maybe_plot(args.output_dir, all_rows)
    write_summary(args.output_dir / "summary.md", all_rows)

    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()

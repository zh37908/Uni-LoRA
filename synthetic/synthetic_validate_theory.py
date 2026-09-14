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

Hidden-P / non-oracle projection mode
-------------------------------------
To address the reviewer concern that ProLoSA cannot observe the true
generating projection, use --experiment hidden_p. Ground truth is

    theta_star = P_T z_star + gamma q_star

where P_T is used only to synthesize theta_star. Learners may only access a
misspecified projection P_M != P_T (controlled by --pm-angle-deg). Compared
methods are:

    lora              full-space estimator
    unilora           Uni-LoRA with P_M
    prolosa           ProLoSA with P_M + warmup SNIP residual
    unilora_oracle    Uni-LoRA with P_T (upper bound only)
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

from fixed_target import target_id

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
    P_teacher: FloatArray | None = None
    pm_mode: str = "rotated"
    pm_angle_deg: float = 0.0
    subspace_overlap: float = 1.0

    @property
    def P_learner(self) -> FloatArray:
        return self.P


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
    pm_mode: str = "rotated"
    pm_angle_deg: float = 0.0


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


def orthonormalize(matrix: FloatArray) -> FloatArray:
    q, r = np.linalg.qr(matrix, mode="reduced")
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs


def rotate_subspace(
    P_teacher: FloatArray,
    angle_deg: float,
    rng: np.random.Generator,
) -> tuple[FloatArray, float]:
    """Build P_M by rotating each teacher column toward an orthogonal direction.

    angle_deg=0 keeps P_M = P_T. angle_deg=90 yields a fully orthogonal learner
    subspace (up to numerical error). Intermediate angles create a controlled
    non-oracle projection mismatch while preserving a shared component.
    """
    D, d = P_teacher.shape
    if not 0.0 <= angle_deg <= 90.0:
        raise ValueError(f"pm_angle_deg must lie in [0, 90], got {angle_deg}.")

    if math.isclose(angle_deg, 0.0):
        return P_teacher.copy(), 1.0

    angle = math.radians(angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    # Sample an orthonormal frame orthogonal to col(P_T).
    ambient = rng.normal(size=(D, d))
    ambient = ambient - P_teacher @ (P_teacher.T @ ambient)
    Q_perp = orthonormalize(ambient)

    P_model = orthonormalize(cos_a * P_teacher + sin_a * Q_perp)
    gram = P_teacher.T @ P_model
    # Mean squared canonical cosine as a scalar overlap summary.
    singular_values = np.linalg.svd(gram, compute_uv=False)
    overlap = float(np.mean(singular_values**2))
    return P_model, overlap


def subspace_overlap(P_teacher: FloatArray, P_model: FloatArray) -> float:
    """Mean squared canonical cosine between teacher and learner subspaces."""
    singular_values = np.linalg.svd(P_teacher.T @ P_model, compute_uv=False)
    return float(np.mean(singular_values**2))


def make_model_projection(
    P_teacher: FloatArray,
    pm_mode: str,
    angle_deg: float,
    rng: np.random.Generator,
) -> tuple[FloatArray, float]:
    if pm_mode == "rotated":
        return rotate_subspace(P_teacher, angle_deg, rng)
    if pm_mode == "independent":
        P_model = orthonormal_projection(P_teacher.shape[0], P_teacher.shape[1], rng)
        return P_model, subspace_overlap(P_teacher, P_model)
    raise ValueError(f"Unknown pm_mode: {pm_mode}")


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
    *,
    hidden_p: bool = False,
    pm_mode: str = "rotated",
    pm_angle_deg: float = 0.0,
) -> Problem:
    rng = np.random.default_rng(seed)
    P_teacher = orthonormal_projection(D, d, rng)
    z_star = rng.normal(size=d)

    if hidden_p:
        # Residual q_star is sparse / coordinate-concentrated and orthogonal to
        # the teacher subspace, so ProLoSA can still recover it through R even
        # when learners only see a misspecified P_M.
        if sparse_mismatch > 0:
            q_star = np.zeros(D)
            sparse_support = np.sort(rng.choice(D, size=sparse_budget, replace=False))
            q_star[sparse_support] = rng.normal(size=sparse_budget)
            # Remove any accidental teacher-subspace component, then renormalize.
            q_star = q_star - P_teacher @ (P_teacher.T @ q_star)
            q_norm = np.linalg.norm(q_star)
            if q_norm <= 1e-12:
                raise RuntimeError("Failed to build a sparse residual orthogonal to P_T.")
            q_star /= q_norm
        else:
            q_star = unit_vector_orthogonal_to(P_teacher, rng)
            sparse_support = np.array([], dtype=np.int64)
        theta_star = P_teacher @ z_star + alpha * q_star
        # Keep the teacher RNG independent of the learner projection. Changing
        # angle/mode must not change theta_star or consume its random stream.
        model_rng = np.random.default_rng(stable_seed(seed, "learner_projection", pm_mode, pm_angle_deg))
        P_model, overlap = make_model_projection(P_teacher, pm_mode, pm_angle_deg, model_rng)
        return Problem(
            P=P_model,
            theta_star=theta_star,
            dense_mismatch=q_star,
            sparse_mismatch=q_star if sparse_mismatch > 0 else np.zeros(D),
            sparse_support=sparse_support,
            P_teacher=P_teacher,
            pm_mode=pm_mode,
            pm_angle_deg=pm_angle_deg,
            subspace_overlap=overlap,
        )

    P = P_teacher
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


def prolosa_basis(P: FloatArray, support: IntArray, tol: float = 1e-8) -> FloatArray:
    """Concatenate P with coordinate residuals orthogonalized against P.

    Orthogonalization avoids near-singular Gram matrices when selected
    coordinates already lie mostly inside col(P).
    """
    if len(support) == 0:
        return P
    sparse = coordinate_basis(P.shape[0], support)
    residual = sparse - P @ (P.T @ sparse)
    keep = []
    for col in residual.T:
        norm = np.linalg.norm(col)
        if norm > tol:
            keep.append(col / norm)
    if not keep:
        return P
    return np.concatenate([P, np.column_stack(keep)], axis=1)


def choose_sparse_support(
    mode: str,
    X: FloatArray,
    y: FloatArray,
    P: FloatArray,
    theta_star: FloatArray,
    theta_unilora: FloatArray,
    theta_observed: FloatArray | None,
    k: int,
    rng: np.random.Generator,
) -> IntArray:
    D = theta_star.shape[0]
    if k <= 0:
        return np.array([], dtype=np.int64)

    if mode == "oracle":
        residual = theta_star - P @ (P.T @ theta_star)
        return np.argpartition(np.abs(residual), -k)[-k:]

    if mode in {"gradient", "snip"}:
        if X.shape[0] > 0:
            # Data-driven proxy: coordinates with largest empirical residual gradient.
            residual = X @ theta_unilora - y
            gradient = X.T @ residual / X.shape[0]
        elif theta_observed is not None:
            # In quadratic_noise mode, the local objective is 0.5 ||theta - theta_observed||^2.
            gradient = theta_unilora - theta_observed
        else:
            raise ValueError(f"{mode} support requires data or an observed noisy optimum.")

    if mode == "gradient":
        return np.argpartition(np.abs(gradient), -k)[-k:]

    if mode == "snip":
        # The sparse branch is opened from zero, so its one-step candidate
        # weight is proportional to -gradient. SNIP saliency |g * w| therefore
        # reduces to gradient energy for this linear synthetic model.
        saliency = gradient**2
        return np.argpartition(saliency, -k)[-k:]

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
    from fixed_target import fixed_target_decomposition
    _, bias, variance = fixed_target_decomposition(estimates, theta_star)
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
        config.experiment,
        config.pm_mode,
        config.pm_angle_deg,
    )
    if config.experiment == "hidden_p":
        problem_seed = stable_seed(config.seed, "fixed_hidden_teacher_v2", config.D,
                                   config.d, config.sparse_budget, config.sparse_mismatch)
    problem = make_problem(
        D=config.D,
        d=config.d,
        alpha=config.alpha,
        sparse_budget=config.sparse_budget,
        sparse_mismatch=config.sparse_mismatch,
        seed=problem_seed,
        hidden_p=config.experiment == "hidden_p",
        pm_mode=config.pm_mode,
        pm_angle_deg=config.pm_angle_deg,
    )
    P_learner = problem.P
    P_teacher = problem.P_teacher if problem.P_teacher is not None else problem.P

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
            config.experiment,
            config.pm_mode,
            config.pm_angle_deg,
        )
        if config.experiment == "hidden_p":
            trial_seed = stable_seed(config.seed, "fixed_hidden_trial_v2", trial,
                                     config.D, config.d, config.n, config.sparse_mismatch)
        rng = np.random.default_rng(trial_seed)

        if config.experiment in {"linear_regression", "hidden_p"}:
            X = rng.normal(size=(config.n, config.D))
            y = X @ problem.theta_star + config.noise_std * rng.normal(size=config.n)

            theta_unilora = None
            if "lora" in methods:
                theta_lora = fit_full_ridge(X, y, config.ridge)
                estimates["lora"].append(theta_lora)

            if "unilora" in methods or "prolosa" in methods:
                theta_unilora = fit_subspace_ridge(X, y, P_learner, config.ridge)
                if "unilora" in methods:
                    estimates["unilora"].append(theta_unilora)

            if "unilora_oracle" in methods:
                theta_oracle = fit_subspace_ridge(X, y, P_teacher, config.ridge)
                estimates["unilora_oracle"].append(theta_oracle)

            if "prolosa" in methods:
                if theta_unilora is None:
                    raise RuntimeError("PROLOSA support selection requires Uni-LoRA fit.")
                support = choose_sparse_support(
                    mode=config.support,
                    X=X,
                    y=y,
                    P=P_learner,
                    theta_star=problem.theta_star,
                    theta_unilora=theta_unilora,
                    theta_observed=None,
                    k=config.sparse_budget,
                    rng=rng,
                )
                basis = prolosa_basis(P_learner, support)
                theta_prolosa = fit_subspace_ridge(X, y, basis, config.ridge)
                estimates["prolosa"].append(theta_prolosa)
        elif config.experiment == "quadratic_noise":
            noisy_theta = problem.theta_star + (config.noise_std / math.sqrt(config.n)) * rng.normal(
                size=config.D
            )
            theta_unilora = P_learner @ (P_learner.T @ noisy_theta)

            if "lora" in methods:
                estimates["lora"].append(noisy_theta)

            if "unilora" in methods:
                estimates["unilora"].append(theta_unilora)

            if "unilora_oracle" in methods:
                estimates["unilora_oracle"].append(P_teacher @ (P_teacher.T @ noisy_theta))

            if "prolosa" in methods:
                support = choose_sparse_support(
                    mode=config.support,
                    X=np.empty((0, config.D)),
                    y=np.empty(0),
                    P=P_learner,
                    theta_star=problem.theta_star,
                    theta_unilora=theta_unilora,
                    theta_observed=noisy_theta,
                    k=config.sparse_budget,
                    rng=rng,
                )
                basis = prolosa_basis(P_learner, support)
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
                "target_id": target_id(problem.theta_star),
                "target_protocol": "fixed_hidden_teacher_v2" if config.experiment == "hidden_p" else "fixed_per_config",
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
                "pm_mode": problem.pm_mode,
                "subspace_overlap": problem.subspace_overlap,
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


def select_plot_rows(rows: list[dict[str, object]]) -> tuple[int, int, float, float, list[dict[str, object]]]:
    sample_sizes = sorted({int(row["n"]) for row in rows})
    dims = sorted({int(row["d"]) for row in rows})
    selected_n = sample_sizes[0]
    selected_d = dims[0]
    alphas = sorted({float(row["alpha"]) for row in rows})
    selected_alpha = alphas[len(alphas) // 2]
    noise_stds = sorted({float(row["noise_std"]) for row in rows})
    selected_noise = noise_stds[len(noise_stds) // 2]
    selected = [
        row
        for row in rows
        if int(row["n"]) == selected_n
        and int(row["d"]) == selected_d
        and math.isclose(float(row["noise_std"]), selected_noise)
    ]
    return selected_n, selected_d, selected_alpha, selected_noise, selected


def svg_polyline(points: list[tuple[float, float]], color: str) -> str:
    encoded = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    circles = "\n".join(
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{color}" />' for x, y in points
    )
    return f'<polyline points="{encoded}" fill="none" stroke="{color}" stroke-width="2.2" />\n{circles}'


def write_alpha_svg(path: Path, rows: list[dict[str, object]]) -> None:
    selected_n, selected_d, _, selected_noise, selected = select_plot_rows(rows)
    methods = sorted({str(row["method"]) for row in selected})
    alphas = sorted({float(row["alpha"]) for row in selected})
    max_risk = max(float(row["risk_mean"]) for row in selected)
    min_alpha, max_alpha = min(alphas), max(alphas)
    width, height = 760, 460
    left, right, top, bottom = 72, 28, 44, 68
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = {"lora": "#3b82f6", "unilora": "#ef4444", "prolosa": "#10b981", "unilora_oracle": "#8b5cf6"}

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
<text x="{left}" y="30" font-size="16" font-weight="700" fill="#111827">Excess risk vs. subspace mismatch (n={selected_n}, d={selected_d}, noise={selected_noise:g})</text>
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


def write_noise_svg(path: Path, rows: list[dict[str, object]]) -> None:
    selected_n, selected_d, selected_alpha, _, _ = select_plot_rows(rows)
    selected = [
        row
        for row in rows
        if int(row["n"]) == selected_n
        and int(row["d"]) == selected_d
        and math.isclose(float(row["alpha"]), selected_alpha)
    ]
    methods = sorted({str(row["method"]) for row in selected})
    noise_stds = sorted({float(row["noise_std"]) for row in selected})
    max_risk = max(float(row["risk_mean"]) for row in selected)
    min_noise, max_noise = min(noise_stds), max(noise_stds)
    width, height = 760, 460
    left, right, top, bottom = 72, 28, 44, 68
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = {"lora": "#3b82f6", "unilora": "#ef4444", "prolosa": "#10b981", "unilora_oracle": "#8b5cf6"}
    y_den = max_risk if max_risk > 0 else 1.0

    def x_scale(noise_std: float) -> float:
        if math.isclose(max_noise, min_noise):
            return left + plot_w / 2
        return left + (noise_std - min_noise) / (max_noise - min_noise) * plot_w

    def y_scale(value: float) -> float:
        return top + plot_h - value / y_den * plot_h

    grid = []
    for i in range(6):
        y = top + i * plot_h / 5
        value = y_den * (1 - i / 5)
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
            key=lambda row: float(row["noise_std"]),
        )
        color = colors.get(method, "#6b7280")
        points = [(x_scale(float(row["noise_std"])), y_scale(float(row["risk_mean"]))) for row in method_rows]
        lines.append(svg_polyline(points, color))
        legend_x = left + i * 120
        legend.append(f'<circle cx="{legend_x}" cy="26" r="4" fill="{color}" />')
        legend.append(
            f'<text x="{legend_x + 10}" y="30" font-size="13" fill="#111827">{escape(method)}</text>'
        )

    xticks = []
    for noise_std in noise_stds:
        x = x_scale(noise_std)
        xticks.append(f'<line x1="{x:.2f}" y1="{height - bottom}" x2="{x:.2f}" y2="{height - bottom + 5}" stroke="#111827" />')
        xticks.append(
            f'<text x="{x:.2f}" y="{height - bottom + 22}" text-anchor="middle" '
            f'font-size="11" fill="#4b5563">{noise_std:g}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white" />
<text x="{left}" y="30" font-size="16" font-weight="700" fill="#111827">Excess risk vs. noise level (n={selected_n}, d={selected_d}, alpha={selected_alpha:g})</text>
{''.join(legend)}
{''.join(grid)}
<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#111827" />
<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#111827" />
{''.join(xticks)}
{''.join(lines)}
<text x="{left + plot_w / 2:.2f}" y="{height - 18}" text-anchor="middle" font-size="13" fill="#111827">noise std</text>
<text x="18" y="{top + plot_h / 2:.2f}" text-anchor="middle" font-size="13" fill="#111827" transform="rotate(-90 18 {top + plot_h / 2:.2f})">population excess risk</text>
</svg>
"""
    path.write_text(svg)


def write_biasvar_svg(path: Path, rows: list[dict[str, object]]) -> None:
    selected_n, selected_d, selected_alpha, selected_noise, _ = select_plot_rows(rows)
    selected = [
        row
        for row in rows
        if int(row["n"]) == selected_n
        and int(row["d"]) == selected_d
        and math.isclose(float(row["alpha"]), selected_alpha)
        and math.isclose(float(row["noise_std"]), selected_noise)
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
<text x="{left}" y="28" font-size="16" font-weight="700" fill="#111827">Bias-variance decomposition (n={selected_n}, d={selected_d}, alpha={selected_alpha:g}, noise={selected_noise:g})</text>
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
    noise_stds = sorted({float(row["noise_std"]) for row in rows})
    largest_n = sample_sizes[-1]
    key_alphas = [alphas[0], alphas[len(alphas) // 2], alphas[-1]]
    selected_noise = noise_stds[len(noise_stds) // 2]
    methods_present = sorted({str(row["method"]) for row in rows})
    method_header = " | ".join(
        {
            "lora": "LoRA",
            "unilora": "Uni-LoRA",
            "prolosa": "ProLoSA",
            "unilora_oracle": "Uni-LoRA (oracle P_T)",
        }.get(m, m)
        for m in ["lora", "unilora", "prolosa", "unilora_oracle"]
        if m in methods_present
    )
    lines = [
        "# Synthetic Theory Validation Summary",
        "",
        f"- Experiment: `{rows[0]['experiment']}`",
        f"- D={rows[0]['D']}, d={rows[0]['d']}, trials={rows[0]['trials']}, noise_stds={noise_stds}",
        f"- Sparse mismatch={rows[0]['sparse_mismatch']}, sparse budget={rows[0]['sparse_budget']}, support=`{rows[0]['support']}`",
    ]
    if rows[0].get("experiment") == "hidden_p":
        lines.append(
            f"- Hidden-P P_M mode=`{rows[0].get('pm_mode', 'rotated')}`, "
            f"angle={rows[0].get('pm_angle_deg', 0)} deg, "
            f"subspace overlap={float(rows[0].get('subspace_overlap', float('nan'))):.4f}"
        )
    lines.extend(
        [
            "",
            f"## Key Risk Values at noise_std={selected_noise:g}",
            "",
            f"| n | alpha | {method_header} |",
            "|---:|---:|" + "|".join(["---:" for _ in method_header.split(" | ")]) + "|",
        ]
    )
    display_methods = [m for m in ["lora", "unilora", "prolosa", "unilora_oracle"] if m in methods_present]
    for n in dict.fromkeys([sample_sizes[0], largest_n]):
        for alpha in key_alphas:
            selected = {
                str(row["method"]): float(row["risk_mean"])
                for row in rows
                if int(row["n"]) == n and math.isclose(float(row["alpha"]), alpha)
                and math.isclose(float(row["noise_std"]), selected_noise)
            }
            values = " | ".join(f"{selected.get(m, float('nan')):.4g}" for m in display_methods)
            lines.append(f"| {n} | {alpha:g} | {values} |")

    lines.extend(["", "## Interpretation", ""])
    if rows[0].get("experiment") == "hidden_p":
        lines.append(
            "- Teacher projection P_T generates theta_star but is hidden from Uni-LoRA/ProLoSA, which only see P_M."
        )
        lines.append(
            "- unilora_oracle uses P_T and is reported only as an upper bound, not as a deployable method."
        )
        lines.append(
            "- If the residual is sparse/coordinate-concentrated, ProLoSA with P_M + SNIP should still beat Uni-LoRA with P_M."
        )
    else:
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
        write_noise_svg(output_dir / "synthetic_noise.svg", rows)
        write_biasvar_svg(output_dir / "synthetic_biasvar.svg", rows)
        return

    # Plot risk versus alpha at the median noise level for the smallest n and first d.
    sample_sizes = sorted({int(row["n"]) for row in rows})
    dims = sorted({int(row["d"]) for row in rows})
    alphas = sorted({float(row["alpha"]) for row in rows})
    noise_stds = sorted({float(row["noise_std"]) for row in rows})
    selected_n = sample_sizes[0]
    selected_d = dims[0]
    selected_alpha = alphas[len(alphas) // 2]
    selected_noise = noise_stds[len(noise_stds) // 2]
    selected = [
        row
        for row in rows
        if int(row["n"]) == selected_n
        and int(row["d"]) == selected_d
        and math.isclose(float(row["noise_std"]), selected_noise)
    ]

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
    ax.set_title(
        f"Excess risk vs. subspace mismatch (n={selected_n}, d={selected_d}, noise={selected_noise:g})"
    )
    ax.set_xlabel("subspace mismatch alpha")
    ax.set_ylabel("population excess risk")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "synthetic_alpha.png", dpi=200)
    plt.close(fig)

    # Plot risk versus noise level at the median alpha.
    selected = [
        row
        for row in rows
        if int(row["n"]) == selected_n
        and int(row["d"]) == selected_d
        and math.isclose(float(row["alpha"]), selected_alpha)
    ]
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for method in sorted({str(row["method"]) for row in selected}):
        method_rows = sorted(
            [row for row in selected if row["method"] == method],
            key=lambda row: float(row["noise_std"]),
        )
        ax.errorbar(
            [float(row["noise_std"]) for row in method_rows],
            [float(row["risk_mean"]) for row in method_rows],
            yerr=[float(row["risk_std"]) for row in method_rows],
            marker="o",
            capsize=2,
            label=method,
        )
    ax.set_title(f"Excess risk vs. noise level (n={selected_n}, d={selected_d}, alpha={selected_alpha:g})")
    ax.set_xlabel("noise std")
    ax.set_ylabel("population excess risk")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "synthetic_noise.png", dpi=200)
    plt.close(fig)

    # Bias-variance decomposition at the median alpha and median noise level.
    selected = [
        row
        for row in rows
        if int(row["n"]) == selected_n
        and int(row["d"]) == selected_d
        and math.isclose(float(row["alpha"]), selected_alpha)
        and math.isclose(float(row["noise_std"]), selected_noise)
    ]
    methods = [str(row["method"]) for row in selected]
    bias = [float(row["bias"]) for row in selected]
    variance = [float(row["variance"]) for row in selected]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(len(methods))
    ax.bar(x, bias, label="bias")
    ax.bar(x, variance, bottom=bias, label="variance")
    ax.set_title(f"Bias-variance decomposition (alpha={selected_alpha:g}, noise={selected_noise:g})")
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
        choices=["quadratic_noise", "linear_regression", "hidden_p"],
        default="quadratic_noise",
        help=(
            "quadratic_noise directly simulates the local quadratic theory; "
            "linear_regression reproduces the teacher-student regression setup; "
            "hidden_p uses teacher P_T to generate theta_star while learners only see P_M."
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
        help="Off-subspace residual strength gamma (alpha in code).",
    )
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument(
        "--noise-std",
        type=float,
        default=1.0,
        help=(
            "Estimator noise scale for quadratic_noise; label noise std for "
            "linear_regression/hidden_p. Use 1.0 for clear bias-variance crossings."
        ),
    )
    parser.add_argument(
        "--noise-stds",
        nargs="+",
        default=None,
        help="Optional list of noise std values to sweep. Overrides --noise-std when provided.",
    )
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["lora", "unilora", "prolosa", "unilora_oracle"],
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
            "For linear_regression/quadratic_noise: optional sparse off-subspace "
            "component scale. For hidden_p: if >0, residual q_star is sparse with "
            "this budget; the residual strength itself is controlled by --alphas."
        ),
    )
    parser.add_argument(
        "--pm-mode",
        choices=["rotated", "independent"],
        default="rotated",
        help=(
            "Hidden-P only: how to construct learner P_M. rotated gives a controlled "
            "angle from P_T; independent samples a fresh random subspace as a stress test."
        ),
    )
    parser.add_argument(
        "--pm-angle-deg",
        type=float,
        default=30.0,
        help=(
            "Hidden-P only: principal angle (degrees) between teacher P_T and "
            "learner P_M. 0 keeps P_M=P_T; 90 makes them orthogonal."
        ),
    )
    parser.add_argument(
        "--support",
        choices=["oracle", "gradient", "snip", "random"],
        default="oracle",
        help=(
            "Sparse support selection for the PROLOSA-style estimator. "
            "snip uses sparse-branch one-step gradient energy."
        ),
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
    noise_stds = parse_float_list(args.noise_stds) if args.noise_stds is not None else [args.noise_std]
    methods = set(args.methods)
    if args.experiment == "hidden_p":
        methods.add("unilora_oracle")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    total = len(dims) * len(sample_sizes) * len(alphas) * len(noise_stds)
    finished = 0

    for d in dims:
        for n in sample_sizes:
            for alpha in alphas:
                for noise_std in noise_stds:
                    finished += 1
                    config = Config(
                        experiment=args.experiment,
                        D=args.D,
                        d=d,
                        n=n,
                        alpha=alpha,
                        sparse_budget=args.sparse_budget,
                        sparse_mismatch=args.sparse_mismatch,
                        noise_std=noise_std,
                        ridge=args.ridge,
                        trials=args.trials,
                        seed=args.seed,
                        support=args.support,
                        pm_mode=args.pm_mode,
                        pm_angle_deg=args.pm_angle_deg,
                    )
                    print(
                        f"[{finished}/{total}] D={args.D} d={d} n={n} "
                        f"alpha={alpha:g} noise={noise_std:g}"
                        + (
                            f" pm_mode={args.pm_mode} pm_angle={args.pm_angle_deg:g}"
                            if args.experiment == "hidden_p"
                            else ""
                        )
                    )
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

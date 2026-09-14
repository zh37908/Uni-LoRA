#!/usr/bin/env python
"""Sweep capacity, sample size, and support quality to reduce estimator bias."""

from __future__ import annotations

import csv
import importlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from synthetic_contaminated_gaussian_noise_biasvar import Config, run_experiment


OUTPUT_DIR = Path("results_contaminated_gaussian_bias_sweep")
TRIALS = 100
CONTAMINATION_RATES = [0.0, 0.01, 0.05]
NOISE_STDS = [1.5]


@dataclass(frozen=True)
class Strategy:
    name: str
    d: int
    sparse_budget: int
    n_eff: int
    support: str = "noisy_residual"


STRATEGIES = [
    Strategy("baseline", d=16, sparse_budget=32, n_eff=512),
    Strategy("samples_4x", d=16, sparse_budget=32, n_eff=2048),
    Strategy("samples_16x", d=16, sparse_budget=32, n_eff=8192),
    Strategy("d_64", d=64, sparse_budget=32, n_eff=512),
    Strategy("d_256", d=256, sparse_budget=32, n_eff=512),
    Strategy("K_64", d=16, sparse_budget=64, n_eff=512),
    Strategy("K_128", d=16, sparse_budget=128, n_eff=512),
    Strategy("combined", d=64, sparse_budget=128, n_eff=2048),
    Strategy("oracle_K32", d=16, sparse_budget=32, n_eff=512, support="oracle"),
    Strategy("oracle_K128", d=16, sparse_budget=128, n_eff=512, support="oracle"),
]


def corrected_bias(raw_bias: float, variance: float, trials: int) -> float:
    """Remove variance/trials in the finite-trial estimate of squared mean error."""
    return max(raw_bias - variance / trials, 0.0)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, object]] = []

    for strategy in STRATEGIES:
        config = Config(
            D=4096,
            d=strategy.d,
            n_eff=strategy.n_eff,
            sparse_budget=strategy.sparse_budget,
            source_std=0.0625,
            contamination_rates=CONTAMINATION_RATES,
            outlier_std_multiplier=10.0,
            outlier_df=3.0,
            noise_stds=NOISE_STDS,
            trials=TRIALS,
            support=strategy.support,
            ridge=0.0,
            seed=13,
            output_dir=OUTPUT_DIR,
        )
        rows, _ = run_experiment(config)
        for row in rows:
            raw_bias = float(row["bias"])
            variance = float(row["variance"])
            output_rows.append(
                {
                    "strategy": strategy.name,
                    "d": strategy.d,
                    "sparse_budget": strategy.sparse_budget,
                    "n_eff": strategy.n_eff,
                    "support": strategy.support,
                    "contamination_rate": float(row["contamination_rate"]),
                    "noise_std": float(row["noise_std"]),
                    "method": str(row["method"]),
                    "risk_mean": float(row["risk_mean"]),
                    "raw_bias": raw_bias,
                    "corrected_bias": corrected_bias(raw_bias, variance, TRIALS),
                    "variance": variance,
                    "outlier_recall": float(row["outlier_recall"]),
                }
            )

    with (OUTPUT_DIR / "bias_sweep_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    by_case = {
        (str(row["strategy"]), float(row["contamination_rate"]), str(row["method"])): row
        for row in output_rows
    }
    best_by_case: dict[tuple[float, str], dict[str, object]] = {}
    for contamination_rate in CONTAMINATION_RATES:
        for method in ("unilora", "prolosa"):
            candidates = [
                row
                for row in output_rows
                if float(row["contamination_rate"]) == contamination_rate
                and row["method"] == method
            ]
            best_by_case[(contamination_rate, method)] = min(
                candidates,
                key=lambda row: float(row["corrected_bias"]),
            )

    lines = [
        "# Contaminated-Gaussian bias-reduction sweep",
        "",
        (
            "Fixed settings: D=4096, source_std=0.0625, outlier std multiplier=10, "
            "noise_std=1.5, trials=100."
        ),
        "",
        (
            "`corrected bias = max(raw bias - variance / trials, 0)` removes the "
            "finite-Monte-Carlo contribution to squared mean error."
        ),
        "",
    ]
    for contamination_rate in CONTAMINATION_RATES:
        lines.extend(
            [
                f"## epsilon={contamination_rate:g}",
                "",
                (
                    "| strategy | d | K | n_eff | support | Uni bias | ProLoSA bias | "
                    "ProLoSA variance | outlier recall |"
                ),
                "|:---|---:|---:|---:|:---|---:|---:|---:|---:|",
            ]
        )
        for strategy in STRATEGIES:
            uni = by_case[(strategy.name, contamination_rate, "unilora")]
            pro = by_case[(strategy.name, contamination_rate, "prolosa")]
            lines.append(
                f"| {strategy.name} | {strategy.d} | {strategy.sparse_budget} | "
                f"{strategy.n_eff} | {strategy.support} | "
                f"{float(uni['corrected_bias']):.6g} | "
                f"{float(pro['corrected_bias']):.6g} | "
                f"{float(pro['variance']):.6g} | "
                f"{float(pro['outlier_recall']):.4f} |"
            )
        best_uni = best_by_case[(contamination_rate, "unilora")]
        best_pro = best_by_case[(contamination_rate, "prolosa")]
        lines.extend(
            [
                "",
                (
                    f"- Best Uni-LoRA bias: {float(best_uni['corrected_bias']):.6g} "
                    f"with `{best_uni['strategy']}`."
                ),
                (
                    f"- Best ProLoSA bias: {float(best_pro['corrected_bias']):.6g} "
                    f"with `{best_pro['strategy']}`."
                ),
                "",
            ]
        )
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines) + "\n")

    plt = importlib.import_module("matplotlib.pyplot")
    fig, axes = plt.subplots(1, len(CONTAMINATION_RATES), figsize=(16, 5), squeeze=False)
    x = np.arange(len(STRATEGIES))
    width = 0.38
    for ax, contamination_rate in zip(axes[0], CONTAMINATION_RATES):
        uni_bias = [
            float(by_case[(strategy.name, contamination_rate, "unilora")]["corrected_bias"])
            for strategy in STRATEGIES
        ]
        pro_bias = [
            float(by_case[(strategy.name, contamination_rate, "prolosa")]["corrected_bias"])
            for strategy in STRATEGIES
        ]
        ax.bar(x - width / 2, uni_bias, width, label="Uni-LoRA", color="#f97316")
        ax.bar(x + width / 2, pro_bias, width, label="ProLoSA", color="#16a34a")
        ax.set_title(rf"$\epsilon={contamination_rate:g}$", fontweight="bold")
        ax.set_xticks(x, [strategy.name for strategy in STRATEGIES], rotation=60, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("Corrected squared-bias risk")
    axes[0, -1].legend(frameon=False)
    fig.suptitle("Bias reduction by samples, subspace size, and sparse budget", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "bias_sweep_comparison.png", dpi=240)
    fig.savefig(OUTPUT_DIR / "bias_sweep_comparison.pdf")
    plt.close(fig)
    print(f"Saved bias sweep to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

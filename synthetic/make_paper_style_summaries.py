#!/usr/bin/env python
# coding: utf-8

"""Create paper-style three-panel summaries and data tables for synthetic runs."""

from __future__ import annotations

import csv
import math
import textwrap
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "figure_summaries"

KNOWN_P_CSV = ROOT / "results_synthetic_theory_snip_better" / "synthetic_theory_results.csv"
HIDDEN_P_ROOT = ROOT / "results_synthetic_theory_hidden_p_sweep"

METHOD_ORDER = ["lora", "unilora", "prolosa", "unilora_oracle"]
METHOD_LABELS = {
    "lora": "LoRA",
    "unilora": "Uni-LoRA",
    "prolosa": "ProLoSA",
    "unilora_oracle": "Oracle Uni-LoRA",
}
METHOD_COLORS = {
    "lora": "#1f77b4",
    "unilora": "#ff7f0e",
    "prolosa": "#2ca02c",
    "unilora_oracle": "#9467bd",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_hidden_p_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for csv_path in sorted(HIDDEN_P_ROOT.glob("*/synthetic_theory_results.csv")):
        with csv_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                row["scenario"] = scenario_label(row)
                rows.append(row)
    return rows


def scenario_label(row: dict[str, str]) -> str:
    if row.get("pm_mode") == "independent":
        return "independent"
    return f"rotated {float(row.get('pm_angle_deg', 0.0)):g} deg"


def scenario_sort_key(label: str) -> tuple[int, float]:
    if label == "independent":
        return (1, 999.0)
    return (0, float(label.split()[1]))


def fnum(row: dict[str, str], key: str) -> float:
    return float(row[key])


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def filter_rows(
    rows: Iterable[dict[str, str]],
    *,
    n: int | None = None,
    gamma: float | None = None,
    noise_std: float | None = None,
    scenario: str | None = None,
) -> list[dict[str, str]]:
    selected = []
    for row in rows:
        if n is not None and int(float(row["n"])) != n:
            continue
        alpha_key = "gamma" if "gamma" in row else "alpha"
        if gamma is not None and not close(float(row[alpha_key]), gamma):
            continue
        if noise_std is not None and not close(float(row["noise_std"]), noise_std):
            continue
        if scenario is not None and row.get("scenario") != scenario:
            continue
        selected.append(row)
    return selected


def method_rows(rows: list[dict[str, str]], method: str) -> list[dict[str, str]]:
    alpha_key = "gamma" if rows and "gamma" in rows[0] else "alpha"
    return sorted([row for row in rows if row["method"] == method], key=lambda row: float(row[alpha_key]))


def write_csv_table(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def write_md_table(path: Path, rows: list[dict[str, object]], fields: list[str], title: str) -> None:
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("|" + "|".join(["---"] * len(fields)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(field, "")) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n")


def add_caption(fig, caption: str) -> None:
    wrapped = textwrap.fill(caption, width=132)
    fig.text(0.03, 0.02, wrapped, ha="left", va="bottom", fontsize=10.5, family="serif")


def plot_risk_panel(ax, rows: list[dict[str, str]], *, n: int, title: str, include_oracle: bool = False) -> None:
    selected = filter_rows(rows, n=n, noise_std=0.5)
    methods = METHOD_ORDER if include_oracle else METHOD_ORDER[:3]
    for method in methods:
        points = method_rows(selected, method)
        if not points:
            continue
        alpha_key = "gamma" if "gamma" in points[0] else "alpha"
        label = METHOD_LABELS[method]
        ax.errorbar(
            [float(row[alpha_key]) for row in points],
            [float(row["risk_mean"]) for row in points],
            yerr=[float(row.get("risk_std", 0.0)) for row in points],
            marker="o",
            linewidth=2.0,
            capsize=2.0,
            color=METHOD_COLORS[method],
            label=label,
        )
    ax.set_title(title)
    ax.set_xlabel(r"Mismatch $\gamma$")
    ax.grid(alpha=0.28)


def plot_bias_variance_panel(
    ax,
    rows: list[dict[str, str]],
    *,
    n: int,
    gamma: float,
    title: str,
    include_oracle: bool = False,
) -> None:
    selected = filter_rows(rows, n=n, gamma=gamma, noise_std=0.5)
    methods = [method for method in METHOD_ORDER if include_oracle or method != "unilora_oracle"]
    selected_by_method = {row["method"]: row for row in selected}
    methods = [method for method in methods if method in selected_by_method]
    x = list(range(len(methods)))
    bias = [float(selected_by_method[method]["bias"]) for method in methods]
    variance = [float(selected_by_method[method]["variance"]) for method in methods]
    ax.bar(x, bias, color="#1f77b4", label="Bias")
    ax.bar(x, variance, bottom=bias, color="#ff7f0e", label="Variance")
    ax.set_title(title)
    ax.set_xticks(x, [METHOD_LABELS[method] for method in methods], rotation=25, ha="right")
    ax.set_ylabel("Components")
    ax.grid(axis="y", alpha=0.28)
    ax.legend(fontsize=9)


def figure_known_p() -> dict[str, str]:
    import matplotlib.pyplot as plt

    rows = read_csv(KNOWN_P_CSV)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.9))
    plot_risk_panel(axes[0], rows, n=64, title=r"Risk, $n=64$")
    plot_risk_panel(axes[1], rows, n=256, title=r"Risk, $n=256$")
    plot_bias_variance_panel(
        axes[2],
        rows,
        n=64,
        gamma=1.0,
        title=r"Bias--variance, $n=64, \gamma=1$",
    )
    axes[0].set_ylabel("Population excess risk")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.36, 1.02))
    caption = (
        "Figure S1: Known-projection synthetic validation of the bias--variance theory. "
        "Left and middle: population excess risk as subspace mismatch gamma increases under "
        "different sample sizes. Right: empirical bias--variance decomposition at n=64 and "
        "gamma=1. ProLoSA uses warmup SNIP scores to select sparse residual coordinates."
    )
    add_caption(fig, caption)
    fig.tight_layout(rect=(0, 0.18, 1, 0.92))
    png = OUT_DIR / "figure_s1_known_projection.png"
    pdf = OUT_DIR / "figure_s1_known_projection.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    table_rows: list[dict[str, object]] = []
    for panel, n in [("risk_n64", 64), ("risk_n256", 256)]:
        for row in filter_rows(rows, n=n, noise_std=0.5):
            table_rows.append(
                {
                    "panel": panel,
                    "n": n,
                    "gamma": fnum(row, "alpha"),
                    "method": METHOD_LABELS[row["method"]],
                    "risk_mean": fnum(row, "risk_mean"),
                    "risk_std": fnum(row, "risk_std"),
                    "bias": fnum(row, "bias"),
                    "variance": fnum(row, "variance"),
                }
            )
    for row in filter_rows(rows, n=64, gamma=1.0, noise_std=0.5):
        table_rows.append(
            {
                "panel": "bias_variance_n64_gamma1",
                "n": 64,
                "gamma": 1.0,
                "method": METHOD_LABELS[row["method"]],
                "risk_mean": fnum(row, "risk_mean"),
                "risk_std": fnum(row, "risk_std"),
                "bias": fnum(row, "bias"),
                "variance": fnum(row, "variance"),
            }
        )
    fields = ["panel", "n", "gamma", "method", "risk_mean", "risk_std", "bias", "variance"]
    write_csv_table(OUT_DIR / "figure_s1_known_projection_data.csv", table_rows, fields)
    write_md_table(OUT_DIR / "figure_s1_known_projection_data.md", table_rows, fields, "Figure S1 Data")
    return {"figure": str(png), "table": str(OUT_DIR / "figure_s1_known_projection_data.md"), "caption": caption}


def figure_hidden_p_controlled() -> dict[str, str]:
    import matplotlib.pyplot as plt

    rows = read_hidden_p_rows()
    selected = filter_rows(rows, scenario="rotated 15 deg")
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 5.1))
    plot_risk_panel(axes[0], selected, n=256, title=r"Risk, $n=256$", include_oracle=True)
    plot_risk_panel(axes[1], selected, n=1024, title=r"Risk, $n=1024$", include_oracle=True)
    plot_bias_variance_panel(
        axes[2],
        selected,
        n=1024,
        gamma=1.0,
        title=r"Bias--variance, $n=1024, \gamma=1$",
        include_oracle=True,
    )
    axes[0].set_ylabel("Population excess risk")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.42, 1.02))
    caption = (
        "Figure S2: Hidden-P non-oracle projection with controlled mismatch. The teacher uses "
        "P_T to generate the update, while Uni-LoRA and ProLoSA only see P_M, here a 15-degree "
        "rotation of P_T (mean squared canonical overlap 0.933). ProLoSA remains below Uni-LoRA "
        "for larger gamma at n=256 and n=1024; oracle Uni-LoRA with P_T is shown only as an upper bound."
    )
    add_caption(fig, caption)
    fig.tight_layout(rect=(0, 0.2, 1, 0.92))
    png = OUT_DIR / "figure_s2_hidden_p_controlled.png"
    pdf = OUT_DIR / "figure_s2_hidden_p_controlled.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    table_rows: list[dict[str, object]] = []
    for panel, n in [("risk_n256", 256), ("risk_n1024", 1024)]:
        for row in filter_rows(selected, n=n, noise_std=0.5):
            table_rows.append(
                {
                    "panel": panel,
                    "scenario": row["scenario"],
                    "overlap": fnum(row, "subspace_overlap"),
                    "n": n,
                    "gamma": fnum(row, "alpha"),
                    "method": METHOD_LABELS[row["method"]],
                    "risk_mean": fnum(row, "risk_mean"),
                    "risk_std": fnum(row, "risk_std"),
                    "bias": fnum(row, "bias"),
                    "variance": fnum(row, "variance"),
                }
            )
    for row in filter_rows(selected, n=1024, gamma=1.0, noise_std=0.5):
        table_rows.append(
            {
                "panel": "bias_variance_n1024_gamma1",
                "scenario": row["scenario"],
                "overlap": fnum(row, "subspace_overlap"),
                "n": 1024,
                "gamma": 1.0,
                "method": METHOD_LABELS[row["method"]],
                "risk_mean": fnum(row, "risk_mean"),
                "risk_std": fnum(row, "risk_std"),
                "bias": fnum(row, "bias"),
                "variance": fnum(row, "variance"),
            }
        )
    fields = ["panel", "scenario", "overlap", "n", "gamma", "method", "risk_mean", "risk_std", "bias", "variance"]
    write_csv_table(OUT_DIR / "figure_s2_hidden_p_controlled_data.csv", table_rows, fields)
    write_md_table(OUT_DIR / "figure_s2_hidden_p_controlled_data.md", table_rows, fields, "Figure S2 Data")
    return {"figure": str(png), "table": str(OUT_DIR / "figure_s2_hidden_p_controlled_data.md"), "caption": caption}


def read_sweep_summary() -> list[dict[str, str]]:
    path = HIDDEN_P_ROOT / "hidden_p_results_noise_0.5.tsv"
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_win_rates() -> list[dict[str, str]]:
    path = HIDDEN_P_ROOT / "hidden_p_win_rates.tsv"
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def figure_projection_sweep() -> dict[str, str]:
    import matplotlib.pyplot as plt

    sweep = read_sweep_summary()
    wins = read_win_rates()
    scenarios = sorted({row["scenario"] for row in sweep}, key=scenario_sort_key)

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 5.0))

    gamma_wins = [row for row in wins if row["subset"] == "gamma>=1"]
    gamma_wins = sorted(gamma_wins, key=lambda row: scenario_sort_key(row["scenario"]))
    x = list(range(len(gamma_wins)))
    axes[0].bar(x, [float(row["win_rate"]) for row in gamma_wins], color="#4c78a8")
    axes[0].set_xticks(x, [row["scenario"].replace(" ", "\n") for row in gamma_wins])
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("ProLoSA win rate")
    axes[0].set_title(r"Win rate, $\gamma \geq 1$")
    axes[0].grid(axis="y", alpha=0.28)

    for ax, n in [(axes[1], 256), (axes[2], 1024)]:
        for scenario in scenarios:
            rows = sorted(
                [
                    row
                    for row in sweep
                    if row["scenario"] == scenario and int(row["n"]) == n
                ],
                key=lambda row: float(row["gamma"]),
            )
            ax.plot(
                [float(row["gamma"]) for row in rows],
                [float(row["delta_unilora_minus_prolosa"]) for row in rows],
                marker="o",
                linewidth=1.8,
                label=scenario,
            )
        ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1.0)
        ax.set_title(rf"Uni-LoRA $-$ ProLoSA, $n={n}$")
        ax.set_xlabel(r"Mismatch $\gamma$")
        ax.grid(alpha=0.28)
    axes[1].set_ylabel("Risk difference")
    axes[2].legend(fontsize=8, loc="best")

    caption = (
        "Figure S3: Projection-mismatch robustness in the Hidden-P setting. Left: ProLoSA win "
        "rate over Uni-LoRA for gamma >= 1 across all sample sizes and noise levels. Middle and "
        "right: Uni-LoRA minus ProLoSA risk at noise std 0.5 for n=256 and n=1024; values above "
        "zero indicate that ProLoSA is better. Independent P_M is an extreme random-subspace stress test."
    )
    add_caption(fig, caption)
    fig.tight_layout(rect=(0, 0.21, 1, 0.97))
    png = OUT_DIR / "figure_s3_projection_mismatch_sweep.png"
    pdf = OUT_DIR / "figure_s3_projection_mismatch_sweep.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    table_rows: list[dict[str, object]] = []
    for row in gamma_wins:
        table_rows.append(
            {
                "panel": "win_rate_gamma_ge_1",
                "scenario": row["scenario"],
                "overlap": float(row["subspace_overlap"]),
                "n": "all",
                "gamma": ">=1",
                "win_rate": float(row["win_rate"]),
                "wins": int(row["wins"]),
                "total": int(row["total"]),
                "delta_unilora_minus_prolosa": "",
            }
        )
    for n in [256, 1024]:
        for row in sweep:
            if int(row["n"]) != n:
                continue
            table_rows.append(
                {
                    "panel": f"delta_n{n}",
                    "scenario": row["scenario"],
                    "overlap": float(row["subspace_overlap"]),
                    "n": n,
                    "gamma": float(row["gamma"]),
                    "win_rate": "",
                    "wins": "",
                    "total": "",
                    "delta_unilora_minus_prolosa": float(row["delta_unilora_minus_prolosa"]),
                }
            )
    fields = [
        "panel",
        "scenario",
        "overlap",
        "n",
        "gamma",
        "win_rate",
        "wins",
        "total",
        "delta_unilora_minus_prolosa",
    ]
    write_csv_table(OUT_DIR / "figure_s3_projection_mismatch_sweep_data.csv", table_rows, fields)
    write_md_table(OUT_DIR / "figure_s3_projection_mismatch_sweep_data.md", table_rows, fields, "Figure S3 Data")
    return {"figure": str(png), "table": str(OUT_DIR / "figure_s3_projection_mismatch_sweep_data.md"), "caption": caption}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [figure_known_p(), figure_hidden_p_controlled(), figure_projection_sweep()]
    lines = ["# Synthetic Figure Summaries", ""]
    for i, output in enumerate(outputs, start=1):
        lines.extend(
            [
                f"## Figure S{i}",
                "",
                f"- Figure: `{output['figure']}`",
                f"- Data table: `{output['table']}`",
                f"- Summary: {output['caption']}",
                "",
            ]
        )
    (OUT_DIR / "figure_summaries.md").write_text("\n".join(lines))
    print(f"Wrote figures and tables to {OUT_DIR}")


if __name__ == "__main__":
    main()

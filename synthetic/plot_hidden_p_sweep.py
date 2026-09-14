#!/usr/bin/env python
# coding: utf-8

"""Summarize and plot Hidden-P / non-oracle projection sweep results."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


METHOD_ORDER = ["lora", "unilora", "prolosa", "unilora_oracle"]
METHOD_LABELS = {
    "lora": "LoRA",
    "unilora": "Uni-LoRA",
    "prolosa": "ProLoSA",
    "unilora_oracle": "Uni-LoRA (oracle)",
}
METHOD_COLORS = {
    "lora": "#1f77b4",
    "unilora": "#ff7f0e",
    "prolosa": "#2ca02c",
    "unilora_oracle": "#9467bd",
}


def scenario_label(row: dict[str, str]) -> str:
    mode = row.get("pm_mode", "rotated")
    if mode == "independent":
        return "independent"
    angle = float(row.get("pm_angle_deg", 0.0))
    return f"rotated {angle:g} deg"


def scenario_sort_key(label: str) -> tuple[int, float]:
    if label == "independent":
        return (1, 999.0)
    try:
        angle = float(label.split()[1])
    except (IndexError, ValueError):
        angle = 0.0
    return (0, angle)


def read_result_rows(results_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for csv_path in sorted(results_root.glob("*/synthetic_theory_results.csv")):
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row["source_dir"] = csv_path.parent.name
                row["scenario"] = scenario_label(row)
                rows.append(row)
    if not rows:
        raise ValueError(f"No synthetic_theory_results.csv files found below {results_root}.")
    return rows


def group_method_rows(rows: list[dict[str, str]]) -> dict[tuple[str, int, float, float], dict[str, dict[str, str]]]:
    grouped: dict[tuple[str, int, float, float], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        key = (
            row["scenario"],
            int(float(row["n"])),
            float(row["alpha"]),
            float(row["noise_std"]),
        )
        grouped[key][row["method"]] = row
    return grouped


def metric_records(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped = group_method_rows(rows)
    records: list[dict[str, object]] = []
    for (scenario, n, gamma, noise_std), methods in sorted(grouped.items()):
        if "unilora" not in methods or "prolosa" not in methods:
            continue
        unilora = float(methods["unilora"]["risk_mean"])
        prolosa = float(methods["prolosa"]["risk_mean"])
        record: dict[str, object] = {
            "scenario": scenario,
            "pm_mode": methods["unilora"].get("pm_mode", "rotated"),
            "pm_angle_deg": float(methods["unilora"].get("pm_angle_deg", 0.0)),
            "subspace_overlap": float(methods["unilora"].get("subspace_overlap", "nan")),
            "n": n,
            "gamma": gamma,
            "noise_std": noise_std,
            "unilora": unilora,
            "prolosa": prolosa,
            "delta_unilora_minus_prolosa": unilora - prolosa,
            "prolosa_wins": prolosa < unilora,
        }
        for method in METHOD_ORDER:
            if method in methods:
                record[method] = float(methods[method]["risk_mean"])
        records.append(record)
    return records


def fmt(value: object) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.4g}"
    return str(value)


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, rows: list[dict[str, object]], fields: list[str], title: str) -> None:
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("|" + "|".join(["---" for _ in fields]) + "|")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(field, "")) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n")


def win_rate_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    scenarios = sorted({str(record["scenario"]) for record in records}, key=scenario_sort_key)
    for scenario in scenarios:
        scenario_records = [record for record in records if record["scenario"] == scenario]
        strong_records = [record for record in scenario_records if float(record["gamma"]) >= 1.0]
        for label, subset in [("all", scenario_records), ("gamma>=1", strong_records)]:
            wins = sum(bool(record["prolosa_wins"]) for record in subset)
            total = len(subset)
            first = subset[0]
            output.append(
                {
                    "scenario": scenario,
                    "subset": label,
                    "subspace_overlap": float(first["subspace_overlap"]),
                    "wins": wins,
                    "total": total,
                    "win_rate": wins / total if total else float("nan"),
                }
            )
    return output


def plot_delta(records: list[dict[str, object]], path: Path, noise_std: float) -> None:
    import matplotlib.pyplot as plt

    selected = [record for record in records if math.isclose(float(record["noise_std"]), noise_std)]
    scenarios = sorted({str(record["scenario"]) for record in selected}, key=scenario_sort_key)
    sample_sizes = sorted({int(record["n"]) for record in selected})
    fig, axes = plt.subplots(1, len(scenarios), figsize=(4.0 * len(scenarios), 3.2), sharey=True)
    if len(scenarios) == 1:
        axes = [axes]
    for ax, scenario in zip(axes, scenarios):
        scenario_rows = [record for record in selected if record["scenario"] == scenario]
        overlap = float(scenario_rows[0]["subspace_overlap"])
        for n in sample_sizes:
            rows = sorted(
                [record for record in scenario_rows if int(record["n"]) == n],
                key=lambda record: float(record["gamma"]),
            )
            ax.plot(
                [float(record["gamma"]) for record in rows],
                [float(record["delta_unilora_minus_prolosa"]) for record in rows],
                marker="o",
                label=f"n={n}",
            )
        ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")
        ax.set_title(f"{scenario}\noverlap={overlap:.3f}")
        ax.set_xlabel("mismatch gamma")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Uni-LoRA risk - ProLoSA risk\n(positive means ProLoSA wins)")
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle(f"Hidden-P robustness at noise std={noise_std:g}", y=1.04)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_win_rates(win_rows: list[dict[str, object]], path: Path) -> None:
    import matplotlib.pyplot as plt

    scenarios = sorted({str(row["scenario"]) for row in win_rows}, key=scenario_sort_key)
    all_rows = {str(row["scenario"]): row for row in win_rows if row["subset"] == "all"}
    strong_rows = {str(row["scenario"]): row for row in win_rows if row["subset"] == "gamma>=1"}
    x = range(len(scenarios))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    ax.bar([i - width / 2 for i in x], [float(all_rows[s]["win_rate"]) for s in scenarios], width, label="all")
    ax.bar(
        [i + width / 2 for i in x],
        [float(strong_rows[s]["win_rate"]) for s in scenarios],
        width,
        label="gamma>=1",
    )
    ax.set_xticks(list(x), scenarios, rotation=25, ha="right")
    ax.set_ylabel("ProLoSA win rate vs Uni-LoRA")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_representative_risk(records: list[dict[str, object]], path: Path, noise_std: float) -> None:
    import matplotlib.pyplot as plt

    preferred = ["rotated 15 deg", "rotated 30 deg", "independent"]
    scenarios = [s for s in preferred if any(record["scenario"] == s for record in records)]
    if not scenarios:
        scenarios = sorted({str(record["scenario"]) for record in records}, key=scenario_sort_key)[:3]
    selected_n = max(int(record["n"]) for record in records)
    fig, axes = plt.subplots(1, len(scenarios), figsize=(4.3 * len(scenarios), 3.4), sharey=True)
    if len(scenarios) == 1:
        axes = [axes]
    for ax, scenario in zip(axes, scenarios):
        rows = [
            record
            for record in records
            if record["scenario"] == scenario
            and int(record["n"]) == selected_n
            and math.isclose(float(record["noise_std"]), noise_std)
        ]
        for method in METHOD_ORDER:
            if method not in rows[0]:
                continue
            method_label = METHOD_LABELS[method]
            ax.plot(
                [float(record["gamma"]) for record in rows],
                [float(record[method]) for record in rows],
                marker="o",
                label=method_label,
                color=METHOD_COLORS[method],
            )
        ax.set_title(scenario)
        ax.set_xlabel("mismatch gamma")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(f"population excess risk (n={selected_n}, noise={noise_std:g})")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--noise-std", type=float, default=0.5)
    args = parser.parse_args()

    rows = read_result_rows(args.results_root)
    records = metric_records(rows)
    selected = [
        record
        for record in records
        if math.isclose(float(record["noise_std"]), args.noise_std)
    ]
    selected = sorted(
        selected,
        key=lambda record: (
            scenario_sort_key(str(record["scenario"])),
            int(record["n"]),
            float(record["gamma"]),
        ),
    )

    fields = [
        "scenario",
        "subspace_overlap",
        "n",
        "gamma",
        "noise_std",
        "lora",
        "unilora",
        "prolosa",
        "unilora_oracle",
        "delta_unilora_minus_prolosa",
        "prolosa_wins",
    ]
    write_tsv(args.results_root / f"hidden_p_results_noise_{args.noise_std:g}.tsv", selected, fields)
    write_markdown_table(
        args.results_root / f"hidden_p_results_noise_{args.noise_std:g}.md",
        selected,
        fields,
        f"Hidden-P Results at noise_std={args.noise_std:g}",
    )

    win_rows = win_rate_records(records)
    win_fields = ["scenario", "subset", "subspace_overlap", "wins", "total", "win_rate"]
    write_tsv(args.results_root / "hidden_p_win_rates.tsv", win_rows, win_fields)
    write_markdown_table(args.results_root / "hidden_p_win_rates.md", win_rows, win_fields, "Hidden-P Win Rates")

    plot_delta(records, args.results_root / f"hidden_p_delta_noise_{args.noise_std:g}.png", args.noise_std)
    plot_win_rates(win_rows, args.results_root / "hidden_p_win_rates.png")
    plot_representative_risk(
        records,
        args.results_root / f"hidden_p_representative_risk_noise_{args.noise_std:g}.png",
        args.noise_std,
    )

    print(f"Wrote summaries and plots to {args.results_root}")


if __name__ == "__main__":
    main()

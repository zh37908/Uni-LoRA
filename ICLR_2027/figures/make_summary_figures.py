"""Reproduce the main-paper figures from existing results; no experiments run.

Requirements: matplotlib, numpy. Run from any directory.
Sequence CSVs are copied unchanged from synthetic/results_matched_budget_baselines
and synthetic/results_gaussian_synthetic_theory_std_0.125. Real-model figures use paired raw-run reanalysis outputs in
revision/existing_evidence (run revision/reanalyze_existing_evidence.py first).
"""
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5,
                     "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "legend.fontsize": 6.5, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
COLORS = {"lora": "#303f66", "unilora_matched": "#da8530",
          "unilora": "#da8530", "prolosa": "#24866a",
          "prolosa_random": "#9073a7", "prolosa_oracle": "#ba5261"}


def read_csv(name):
    with (ROOT / "data" / name).open() as f:
        return list(csv.DictReader(f))


def curve(ax, rows, method, label, style="-"):
    data = sorted((r for r in rows if r["method"] == method),
                  key=lambda r: float(r["noise_std"]))
    x = np.array([float(r["noise_std"]) for r in data])
    y = np.array([float(r["risk_mean"]) for r in data])
    ax.plot(x, y, style, color=COLORS[method], label=label, lw=1.45)
    ax.set_xlabel(r"Noise $\sigma_y$")
    ax.grid(alpha=.17)


def sequence():
    matched = read_csv("matched_budget_results.csv")
    diffuse = read_csv("gaussian_std0125_results.csv")
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.25), layout="constrained")
    a, b, c = axes
    curve(a, matched, "lora", "Full space")
    curve(a, matched, "unilora_matched", "Compressed (48)")
    # Conditional threshold for the B=48 projection, reported in the source text.
    a.axvline(np.sqrt(512 * 2.6427e-3), color=".45", ls=":", lw=1)
    a.set_title("(a) Exact crossover")
    a.set_ylabel("Population excess risk")
    a.legend(loc="upper left", frameon=False)
    for m, label, style in [("unilora_matched", "Compressed", "-"),
                            ("prolosa_random", "Random", "--"),
                            ("prolosa", "Noisy support", "-"),
                            ("prolosa_oracle", "Oracle", ":")]:
        curve(b, matched, m, label, style)
    b.set_yscale("log")
    b.set_title("(b) Equal budget (48)")
    b.legend(loc="center right", frameon=False)
    for m, label in [("lora", "Full space"), ("unilora", "Compressed (16)"),
                     ("prolosa", "Hybrid (48)")]:
        curve(c, diffuse, m, label)
    c.set_title("(c) Diffuse failure")
    c.legend(loc="lower right", frameon=False)
    fig.savefig(ROOT / "sequence_summary.pdf")
    fig.savefig(ROOT / "sequence_summary.png", dpi=180)
    plt.close(fig)


def real_model():
    data_dir = ROOT.parent / "revision" / "existing_evidence"
    e1 = json.loads((data_dir / "e1.json").read_text())
    e3 = json.loads((data_dir / "e3.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.65), layout="constrained")
    for j, task in enumerate(("SST-2", "QNLI")):
        ax = axes[0, j]
        obj = e1[("sst2", "qnli")[j]]
        pts = obj["points"]
        fractions = np.array([100*p["fraction"] for p in pts])
        gaps = np.array([p["mean"] for p in pts])
        errors = np.array([[p["mean"]-p["ci_low"] for p in pts],
                           [p["ci_high"]-p["mean"] for p in pts]])
        ax.errorbar(fractions, gaps, yerr=errors, fmt="o", ms=3,
                    capsize=2, color=COLORS["lora"], lw=1, label="Paired mean, 95% CI")
        for mode, style, color, label in [("free", ":", ".45", "Free fit"),
                  ("constrained", "--", "#24866a", r"Fit: $a,b\geq0$")]:
            f = obj["fits"][mode]
            xx = np.geomspace(fractions.min(), fractions.max(), 100)
            ns = np.interp(xx, fractions, [p["n"] for p in pts])
            ax.plot(xx, f["a"]/ns-f["b"], style, color=color, lw=1, label=label)
        if j == 0:
            ax.legend(frameon=False, fontsize=5.8, loc="upper right")
        ax.axhline(0, color=".5", lw=.7, ls="--")
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 25, 100], ["1", "5", "25", "100"])
        ax.set_ylim(-.12, .26)
        ax.set_xlabel("Training data (%)")
        ax.set_title(f"({chr(97+j)}) {task}: loss gap")
        ax.set_ylabel(r"$\Delta$ (positive: compression wins)", fontsize=6.5)
        ax.grid(alpha=.15)
    for j, task in enumerate(("CoLA", "MRPC")):
        ax = axes[1, j]
        block = [e3[("cola", "mrpc")[j]]["methods"][m]
                 for m in ("lora", "unilora", "prolosa")]
        x = np.arange(3)
        bottom = np.zeros(3)
        for key, color, label in [("Bias2", "#da8530", "Squared bias proxy"),
              ("V_stat", "#24866a", "Data-resampling variance"),
              ("V_opt", "#657fac", "Training-seed variance")]:
            values = np.array([row[key] for row in block])
            ax.bar(x, values, bottom=bottom, color=color, width=.58, label=label)
            bottom += values
        ax.set_xticks(x, ["LoRA", "Uni-LoRA", "ProLoSA"])
        ax.set_ylim(0, .115)
        ax.set_title(f"({chr(99+j)}) {task}: decomposition")
        ax.set_ylabel("Predictive squared error")
        ax.grid(axis="y", alpha=.15)
        if j == 0:
            ax.legend(loc="upper right", frameon=False, fontsize=5.7)
    fig.savefig(ROOT / "real_mechanism.pdf")
    fig.savefig(ROOT / "real_mechanism.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    sequence()
    real_model()

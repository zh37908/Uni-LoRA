#!/usr/bin/env python3
"""E3: empirical bias-variance decomposition with dataset bootstrap x optimizer seeds.

Runs live under  <root>/<model>/<task>/<method_dir>/boot_<B>/seed_<S>/  and carry
`*_theory.pt` artifacts with `delta_theta` (flattened per-module [A, B]) and
`eval_logits` on the clean dev set.

Decomposition (per method), with z_{b,s} the estimator for bootstrap b, seed s:
    zbar_b  = mean_s z_{b,s}
    zbar    = mean_b zbar_b
    V_opt   = mean_{b,s} ||z_{b,s} - zbar_b||^2      (optimisation variance)
    V_stat  = mean_b     ||zbar_b - zbar||^2         (statistical variance)
    Bias^2  = ||zbar - z_ref||^2                      (to a high-resource reference)

Two representations are reported:
  * functional (PRIMARY): z = dev-set log-probabilities; distances are averaged over
    examples. Also reports predictive KL(p_ref || p) for interpretability.
  * parameter space in the MERGED update  Delta W_l = B_l A_l  (gauge invariant; raw
    A/B coordinates are not comparable across runs because BA = (BR)(R^-1 A)).
    Optionally Fisher-weighted if --fisher_w task=path.pt provides a diagonal in
    the same merged ordering.

Reference theta_ref: full-data LoRA runs from --ref_roots (e.g. the E4 projection-sweep
root, rank 4, dedicated lr) plus `lora_ref_*` (rank 64) runs inside the E3 root whose
score is not collapsed (>= --ref_min_score_frac of the best reference score).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HIDDEN = {"roberta-large": (1024, 4096), "roberta-base": (768, 3072)}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_pt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def method_of(path: Path):
    p = path.as_posix()
    if "/lora_ref_r" in p:
        return "lora_ref"
    if "/lora_r" in p:
        return "lora"
    if "/unilora_td" in p:
        return "unilora"
    if "/prolosa_td" in p:
        return "prolosa"
    return None


def sidecar(path: Path):
    j = path.with_name(path.name.replace("_theory.pt", ".json"))
    if not j.exists():
        return {}
    try:
        return json.loads(j.read_text())
    except Exception:
        return {}


def scan(root: Path, model: str):
    """Return list of dict(task, method, boot, seed, path, score, dedicated_lr)."""
    out = []
    for path in sorted(root.rglob("*_theory.pt")):
        p = path.as_posix()
        if f"/{model}/" not in p:
            continue
        m = method_of(path)
        if m is None:
            continue
        task = p.split(f"/{model}/")[1].split("/")[0]
        boot = re.search(r"/boot_(\d+)/", p)
        seed = re.search(r"/seed_(\d+)/", p)
        js = sidecar(path)
        out.append(
            {
                "task": task,
                "method": m,
                "boot": int(boot.group(1)) if boot else None,
                "seed": int(seed.group(1)) if seed else None,
                "path": path,
                "score": js.get("best_score"),
                "dedicated_lr": bool(re.search(r"/lora(?:_ref)?_r\d+_lr", p)),
            }
        )
    return out


def drop_legacy_lora(items):
    """Prefer dedicated-lr LoRA runs over legacy shared-lr ones for the same task/method."""
    has = {(i["task"], i["method"]) for i in items if i["method"] in ("lora", "lora_ref") and i["dedicated_lr"]}
    return [
        i for i in items
        if not (i["method"] in ("lora", "lora_ref") and not i["dedicated_lr"] and (i["task"], i["method"]) in has)
    ]


# ---------------------------------------------------------------------------
# representations
# ---------------------------------------------------------------------------

def probs(obj):
    """Functional representation: dev-set predictive probabilities (bounded, so the
    decomposition is not dominated by log-tails of confidently-rejected classes)."""
    z = obj.get("eval_logits")
    if z is None:
        return None
    z = z.float()
    if z.shape[-1] == 1:  # regression: use raw prediction
        return z.reshape(-1).double().numpy()
    return F.softmax(z, dim=-1).double().numpy().reshape(-1)


def module_shape(name: str, model: str):
    h, inter = HIDDEN[model]
    if name.endswith("intermediate.dense"):
        return inter, h  # out, in
    if name.endswith("attention.output.dense"):
        return h, h
    if name.endswith("output.dense"):
        return h, inter
    return h, h  # query/key/value


def merged_delta_w(obj, model: str) -> np.ndarray:
    """Concatenate flattened B_l A_l over modules (float64)."""
    flat = obj["delta_theta"].float()
    r = int(obj.get("rank") or 4)
    off = 0
    parts = []
    for name in obj["module_names"]:
        out_f, in_f = module_shape(name, model)
        n_a, n_b = r * in_f, out_f * r
        A = flat[off : off + n_a].view(r, in_f)
        B = flat[off + n_a : off + n_a + n_b].view(out_f, r)
        off += n_a + n_b
        parts.append((B @ A).reshape(-1))
    assert off == flat.numel(), f"consumed {off} of {flat.numel()}"
    # float32 to keep 10 x 302M-dim vectors per method within memory
    return torch.cat(parts).numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# decomposition
# ---------------------------------------------------------------------------

def decompose(vectors_by_boot: dict, ref: np.ndarray | None, w: np.ndarray | None, per_dim: float):
    """vectors_by_boot: {boot: [vec, ...]} ; returns dict of metrics (all divided by per_dim)."""

    def sq(x):
        x = np.asarray(x)
        if w is not None:
            return float(np.einsum("i,i,i->", w, x, x, dtype=np.float64))
        return float(np.einsum("i,i->", x, x, dtype=np.float64))

    boot_means = {}
    v_opt_terms = []
    for b, vecs in vectors_by_boot.items():
        m = np.mean(vecs, axis=0)
        boot_means[b] = m
        for v in vecs:
            v_opt_terms.append(sq(v - m))
    grand = np.mean(list(boot_means.values()), axis=0)
    v_stat = float(np.mean([sq(m - grand) for m in boot_means.values()]))
    v_opt = float(np.mean(v_opt_terms)) if v_opt_terms else float("nan")
    # Unbiased small-sample corrections: V_opt uses S-1 in the denominator per boot,
    # V_stat removes the leaked optimisation variance (V_opt / S).
    S = np.mean([len(v) for v in vectors_by_boot.values()])
    Bn = len(vectors_by_boot)
    v_opt_unb = v_opt * S / max(S - 1, 1)
    v_stat_raw = v_stat * Bn / max(Bn - 1, 1)
    # Between-boot variance of means leaks V_opt / S; remove it (clipped at 0).
    v_stat_unb = max(v_stat_raw - v_opt_unb / S, 0.0)
    out = {
        "V_opt": v_opt_unb / per_dim,
        "V_stat_raw": v_stat_raw / per_dim,
        "V_stat": v_stat_unb / per_dim,
        "V_total": (v_opt_unb + v_stat_unb) / per_dim,
        "n_boot": Bn,
        "n_seed": S,
        "norm2_mean": sq(grand) / per_dim,
    }
    if ref is not None:
        out["Bias2"] = sq(grand - ref) / per_dim
        out["Bias2+V"] = out["Bias2"] + out["V_total"]
    return out, grand


def decompose_streaming(paths_by_boot: dict, loader, ref, w, per_dim: float):
    """Memory-lean variant of `decompose` for ~3e8-dim vectors: loads one bootstrap
    group at a time and only keeps the per-boot means."""

    def sq(x):
        if w is not None:
            return float(np.einsum("i,i,i->", w, x, x, dtype=np.float64))
        return float(np.einsum("i,i->", x, x, dtype=np.float64))

    boot_means = []
    v_opt_terms = []
    n_seeds = []
    for b, paths in paths_by_boot.items():
        vecs = [loader(p) for p in paths]
        m = np.mean(vecs, axis=0, dtype=np.float32) if len(vecs) > 1 else vecs[0]
        for v in vecs:
            v_opt_terms.append(sq(v - m))
        del vecs
        boot_means.append(m)
        n_seeds.append(len(paths))
    grand = np.mean(boot_means, axis=0, dtype=np.float32)
    v_stat = float(np.mean([sq(m - grand) for m in boot_means]))
    del boot_means
    v_opt = float(np.mean(v_opt_terms)) if v_opt_terms else float("nan")
    S = float(np.mean(n_seeds))
    Bn = len(paths_by_boot)
    v_opt_unb = v_opt * S / max(S - 1, 1)
    v_stat_raw = v_stat * Bn / max(Bn - 1, 1)
    v_stat_unb = max(v_stat_raw - v_opt_unb / S, 0.0)
    out = {
        "V_opt": v_opt_unb / per_dim,
        "V_stat_raw": v_stat_raw / per_dim,
        "V_stat": v_stat_unb / per_dim,
        "V_total": (v_opt_unb + v_stat_unb) / per_dim,
        "n_boot": Bn,
        "n_seed": S,
        "norm2_mean": sq(grand) / per_dim,
    }
    if ref is not None:
        out["Bias2"] = sq(grand - ref) / per_dim
        out["Bias2+V"] = out["Bias2"] + out["V_total"]
    return out, grand


def kl_to_ref(p_runs: list[np.ndarray], p_ref: np.ndarray, n_classes: int, eps: float = 1e-8) -> float:
    """Mean KL(p_ref || p_run) over runs and examples (classification only).
    Inputs are probability vectors; each row is renormalised so averaged
    distributions remain valid."""
    if n_classes < 2:
        return float("nan")
    ref = np.clip(p_ref.reshape(-1, n_classes), eps, None)
    ref = ref / ref.sum(axis=1, keepdims=True)
    vals = []
    for p in p_runs:
        run = np.clip(p.reshape(-1, n_classes), eps, None)
        run = run / run.sum(axis=1, keepdims=True)
        vals.append(float(np.mean(np.sum(ref * (np.log(ref) - np.log(run)), axis=1))))
    return float(np.mean(vals))


def fmt(v, d=5):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "-"
    if isinstance(v, (int, np.integer)):
        return str(v)
    return f"{v:.{d}g}"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results_theory_e3_bias_variance")
    ap.add_argument("--ref_roots", nargs="*", default=["results_theory_e4_projection_sweep"],
                    help="Roots with full-data LoRA runs used as theta_ref (rank 4).")
    ap.add_argument("--model", default="roberta-large")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--ref_min_score_frac", type=float, default=0.85,
                    help="Drop lora_ref seeds scoring below this fraction of the best reference score.")
    ap.add_argument("--skip_param_space", action="store_true", help="Only functional metrics (fast).")
    ap.add_argument("--fisher_w", nargs="*", default=None,
                    help="Optional task=path.pt diagonal weights in merged Delta-W ordering.")
    ap.add_argument("--output_md", default="results_theory_e3_bias_variance/e3_decomposition.md")
    args = ap.parse_args()

    root = Path(args.root)
    items = drop_legacy_lora(scan(root, args.model))
    ref_items = []
    for rr in args.ref_roots:
        ref_items += [i for i in drop_legacy_lora(scan(Path(rr), args.model)) if i["method"] == "lora" and i["boot"] is None]

    fisher = {}
    for spec in args.fisher_w or []:
        t, _, p = spec.partition("=")
        obj = torch.load(p, map_location="cpu")
        fisher[t] = np.asarray(obj["fisher_diag"] if isinstance(obj, dict) else obj, dtype=np.float64)

    tasks = args.tasks or sorted({i["task"] for i in items})
    lines = ["# E3 empirical bias-variance decomposition (bootstrap x seeds)", ""]
    lines.append(
        "V_opt = optimisation variance (same data, different seeds); V_stat = statistical variance "
        "(different data subsamples, ratio 0.8); Bias^2 = distance of the grand mean to the reference. "
        "Functional metrics are per-example averages in predictive-probability space (PRIMARY); parameter "
        "metrics are per-coordinate averages in the merged Delta W = BA space (x1e-9 shown as e-9)."
    )
    lines.append("")

    for task in tasks:
        t_items = [i for i in items if i["task"] == task]
        if not t_items:
            continue
        lines.append(f"## {task}")
        lines.append("")

        # ---------------- references ----------------
        refs_lp, refs_dw, ref_desc = [], [], []
        ref_r4 = [i for i in ref_items if i["task"] == task]
        ref_r64 = [i for i in t_items if i["method"] == "lora_ref"]
        best = max([i["score"] for i in ref_r4 + ref_r64 if i["score"] is not None] or [None])
        kept_r64 = [i for i in ref_r64 if i["score"] is not None and best and i["score"] >= args.ref_min_score_frac * best]
        ref_dw_sum, n_ref_dw = None, 0
        for i in ref_r4 + kept_r64:
            obj = load_pt(i["path"])
            lp = probs(obj)
            if lp is not None:
                refs_lp.append(lp)
            if not args.skip_param_space:
                dw = merged_delta_w(obj, args.model)
                ref_dw_sum = dw if ref_dw_sum is None else ref_dw_sum + dw
                n_ref_dw += 1
                del dw
            ref_desc.append(f"{i['method']}(seed {i['seed']}, score {i['score']:.3f})")
        lines.append(f"Reference set ({len(ref_desc)} runs; dropped {len(ref_r64) - len(kept_r64)} collapsed lora_ref seeds): "
                     + ", ".join(ref_desc))
        lines.append("")
        ref_lp = np.mean(refs_lp, axis=0) if refs_lp else None
        ref_dw = (ref_dw_sum / n_ref_dw).astype(np.float32) if n_ref_dw else None
        del ref_dw_sum

        # ---------------- per-method decomposition ----------------
        func_rows, param_rows = [], []
        n_classes = None
        for method in ("lora", "unilora", "prolosa"):
            m_items = [i for i in t_items if i["method"] == method and i["boot"] is not None]
            if not m_items:
                continue
            lp_by_boot, paths_by_boot, all_lp = defaultdict(list), defaultdict(list), []
            for i in m_items:
                obj = load_pt(i["path"])
                lp = probs(obj)
                if lp is not None:
                    lp_by_boot[i["boot"]].append(lp)
                    all_lp.append(lp)
                    if n_classes is None:
                        n_classes = int(obj["eval_logits"].shape[-1])
                paths_by_boot[i["boot"]].append(i["path"])
                del obj
            scores = [i["score"] for i in m_items if i["score"] is not None]
            if lp_by_boot:
                n_ex = next(iter(lp_by_boot.values()))[0].size / max(n_classes or 1, 1)
                f, grand_lp = decompose(lp_by_boot, ref_lp, None, per_dim=n_ex)
                kl = kl_to_ref([grand_lp], ref_lp, n_classes) if ref_lp is not None else float("nan")
                kl_runs = kl_to_ref(all_lp, ref_lp, n_classes) if ref_lp is not None else float("nan")
                func_rows.append((method, f, kl, kl_runs, np.mean(scores) if scores else float("nan")))
            if not args.skip_param_space and paths_by_boot:
                w = fisher.get(task)
                loader = lambda path: merged_delta_w(load_pt(path), args.model)
                D = ref_dw.size if ref_dw is not None else loader(next(iter(paths_by_boot.values()))[0]).size
                p, _ = decompose_streaming(paths_by_boot, loader, ref_dw, w, per_dim=D)
                param_rows.append((method, p))

        lines.append("### Functional (dev predictive-probability space, per example)")
        lines.append("")
        lines.append("| method | score | V_opt | V_stat(raw) | V_stat(corr) | V_total | Bias^2 | Bias^2+V | KL(ref||mean) | KL(ref||run) | n_boot x n_seed |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for method, f, kl, kl_runs, sc in func_rows:
            lines.append(
                f"| {method} | {fmt(sc, 4)} | {fmt(f['V_opt'])} | {fmt(f['V_stat_raw'])} | {fmt(f['V_stat'])} | {fmt(f['V_total'])} | "
                f"{fmt(f.get('Bias2'))} | {fmt(f.get('Bias2+V'))} | {fmt(kl)} | {fmt(kl_runs)} | "
                f"{f['n_boot']} x {f['n_seed']:.0f} |"
            )
        lines.append("")
        if param_rows:
            wtag = "Fisher-weighted" if task in fisher else "Euclidean"
            lines.append(f"### Parameter space (merged Delta W = BA, per coordinate, {wtag})")
            lines.append("")
            lines.append("| method | V_opt | V_stat(raw) | V_stat(corr) | V_total | Bias^2 | Bias^2+V | ||mean||^2 |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
            for method, p in param_rows:
                lines.append(
                    f"| {method} | {fmt(p['V_opt'])} | {fmt(p['V_stat_raw'])} | {fmt(p['V_stat'])} | {fmt(p['V_total'])} | "
                    f"{fmt(p.get('Bias2'))} | {fmt(p.get('Bias2+V'))} | {fmt(p['norm2_mean'])} |"
                )
            lines.append("")

    text = "\n".join(lines) + "\n"
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text(text)
    print(text)


if __name__ == "__main__":
    main()

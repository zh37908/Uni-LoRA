#!/usr/bin/env python3
"""E4/E5 analysis over saved LoRA-space updates (`*_theory.pt`).

Revised per bias_variance_glue_theory_experiment_summary.md:

E4-A (cross-task, secondary): off-subspace energy ratio of the LoRA update under
    Uni-LoRA's projection vs the LoRA-minus-Uni-LoRA gap (score and dev loss).
E4-B (controlled, primary): fixed task/d, varying only the projection seed P;
    correlate B_off(P) with the per-P compression excess dev loss.
E5: energy concentration of the OFF-SUBSPACE residual q = (I - Pi_P) delta_theta
    (not the total update): oracle C_K(q), SNIP captured-energy ratio rho_I^2,
    random-support baselines K/D and K/(D-d), enrichment, top-frac curves vs an
    isotropic Gaussian baseline, kurtosis and Gini.

Optional curvature weighting: pass --fisher task=path.pt (from
estimate_fisher_diag_glue.py); all energy statistics are then also reported in
the H-weighted metric (energy_i = H_ii * x_i^2).
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


# ---------------------------------------------------------------------------
# basic statistics
# ---------------------------------------------------------------------------

def gini(x: np.ndarray) -> float:
    x = np.sort(np.abs(x).astype(np.float64))
    n = x.size
    if n == 0 or x.sum() <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * x) / (n * x.sum())) - (n + 1) / n)


def kurtosis(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    if x.size < 4:
        return float("nan")
    m = x.mean()
    v = x.var()
    if v <= 0:
        return float("nan")
    return float(np.mean((x - m) ** 4) / (v ** 2))


def topfrac_energy(energy: np.ndarray, fracs=(0.001, 0.01, 0.05)) -> dict:
    total = energy.sum()
    if total <= 0:
        return {f: 0.0 for f in fracs}
    order = np.sort(energy)[::-1]
    csum = np.cumsum(order)
    out = {}
    for f in fracs:
        k = max(1, int(round(energy.size * f)))
        out[f] = float(csum[k - 1] / total)
    return out


def topk_energy_ratio(energy: np.ndarray, k: int) -> float:
    """Oracle: fraction of total energy carried by the top-k coordinates."""
    total = energy.sum()
    if total <= 0 or k <= 0:
        return float("nan")
    k = min(k, energy.size)
    top = np.partition(energy, -k)[-k:]
    return float(top.sum() / total)


def pearson(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return pearson(rx, ry)


# ---------------------------------------------------------------------------
# projection / support helpers
# ---------------------------------------------------------------------------

def reconstruct_assignment(index_meta: dict, module_names: list, dim: int) -> np.ndarray:
    """Length-D assignment of LoRA coordinates onto theta_d buckets, in the
    exact flattening order used by extract_lora_space_delta (module order, A then B)."""
    assignment = np.full(dim, -1, dtype=np.int64)
    offset = 0
    names = module_names if module_names else list(index_meta.keys())
    for name in names:
        meta = index_meta.get(name)
        if meta is None:
            continue
        idx_a = meta["indices_A"].reshape(-1).cpu().numpy()
        idx_b = meta["indices_B"].reshape(-1).cpu().numpy()
        assignment[offset : offset + idx_a.size] = idx_a
        offset += idx_a.size
        assignment[offset : offset + idx_b.size] = idx_b
        offset += idx_b.size
    return assignment


def bucket_project(delta: np.ndarray, assignment: np.ndarray, theta_d_length: int) -> np.ndarray:
    """Projection of delta onto Uni-LoRA's (normalized one-hot) subspace:
    per-bucket mean broadcast back to coordinates."""
    d = delta.astype(np.float64)
    proj = np.zeros(theta_d_length, dtype=np.float64)
    counts = np.zeros(theta_d_length, dtype=np.float64)
    valid = assignment >= 0
    np.add.at(proj, assignment[valid], d[valid])
    np.add.at(counts, assignment[valid], 1.0)
    counts = np.maximum(counts, 1.0)
    reconstructed = np.zeros_like(d)
    reconstructed[valid] = (proj / counts)[assignment[valid]]
    return reconstructed


def off_subspace_residual(delta: np.ndarray, assignment: np.ndarray, theta_d_length: int) -> np.ndarray:
    return delta.astype(np.float64) - bucket_project(delta, assignment, theta_d_length)


def snip_support_in_delta_order(obj: dict) -> np.ndarray | None:
    """Map the global ProLoSA sparse mask into the flattened delta ordering.

    Requires per-module offsets (offsets_A/offsets_B) saved in index_meta; artifacts
    produced before this field existed return None (alignment cannot be guaranteed)."""
    mask = obj.get("sparse_mask")
    meta = obj.get("index_meta") or {}
    module_names = obj.get("module_names") or []
    if mask is None or not meta or not module_names:
        return None
    mask = mask.reshape(-1)
    pieces = []
    for name in module_names:
        m = meta.get(name)
        if m is None or "offsets_A" not in m:
            return None
        pieces.append(mask[m["offsets_A"].reshape(-1).long()])
        pieces.append(mask[m["offsets_B"].reshape(-1).long()])
    support = torch.cat(pieces).cpu().numpy()
    return support > 0.5


# ---------------------------------------------------------------------------
# artifact loading
# ---------------------------------------------------------------------------

PROJ_RE = re.compile(r"_proj(\d+)")


def classify_method(path: Path) -> tuple[str, int | None]:
    """Return (method, proj_seed) from the artifact path."""
    parts = path.as_posix()
    proj_seed = None
    m = PROJ_RE.search(parts)
    if m:
        proj_seed = int(m.group(1))
    if "/lora_ref_r" in parts:
        return "lora_ref", proj_seed
    if "/lora_r" in parts:
        return "lora", proj_seed
    if "/unilora_td" in parts:
        return "unilora", proj_seed
    if "/prolosa_td" in parts:
        return "prolosa", proj_seed
    return "other", proj_seed


def sidecar_metrics(pt_path: Path) -> dict:
    json_path = pt_path.with_name(pt_path.name.replace("_theory.pt", ".json"))
    if not json_path.exists():
        return {}
    try:
        data = json.loads(json_path.read_text())
    except Exception:
        return {}
    out = {
        "best_score": data.get("best_score"),
        "best_val_loss": data.get("best_val_loss"),
        "min_val_loss": data.get("min_val_loss"),
    }
    # Older runs lack the explicit loss fields; recover them from history.
    if out["min_val_loss"] is None or out["best_val_loss"] is None:
        history = data.get("history") or []
        losses = [h.get("val_loss") for h in history if h.get("val_loss") is not None]
        if losses:
            if out["min_val_loss"] is None:
                out["min_val_loss"] = float(min(losses))
            if out["best_val_loss"] is None:
                best_h = max(history, key=lambda h: h.get("score", -1e18))
                out["best_val_loss"] = best_h.get("val_loss")
    return out


def load_all(roots) -> list[dict]:
    records = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for path in sorted(root.rglob("*_theory.pt")):
            method, proj_seed = classify_method(path)
            if method == "other":
                continue
            try:
                try:
                    obj = torch.load(path, map_location="cpu", weights_only=False)
                except TypeError:
                    obj = torch.load(path, map_location="cpu")
            except Exception:
                continue
            if obj.get("delta_theta") is None:
                continue
            rec = {
                "path": path,
                "root": str(root),
                "task": obj.get("task"),
                "method": method,
                "proj_seed": proj_seed if proj_seed is not None else obj.get("proj_seed"),
                # Only paths like .../unilora_td23040_proj12/... come from the dedicated
                # controlled sweep (fixed opt seeds, varying P). Elsewhere proj_seed
                # defaults to the run seed and is confounded with init/data order.
                "proj_from_path": proj_seed is not None,
                "obj": obj,
            }
            rec.update(sidecar_metrics(path))
            rec["lora_dedicated_lr"] = bool(re.search(r"/lora(?:_ref)?_r\d+_lr", path.as_posix()))
            records.append(rec)

    # LoRA baselines trained with a dedicated adapter lr (theory_common_lora_lr.sh)
    # supersede legacy runs that used the shared head lr (which diverged on some tasks).
    tasks_with_dedicated = {
        (r["task"], r["method"]) for r in records
        if r["method"] in ("lora", "lora_ref") and r["lora_dedicated_lr"]
    }
    if tasks_with_dedicated:
        before = len(records)
        records = [
            r for r in records
            if not (
                r["method"] in ("lora", "lora_ref")
                and not r["lora_dedicated_lr"]
                and (r["task"], r["method"]) in tasks_with_dedicated
            )
        ]
        print(f"Dropped {before - len(records)} legacy shared-lr LoRA artifacts superseded by dedicated-lr runs.")
    return records


def load_fisher(specs) -> dict:
    fisher = {}
    for spec in specs or []:
        task, _, path = spec.partition("=")
        try:
            obj = torch.load(path, map_location="cpu")
        except Exception as exc:
            print(f"WARNING: cannot load fisher file {path}: {exc}")
            continue
        vec = obj.get("fisher_diag") if isinstance(obj, dict) else obj
        fisher[task] = np.asarray(vec, dtype=np.float64).reshape(-1)
    return fisher


def energy_of(x: np.ndarray, h: np.ndarray | None) -> np.ndarray:
    e = x.astype(np.float64) ** 2
    if h is not None and h.size == e.size:
        e = h * e
    return e


def fmt(v, digits=4):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "-"
    return f"{v:.{digits}f}"


# ---------------------------------------------------------------------------
# main analyses
# ---------------------------------------------------------------------------

def mean_lora_delta(records, task) -> np.ndarray | None:
    deltas = [r["obj"]["delta_theta"].numpy() for r in records if r["task"] == task and r["method"] == "lora"]
    if not deltas:
        return None
    return np.mean(deltas, axis=0)


def analyze_cross_task(records, fisher, lines):
    lines.append("## E4-A cross-task: off-subspace energy vs compression gap (secondary)")
    lines.append("")
    lines.append("| task | r_off | r_off^H | gap_score (LoRA-Uni) | gap_loss (Uni-LoRA loss excess) |")
    lines.append("|---|---:|---:|---:|---:|")

    tasks = sorted({r["task"] for r in records if r["task"]})
    xs_bare, xs_w, gaps_score, gaps_loss = [], [], [], []
    for task in tasks:
        lora_delta = mean_lora_delta(records, task)
        uni_recs = [r for r in records if r["task"] == task and r["method"] == "unilora"]
        if lora_delta is None or not uni_recs:
            continue
        h = fisher.get(task)

        r_offs, r_offs_w = [], []
        for r in uni_recs:
            obj = r["obj"]
            meta = obj.get("index_meta") or {}
            if not meta:
                continue
            assignment = reconstruct_assignment(meta, obj.get("module_names"), lora_delta.size)
            L = int(obj.get("theta_d_length") or 0) or int(assignment.max()) + 1
            q = off_subspace_residual(lora_delta, assignment, L)
            e_tot = energy_of(lora_delta, None).sum()
            r_offs.append(float(energy_of(q, None).sum() / e_tot) if e_tot > 0 else float("nan"))
            if h is not None:
                ew_tot = energy_of(lora_delta, h).sum()
                r_offs_w.append(float(energy_of(q, h).sum() / ew_tot) if ew_tot > 0 else float("nan"))

        lora_scores = [r["best_score"] for r in records if r["task"] == task and r["method"] == "lora" and r["best_score"] is not None]
        uni_scores = [r["best_score"] for r in uni_recs if r["best_score"] is not None]
        lora_losses = [r["min_val_loss"] for r in records if r["task"] == task and r["method"] == "lora" and r["min_val_loss"] is not None]
        uni_losses = [r["min_val_loss"] for r in uni_recs if r["min_val_loss"] is not None]

        gap_score = (np.mean(lora_scores) - np.mean(uni_scores)) if lora_scores and uni_scores else float("nan")
        gap_loss = (np.mean(uni_losses) - np.mean(lora_losses)) if lora_losses and uni_losses else float("nan")
        r_off = float(np.mean(r_offs)) if r_offs else float("nan")
        r_off_w = float(np.mean(r_offs_w)) if r_offs_w else float("nan")

        lines.append(f"| {task} | {fmt(r_off)} | {fmt(r_off_w)} | {fmt(gap_score)} | {fmt(gap_loss)} |")
        if not math.isnan(r_off) and not math.isnan(gap_loss):
            xs_bare.append(r_off)
            gaps_loss.append(gap_loss)
        if not math.isnan(r_off) and not math.isnan(gap_score):
            gaps_score.append((r_off, gap_score))
        if not math.isnan(r_off_w):
            xs_w.append(r_off_w)

    lines.append("")
    if len(xs_bare) >= 3:
        lines.append(
            f"- r_off vs gap_loss: Pearson={fmt(pearson(xs_bare, gaps_loss))}, "
            f"Spearman={fmt(spearman(xs_bare, gaps_loss))} (n={len(xs_bare)})"
        )
    if len(gaps_score) >= 3:
        a = [t[0] for t in gaps_score]
        b = [t[1] for t in gaps_score]
        lines.append(
            f"- r_off vs gap_score: Pearson={fmt(pearson(a, b))}, Spearman={fmt(spearman(a, b))} (n={len(a)})"
        )
    lines.append("")


def analyze_projection_sweep(records, fisher, lines):
    lines.append("## E4-B controlled projection sweep (primary): B_off(P) vs excess dev loss")
    lines.append("")

    tasks = sorted({r["task"] for r in records if r["method"] == "unilora" and r["proj_from_path"]})
    any_output = False
    for task in tasks:
        # Only artifacts from an explicit proj sweep directory (multiple P per task).
        sweep = defaultdict(list)
        for r in records:
            if r["task"] == task and r["method"] == "unilora" and r["proj_from_path"]:
                sweep[r["proj_seed"]].append(r)
        if len(sweep) < 3:
            continue
        lora_delta = mean_lora_delta(records, task)
        lora_losses = [r["min_val_loss"] for r in records if r["task"] == task and r["method"] == "lora" and r["min_val_loss"] is not None]
        if lora_delta is None or not lora_losses:
            continue
        lora_loss = float(np.mean(lora_losses))
        h = fisher.get(task)

        any_output = True
        lines.append(f"### {task}")
        lines.append("")
        lines.append("| proj_seed | B_off ratio | B_off^H ratio | Uni dev loss (mean) | excess loss |")
        lines.append("|---:|---:|---:|---:|---:|")

        boffs, boffs_w, excesses = [], [], []
        for ps in sorted(sweep):
            recs = sweep[ps]
            obj = recs[0]["obj"]
            meta = obj.get("index_meta") or {}
            if not meta:
                continue
            assignment = reconstruct_assignment(meta, obj.get("module_names"), lora_delta.size)
            L = int(obj.get("theta_d_length") or 0) or int(assignment.max()) + 1
            q = off_subspace_residual(lora_delta, assignment, L)
            e_tot = energy_of(lora_delta, None).sum()
            b_off = float(energy_of(q, None).sum() / e_tot) if e_tot > 0 else float("nan")
            b_off_w = float("nan")
            if h is not None:
                ew_tot = energy_of(lora_delta, h).sum()
                b_off_w = float(energy_of(q, h).sum() / ew_tot) if ew_tot > 0 else float("nan")

            losses = [r["min_val_loss"] for r in recs if r["min_val_loss"] is not None]
            uni_loss = float(np.mean(losses)) if losses else float("nan")
            excess = uni_loss - lora_loss if losses else float("nan")
            lines.append(f"| {ps} | {fmt(b_off, 6)} | {fmt(b_off_w, 6)} | {fmt(uni_loss)} | {fmt(excess)} |")
            if not math.isnan(b_off) and not math.isnan(excess):
                boffs.append(b_off)
                excesses.append(excess)
                if not math.isnan(b_off_w):
                    boffs_w.append(b_off_w)

        lines.append("")
        if len(boffs) >= 3:
            lines.append(
                f"- B_off vs excess loss: Pearson={fmt(pearson(boffs, excesses))}, "
                f"Spearman={fmt(spearman(boffs, excesses))} (n={len(boffs)})"
            )
            if len(boffs_w) == len(excesses):
                lines.append(
                    f"- B_off^H vs excess loss: Pearson={fmt(pearson(boffs_w, excesses))}, "
                    f"Spearman={fmt(spearman(boffs_w, excesses))}"
                )
        lines.append("")
    if not any_output:
        lines.append("(no projection-sweep artifacts found; run submit_theory_e4_projection_sweep.sh)")
        lines.append("")


def analyze_residual_concentration(records, fisher, lines, rng):
    lines.append("## E5 off-subspace residual concentration (sparse recoverability)")
    lines.append("")
    lines.append(
        "| task | weighting | K | C_K oracle | rho_I^2 SNIP | K/D | K/(D-d) | Enrich_K | "
        "q top0.1% | q top1% | q top5% | gauss top1% | kurt(q) | gini(q) |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    tasks = sorted({r["task"] for r in records if r["task"]})
    for task in tasks:
        lora_delta = mean_lora_delta(records, task)
        pro_recs = [r for r in records if r["task"] == task and r["method"] == "prolosa"]
        if lora_delta is None or not pro_recs:
            continue
        obj = pro_recs[0]["obj"]
        meta = obj.get("index_meta") or {}
        if not meta:
            continue
        assignment = reconstruct_assignment(meta, obj.get("module_names"), lora_delta.size)
        L = int(obj.get("theta_d_length") or 0) or int(assignment.max()) + 1
        q = off_subspace_residual(lora_delta, assignment, L)
        support = snip_support_in_delta_order(obj)
        D = lora_delta.size
        K = int(support.sum()) if support is not None else 0

        weightings = [("bare", None)]
        h = fisher.get(task)
        if h is not None and h.size == D:
            weightings.append(("fisher", h))

        for wname, w in weightings:
            e_q = energy_of(q, w)
            total_q = e_q.sum()
            c_k = topk_energy_ratio(e_q, K) if K > 0 else float("nan")
            rho2 = float(e_q[support].sum() / total_q) if (support is not None and total_q > 0) else float("nan")
            kd = K / float(D) if K > 0 else float("nan")
            kdd = K / float(D - L) if K > 0 and D > L else float("nan")
            enrich = rho2 / kdd if (not math.isnan(rho2) and kdd and kdd > 0) else float("nan")
            fr = topfrac_energy(e_q)
            # isotropic Gaussian baseline with the same dimension
            g = rng.standard_normal(D)
            fr_g = topfrac_energy(energy_of(g, w))
            lines.append(
                f"| {task} | {wname} | {K} | {fmt(c_k)} | {fmt(rho2)} | {fmt(kd, 6)} | {fmt(kdd, 6)} | "
                f"{fmt(enrich, 2)} | {fmt(fr[0.001])} | {fmt(fr[0.01])} | {fmt(fr[0.05])} | "
                f"{fmt(fr_g[0.01])} | {fmt(kurtosis(q), 2)} | {fmt(gini(q))} |"
            )
    lines.append("")
    lines.append(
        "Notes: q = (I - Pi_P) delta_theta_LoRA with ProLoSA's own grouping; K = SNIP budget; "
        "rho_I^2 = off-subspace energy captured by the actual SNIP support; "
        "Enrich_K = rho_I^2 / (K/(D-d)). rho_I^2 requires offsets_A/B in the artifact "
        "(runs after 2026-08-31); older artifacts show '-'."
    )
    lines.append("")


def analyze_total_spectrum(records, fisher, lines):
    lines.append("## E5 (secondary) total-update energy spectrum")
    lines.append("")
    lines.append("| task | method | top0.1% | top1% | top5% | kurtosis | gini |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    tasks = sorted({r["task"] for r in records if r["task"]})
    for task in tasks:
        for method in ("lora", "unilora", "prolosa"):
            deltas = [r["obj"]["delta_theta"].numpy() for r in records if r["task"] == task and r["method"] == method]
            if not deltas:
                continue
            mean_delta = np.mean(deltas, axis=0)
            fr = topfrac_energy(energy_of(mean_delta, fisher.get(task)))
            lines.append(
                f"| {task} | {method} | {fr[0.001]:.4f} | {fr[0.01]:.4f} | {fr[0.05]:.4f} | "
                f"{kurtosis(mean_delta):.2f} | {gini(mean_delta):.4f} |"
            )
    lines.append("")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True, help="Result roots containing *_theory.pt files.")
    parser.add_argument("--output_md", required=True)
    parser.add_argument(
        "--fisher",
        nargs="*",
        default=None,
        help="Optional task=fisher.pt pairs from estimate_fisher_diag_glue.py for curvature weighting.",
    )
    parser.add_argument("--gauss_seed", type=int, default=0)
    args = parser.parse_args()

    records = load_all(args.roots)
    fisher = load_fisher(args.fisher)
    rng = np.random.default_rng(args.gauss_seed)

    print(f"Loaded {len(records)} artifacts from {len(args.roots)} roots; fisher tasks: {sorted(fisher)}")

    lines = ["# E4 / E5 theory artifacts (revised analysis)", ""]
    analyze_cross_task(records, fisher, lines)
    analyze_projection_sweep(records, fisher, lines)
    analyze_residual_concentration(records, fisher, lines, rng)
    analyze_total_spectrum(records, fisher, lines)

    text = "\n".join(lines) + "\n"
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text(text)
    print(text)


if __name__ == "__main__":
    main()

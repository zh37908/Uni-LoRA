"""Analyze the locked, paired ten-teacher Q4 experiment.

This program deliberately refuses partial grids.  Its independent statistical
units are the ten teacher draws, while the twenty fits within each teacher
estimate that teacher's risk and prediction variance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t


HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "prolosa_spike_slab_teacher_20260925"
OUT = HERE / "summary"
SEEDS = tuple(range(100, 110))
METHODS = ("LoRA", "Uni-LoRA", "ProLoSA")
NOISE = (0.0, 0.5, 1.0, 2.0, 4.0)
REPEATS = tuple(range(20))
LONG_NOISE = (1.0, 4.0)
LONG_REPEATS = tuple(range(5))
LONG_STEPS = (1000, 3000)
SHAPE = (1024, 16)
COLORS = {"LoRA": "#555555", "Uni-LoRA": "#0072b2", "ProLoSA": "#d55e00"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def confidence_interval(values, level: float = 0.95) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(SEEDS),) or not np.isfinite(values).all():
        raise ValueError(f"Expected {len(SEEDS)} finite teacher values, got {values.shape}")
    center = float(values.mean())
    standard_error = float(values.std(ddof=1) / math.sqrt(len(values)))
    radius = float(student_t.ppf((1.0 + level) / 2.0, df=len(values) - 1) * standard_error)
    return {"mean": center, "low": center - radius, "high": center + radius,
            "standard_error": standard_error, "level": level, "df": len(values) - 1}


def cell(predictions: np.ndarray, truth: np.ndarray) -> dict:
    if predictions.shape != (len(REPEATS), *SHAPE):
        raise ValueError(f"Unexpected prediction stack shape {predictions.shape}")
    predictions = predictions.astype(np.float64)
    mean_prediction = predictions.mean(axis=0)
    variance = float(predictions.var(axis=0, ddof=1).mean())
    bias2 = float(np.square(mean_prediction - truth).mean() - variance / len(REPEATS))
    risk = float(np.square(predictions - truth).mean())
    if not math.isclose(risk, bias2 + variance, rel_tol=1e-10, abs_tol=1e-12):
        raise AssertionError("R = B2 + V identity failed")
    return {"risk": risk, "bias2": bias2, "variance": variance}


def relative_output(spec_out: str) -> Path:
    relative = Path(spec_out)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"Unsafe result directory in grid: {spec_out}")
    folder = (HERE / relative).resolve()
    if not folder.is_relative_to(HERE):
        raise ValueError(f"Result directory escapes experiment: {spec_out}")
    return folder


def validate_grid(grid: list[dict]) -> dict[int, object]:
    expected = {(seed, method, noise, repeat)
                for seed in SEEDS for method in METHODS for noise in NOISE for repeat in REPEATS}
    if len(grid) != len(expected):
        raise ValueError(f"Incomplete grid: expected 3000 fits, found {len(grid)}")
    actual = set()
    teacher_ids = {}
    outputs = set()
    for item in grid:
        required = ("teacher_seed", "teacher_id", "repeat", "noise_multiple", "method", "out")
        missing = set(required) - item.keys()
        if missing:
            raise ValueError(f"Grid item lacks {sorted(missing)}: {item}")
        seed, method, noise, repeat = (int(item["teacher_seed"]), item["method"],
                                      float(item["noise_multiple"]), int(item["repeat"]))
        key = (seed, method, noise, repeat)
        if key in actual:
            raise ValueError(f"Duplicate fit: {key}")
        actual.add(key)
        expected_long = noise in LONG_NOISE and repeat in LONG_REPEATS
        if item.get("long_control") is not expected_long:
            raise ValueError(f"Long-control selection differs from the locked protocol: {key}")
        teacher_id = item["teacher_id"]
        if seed in teacher_ids and teacher_ids[seed] != teacher_id:
            raise ValueError(f"Teacher {seed} has inconsistent IDs")
        teacher_ids[seed] = teacher_id
        folder = relative_output(item["out"])
        if folder in outputs:
            raise ValueError(f"Two fits share output directory {folder}")
        outputs.add(folder)
    if actual != expected:
        raise ValueError(f"Grid keys differ from protocol; missing={list(expected-actual)[:5]}, "
                         f"extra={list(actual-expected)[:5]}")
    if len(set(map(str, teacher_ids.values()))) != len(SEEDS):
        raise ValueError("Teacher IDs are not unique")
    return teacher_ids


def same_field(records: list[dict], field: str, context: str, required: bool = False) -> object:
    values = [record.get(field) for record in records]
    if required and any(value is None for value in values):
        raise ValueError(f"Missing {field} in {context}")
    if any(value is not None for value in values) and any(value is None for value in values):
        raise ValueError(f"Inconsistent availability of {field} in {context}")
    if len(set(values)) != 1:
        raise ValueError(f"Pairing mismatch for {field} in {context}")
    return values[0]


def load_main(spec: dict):
    grid = spec["grid"]
    teacher_ids = validate_grid(grid)
    spec_hash = sha(HERE / "spec.json")
    if spec.get("teacher_seeds") not in (None, list(SEEDS)):
        raise ValueError("spec.json teacher seeds differ from the locked protocol")
    if spec.get("noise_multiples") not in (None, list(NOISE)):
        raise ValueError("spec.json noise levels differ from the locked protocol")
    if spec.get("methods") not in (None, list(METHODS)):
        raise ValueError("spec.json methods differ from the locked protocol")
    if spec.get("updates") not in (None, 1000) or spec.get("repeats") not in (None, len(REPEATS)):
        raise ValueError("spec.json update or repeat count differs from the locked protocol")
    if spec.get("long_control") is not True or spec.get("long_updates") != 3000:
        raise ValueError("spec.json does not declare the locked 300-trajectory long control")
    if spec.get("test_id") is not None and spec.get("test_sha256") != sha(HERE / "test.pt"):
        raise ValueError("Fixed test inputs no longer match spec.json")
    truths = {}
    truth_hashes = {}
    for seed in SEEDS:
        path = HERE / "teachers" / f"t{seed}" / "truth.npy"
        truth = np.load(path).astype(np.float64)
        if truth.shape != SHAPE or not np.isfinite(truth).all():
            raise ValueError(f"Invalid teacher {seed} truth: {truth.shape}")
        truths[seed] = truth
        truth_hashes[seed] = sha(path)
        if "teacher_manifests" in spec:
            manifest = spec["teacher_manifests"][str(seed)]
            if manifest["truth_sha256"] != truth_hashes[seed]:
                raise ValueError(f"Teacher {seed} truth no longer matches spec.json")
            if manifest["teacher_id"] != teacher_ids[seed]:
                raise ValueError(f"Teacher {seed} ID no longer matches spec.json")

    predictions = {(seed, method, noise): [None] * len(REPEATS)
                   for seed in SEEDS for method in METHODS for noise in NOISE}
    long_risks = {}
    records_by_pair = defaultdict(list)
    records_by_target = defaultdict(list)
    records_by_label = defaultdict(list)
    manifest = []
    for item in grid:
        seed = int(item["teacher_seed"])
        method = item["method"]
        noise = float(item["noise_multiple"])
        repeat = int(item["repeat"])
        folder = relative_output(item["out"])
        result_path = folder / "result.json"
        prediction_path = folder / "predictions.npz"
        if not result_path.is_file() or not prediction_path.is_file():
            raise FileNotFoundError(f"Incomplete fit: {folder}")
        result = read_json(result_path)
        result_spec = result.get("spec")
        if not isinstance(result_spec, dict):
            raise ValueError(f"Missing result spec: {result_path}")
        for key in ("teacher_seed", "teacher_id", "repeat", "noise_multiple", "method", "out"):
            got, want = result_spec.get(key), item[key]
            if key == "noise_multiple":
                equal = got is not None and float(got) == float(want)
            else:
                equal = got == want
            if not equal:
                raise ValueError(f"Result spec mismatch for {key}: {result_path}")
        if result.get("spec_sha256") not in (None, spec_hash):
            raise ValueError(f"Result uses another experiment spec: {result_path}")
        if "configs" in spec and result.get("config") != spec["configs"][method]:
            raise ValueError(f"Selected configuration changed: {result_path}")
        if result.get("effective_parameters") not in (None, 4608 if method == "LoRA" else 576):
            raise ValueError(f"Adapter budget changed: {result_path}")
        if result.get("test_id") not in (None, spec.get("test_id")):
            raise ValueError(f"Test input ID changed: {result_path}")
        if result.get("truth_sha256") not in (None, truth_hashes[seed]):
            raise ValueError(f"Teacher truth changed: {result_path}")
        if "teacher_manifests" in spec:
            expected_target = spec["teacher_manifests"][str(seed)]["teacher_hash"]
            if result.get("teacher_hash") not in (None, expected_target):
                raise ValueError(f"Teacher target changed: {result_path}")
        dependency_fields = (("source_sha256", "runner"),
                             ("train_source_sha256", "training_source"),
                             ("teacher_source_sha256", "teacher_constructor"))
        for result_key, dependency_key in dependency_fields:
            expected_hash = spec.get("dependencies", {}).get(dependency_key)
            if result.get(result_key) not in (None, expected_hash):
                raise ValueError(f"Source hash changed for {result_key}: {result_path}")
        expected_steps = [1000, 3000] if item.get("long_control") else [1000]
        if [int(row["step"]) for row in result.get("rows", [])] != expected_steps:
            raise ValueError(f"Unexpected checkpoints: {result_path}")
        if result.get("predictions_sha256") not in (None, sha(prediction_path)):
            raise ValueError(f"Prediction hash mismatch: {prediction_path}")
        rows = {int(row["step"]): row for row in result["rows"]}
        with np.load(prediction_path) as data:
            expected_keys = {f"step{step}" for step in expected_steps}
            if set(data.files) != expected_keys:
                raise ValueError(f"Prediction checkpoints differ from protocol: {prediction_path}")
            checked = {}
            for step in expected_steps:
                prediction_at_step = data[f"step{step}"].copy()
                if prediction_at_step.shape != SHAPE or not np.isfinite(prediction_at_step).all():
                    raise ValueError(f"Invalid step {step} predictions: {prediction_path}")
                step_risk = float(np.square(prediction_at_step.astype(np.float64) - truths[seed]).mean())
                if not math.isclose(step_risk, float(rows[step]["risk"]),
                                    rel_tol=1e-7, abs_tol=1e-10):
                    raise ValueError(f"Reported step {step} risk does not match predictions: {result_path}")
                checked[step] = (prediction_at_step, step_risk)
        prediction, risk = checked[1000]
        predictions[(seed, method, noise)][repeat] = prediction
        if item["long_control"]:
            for step in LONG_STEPS:
                long_risks[seed, method, noise, repeat, step] = checked[step][1]
        record = {"teacher_seed": seed, "teacher_id": teacher_ids[seed],
                  "repeat": repeat, "noise_multiple": noise, "method": method,
                  "out": item["out"], "risk": risk,
                  "seconds": rows[1000].get("seconds"),
                  "train_x_id": result.get("train_x_id"),
                  "noise_id": result.get("noise_id"),
                  "train_y_id": result.get("train_y_id"),
                  "test_id": result.get("test_id"),
                  "target_id": result.get("target_id", result.get("teacher_hash")),
                  "result_sha256": sha(result_path),
                  "predictions_sha256": sha(prediction_path)}
        records_by_pair[(seed, repeat)].append(record)
        records_by_target[seed].append(record)
        records_by_label[(seed, repeat, noise)].append(record)
        manifest.append(record)

    for key, stack in predictions.items():
        if any(prediction is None for prediction in stack):
            raise ValueError(f"Missing repetitions in {key}")
    x_ids, noise_ids = set(), set()
    for (seed, repeat), records in records_by_pair.items():
        if len(records) != len(METHODS) * len(NOISE):
            raise ValueError(f"Pair ({seed}, {repeat}) has {len(records)} fits")
        x_ids.add(same_field(records, "train_x_id", str((seed, repeat)), required=True))
        noise_ids.add(same_field(records, "noise_id", str((seed, repeat)), required=True))
        same_field(records, "test_id", str((seed, repeat)))
        same_field(records, "target_id", str((seed, repeat)))
    if len(x_ids) != len(SEEDS) * len(REPEATS) or len(noise_ids) != len(x_ids):
        raise ValueError("Independent teacher/repeat X or noise streams were reused")
    test_ids = [record["test_id"] for record in manifest]
    if all(value is not None for value in test_ids) and len(set(test_ids)) != 1:
        raise ValueError("The test inputs differ across teachers")
    targets = []
    for seed in SEEDS:
        target = same_field(records_by_target[seed], "target_id", f"teacher {seed}")
        if target is not None:
            targets.append(target)
    if targets and len(set(targets)) != len(SEEDS):
        raise ValueError("Different teachers share a target hash")
    for key, records in records_by_label.items():
        if len(records) != len(METHODS):
            raise ValueError(f"Methods are not paired for {key}")
        same_field(records, "train_y_id", str(key))
    return teacher_ids, truths, truth_hashes, predictions, long_risks, manifest


def crossings(values: list[float]) -> tuple[list[dict], bool]:
    events = []
    for index, (left, right) in enumerate(zip(values[:-1], values[1:])):
        if left * right < 0:
            events.append({"left": NOISE[index], "right": NOISE[index + 1],
                           "direction": "negative_to_positive" if left < 0 else "positive_to_negative"})
    for index, value in enumerate(values):
        if value == 0:
            events.append({"at": NOISE[index], "direction": "exact_zero"})
    negative_before_positive = any(values[i] < 0 and values[j] > 0
                                   for i in range(len(values)) for j in range(i + 1, len(values)))
    return events, negative_before_positive


def build_statistics(teacher_ids, truths, predictions):
    cells = {}
    cell_rows = []
    for seed in SEEDS:
        for noise in NOISE:
            for method in METHODS:
                prediction_stack = np.stack(predictions[seed, method, noise])
                values = cell(prediction_stack, truths[seed])
                row = {"teacher_seed": seed, "teacher_id": teacher_ids[seed],
                       "noise_multiple": noise, "method": method, **values}
                cells[seed, method, noise] = row
                cell_rows.append(row)

    differences = []
    teacher_rows = []
    for seed in SEEDS:
        by_noise = []
        for noise in NOISE:
            lora, uni, pro = [cells[seed, method, noise] for method in METHODS]
            d = lora["risk"] - uni["risk"]
            g = uni["risk"] - pro["risk"]
            db = lora["bias2"] - uni["bias2"]
            dv = lora["variance"] - uni["variance"]
            gb = uni["bias2"] - pro["bias2"]
            gv = uni["variance"] - pro["variance"]
            if not math.isclose(d, db + dv, rel_tol=1e-9, abs_tol=1e-12):
                raise AssertionError("D components do not sum")
            if not math.isclose(g, gb + gv, rel_tol=1e-9, abs_tol=1e-12):
                raise AssertionError("G components do not sum")
            rho = 100.0 * (1.0 - pro["risk"] / uni["risk"])
            row = {"teacher_seed": seed, "teacher_id": teacher_ids[seed],
                   "noise_multiple": noise, "D_lora_minus_uni": d,
                   "D_bias2": db, "D_variance": dv,
                   "G_uni_minus_pro": g, "G_bias2": gb, "G_variance": gv,
                   "rho_pro_vs_uni_pct": rho}
            differences.append(row)
            by_noise.append(row)
        events, has_crossover = crossings([row["D_lora_minus_uni"] for row in by_noise])
        d0, d4 = by_noise[0]["D_lora_minus_uni"], by_noise[-1]["D_lora_minus_uni"]
        gains = [row["G_uni_minus_pro"] for row in by_noise]
        teacher_rows.append({
            "teacher_seed": seed, "teacher_id": teacher_ids[seed], "reference_only": False,
            "D_0": d0, "D_4": d4, "H_D4_minus_D0": d4 - d0,
            "A_mean_G": float(np.mean(gains)),
            "point_estimate_neg_to_pos_crossover": has_crossover,
            "all_sign_changes": json.dumps(events, ensure_ascii=False),
            "G_signs_by_noise": ",".join("+" if gain > 0 else "-" if gain < 0 else "0" for gain in gains),
            "G_positive_noise_count": sum(gain > 0 for gain in gains),
            **{f"G_{noise:g}": gain for noise, gain in zip(NOISE, gains)},
        })

    primary = {}
    for key, values in (("H", [row["H_D4_minus_D0"] for row in teacher_rows]),
                        ("A", [row["A_mean_G"] for row in teacher_rows])):
        primary[key] = {"teacher_values": values,
                        "ci95": confidence_interval(values, 0.95),
                        "ci97_5_bonferroni": confidence_interval(values, 0.975)}
    table_a = []
    curve = []
    for noise in NOISE:
        row = {"noise_multiple": noise}
        for method in METHODS:
            compact = method.lower().replace("-", "_")
            for measure in ("risk", "bias2", "variance"):
                values = [cells[seed, method, noise][measure] for seed in SEEDS]
                ci = confidence_interval(values)
                curve.append({"noise_multiple": noise, "method": method,
                              "measure": measure, **ci})
                row[f"{compact}_{measure}_mean"] = ci["mean"]
                if measure == "risk":
                    row[f"{compact}_risk_ci95_low"] = ci["low"]
                    row[f"{compact}_risk_ci95_high"] = ci["high"]
        noise_rows = [difference for difference in differences if difference["noise_multiple"] == noise]
        for key, prefix in (("D_lora_minus_uni", "D"), ("G_uni_minus_pro", "G")):
            values = [difference[key] for difference in noise_rows]
            ci95 = confidence_interval(values, 0.95)
            ci99 = confidence_interval(values, 0.99)
            row.update({f"{prefix}_mean": ci95["mean"],
                        f"{prefix}_ci95_low": ci95["low"], f"{prefix}_ci95_high": ci95["high"],
                        f"{prefix}_ci99_low": ci99["low"], f"{prefix}_ci99_high": ci99["high"]})
        rho = confidence_interval([difference["rho_pro_vs_uni_pct"] for difference in noise_rows])
        row.update({"mean_teacher_relative_gain_pct": rho["mean"],
                    "relative_gain_ci95_low": rho["low"],
                    "relative_gain_ci95_high": rho["high"]})
        table_a.append(row)
    significant_reversal_pairs = [(left["noise_multiple"], right["noise_multiple"])
                                  for left in table_a for right in table_a
                                  if left["noise_multiple"] < right["noise_multiple"]
                                  and left["D_ci99_high"] < 0 < right["D_ci99_low"]]
    formal_all_five_gains = all(row["G_ci99_low"] > 0 for row in table_a)
    summary = {
        "independent_unit": "teacher", "teacher_seeds": list(SEEDS),
        "teachers": len(SEEDS), "repeats_per_teacher_cell": len(REPEATS),
        "fits": len(SEEDS) * len(METHODS) * len(NOISE) * len(REPEATS),
        "primary": primary,
        "point_estimate_neg_to_pos_crossover_teachers": sum(
            row["point_estimate_neg_to_pos_crossover"] for row in teacher_rows),
        "A_positive_teachers": sum(row["A_mean_G"] > 0 for row in teacher_rows),
        "formal_average_reversal_supported_by_five_D_99pct_simultaneous_intervals":
            bool(significant_reversal_pairs),
        "supporting_low_high_noise_pairs": significant_reversal_pairs,
        "all_five_G_positive_with_99pct_simultaneous_intervals": formal_all_five_gains,
        "ci_note": "Teacher-level paired Student t intervals, df=9; 97.5% for H/A joint "
                   "Bonferroni and 99% for five noise-point simultaneous claims. "
                   "Small-sample t approximation; fixed test-input design.",
    }
    return cells, cell_rows, differences, teacher_rows, table_a, curve, summary


def build_long_control_statistics(teacher_ids: dict, fit_risks: dict):
    """Summarize only the predeclared 300 extended fits, with teachers as units."""
    expected = {(seed, method, noise, repeat, step)
                for seed in SEEDS for method in METHODS for noise in LONG_NOISE
                for repeat in LONG_REPEATS for step in LONG_STEPS}
    if set(fit_risks) != expected:
        raise ValueError(f"Long-control checkpoints differ from protocol: "
                         f"missing={list(expected - set(fit_risks))[:5]}, "
                         f"extra={list(set(fit_risks) - expected)[:5]}")
    if any(not math.isfinite(float(value)) or float(value) < 0 for value in fit_risks.values()):
        raise ValueError("Long-control risks must be finite and nonnegative")

    teacher_rows = []
    by_teacher = {}
    for seed in SEEDS:
        for noise in LONG_NOISE:
            for step in LONG_STEPS:
                risks = {method: float(np.mean([fit_risks[seed, method, noise, repeat, step]
                                                for repeat in LONG_REPEATS]))
                         for method in METHODS}
                row = {"teacher_seed": seed, "teacher_id": teacher_ids[seed],
                       "noise_multiple": noise, "step": step,
                       "lora_risk": risks["LoRA"],
                       "uni_lora_risk": risks["Uni-LoRA"],
                       "prolosa_risk": risks["ProLoSA"],
                       "D_lora_minus_uni": risks["LoRA"] - risks["Uni-LoRA"],
                       "G_uni_minus_pro": risks["Uni-LoRA"] - risks["ProLoSA"]}
                teacher_rows.append(row)
                by_teacher[seed, noise, step] = row

    risk_fields = ("lora_risk", "uni_lora_risk", "prolosa_risk",
                   "D_lora_minus_uni", "G_uni_minus_pro")
    summary_rows = []
    for noise in LONG_NOISE:
        for step in LONG_STEPS:
            row = {"noise_multiple": noise, "step": step,
                   "teacher_count": len(SEEDS), "repeats_per_teacher": len(LONG_REPEATS)}
            for field in risk_fields:
                interval = confidence_interval([by_teacher[seed, noise, step][field]
                                                for seed in SEEDS])
                row.update({f"{field}_mean": interval["mean"],
                            f"{field}_ci95_low": interval["low"],
                            f"{field}_ci95_high": interval["high"]})
            summary_rows.append(row)

    change_teacher_rows = []
    by_change = {}
    for seed in SEEDS:
        for noise in LONG_NOISE:
            first = by_teacher[seed, noise, 1000]
            last = by_teacher[seed, noise, 3000]
            row = {"teacher_seed": seed, "teacher_id": teacher_ids[seed],
                   "noise_multiple": noise,
                   **{f"{field}_3000_minus_1000": last[field] - first[field]
                      for field in risk_fields}}
            change_teacher_rows.append(row)
            by_change[seed, noise] = row
    change_summary_rows = []
    for noise in LONG_NOISE:
        row = {"noise_multiple": noise, "teacher_count": len(SEEDS),
               "repeats_per_teacher": len(LONG_REPEATS)}
        for field in risk_fields:
            name = f"{field}_3000_minus_1000"
            interval = confidence_interval([by_change[seed, noise][name] for seed in SEEDS])
            row.update({f"{name}_mean": interval["mean"],
                        f"{name}_ci95_low": interval["low"],
                        f"{name}_ci95_high": interval["high"]})
        change_summary_rows.append(row)

    metadata = {"status": "complete", "analysis": "secondary_optimization_control",
                "independent_unit": "teacher", "teacher_seeds": list(SEEDS),
                "noise_multiples": list(LONG_NOISE), "repeats": list(LONG_REPEATS),
                "steps": list(LONG_STEPS), "extended_trajectories": len(expected) // len(LONG_STEPS),
                "prediction_checkpoints": len(expected),
                "interval": "Two-sided 95% teacher-level paired Student t, df=9; descriptive, "
                            "without multiplicity adjustment. Fixed test inputs and five "
                            "predeclared repeats per teacher/noise/method."}
    return teacher_rows, summary_rows, change_teacher_rows, change_summary_rows, metadata


def write_long_control_report(summary_rows: list[dict], change_summary_rows: list[dict]) -> None:
    def estimate(row: dict, field: str) -> str:
        return (f"{row[f'{field}_mean']:.6g} "
                f"[{row[f'{field}_ci95_low']:.6g}, {row[f'{field}_ci95_high']:.6g}]")

    lines = ["# Q4 secondary optimization control", "",
             "The 300 extended trajectories were specified before outcomes were observed: "
             "ten new teachers, noise levels 1 and 4, the first five paired repeats, and all "
             "three methods. Every trajectory contributes predictions at 1000 and 3000 "
             "optimizer updates. The 1000-step column below uses only these same five repeats, "
             "so each change is paired. It does not replace the 20-repeat primary analysis.", "",
             "## Clean test risk and Uni-LoRA minus ProLoSA gain", "",
             "| Noise | Step | LoRA risk (95% CI) | Uni-LoRA risk (95% CI) | "
             "ProLoSA risk (95% CI) | Uni−Pro gain (95% CI) |",
             "|---:|---:|---:|---:|---:|---:|"]
    for row in summary_rows:
        lines.append(f"| {row['noise_multiple']:g} | {row['step']} | "
                     f"{estimate(row, 'lora_risk')} | {estimate(row, 'uni_lora_risk')} | "
                     f"{estimate(row, 'prolosa_risk')} | "
                     f"{estimate(row, 'G_uni_minus_pro')} |")
    lines.extend(["", "## Change from 1000 to 3000 updates", "",
                  "Negative risk changes indicate improvement; a positive gain change indicates "
                  "a larger ProLoSA advantage. Each interval is calculated from ten paired "
                  "teacher-level changes.", "",
                  "| Noise | ΔLoRA risk (95% CI) | ΔUni-LoRA risk (95% CI) | "
                  "ΔProLoSA risk (95% CI) | Δ(Uni−Pro gain) (95% CI) |",
                  "|---:|---:|---:|---:|---:|"])
    for row in change_summary_rows:
        lines.append(f"| {row['noise_multiple']:g} | "
                     f"{estimate(row, 'lora_risk_3000_minus_1000')} | "
                     f"{estimate(row, 'uni_lora_risk_3000_minus_1000')} | "
                     f"{estimate(row, 'prolosa_risk_3000_minus_1000')} | "
                     f"{estimate(row, 'G_uni_minus_pro_3000_minus_1000')} |")
    lines.extend(["", "Intervals use the ten teachers as independent units (df=9), with five "
                  "repeats averaged within each teacher. These descriptive secondary intervals "
                  "are unadjusted for multiple comparisons and conditional on the fixed test "
                  "inputs and selected configurations. The individual teacher results, including "
                  "all signs and negative results, are in [long_control_teachers.csv]"
                  "(long_control_teachers.csv) and [long_control_changes_teachers.csv]"
                  "(long_control_changes_teachers.csv).", "",
                  "Machine-readable intervals: [long_control_summary.csv]"
                  "(long_control_summary.csv), [long_control_changes_summary.csv]"
                  "(long_control_changes_summary.csv).", ""])
    (OUT / "LONG_CONTROL_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def old_reference() -> dict | None:
    selection_path = OLD / "selection.json"
    truth_path = OLD / "truth.npy"
    if not selection_path.is_file() or not truth_path.is_file():
        return None
    selection = read_json(selection_path)
    grid = selection["main_grid"]
    if len(grid) != len(METHODS) * len(NOISE) * len(REPEATS):
        return None
    if any(not (OLD / "main" / f"{index:04d}" / "predictions.npz").is_file()
           for index in range(len(grid))):
        return None
    truth = np.load(truth_path).astype(np.float64)
    if truth.shape != SHAPE:
        raise ValueError("Old seed-13 test truth has the wrong shape")
    grouped = {(method, noise): [None] * len(REPEATS) for method in METHODS for noise in NOISE}
    for index, item in enumerate(grid):
        folder = OLD / "main" / f"{index:04d}"
        with np.load(folder / "predictions.npz") as archive:
            prediction = archive["step1000"].copy()
        grouped[item["method"], float(item["noise_multiple"])][int(item["repeat"])] = prediction
    old_cells = {key: cell(np.stack(stack), truth) for key, stack in grouped.items()}
    d = [old_cells["LoRA", noise]["risk"] - old_cells["Uni-LoRA", noise]["risk"] for noise in NOISE]
    g = [old_cells["Uni-LoRA", noise]["risk"] - old_cells["ProLoSA", noise]["risk"] for noise in NOISE]
    events, crossed = crossings(d)
    return {"teacher_seed": 13, "teacher_id": "original_seed13", "reference_only": True,
            "D_0": d[0], "D_4": d[-1], "H_D4_minus_D0": d[-1] - d[0],
            "A_mean_G": float(np.mean(g)), "point_estimate_neg_to_pos_crossover": crossed,
            "all_sign_changes": json.dumps(events, ensure_ascii=False),
            "G_signs_by_noise": ",".join("+" if value > 0 else "-" if value < 0 else "0" for value in g),
            "G_positive_noise_count": sum(value > 0 for value in g),
            **{f"G_{noise:g}": value for noise, value in zip(NOISE, g)},
            "note": "Original single teacher; excluded from new-teacher means and confidence intervals."}


def plot_figures(cells, differences, curve) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    titles = {"risk": "Clean test MSE", "bias2": "Prediction bias squared",
              "variance": "Prediction variance"}
    for ax, measure in zip(axes, ("risk", "bias2", "variance")):
        for method in METHODS:
            entries = [next(row for row in curve if row["measure"] == measure
                            and row["method"] == method and row["noise_multiple"] == noise)
                       for noise in NOISE]
            ax.plot(NOISE, [row["mean"] for row in entries], "o-", color=COLORS[method],
                    lw=1.8, markersize=4, label=method)
            ax.fill_between(NOISE, [row["low"] for row in entries],
                            [row["high"] for row in entries], color=COLORS[method], alpha=0.14)
        ax.set_title(titles[measure])
        ax.set_xlabel("Noise / teacher displacement RMS")
        ax.grid(alpha=0.2)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0].legend(fontsize=9)
    fig.suptitle("Ten independent teachers; teacher-level 95% t intervals", fontsize=11)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"figure_a_risk_bias_variance.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title, color in zip(axes,
                                     ("D_lora_minus_uni", "G_uni_minus_pro"),
                                     ("LoRA minus Uni-LoRA risk", "Uni-LoRA minus ProLoSA risk"),
                                     (COLORS["Uni-LoRA"], COLORS["ProLoSA"])):
        for seed in SEEDS:
            rows = sorted((row for row in differences if row["teacher_seed"] == seed),
                          key=lambda row: row["noise_multiple"])
            ax.plot(NOISE, [row[key] for row in rows], color=color, alpha=0.25, lw=0.85)
        means = [float(np.mean([row[key] for row in differences
                                if row["noise_multiple"] == noise])) for noise in NOISE]
        ax.plot(NOISE, means, "o-", color=color, lw=2.5, markersize=5,
                label="Mean across teachers")
        ax.axhline(0, color="black", lw=0.9)
        ax.set_title(title)
        ax.set_xlabel("Noise / teacher displacement RMS")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.suptitle("Thin lines: independent teachers; thick line: teacher mean", fontsize=11)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"figure_b_paired_differences.{extension}", dpi=220,
                    bbox_inches="tight")
    plt.close(fig)


def write_report(table_a: list[dict], teacher_rows: list[dict], summary: dict) -> None:
    def ci(value: dict) -> str:
        return f"{value['mean']:.6g} [{value['low']:.6g}, {value['high']:.6g}]"

    primary = summary["primary"]
    lines = ["# Q4: independent Transformer teachers", "",
             "Frozen configurations from the original seed-13 development search; ten new teachers "
             "(100–109), twenty paired training-data/noise repeats per teacher and noise level. "
             "The original teacher is a separate reference row.", "",
             f"H = D(4) − D(0): {ci(primary['H']['ci95'])} (95% teacher t interval); "
             f"Bonferroni 97.5% interval {ci(primary['H']['ci97_5_bonferroni'])}.", "",
             f"A = mean over five noise levels of (Uni risk − Pro risk): "
             f"{ci(primary['A']['ci95'])} (95% teacher t interval); "
             f"Bonferroni 97.5% interval {ci(primary['A']['ci97_5_bonferroni'])}.", "",
             f"Point-estimate negative-to-positive LoRA–Uni crossover: "
             f"{summary['point_estimate_neg_to_pos_crossover_teachers']}/10 teachers. "
             f"A > 0: {summary['A_positive_teachers']}/10 teachers.", "",
             "The five D intervals used for a formal average reversal are Bonferroni-simultaneous "
             "99% intervals. Formal reversal supported: "
             f"{summary['formal_average_reversal_supported_by_five_D_99pct_simultaneous_intervals']}. "
             "All five G benefits supported with simultaneous 99% intervals: "
             f"{summary['all_five_G_positive_with_99pct_simultaneous_intervals']}.", "",
             "## Table A: clean test MSE and paired differences", "",
             "| Noise | LoRA MSE | Uni-LoRA MSE | ProLoSA MSE | D = LoRA−Uni (95% CI) | "
             "G = Uni−Pro (95% CI) | Mean per-teacher Pro risk reduction |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in table_a:
        lines.append(f"| {row['noise_multiple']:g} | {row['lora_risk_mean']:.6g} | "
                     f"{row['uni_lora_risk_mean']:.6g} | {row['prolosa_risk_mean']:.6g} | "
                     f"{row['D_mean']:.6g} [{row['D_ci95_low']:.6g}, {row['D_ci95_high']:.6g}] | "
                     f"{row['G_mean']:.6g} [{row['G_ci95_low']:.6g}, {row['G_ci95_high']:.6g}] | "
                     f"{row['mean_teacher_relative_gain_pct']:.3g}% |")
    lines.extend(["", "The percentage column averages each teacher's 100×(1−Pro risk/Uni risk); "
                  "it is not a ratio of grand means. Confidence intervals use teachers as units "
                  "(T=10, df=9), with M=20 repeats within each teacher and a fixed test design.", "",
                  "![Risk, bias squared, and variance](figure_a_risk_bias_variance.png)", "",
                  "![Paired differences across teachers](figure_b_paired_differences.png)", "",
                  "## Table B: individual teachers", "",
                  "| Teacher | D(0) | D(4) | H | A | Negative-to-positive crossover | "
                  "All adjacent sign changes | G signs (noise 0,.5,1,2,4) |",
                  "|---:|---:|---:|---:|---:|---|---|---|"])
    for row in teacher_rows:
        name = "13 (reference)" if row["reference_only"] else str(row["teacher_seed"])
        lines.append(f"| {name} | {row['D_0']:.6g} | {row['D_4']:.6g} | "
                     f"{row['H_D4_minus_D0']:.6g} | {row['A_mean_G']:.6g} | "
                     f"{row['point_estimate_neg_to_pos_crossover']} | "
                     f"{row['all_sign_changes']} | {row['G_signs_by_noise']} |")
    lines.extend(["", "The seed-13 reference is excluded from all ten-teacher means and intervals. "
                  "A positive H indicates a direction change, not by itself a risk-order reversal. "
                  "A positive A indicates an average benefit across the five specified noise levels, "
                  "not a benefit at every level. Negative corrected bias² values, if any, are retained. "
                  "These small-sample t intervals are conditional on the fixed test inputs and selected "
                  "configurations; the earlier ProLoSA development search was larger than the LoRA/Uni "
                  "search.", "",
                  "Full numeric results: [table_a.csv](table_a.csv), "
                  "[table_b_teachers.csv](table_b_teachers.csv), "
                  "[teacher_cells.csv](teacher_cells.csv), "
                  "[teacher_differences.csv](teacher_differences.csv), "
                  "[fit_manifest.csv](fit_manifest.csv).", "",
                  "The predeclared 300-trajectory extension to 3000 updates is reported "
                  "separately in [LONG_CONTROL_RESULTS.md](LONG_CONTROL_RESULTS.md).", ""])
    (OUT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    spec_path = HERE / "spec.json"
    spec = read_json(spec_path)
    teacher_ids, truths, truth_hashes, predictions, long_risks, manifest = load_main(spec)
    cells, cell_rows, differences, teacher_rows, table_a, curve, summary = build_statistics(
        teacher_ids, truths, predictions)
    (long_teacher_rows, long_summary_rows, long_change_teacher_rows,
     long_change_summary_rows, long_metadata) = build_long_control_statistics(
        teacher_ids, long_risks)
    long_metadata["spec_sha256"] = sha(spec_path)
    long_metadata["analysis_source_sha256"] = sha(Path(__file__))
    reference = old_reference()
    if reference is not None:
        teacher_rows.append(reference)
        summary["seed13_reference"] = reference
    OUT.mkdir(exist_ok=True)
    write_csv(OUT / "fit_manifest.csv", manifest)
    write_csv(OUT / "teacher_cells.csv", cell_rows)
    write_csv(OUT / "teacher_differences.csv", differences)
    write_csv(OUT / "table_a.csv", table_a)
    write_csv(OUT / "table_b_teachers.csv", teacher_rows)
    write_csv(OUT / "figure_a_curve_values.csv", curve)
    write_csv(OUT / "long_control_teachers.csv", long_teacher_rows)
    write_csv(OUT / "long_control_summary.csv", long_summary_rows)
    write_csv(OUT / "long_control_changes_teachers.csv", long_change_teacher_rows)
    write_csv(OUT / "long_control_changes_summary.csv", long_change_summary_rows)
    write_long_control_report(long_summary_rows, long_change_summary_rows)
    write_json(OUT / "long_control_summary.json", long_metadata)
    write_csv(OUT / "primary_metrics.csv", [
        {"metric": key, "teacher_count": len(SEEDS),
         **{f"ci95_{field}": value for field, value in data["ci95"].items()},
         **{f"ci97_5_bonferroni_{field}": value
            for field, value in data["ci97_5_bonferroni"].items()}}
        for key, data in summary["primary"].items()])
    plot_figures(cells, differences, curve)
    write_report(table_a, teacher_rows, summary)
    summary["spec_sha256"] = sha(spec_path)
    summary["analysis_source_sha256"] = sha(Path(__file__))
    summary["truth_sha256"] = {str(seed): truth_hashes[seed] for seed in SEEDS}
    summary["secondary_long_control"] = "long_control_summary.json"
    summary["output_sha256"] = {path.name: sha(path) for path in OUT.iterdir()
                                if path.is_file() and path.name != "summary.json"}
    write_json(OUT / "summary.json", summary)
    print(f"Validated {len(manifest)} fits and wrote {OUT}", flush=True)
    print("H:", summary["primary"]["H"]["ci95"], flush=True)
    print("A:", summary["primary"]["A"]["ci95"], flush=True)
    print("Secondary 300-trajectory long control: LONG_CONTROL_RESULTS.md", flush=True)


def self_test_long_control() -> None:
    """Exercise teacher aggregation, pairing, intervals, and grid rejection."""
    teacher_ids = {seed: seed - SEEDS[0] for seed in SEEDS}
    truths = {seed: np.zeros((2, 3), dtype=np.float64) for seed in SEEDS}
    fit_risks = {}
    for seed in SEEDS:
        teacher_index = seed - SEEDS[0]
        for noise in LONG_NOISE:
            for repeat in LONG_REPEATS:
                repeat_offset = 0.01 * (repeat - 2)
                base = 1.0 + 0.1 * teacher_index + 0.05 * noise + repeat_offset
                at_step = {
                    1000: {"LoRA": base + 0.5, "Uni-LoRA": base + 0.3,
                           "ProLoSA": base + 0.1 - 0.01 * teacher_index},
                    3000: {"LoRA": base + 0.25, "Uni-LoRA": base + 0.15,
                           "ProLoSA": base + 0.02 - 0.02 * teacher_index},
                }
                for step, method_risks in at_step.items():
                    for method, target_risk in method_risks.items():
                        prediction = np.full((2, 3), math.sqrt(target_risk), dtype=np.float64)
                        fit_risks[seed, method, noise, repeat, step] = float(
                            np.square(prediction - truths[seed]).mean())
    teacher_rows, summary_rows, change_teacher_rows, change_rows, metadata = (
        build_long_control_statistics(teacher_ids, fit_risks))
    assert len(teacher_rows) == 40 and len(summary_rows) == 4
    assert len(change_teacher_rows) == 20 and len(change_rows) == 2
    assert metadata["extended_trajectories"] == 300
    first = next(row for row in summary_rows if row["noise_multiple"] == 1.0
                 and row["step"] == 1000)
    last = next(row for row in summary_rows if row["noise_multiple"] == 1.0
                and row["step"] == 3000)
    change = next(row for row in change_rows if row["noise_multiple"] == 1.0)
    np.testing.assert_allclose(first["lora_risk_mean"], 2.0)
    np.testing.assert_allclose(first["G_uni_minus_pro_mean"], 0.245)
    np.testing.assert_allclose(last["G_uni_minus_pro_mean"], 0.22)
    np.testing.assert_allclose(change["G_uni_minus_pro_3000_minus_1000_mean"], -0.025)
    for row in teacher_rows:
        np.testing.assert_allclose(row["G_uni_minus_pro"],
                                   row["uni_lora_risk"] - row["prolosa_risk"])
    expected_low = 0.245 - student_t.ppf(0.975, df=9) * (
        np.std([0.2 + 0.01 * index for index in range(10)], ddof=1) / math.sqrt(10))
    np.testing.assert_allclose(first["G_uni_minus_pro_ci95_low"], expected_low)
    try:
        missing = fit_risks.copy()
        del missing[100, "LoRA", 1.0, 0, 3000]
        build_long_control_statistics(teacher_ids, missing)
    except ValueError:
        pass
    else:
        raise AssertionError("Missing long-control checkpoint was accepted")
    print("Long-control synthetic prediction self-test passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true",
                        help="verify secondary analysis with synthetic predictions")
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test_long_control()
    else:
        main()

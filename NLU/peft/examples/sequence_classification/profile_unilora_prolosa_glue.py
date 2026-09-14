#!/usr/bin/env python
# coding: utf-8

"""Profile LoRA/Uni-LoRA/ProLoSA GLUE runs without modifying the training script.

This runner launches ``run_unilora_variants_glue.py`` as a subprocess, records
wall-clock time, samples GPU memory with ``nvidia-smi``, and writes a compact
JSON profile that can be summarized into the reviewer-facing efficiency table.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


TASKS = ("cola", "mrpc", "sst2", "qnli", "rte", "stsb")
METHOD_TO_VARIANT = {
    "lora": "lora",
    "unilora": "unilora",
    "prolosa": "unilora_rosa_snip",
    "lora_rosa": "lora_rosa",
    "lora_rosa_snip": "lora_rosa_snip",
    "lora_rosa_random": "lora_rosa_random",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    run_parser = subparsers.add_parser("run", help="Run and profile one training job.")
    run_parser.add_argument("--train_script", type=str, default="run_unilora_variants_glue.py")
    run_parser.add_argument("--method", choices=sorted(METHOD_TO_VARIANT), required=True)
    run_parser.add_argument("--model_name", choices=["roberta-base", "roberta-large"], default="roberta-large")
    run_parser.add_argument("--task", choices=TASKS, required=True)
    run_parser.add_argument("--head_lr", type=str, required=True)
    run_parser.add_argument("--seed", type=int, required=True)
    run_parser.add_argument("--out_dir", type=str, required=True)
    run_parser.add_argument("--batch_size", type=int, default=32)
    run_parser.add_argument("--rank", type=int, default=4)
    run_parser.add_argument("--theta_d_length", type=int, default=23040)
    run_parser.add_argument("--theta_d_lr", type=str, default="5e-3")
    run_parser.add_argument("--init_theta_d_bound", type=str, default="0.02")
    run_parser.add_argument("--unilora_dropout", type=str, default=None)
    run_parser.add_argument("--num_epochs", type=int, default=None)
    run_parser.add_argument("--warmup_ratio", type=str, default=None)
    run_parser.add_argument("--scheduler_type", choices=["linear", "cosine"], default=None)
    run_parser.add_argument("--weight_decay", type=str, default=None)

    run_parser.add_argument("--sparse_budget", type=int, default=0)
    run_parser.add_argument("--total_sparse_positions", type=int, default=None)
    run_parser.add_argument("--rosa_density", type=str, default=None)
    run_parser.add_argument("--rosa_warmup_steps", type=int, default=128)
    run_parser.add_argument("--rosa_mask_steps", type=int, default=1)
    run_parser.add_argument("--rosa_sparse_lr", type=str, default=None)
    run_parser.add_argument("--rosa_reset_optimizer_on_mask", action="store_true")
    run_parser.add_argument("--rosa_decay_sparse_lr_after_activation", action="store_true")

    run_parser.add_argument("--gpu_setup", type=str, default=os.environ.get("GPU_SETUP", "1xNVIDIA_L20"))
    run_parser.add_argument("--gpu_count", type=int, default=int(os.environ.get("GPU_COUNT", "1")))
    run_parser.add_argument(
        "--gpu_query_ids",
        type=str,
        default="auto",
        help="Comma-separated IDs for nvidia-smi fallback device-memory polling; 'auto' uses CUDA_VISIBLE_DEVICES.",
    )
    run_parser.add_argument("--memory_sample_interval", type=float, default=1.0)
    run_parser.add_argument("--profile_json", type=str, default=None)
    run_parser.add_argument("--train_log", type=str, default=None)
    run_parser.add_argument("--skip_existing_profile", action="store_true")
    run_parser.add_argument(
        "--extra_train_args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional arguments passed to the underlying training script after '--extra_train_args'.",
    )

    summarize_parser = subparsers.add_parser("summarize", help="Summarize profile JSON files.")
    summarize_parser.add_argument("--input_root", type=str, required=True)
    summarize_parser.add_argument("--output_csv", type=str, required=True)
    summarize_parser.add_argument("--output_md", type=str, required=True)

    return parser.parse_args()


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", value)


def default_profile_path(args: argparse.Namespace) -> Path:
    lr = sanitize_filename(str(args.head_lr))
    name = f"profile_{args.method}_{args.task}_{args.model_name}_lr{lr}_seed{args.seed}.json"
    return Path(args.out_dir) / name


def default_train_log_path(args: argparse.Namespace) -> Path:
    lr = sanitize_filename(str(args.head_lr))
    return Path(args.out_dir) / f"profile_train_{args.method}_{args.task}_lr{lr}_seed{args.seed}.log"


def parse_int_token(value: str) -> int | None:
    match = re.search(r"(\d+)", value)
    if not match:
        return None
    return int(match.group(1))


class GpuMemoryMonitor:
    """Poll nvidia-smi for process memory, with device-memory fallback."""

    def __init__(self, pid: int, interval: float, gpu_query_ids: list[str]) -> None:
        self.pid = pid
        self.interval = max(0.2, float(interval))
        self.gpu_query_ids = gpu_query_ids
        self.peak_process_memory_mb = 0
        self.peak_device_memory_mb = 0
        self.process_samples = 0
        self.device_samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.interval + 2.0)

    @property
    def peak_memory_mb(self) -> int:
        if self.process_samples > 0:
            return self.peak_process_memory_mb
        return self.peak_device_memory_mb

    @property
    def method(self) -> str:
        if self.process_samples > 0:
            return "nvidia-smi compute-apps process memory"
        if self.device_samples > 0:
            return "nvidia-smi device memory fallback"
        return "unavailable"

    def _run(self) -> None:
        while not self._stop.is_set():
            process_memory = self._query_process_memory()
            if process_memory is not None:
                self.process_samples += 1
                self.peak_process_memory_mb = max(self.peak_process_memory_mb, process_memory)

            device_memory = self._query_device_memory()
            if device_memory is not None:
                self.device_samples += 1
                self.peak_device_memory_mb = max(self.peak_device_memory_mb, device_memory)

            self._stop.wait(self.interval)

    def _query_process_memory(self) -> int | None:
        cmd = [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ]
        try:
            output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=5)
        except (subprocess.SubprocessError, FileNotFoundError):
            return None

        total = 0
        matched = False
        for line in output.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[0])
            except ValueError:
                continue
            if pid != self.pid:
                continue
            memory_mb = parse_int_token(parts[1])
            if memory_mb is None:
                continue
            total += memory_mb
            matched = True
        return total if matched else None

    def _query_device_memory(self) -> int | None:
        cmd = [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ]
        if self.gpu_query_ids:
            cmd.extend(["-i", ",".join(self.gpu_query_ids)])
        try:
            output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=5)
        except (subprocess.SubprocessError, FileNotFoundError):
            return None

        values = [parse_int_token(line) for line in output.splitlines()]
        values = [value for value in values if value is not None]
        if not values:
            return None
        return max(values)


def resolve_gpu_query_ids(value: str) -> list[str]:
    if value != "auto":
        return [item.strip() for item in value.split(",") if item.strip()]

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible not in {"NoDevFiles", "-1"}:
        return [item.strip() for item in visible.split(",") if item.strip()]
    return ["0"]


def build_train_command(args: argparse.Namespace) -> list[str]:
    variant = METHOD_TO_VARIANT[args.method]
    cmd = [
        sys.executable,
        "-u",
        args.train_script,
        "--variant",
        variant,
        "--model_name",
        args.model_name,
        "--task",
        args.task,
        "--batch_size",
        str(args.batch_size),
        "--rank",
        str(args.rank),
        "--theta_d_length",
        str(args.theta_d_length),
        "--theta_d_lr",
        str(args.theta_d_lr),
        "--init_theta_d_bound",
        str(args.init_theta_d_bound),
        "--head_lr",
        str(args.head_lr),
        "--seed",
        str(args.seed),
        "--out_dir",
        args.out_dir,
    ]

    optional_pairs = [
        ("--num_epochs", args.num_epochs),
        ("--warmup_ratio", args.warmup_ratio),
        ("--scheduler_type", args.scheduler_type),
        ("--weight_decay", args.weight_decay),
        ("--unilora_dropout", args.unilora_dropout),
    ]
    for flag, value in optional_pairs:
        if value is not None:
            cmd.extend([flag, str(value)])

    if args.method in {"prolosa", "lora_rosa", "lora_rosa_snip", "lora_rosa_random"}:
        if args.rosa_density is None:
            raise ValueError(f"{args.method} profiling requires --rosa_density.")
        cmd.extend(
            [
                "--rosa_density",
                str(args.rosa_density),
                "--rosa_sparse_budget",
                str(args.sparse_budget),
                "--rosa_warmup_steps",
                str(args.rosa_warmup_steps),
                "--rosa_mask_steps",
                str(args.rosa_mask_steps),
            ]
        )
        if args.rosa_sparse_lr is not None:
            cmd.extend(["--rosa_sparse_lr", str(args.rosa_sparse_lr)])
        if args.rosa_reset_optimizer_on_mask:
            cmd.append("--rosa_reset_optimizer_on_mask")
        if args.rosa_decay_sparse_lr_after_activation:
            cmd.append("--rosa_decay_sparse_lr_after_activation")

    cmd.extend(args.extra_train_args or [])
    return cmd


def compute_buffer_notes(args: argparse.Namespace) -> dict[str, Any]:
    total_sparse_positions = args.total_sparse_positions
    sparse_budget = int(args.sparse_budget or 0)
    uses_sparse_branch = args.method in {"prolosa", "lora_rosa", "lora_rosa_snip", "lora_rosa_random"}
    if args.method == "lora" and total_sparse_positions is not None:
        adapter_reported_params = int(total_sparse_positions)
    else:
        adapter_reported_params = int(args.theta_d_length) + (sparse_budget if uses_sparse_branch else 0)

    notes = {
        "adapter_reported_params": adapter_reported_params,
        "theta_d_length": int(args.theta_d_length),
        "sparse_budget": sparse_budget if uses_sparse_branch else 0,
        "uses_dense_base_gradient_buffer": False,
        "full_space_gradient_buffer_note": (
            "No dense base-weight gradient buffer is required. ProLoSA captures gradients on the "
            "low-rank A/B adapter tensors via autograd hooks and writes saliency into a flat "
            "adapter-offset score buffer; base model weights remain frozen."
        ),
    }

    if args.method == "lora":
        notes.update(
            {
                "theta_d_length": 0,
                "full_space_gradient_buffer_note": (
                    "Standard LoRA trains only low-rank A/B adapter tensors; base model weights remain frozen."
                ),
            }
        )
    elif args.method == "prolosa" and total_sparse_positions is not None:
        notes.update(
            {
                "adapter_offset_score_buffer_positions": int(total_sparse_positions),
                "adapter_offset_score_buffer_bytes_fp32": int(total_sparse_positions) * 4,
                "adapter_offset_mask_bytes_bool": int(total_sparse_positions),
            }
        )
    elif args.method in {"lora_rosa", "lora_rosa_snip", "lora_rosa_random"} and total_sparse_positions is not None:
        notes.update(
            {
                "base_weight_sparse_candidate_positions": int(total_sparse_positions),
                "base_weight_sparse_budget": sparse_budget,
                "full_space_gradient_buffer_note": (
                    "LoRA-RoSA collects sparse scores on frozen base-weight coordinates during the configured "
                    "warmup/mask window, then trains only the selected sparse value vector plus LoRA A/B factors."
                ),
            }
        )
    return notes


def is_complete_profile(path: Path) -> bool:
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        profile.get("status") == "ok"
        and profile.get("return_code") == 0
        and profile.get("best_score") is not None
    )


def run_profile(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_path = Path(args.profile_json) if args.profile_json else default_profile_path(args)
    train_log_path = Path(args.train_log) if args.train_log else default_train_log_path(args)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    train_log_path.parent.mkdir(parents=True, exist_ok=True)

    if args.skip_existing_profile and is_complete_profile(profile_path):
        print(f"Skip existing profile: {profile_path}")
        return 0

    cmd = build_train_command(args)
    mask_preloaded = "--rosa_mask_load_path" in (args.extra_train_args or [])
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    print("=" * 80)
    print(f"Profiling method={args.method} task={args.task} model={args.model_name} seed={args.seed}")
    print(f"GPU setup: {args.gpu_setup}")
    print(f"Train log: {train_log_path}")
    print(f"Profile JSON: {profile_path}")
    print("Command:")
    print(" ".join(cmd))
    print("=" * 80)

    start = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    monitor = GpuMemoryMonitor(
        pid=proc.pid,
        interval=args.memory_sample_interval,
        gpu_query_ids=resolve_gpu_query_ids(args.gpu_query_ids),
    )
    monitor.start()

    train_loop_start: float | None = None
    mask_activation_time: float | None = None
    best_score: float | None = None
    train_result_path: str | None = None
    best_saved_time: float | None = None

    best_score_re = re.compile(r"Best score:\s*([-+eE0-9.]+)\s*saved to\s*(.+)$")
    with train_log_path.open("w", encoding="utf-8") as log_file:
        assert proc.stdout is not None
        for line in proc.stdout:
            now = time.perf_counter()
            log_file.write(line)
            log_file.flush()
            sys.stdout.write(line)
            sys.stdout.flush()

            if train_loop_start is None and "TensorBoard logging to:" in line:
                train_loop_start = now
            if mask_activation_time is None and "sparse compensation:" in line and line.startswith("Activated "):
                mask_activation_time = now
            match = best_score_re.search(line.strip())
            if match:
                best_saved_time = now
                try:
                    best_score = float(match.group(1))
                except ValueError:
                    best_score = None
                train_result_path = match.group(2)

    return_code = proc.wait()
    end = time.perf_counter()
    monitor.stop()

    train_loop_end = best_saved_time if best_saved_time is not None else end
    train_loop_wall_clock_s = None
    if train_loop_start is not None:
        train_loop_wall_clock_s = max(0.0, train_loop_end - train_loop_start)

    warmup_selection_time_s = None
    warmup_extra_cost_pct = None
    if train_loop_start is not None and mask_activation_time is not None:
        warmup_selection_time_s = max(0.0, mask_activation_time - train_loop_start)
        if train_loop_wall_clock_s and train_loop_wall_clock_s > 0:
            warmup_extra_cost_pct = 100.0 * warmup_selection_time_s / train_loop_wall_clock_s

    profile = {
        "status": "ok" if return_code == 0 else "failed",
        "return_code": return_code,
        "method": args.method,
        "variant": METHOD_TO_VARIANT[args.method],
        "model_name": args.model_name,
        "task": args.task,
        "seed": args.seed,
        "head_lr": args.head_lr,
        "batch_size": args.batch_size,
        "rank": args.rank,
        "gpu_setup": args.gpu_setup,
        "gpu_count": args.gpu_count,
        "wall_clock_total_s": end - start,
        "train_loop_wall_clock_s": train_loop_wall_clock_s,
        "peak_gpu_memory_mb": monitor.peak_memory_mb,
        "memory_monitor_method": monitor.method,
        "memory_process_samples": monitor.process_samples,
        "memory_device_samples": monitor.device_samples,
        "warmup_selection_steps": (
            0
            if mask_preloaded
            else int(args.rosa_warmup_steps) + int(args.rosa_mask_steps)
            if args.method in {"prolosa", "lora_rosa", "lora_rosa_snip", "lora_rosa_random"}
            else 0
        ),
        "mask_preloaded": mask_preloaded,
        "warmup_selection_time_s": warmup_selection_time_s,
        "warmup_extra_cost_pct": warmup_extra_cost_pct,
        "warmup_extra_cost_definition": (
            "For sparse-branch methods, time from training-loop start to sparse-mask activation divided by measured "
            "training-loop wall-clock time. Use matched baseline vs sparse-method total times for end-to-end overhead."
        ),
        "best_score": best_score,
        "train_result_path": train_result_path,
        "train_log_path": str(train_log_path),
        "profile_json_path": str(profile_path),
        "command": cmd,
        "args": vars(args),
    }
    profile.update(compute_buffer_notes(args))

    temporary_profile_path = profile_path.with_name(profile_path.name + ".tmp")
    with temporary_profile_path.open("w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary_profile_path, profile_path)

    print(f"Profile saved to {profile_path}")
    return return_code


def load_profiles(input_root: str) -> list[dict[str, Any]]:
    profiles = []
    for path in Path(input_root).rglob("profile_*.json"):
        try:
            with path.open("r", encoding="utf-8") as f:
                profile = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if profile.get("status") != "ok":
            continue
        profiles.append(profile)
    return profiles


def mean_or_none(values: list[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return statistics.mean(clean)


def std_or_none(values: list[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if len(clean) < 2:
        return None
    return statistics.stdev(clean)


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "NA"
    if seconds >= 3600:
        return f"{seconds / 3600:.2f} h"
    if seconds >= 60:
        return f"{seconds / 60:.2f} min"
    return f"{seconds:.1f} s"


def fmt_memory(mb: float | None) -> str:
    if mb is None:
        return "NA"
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    return f"{mb:.0f} MB"


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.2f}%"


def summarize_profiles(args: argparse.Namespace) -> int:
    profiles = load_profiles(args.input_root)
    if not profiles:
        print(f"No successful profile JSON files found under {args.input_root}", file=sys.stderr)
        return 1

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for profile in profiles:
        key = (profile["method"], profile["model_name"], profile["task"])
        groups.setdefault(key, []).append(profile)

    rows = []
    for (method, model_name, task), items in sorted(groups.items()):
        seeds = sorted({int(item["seed"]) for item in items})
        rows.append(
            {
                "method": method,
                "model_name": model_name,
                "task": task,
                "num_runs": len(items),
                "seeds": " ".join(str(seed) for seed in seeds),
                "params": int(round(mean_or_none([item.get("adapter_reported_params") for item in items]) or 0)),
                "wall_clock_total_mean_s": mean_or_none([item.get("wall_clock_total_s") for item in items]),
                "wall_clock_total_std_s": std_or_none([item.get("wall_clock_total_s") for item in items]),
                "train_loop_mean_s": mean_or_none([item.get("train_loop_wall_clock_s") for item in items]),
                "peak_gpu_memory_mean_mb": mean_or_none([item.get("peak_gpu_memory_mb") for item in items]),
                "peak_gpu_memory_std_mb": std_or_none([item.get("peak_gpu_memory_mb") for item in items]),
                "warmup_selection_time_mean_s": mean_or_none([item.get("warmup_selection_time_s") for item in items]),
                "warmup_extra_cost_pct_mean": mean_or_none([item.get("warmup_extra_cost_pct") for item in items]),
                "gpu_setup": items[0].get("gpu_setup", "NA"),
                "uses_dense_base_gradient_buffer": items[0].get("uses_dense_base_gradient_buffer", False),
                "buffer_note": items[0].get("full_space_gradient_buffer_note", ""),
            }
        )

    csv_path = Path(args.output_csv)
    md_path = Path(args.output_md)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# LoRA / Uni-LoRA / ProLoSA Efficiency Profiling\n\n")
        f.write(
            "| Method | Model | Task | Params | Time | Peak Memory | Extra Warmup Cost | GPU setup |\n"
        )
        f.write("|---|---|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            f.write(
                "| {method} | {model_name} | {task} | {params} | {time} | {memory} | {warmup} | {gpu} |\n".format(
                    method=row["method"],
                    model_name=row["model_name"],
                    task=row["task"],
                    params=row["params"],
                    time=fmt_duration(row["wall_clock_total_mean_s"]),
                    memory=fmt_memory(row["peak_gpu_memory_mean_mb"]),
                    warmup=fmt_pct(row["warmup_extra_cost_pct_mean"]),
                    gpu=row["gpu_setup"],
                )
            )

        f.write("\n")
        f.write("Notes:\n\n")
        f.write(
            "- Time is end-to-end subprocess wall-clock time, including model/data loading, training, evaluation, and save.\n"
        )
        f.write(
            "- Extra Warmup Cost is reported for sparse methods as time from training-loop start to sparse-mask activation divided by training-loop wall-clock time; LoRA and Uni-LoRA have no support-selection warmup.\n"
        )
        f.write(
            "- ProLoSA does not require dense base-weight gradient buffers: saliency is collected from low-rank A/B adapter tensor hooks into an adapter-offset score buffer while base weights stay frozen.\n"
        )

    print(f"Wrote CSV summary to {csv_path}")
    print(f"Wrote Markdown summary to {md_path}")
    return 0


def main() -> int:
    args = parse_args()
    if args.mode == "run":
        return run_profile(args)
    if args.mode == "summarize":
        return summarize_profiles(args)
    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())

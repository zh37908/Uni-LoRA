#!/usr/bin/env python
# coding: utf-8

"""Profile LLaMA2-7B instruction tuning for Uni-LoRA and ProLoSA.

The script is intentionally an outer runner: it does not modify qlora_unilora.py.
It measures wall-clock time, peak GPU memory, GPU utilization, and ProLoSA
support-selection warmup time by streaming the underlying training log.
"""

from __future__ import print_function

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


METHOD_TO_VARIANT = {
    "unilora": "unilora",
    "prolosa": "unilora_rosa_snip",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode")
    subparsers.required = True

    run_parser = subparsers.add_parser("run", help="Run and profile one instruction-tuning job.")
    run_parser.add_argument("--train_script", default="qlora_unilora.py")
    run_parser.add_argument("--method", choices=sorted(METHOD_TO_VARIANT), required=True)
    run_parser.add_argument("--model_name_or_path", default="meta-llama/Llama-2-7b-hf")
    run_parser.add_argument("--output_dir", required=True)
    run_parser.add_argument("--profile_json", default=None)
    run_parser.add_argument("--train_log", default=None)
    run_parser.add_argument("--seed", type=int, default=0)

    run_parser.add_argument("--lora_r", type=int, default=4)
    run_parser.add_argument("--num_vectors", type=int, default=2048)
    run_parser.add_argument("--theta_d_length", type=int, default=507904)
    run_parser.add_argument("--total_trainable_budget", type=int, default=524288)
    run_parser.add_argument("--sparse_budget", type=int, default=16384)
    run_parser.add_argument("--theta_d_lr", default="8e-4")
    run_parser.add_argument("--init_theta_d_bound", default="0.02")
    run_parser.add_argument("--rosa_sparse_lr_mult", default="0.2")
    run_parser.add_argument("--rosa_warmup_steps", type=int, default=128)
    run_parser.add_argument("--rosa_mask_steps", type=int, default=1)
    run_parser.add_argument("--rosa_reset_optimizer_on_mask", default="True")
    run_parser.add_argument("--rosa_decay_sparse_lr_after_activation", default="True")

    run_parser.add_argument("--dataset", default="alpaca-clean")
    run_parser.add_argument("--source_max_len", type=int, default=16)
    run_parser.add_argument("--target_max_len", type=int, default=512)
    run_parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    run_parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    run_parser.add_argument("--num_train_epochs", default="1")
    run_parser.add_argument("--max_train_samples", type=int, default=None)
    run_parser.add_argument("--learning_rate", default="0")
    run_parser.add_argument("--warmup_ratio", default="0.1")
    run_parser.add_argument("--lr_scheduler_type", default="linear")
    run_parser.add_argument("--lora_dropout", default="0.05")
    run_parser.add_argument("--weight_decay", default="0.0")
    run_parser.add_argument("--adam_beta2", default="0.999")
    run_parser.add_argument("--max_grad_norm", default="0.3")
    run_parser.add_argument("--logging_steps", type=int, default=20)
    run_parser.add_argument("--dataloader_num_workers", type=int, default=1)
    run_parser.add_argument("--max_memory_MB", type=int, default=80000)

    run_parser.add_argument("--gpu_setup", default=os.environ.get("GPU_SETUP", "1xNVIDIA L20"))
    run_parser.add_argument("--gpu_count", type=int, default=int(os.environ.get("GPU_COUNT", "1")))
    run_parser.add_argument("--gpu_query_ids", default="auto")
    run_parser.add_argument("--memory_sample_interval", type=float, default=1.0)
    run_parser.add_argument("--skip_existing_profile", action="store_true")
    run_parser.add_argument("--extra_train_args", nargs=argparse.REMAINDER, default=[])

    summarize_parser = subparsers.add_parser("summarize", help="Summarize profile JSON files.")
    summarize_parser.add_argument("--input_root", required=True)
    summarize_parser.add_argument("--output_csv", required=True)
    summarize_parser.add_argument("--output_md", required=True)

    return parser.parse_args()


def sanitize_filename(value):
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(value))


def default_profile_path(args):
    name = "profile_llama2_7b_{method}_seed{seed}.json".format(method=args.method, seed=args.seed)
    return Path(args.output_dir) / name


def default_train_log_path(args):
    name = "profile_train_llama2_7b_{method}_seed{seed}.log".format(method=args.method, seed=args.seed)
    return Path(args.output_dir) / name


def parse_int_token(value):
    match = re.search(r"(\d+)", str(value))
    if match is None:
        return None
    return int(match.group(1))


class GpuMonitor(object):
    def __init__(self, pid, interval, gpu_query_ids):
        self.pid = int(pid)
        self.interval = max(0.2, float(interval))
        self.gpu_query_ids = gpu_query_ids
        self.peak_process_memory_mb = 0
        self.peak_device_memory_mb = 0
        self.peak_gpu_util_pct = 0
        self.peak_mem_util_pct = 0
        self.gpu_util_samples = []
        self.mem_util_samples = []
        self.process_samples = 0
        self.device_samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run)
        self._thread.daemon = True

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=self.interval + 2.0)

    @property
    def peak_memory_mb(self):
        if self.process_samples > 0:
            return self.peak_process_memory_mb
        return self.peak_device_memory_mb

    @property
    def memory_method(self):
        if self.process_samples > 0:
            return "nvidia-smi compute-apps process memory"
        if self.device_samples > 0:
            return "nvidia-smi device memory fallback"
        return "unavailable"

    @property
    def avg_gpu_util_pct(self):
        if not self.gpu_util_samples:
            return None
        return statistics.mean(self.gpu_util_samples)

    @property
    def avg_mem_util_pct(self):
        if not self.mem_util_samples:
            return None
        return statistics.mean(self.mem_util_samples)

    def _run(self):
        while not self._stop.is_set():
            process_memory = self._query_process_memory()
            if process_memory is not None:
                self.process_samples += 1
                self.peak_process_memory_mb = max(self.peak_process_memory_mb, process_memory)

            device_stats = self._query_device_stats()
            if device_stats is not None:
                used_memory, gpu_util, mem_util = device_stats
                self.device_samples += 1
                self.peak_device_memory_mb = max(self.peak_device_memory_mb, used_memory)
                self.peak_gpu_util_pct = max(self.peak_gpu_util_pct, gpu_util)
                self.peak_mem_util_pct = max(self.peak_mem_util_pct, mem_util)
                self.gpu_util_samples.append(gpu_util)
                self.mem_util_samples.append(mem_util)

            self._stop.wait(self.interval)

    def _query_process_memory(self):
        cmd = [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ]
        try:
            output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=5)
        except (subprocess.SubprocessError, OSError):
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

    def _query_device_stats(self):
        cmd = [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu,utilization.memory",
            "--format=csv,noheader,nounits",
        ]
        if self.gpu_query_ids:
            cmd.extend(["-i", ",".join(self.gpu_query_ids)])
        try:
            output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=5)
        except (subprocess.SubprocessError, OSError):
            return None

        rows = []
        for line in output.splitlines():
            parts = [parse_int_token(part) for part in line.split(",")]
            if len(parts) < 3 or any(part is None for part in parts[:3]):
                continue
            rows.append((parts[0], parts[1], parts[2]))
        if not rows:
            return None
        return max(rows, key=lambda item: item[0])


def resolve_gpu_query_ids(value):
    if value != "auto":
        return [item.strip() for item in value.split(",") if item.strip()]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible not in {"NoDevFiles", "-1"}:
        return [item.strip() for item in visible.split(",") if item.strip()]
    return ["0"]


def str_bool(value):
    return "True" if str(value).lower() in {"1", "true", "yes", "y"} else "False"


def build_train_command(args):
    variant = METHOD_TO_VARIANT[args.method]
    cmd = [
        sys.executable,
        "-u",
        args.train_script,
        "--model_name_or_path",
        args.model_name_or_path,
        "--use_auth_token",
        "True",
        "--output_dir",
        args.output_dir,
        "--logging_steps",
        str(args.logging_steps),
        "--save_strategy",
        "no",
        "--data_seed",
        "42",
        "--save_steps",
        "500",
        "--save_total_limit",
        "40",
        "--evaluation_strategy",
        "no",
        "--max_new_tokens",
        "32",
        "--dataloader_num_workers",
        str(args.dataloader_num_workers),
        "--group_by_length",
        "--logging_strategy",
        "steps",
        "--remove_unused_columns",
        "False",
        "--do_train",
        "--unilora_variant",
        variant,
        "--lora_r",
        str(args.lora_r),
        "--lora_modules",
        "all",
        "--double_quant",
        "--quant_type",
        "nf4",
        "--bf16",
        "--bits",
        "4",
        "--warmup_ratio",
        str(args.warmup_ratio),
        "--lr_scheduler_type",
        args.lr_scheduler_type,
        "--gradient_checkpointing",
        "--dataset",
        args.dataset,
        "--source_max_len",
        str(args.source_max_len),
        "--target_max_len",
        str(args.target_max_len),
        "--per_device_train_batch_size",
        str(args.per_device_train_batch_size),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--num_train_epochs",
        str(args.num_train_epochs),
        "--learning_rate",
        str(args.learning_rate),
        "--learning_rate_vector_bank",
        str(args.theta_d_lr),
        "--adam_beta2",
        str(args.adam_beta2),
        "--max_grad_norm",
        str(args.max_grad_norm),
        "--lora_dropout",
        str(args.lora_dropout),
        "--weight_decay",
        str(args.weight_decay),
        "--seed",
        str(args.seed),
        "--max_memory_MB",
        str(args.max_memory_MB),
    ]

    if args.max_train_samples is not None:
        cmd.extend(["--max_train_samples", str(args.max_train_samples)])

    if args.method == "unilora":
        cmd.extend(["--num_vectors", str(args.num_vectors)])
    else:
        cmd.extend(
            [
                "--theta_d_length",
                str(args.theta_d_length),
                "--init_theta_d_bound",
                str(args.init_theta_d_bound),
                "--rosa_sparse_budget",
                str(args.sparse_budget),
                "--rosa_warmup_steps",
                str(args.rosa_warmup_steps),
                "--rosa_mask_steps",
                str(args.rosa_mask_steps),
                "--rosa_sparse_lr_mult",
                str(args.rosa_sparse_lr_mult),
                "--rosa_reset_optimizer_on_mask",
                str_bool(args.rosa_reset_optimizer_on_mask),
                "--rosa_decay_sparse_lr_after_activation",
                str_bool(args.rosa_decay_sparse_lr_after_activation),
                "--learning_rate_theta_d",
                str(args.theta_d_lr),
            ]
        )

    cmd.extend(args.extra_train_args or [])
    return cmd


def method_params(args):
    if args.method == "unilora":
        return int(args.num_vectors) * 256
    return int(args.theta_d_length) + int(args.sparse_budget)


def buffer_notes(args):
    if args.method != "prolosa":
        return {
            "uses_dense_base_gradient_buffer": False,
            "full_space_gradient_buffer_note": "Uni-LoRA does not run support selection.",
        }
    return {
        "uses_dense_base_gradient_buffer": False,
        "full_space_gradient_buffer_note": (
            "ProLoSA does not store dense base-model gradients. The implementation captures "
            "low-rank A/B adapter gradients with autograd hooks and accumulates SNIP saliency "
            "in an adapter-offset buffer; the quantized base model remains frozen."
        ),
    }


def read_metrics(output_dir):
    metrics_path = Path(output_dir) / "metrics.json"
    if not metrics_path.exists():
        return {}
    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def run_profile(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = Path(args.profile_json) if args.profile_json else default_profile_path(args)
    train_log_path = Path(args.train_log) if args.train_log else default_train_log_path(args)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    train_log_path.parent.mkdir(parents=True, exist_ok=True)

    if args.skip_existing_profile and profile_path.exists() and profile_path.stat().st_size > 0:
        print("Skip existing profile: {}".format(profile_path))
        return 0

    cmd = build_train_command(args)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    print("=" * 80)
    print("Profiling LLaMA2-7B method={} seed={}".format(args.method, args.seed))
    print("GPU setup: {}".format(args.gpu_setup))
    print("Output dir: {}".format(output_dir))
    print("Train log: {}".format(train_log_path))
    print("Profile JSON: {}".format(profile_path))
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

    monitor = GpuMonitor(
        pid=proc.pid,
        interval=args.memory_sample_interval,
        gpu_query_ids=resolve_gpu_query_ids(args.gpu_query_ids),
    )
    monitor.start()

    train_loop_start = None
    mask_activation_time = None
    train_end_time = None
    train_begin_patterns = (
        "***** Running training *****",
        "Num examples =",
        "trainable params:",
    )

    with train_log_path.open("w", encoding="utf-8") as log_file:
        for line in proc.stdout:
            now = time.perf_counter()
            log_file.write(line)
            log_file.flush()
            sys.stdout.write(line)
            sys.stdout.flush()

            if train_loop_start is None and any(pattern in line for pattern in train_begin_patterns):
                train_loop_start = now
            if mask_activation_time is None and "Activated UniLoRA-RoSA-SNIP sparse compensation:" in line:
                mask_activation_time = now
            if "train_runtime" in line or "Saving PEFT checkpoint" in line:
                train_end_time = now

    return_code = proc.wait()
    end = time.perf_counter()
    monitor.stop()

    if train_loop_start is None:
        train_loop_start = start
    if train_end_time is None:
        train_end_time = end
    train_loop_wall_clock_s = max(0.0, train_end_time - train_loop_start)

    warmup_selection_time_s = None
    warmup_extra_cost_pct = None
    if mask_activation_time is not None:
        warmup_selection_time_s = max(0.0, mask_activation_time - train_loop_start)
        if train_loop_wall_clock_s > 0:
            warmup_extra_cost_pct = 100.0 * warmup_selection_time_s / train_loop_wall_clock_s

    metrics = read_metrics(output_dir)
    profile = {
        "status": "ok" if return_code == 0 else "failed",
        "return_code": return_code,
        "method": args.method,
        "variant": METHOD_TO_VARIANT[args.method],
        "model_name_or_path": args.model_name_or_path,
        "model_size": "llama2-7b",
        "dataset": args.dataset,
        "seed": args.seed,
        "adapter_reported_params": method_params(args),
        "total_trainable_budget": args.total_trainable_budget,
        "wall_clock_total_s": end - start,
        "train_loop_wall_clock_s": train_loop_wall_clock_s,
        "hf_train_runtime_s": metrics.get("train_runtime"),
        "hf_train_samples_per_second": metrics.get("train_samples_per_second"),
        "hf_train_steps_per_second": metrics.get("train_steps_per_second"),
        "train_loss": metrics.get("train_loss"),
        "peak_gpu_memory_mb": monitor.peak_memory_mb,
        "memory_monitor_method": monitor.memory_method,
        "peak_gpu_util_pct": monitor.peak_gpu_util_pct,
        "avg_gpu_util_pct": monitor.avg_gpu_util_pct,
        "peak_mem_util_pct": monitor.peak_mem_util_pct,
        "avg_mem_util_pct": monitor.avg_mem_util_pct,
        "memory_process_samples": monitor.process_samples,
        "memory_device_samples": monitor.device_samples,
        "warmup_selection_steps": (
            int(args.rosa_warmup_steps) + int(args.rosa_mask_steps) if args.method == "prolosa" else 0
        ),
        "warmup_selection_time_s": warmup_selection_time_s,
        "warmup_extra_cost_pct": warmup_extra_cost_pct,
        "warmup_extra_cost_definition": (
            "For ProLoSA, time from training-loop start to sparse-mask activation divided by "
            "measured training-loop wall-clock time."
        ),
        "gpu_setup": args.gpu_setup,
        "gpu_count": args.gpu_count,
        "output_dir": str(output_dir),
        "train_log_path": str(train_log_path),
        "profile_json_path": str(profile_path),
        "command": cmd,
        "args": vars(args),
        "train_metrics": metrics,
    }
    profile.update(buffer_notes(args))

    with profile_path.open("w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, sort_keys=True)

    print("Profile saved to {}".format(profile_path))
    return return_code


def load_profiles(input_root):
    profiles = []
    for path in Path(input_root).rglob("profile_llama2_7b_*.json"):
        try:
            with path.open("r", encoding="utf-8") as f:
                profile = json.load(f)
        except (OSError, ValueError):
            continue
        if profile.get("status") != "ok":
            continue
        profiles.append(profile)
    return profiles


def mean_or_none(values):
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return statistics.mean(clean)


def std_or_none(values):
    clean = [float(value) for value in values if value is not None]
    if len(clean) < 2:
        return None
    return statistics.stdev(clean)


def fmt_duration(seconds):
    if seconds is None:
        return "NA"
    seconds = float(seconds)
    if seconds >= 3600:
        return "{:.2f} h".format(seconds / 3600.0)
    if seconds >= 60:
        return "{:.2f} min".format(seconds / 60.0)
    return "{:.1f} s".format(seconds)


def fmt_memory(mb):
    if mb is None:
        return "NA"
    mb = float(mb)
    if mb >= 1024:
        return "{:.2f} GB".format(mb / 1024.0)
    return "{:.0f} MB".format(mb)


def fmt_pct(value):
    if value is None:
        return "NA"
    return "{:.2f}%".format(float(value))


def summarize_profiles(args):
    profiles = load_profiles(args.input_root)
    if not profiles:
        print("No successful profile JSON files found under {}".format(args.input_root), file=sys.stderr)
        return 1

    groups = {}
    for profile in profiles:
        key = (profile.get("method"), profile.get("model_size"), profile.get("dataset"))
        groups.setdefault(key, []).append(profile)

    rows = []
    for (method, model_size, dataset), items in sorted(groups.items()):
        seeds = sorted({int(item.get("seed", 0)) for item in items})
        rows.append(
            {
                "method": method,
                "model_size": model_size,
                "dataset": dataset,
                "num_runs": len(items),
                "seeds": " ".join(str(seed) for seed in seeds),
                "params": int(round(mean_or_none([item.get("adapter_reported_params") for item in items]) or 0)),
                "wall_clock_total_mean_s": mean_or_none([item.get("wall_clock_total_s") for item in items]),
                "wall_clock_total_std_s": std_or_none([item.get("wall_clock_total_s") for item in items]),
                "train_loop_mean_s": mean_or_none([item.get("train_loop_wall_clock_s") for item in items]),
                "hf_train_runtime_mean_s": mean_or_none([item.get("hf_train_runtime_s") for item in items]),
                "train_loss_mean": mean_or_none([item.get("train_loss") for item in items]),
                "peak_gpu_memory_mean_mb": mean_or_none([item.get("peak_gpu_memory_mb") for item in items]),
                "peak_gpu_memory_std_mb": std_or_none([item.get("peak_gpu_memory_mb") for item in items]),
                "avg_gpu_util_pct_mean": mean_or_none([item.get("avg_gpu_util_pct") for item in items]),
                "peak_gpu_util_pct_mean": mean_or_none([item.get("peak_gpu_util_pct") for item in items]),
                "avg_mem_util_pct_mean": mean_or_none([item.get("avg_mem_util_pct") for item in items]),
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
        f.write("# LLaMA2-7B Instruction Tuning Efficiency Profiling\n\n")
        f.write(
            "| Method | Model | Dataset | Params | Time | Peak Memory | Avg GPU Util | Extra Warmup Cost | GPU setup |\n"
        )
        f.write("|---|---|---|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            f.write(
                "| {method} | {model} | {dataset} | {params} | {time} | {memory} | {gpu_util} | {warmup} | {gpu} |\n".format(
                    method=row["method"],
                    model=row["model_size"],
                    dataset=row["dataset"],
                    params=row["params"],
                    time=fmt_duration(row["wall_clock_total_mean_s"]),
                    memory=fmt_memory(row["peak_gpu_memory_mean_mb"]),
                    gpu_util=fmt_pct(row["avg_gpu_util_pct_mean"]),
                    warmup=fmt_pct(row["warmup_extra_cost_pct_mean"]),
                    gpu=row["gpu_setup"],
                )
            )

        f.write("\n")
        f.write("Notes:\n\n")
        f.write("- Time is end-to-end subprocess wall-clock time, including loading, training, and save.\n")
        f.write("- Peak Memory is sampled via `nvidia-smi`; process memory is used when available, otherwise device memory is used as a fallback.\n")
        f.write("- Avg GPU Util is sampled from `nvidia-smi --query-gpu=utilization.gpu` on the assigned GPU.\n")
        f.write("- ProLoSA Extra Warmup Cost is time from training-loop start to sparse-mask activation divided by training-loop wall-clock time.\n")
        f.write("- ProLoSA does not require dense base-weight gradient buffers; support scores are accumulated from low-rank adapter-gradient hooks.\n")

    print("Wrote CSV summary to {}".format(csv_path))
    print("Wrote Markdown summary to {}".format(md_path))
    return 0


def main():
    args = parse_args()
    if args.mode == "run":
        return run_profile(args)
    if args.mode == "summarize":
        return summarize_profiles(args)
    raise ValueError("Unsupported mode: {}".format(args.mode))


if __name__ == "__main__":
    raise SystemExit(main())

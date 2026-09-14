#!/usr/bin/env python
"""Short-run time and GPU-memory profiler for math LoRA, UniLoRA, and ProLoSA."""

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path


METHOD_TO_VARIANT = {
    "lora": "lora",
    "unilora": "unilora",
    "prolosa": "unilora_rosa_snip_multi_gpu",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--method", choices=sorted(METHOD_TO_VARIANT), required=True)
    run.add_argument("--train_script", default="intruction_tuning_unilora_multi_gpu.py")
    run.add_argument("--model_name_or_path", default="google/gemma-7b")
    run.add_argument("--data_path", default="meta-math/MetaMathQA")
    run.add_argument("--dataset_field", nargs=2, default=["query", "response"])
    run.add_argument("--output_dir", required=True)
    run.add_argument("--profile_json", required=True)
    run.add_argument("--train_log", required=True)
    run.add_argument("--seed", type=int, default=42)

    run.add_argument("--full_dataset_samples", type=int, default=100000)
    run.add_argument("--full_num_train_epochs", type=float, default=2.0)
    run.add_argument("--profile_max_steps", type=int, default=None)
    run.add_argument(
        "--profile_wall_time_seconds",
        type=float,
        default=None,
        help=(
            "Stop after this many seconds of actual training. Model loading and "
            "tokenization are excluded. A timed stop is treated as a successful profile."
        ),
    )
    run.add_argument("--post_activation_steps", type=int, default=10)
    run.add_argument("--timing_skip_steps", type=int, default=2)
    run.add_argument("--per_device_train_batch_size", type=int, default=1)
    run.add_argument("--gradient_accumulation_steps", type=int, default=64)
    run.add_argument("--model_max_length", type=int, default=512)
    run.add_argument("--preprocessing_num_workers", type=int, default=1)

    run.add_argument("--lora_r", type=int, default=4)
    run.add_argument("--num_vectors", type=int, default=2048)
    run.add_argument("--vector_length", type=int, default=524288)
    run.add_argument("--theta_d_length", type=int, default=507904)
    run.add_argument("--sparse_budget", type=int, default=16384)
    run.add_argument("--rosa_warmup_steps", type=int, default=128)
    run.add_argument("--rosa_mask_steps", type=int, default=1)
    run.add_argument("--learning_rate", default=None)
    run.add_argument("--theta_d_lr", default="8e-4")
    run.add_argument("--rosa_sparse_lr_mult", default="0.2")

    run.add_argument("--max_memory_per_gpu", default="44GiB")
    run.add_argument("--max_memory_cpu", default="128GiB")
    run.add_argument("--gpu_query_ids", default="auto")
    run.add_argument("--memory_sample_interval", type=float, default=0.5)
    run.add_argument("--extra_train_args", nargs=argparse.REMAINDER, default=[])

    summary = subparsers.add_parser("summarize")
    summary.add_argument("--profiles", nargs="+", required=True)
    summary.add_argument("--output_json", required=True)
    summary.add_argument("--output_md", required=True)
    return parser.parse_args()


def parse_number(value):
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
    return float(match.group(0)) if match else None


def resolve_gpu_ids(value):
    if value != "auto":
        return [item.strip() for item in value.split(",") if item.strip()]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible not in {"-1", "NoDevFiles"}:
        return [item.strip() for item in visible.split(",") if item.strip()]
    return []


class GpuMonitor:
    """Sample per-device memory; intended for an exclusive Slurm allocation."""

    def __init__(self, interval, gpu_ids):
        self.interval = max(0.2, interval)
        self.gpu_ids = gpu_ids
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=self.interval + 5)

    def _run(self):
        while not self._stop.is_set():
            cmd = [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
            if self.gpu_ids:
                cmd.extend(["-i", ",".join(self.gpu_ids)])
            try:
                output = subprocess.check_output(
                    cmd, stderr=subprocess.DEVNULL, text=True, timeout=5
                )
                devices = {}
                for line in output.splitlines():
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) < 3:
                        continue
                    memory = parse_number(parts[1])
                    utilization = parse_number(parts[2])
                    if memory is not None:
                        devices[parts[0]] = {
                            "memory_mb": memory,
                            "utilization_pct": utilization,
                        }
                if devices:
                    self.samples.append({"time": time.perf_counter(), "devices": devices})
            except (OSError, subprocess.SubprocessError):
                pass
            self._stop.wait(self.interval)

    def stats(self, start=None, end=None):
        selected = [
            sample
            for sample in self.samples
            if (start is None or sample["time"] >= start)
            and (end is None or sample["time"] <= end)
        ]
        per_gpu_peak = {}
        aggregate_memory = []
        utilization = []
        for sample in selected:
            aggregate_memory.append(
                sum(values["memory_mb"] for values in sample["devices"].values())
            )
            for gpu_id, values in sample["devices"].items():
                per_gpu_peak[gpu_id] = max(
                    per_gpu_peak.get(gpu_id, 0.0), values["memory_mb"]
                )
                if values["utilization_pct"] is not None:
                    utilization.append(values["utilization_pct"])
        return {
            "samples": len(selected),
            "per_gpu_peak_memory_mb": per_gpu_peak,
            "max_peak_memory_mb": max(per_gpu_peak.values()) if per_gpu_peak else None,
            "peak_total_memory_mb": max(aggregate_memory) if aggregate_memory else None,
            "sum_per_gpu_peaks_mb": sum(per_gpu_peak.values()) if per_gpu_peak else None,
            "mean_gpu_utilization_pct": (
                statistics.mean(utilization) if utilization else None
            ),
        }


def total_optimizer_steps(args):
    samples_per_update = (
        args.per_device_train_batch_size * args.gradient_accumulation_steps
    )
    updates_per_epoch = math.ceil(args.full_dataset_samples / samples_per_update)
    return math.ceil(updates_per_epoch * args.full_num_train_epochs)


def profile_steps(args):
    if args.profile_max_steps is not None:
        required = 3
        if args.method == "prolosa":
            required = (
                args.rosa_warmup_steps
                + args.rosa_mask_steps
                + args.post_activation_steps
            )
        if args.profile_max_steps < required:
            raise ValueError(
                "profile_max_steps={} is too small for {} (need at least {}).".format(
                    args.profile_max_steps, args.method, required
                )
            )
        return args.profile_max_steps
    if args.method == "prolosa":
        return (
            args.rosa_warmup_steps
            + args.rosa_mask_steps
            + args.post_activation_steps
        )
    return max(10, args.post_activation_steps)


def build_train_command(args, short_steps):
    profile_samples = min(
        args.full_dataset_samples,
        short_steps
        * args.per_device_train_batch_size
        * args.gradient_accumulation_steps,
    )
    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = "2e-3" if args.method == "unilora" else "2e-4"

    cmd = [
        sys.executable,
        "-u",
        args.train_script,
        "--model_name_or_path",
        args.model_name_or_path,
        "--output_dir",
        args.output_dir,
        "--unilora_variant",
        METHOD_TO_VARIANT[args.method],
        "--lora_r",
        str(args.lora_r),
        "--data_path",
        args.data_path,
        "--dataset_split",
        "train[:{}]".format(profile_samples),
        "--dataset_field",
        *args.dataset_field,
        "--max_steps",
        str(short_steps),
        "--num_train_epochs",
        str(args.full_num_train_epochs),
        "--per_device_train_batch_size",
        str(args.per_device_train_batch_size),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--model_max_length",
        str(args.model_max_length),
        "--preprocessing_num_workers",
        str(args.preprocessing_num_workers),
        "--gradient_checkpointing",
        "True",
        "--save_strategy",
        "no",
        "--logging_steps",
        "1",
        "--learning_rate",
        str(learning_rate),
        "--weight_decay",
        "0",
        "--warmup_ratio",
        "0.02",
        "--lr_scheduler_type",
        "cosine",
        "--bf16",
        "False",
        "--tf32",
        "False",
        "--fp16",
        "True",
        "--device_map",
        "auto",
        "--max_memory_per_gpu",
        args.max_memory_per_gpu,
        "--max_memory_cpu",
        args.max_memory_cpu,
        "--report_to",
        "none",
        "--seed",
        str(args.seed),
    ]
    if args.method == "unilora":
        cmd.extend(
            [
                "--num_vectors",
                str(args.num_vectors),
                "--vector_length",
                str(args.vector_length),
                "--save_only_topk_weights",
                "True",
            ]
        )
    elif args.method == "prolosa":
        cmd.extend(
            [
                "--theta_d_length",
                str(args.theta_d_length),
                "--init_theta_d_bound",
                "0.02",
                "--rosa_sparse_budget",
                str(args.sparse_budget),
                "--rosa_warmup_steps",
                str(args.rosa_warmup_steps),
                "--rosa_mask_steps",
                str(args.rosa_mask_steps),
                "--rosa_sparse_lr_mult",
                str(args.rosa_sparse_lr_mult),
                "--rosa_reset_optimizer_on_mask",
                "True",
                "--rosa_decay_sparse_lr_after_activation",
                "True",
                "--learning_rate_vector_bank",
                str(args.theta_d_lr),
                "--learning_rate_theta_d",
                str(args.theta_d_lr),
            ]
        )
    cmd.extend(args.extra_train_args)
    return cmd


def median_interval(timestamps, skip_intervals=0):
    intervals = [
        right - left for left, right in zip(timestamps[:-1], timestamps[1:])
    ]
    intervals = intervals[skip_intervals:]
    return statistics.median(intervals) if intervals else None


def event_time_for_step(step_events, step):
    if 1 <= step <= len(step_events):
        return step_events[step - 1]
    return None


def run_profile(args):
    full_steps = total_optimizer_steps(args)
    short_steps = full_steps if args.profile_wall_time_seconds else profile_steps(args)
    if not args.profile_wall_time_seconds and full_steps <= short_steps:
        raise ValueError("Full run must contain more steps than the profile run.")

    output_dir = Path(args.output_dir)
    profile_path = Path(args.profile_json)
    train_log_path = Path(args.train_log)
    for path in (output_dir, profile_path.parent, train_log_path.parent):
        path.mkdir(parents=True, exist_ok=True)

    cmd = build_train_command(args, short_steps)
    print("Method: {}".format(args.method))
    print("Short/full optimizer steps: {}/{}".format(short_steps, full_steps))
    print("Command: {}".format(" ".join(cmd)))

    process_start = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    monitor = GpuMonitor(
        args.memory_sample_interval, resolve_gpu_ids(args.gpu_query_ids)
    )
    monitor.start()

    train_start = None
    activation_time = None
    step_events = []
    timed_stop = threading.Event()
    stop_timer = None

    def stop_timed_profile():
        if proc.poll() is None:
            timed_stop.set()
            proc.terminate()

    train_begin_patterns = (
        "***** Running training *****",
        "Num examples =",
        # The local training scripts print this immediately before trainer.train().
        # Transformers may suppress its usual INFO-level training banner.
        "trainable params:",
    )

    def mark_training_started(now, trigger):
        nonlocal train_start, stop_timer
        if train_start is not None:
            return
        train_start = now
        if args.profile_wall_time_seconds:
            print(
                "Timed profile started ({:.0f}s, trigger: {}).".format(
                    args.profile_wall_time_seconds, trigger
                )
            )
            stop_timer = threading.Timer(
                args.profile_wall_time_seconds, stop_timed_profile
            )
            stop_timer.daemon = True
            stop_timer.start()

    with train_log_path.open("w", encoding="utf-8") as log_file:
        for line in proc.stdout:
            now = time.perf_counter()
            log_file.write(line)
            log_file.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
            matched_marker = next(
                (pattern for pattern in train_begin_patterns if pattern in line), None
            )
            if matched_marker is not None:
                mark_training_started(now, matched_marker)
            is_step_event = "'loss':" in line or '"loss":' in line
            if is_step_event:
                # Keep timed profiling functional for custom training scripts that
                # emit neither the Transformers banner nor trainable-parameter output.
                mark_training_started(now, "first optimizer-step loss")
                step_events.append(now)
            if (
                activation_time is None
                and "Activated UniLoRA-RoSA-SNIP sparse compensation:" in line
            ):
                activation_time = now

    return_code = proc.wait()
    if stop_timer is not None:
        stop_timer.cancel()
    process_end = time.perf_counter()
    monitor.stop()

    if len(step_events) < 2:
        raise RuntimeError(
            "Only {} optimizer-step log events detected; see {}.".format(
                len(step_events), train_log_path
            )
        )
    generic_step_s = median_interval(step_events, args.timing_skip_steps)
    if train_start is None:
        train_start = step_events[0] - (generic_step_s or 0.0)
    train_profile_time_s = step_events[-1] - train_start

    warmup = {
        "configured_warmup_steps": 0,
        "mask_collection_steps": 0,
        "warmup_selection_steps": 0,
        "warmup_only_time_s": None,
        "mask_collection_time_s": None,
        "warmup_selection_time_s": None,
    }
    if args.method == "prolosa":
        activation_step = args.rosa_warmup_steps + args.rosa_mask_steps
        if activation_time is not None:
            warmup_end = event_time_for_step(step_events, args.rosa_warmup_steps)
            warmup_selection_time_s = activation_time - train_start
            warmup.update(
                {
                    "completed_during_profile": True,
                    "observed_warmup_steps": min(len(step_events), activation_step),
                    "configured_warmup_steps": args.rosa_warmup_steps,
                    "mask_collection_steps": args.rosa_mask_steps,
                    "warmup_selection_steps": activation_step,
                    "warmup_only_time_s": (
                        warmup_end - train_start if warmup_end is not None else None
                    ),
                    "mask_collection_time_s": (
                        activation_time - warmup_end if warmup_end is not None else None
                    ),
                    "warmup_selection_time_s": warmup_selection_time_s,
                    "time_is_estimated": False,
                }
            )
            post_events = [stamp for stamp in step_events if stamp > activation_time]
            stable_step_s = median_interval(post_events, 1)
            if stable_step_s is None:
                raise RuntimeError("Not enough post-activation steps to estimate ProLoSA.")
            estimated_train_time_s = warmup_selection_time_s + stable_step_s * (
                full_steps - activation_step
            )
            estimate_method = (
                "measured full structural warmup+mask time, then extrapolated only "
                "post-activation stable steps"
            )
        else:
            if not timed_stop.is_set():
                raise RuntimeError(
                    "ProLoSA sparse activation was not observed before training ended."
                )
            stable_step_s = generic_step_s
            estimated_warmup_s = stable_step_s * args.rosa_warmup_steps
            estimated_mask_s = stable_step_s * args.rosa_mask_steps
            warmup.update(
                {
                    "completed_during_profile": False,
                    "observed_warmup_steps": len(step_events),
                    "configured_warmup_steps": args.rosa_warmup_steps,
                    "mask_collection_steps": args.rosa_mask_steps,
                    "warmup_selection_steps": activation_step,
                    "warmup_only_time_s": estimated_warmup_s,
                    "mask_collection_time_s": estimated_mask_s,
                    "warmup_selection_time_s": estimated_warmup_s
                    + estimated_mask_s,
                    "time_is_estimated": True,
                }
            )
            estimated_train_time_s = stable_step_s * full_steps
            estimate_method = (
                "20-minute warmup prefix; full warmup and training time extrapolated "
                "from the observed median warmup step time; post-activation behavior "
                "was not observed"
            )
    else:
        stable_step_s = generic_step_s
        estimated_train_time_s = train_profile_time_s + stable_step_s * (
            full_steps - len(step_events)
        )
        estimate_method = (
            "measured short training prefix, then extrapolated stable step time"
        )

    whole_process_memory = monitor.stats(process_start, process_end)
    training_memory = monitor.stats(train_start, step_events[-1])
    profiled_gpu_count = len(training_memory["per_gpu_peak_memory_mb"])
    profile = {
        "status": "ok" if return_code == 0 or timed_stop.is_set() else "failed",
        "return_code": return_code,
        "timed_profile_stop": timed_stop.is_set(),
        "profile_wall_time_seconds": args.profile_wall_time_seconds,
        "method": args.method,
        "variant": METHOD_TO_VARIANT[args.method],
        "model_name_or_path": args.model_name_or_path,
        "full_dataset_samples": args.full_dataset_samples,
        "full_num_train_epochs": args.full_num_train_epochs,
        "full_optimizer_steps": full_steps,
        "profile_optimizer_steps_requested": short_steps,
        "profile_optimizer_steps_observed": len(step_events),
        "stable_step_time_s": stable_step_s,
        "measured_short_train_time_s": train_profile_time_s,
        "estimated_full_train_time_s": estimated_train_time_s,
        "estimated_full_train_hours": estimated_train_time_s / 3600.0,
        "profiled_gpu_count": profiled_gpu_count,
        "estimated_full_gpu_hours": (
            estimated_train_time_s * profiled_gpu_count / 3600.0
        ),
        "estimate_method": estimate_method,
        "process_wall_time_s": process_end - process_start,
        "setup_before_train_time_s": train_start - process_start,
        "warmup": warmup,
        "gpu_memory": {
            "measurement": (
                "nvidia-smi device memory on the exclusive allocation; "
                "peak_total_memory_mb is the largest simultaneous sum across "
                "all profiled GPUs; max_peak_memory_mb is the largest single-GPU peak"
            ),
            "whole_process": whole_process_memory,
            "training": training_memory,
            "warmup_and_mask": (
                monitor.stats(train_start, activation_time)
                if activation_time is not None
                else (
                    monitor.stats(train_start, step_events[-1])
                    if args.method == "prolosa"
                    else None
                )
            ),
            "post_activation": (
                monitor.stats(activation_time, step_events[-1])
                if activation_time is not None
                else None
            ),
        },
        "train_log": str(train_log_path),
        "command": cmd,
        "args": vars(args),
    }
    with profile_path.open("w", encoding="utf-8") as file:
        json.dump(profile, file, indent=2, sort_keys=True)
    print("Profile written to {}".format(profile_path))
    return 0 if timed_stop.is_set() else return_code


def format_seconds(value):
    if value is None:
        return "N/A"
    if value >= 3600:
        return "{:.2f} h".format(value / 3600)
    if value >= 60:
        return "{:.2f} min".format(value / 60)
    return "{:.2f} s".format(value)


def summarize(args):
    profiles = []
    for path in args.profiles:
        with Path(path).open("r", encoding="utf-8") as file:
            profiles.append(json.load(file))
    if any(profile.get("status") != "ok" for profile in profiles):
        raise RuntimeError("At least one profile run failed.")

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as file:
        json.dump({"profiles": profiles}, file, indent=2, sort_keys=True)

    with output_md.open("w", encoding="utf-8") as file:
        file.write("# Math instruction-tuning efficiency estimate\n\n")
        for profile in profiles:
            training_memory = profile["gpu_memory"]["training"]
            total_memory = training_memory.get(
                "peak_total_memory_mb", training_memory.get("sum_per_gpu_peaks_mb")
            )
            single_gpu_memory = training_memory["max_peak_memory_mb"]
            per_gpu_memory = training_memory["per_gpu_peak_memory_mb"]
            gpu_count = profile.get("profiled_gpu_count", len(per_gpu_memory))
            gpu_hours = profile.get(
                "estimated_full_gpu_hours",
                profile["estimated_full_train_time_s"] * gpu_count / 3600.0,
            )
            file.write("## {}\n\n".format(profile["method"]))
            file.write("- Variant: {}\n".format(profile["variant"]))
            file.write("- Profiled GPUs: {} (model parallel)\n".format(gpu_count))
            file.write(
                "- Short run: {} / {} optimizer steps\n".format(
                    profile["profile_optimizer_steps_observed"],
                    profile["full_optimizer_steps"],
                )
            )
            file.write(
                "- Measured training window: {}\n".format(
                    format_seconds(profile["measured_short_train_time_s"])
                )
            )
            file.write(
                "- Setup before training (load + tokenize): {}\n".format(
                    format_seconds(profile["setup_before_train_time_s"])
                )
            )
            file.write(
                "- Total profiler wall time: {}\n".format(
                    format_seconds(profile["process_wall_time_s"])
                )
            )
            file.write(
                "- Estimated full training time: {}\n".format(
                    format_seconds(profile["estimated_full_train_time_s"])
                )
            )
            file.write("- Estimated full GPU-hours: {:.2f}\n".format(gpu_hours))
            file.write(
                "- Stable optimizer-step time: {}\n".format(
                    format_seconds(profile["stable_step_time_s"])
                )
            )
            file.write(
                "- Peak total GPU memory (simultaneous across all GPUs): {} MiB\n".format(
                    "N/A"
                    if total_memory is None
                    else "{:.0f}".format(total_memory)
                )
            )
            file.write(
                "- Peak memory on largest single GPU: {} MiB\n".format(
                    "N/A"
                    if single_gpu_memory is None
                    else "{:.0f}".format(single_gpu_memory)
                )
            )
            file.write(
                "- Per-GPU peak memory: {}\n".format(
                    ", ".join(
                        "GPU {}: {:.0f} MiB".format(gpu_id, value)
                        for gpu_id, value in sorted(per_gpu_memory.items())
                    )
                    or "N/A"
                )
            )
            file.write("- Estimator: {}\n".format(profile["estimate_method"]))
            if profile["method"] == "prolosa":
                warmup = profile["warmup"]
                warmup_stats = profile["gpu_memory"]["warmup_and_mask"]
                post_stats = profile["gpu_memory"]["post_activation"]
                warmup_memory = (
                    warmup_stats.get(
                        "peak_total_memory_mb",
                        warmup_stats.get("sum_per_gpu_peaks_mb"),
                    )
                    if warmup_stats is not None
                    else None
                )
                post_memory = (
                    post_stats.get(
                        "peak_total_memory_mb",
                        post_stats.get("sum_per_gpu_peaks_mb"),
                    )
                    if post_stats is not None
                    else None
                )
                file.write(
                    "- Warmup completed during profile: {}\n".format(
                        "yes" if warmup.get("completed_during_profile") else "no"
                    )
                )
                file.write(
                    "- Warmup steps observed: {} / {}\n".format(
                        warmup.get("observed_warmup_steps", 0),
                        warmup["warmup_selection_steps"],
                    )
                )
                file.write(
                    "- Structural warmup: {} steps, {}{}\n".format(
                        warmup["configured_warmup_steps"],
                        format_seconds(warmup["warmup_only_time_s"]),
                        " (estimated)" if warmup.get("time_is_estimated") else "",
                    )
                )
                file.write(
                    "- Mask collection: {} steps, {}{}\n".format(
                        warmup["mask_collection_steps"],
                        format_seconds(warmup["mask_collection_time_s"]),
                        " (estimated)" if warmup.get("time_is_estimated") else "",
                    )
                )
                file.write(
                    "- Warmup + mask selection: {} steps, {}{}\n".format(
                        warmup["warmup_selection_steps"],
                        format_seconds(warmup["warmup_selection_time_s"]),
                        " (estimated)" if warmup.get("time_is_estimated") else "",
                    )
                )
                file.write(
                    "- Peak total memory warmup / post-activation: {} / {} MiB\n".format(
                        "N/A"
                        if warmup_memory is None
                        else "{:.0f}".format(warmup_memory),
                        "N/A"
                        if post_memory is None
                        else "{:.0f}".format(post_memory),
                    )
                )
            file.write("\n")
        file.write(
            "Training-time estimates exclude model loading, tokenization, and final save. "
            "Wall-clock time already reflects model parallelism; GPU-hours multiply it by "
            "the number of profiled GPUs. GPU memory is sampled with nvidia-smi and assumes "
            "exclusive use of the allocated GPUs.\n"
        )
    print("Summary written to {} and {}".format(output_json, output_md))
    return 0


def main():
    args = parse_args()
    if args.mode == "run":
        return run_profile(args)
    return summarize(args)


if __name__ == "__main__":
    raise SystemExit(main())

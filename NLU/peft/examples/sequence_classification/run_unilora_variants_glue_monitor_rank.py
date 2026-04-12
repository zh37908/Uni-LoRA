#!/usr/bin/env python
# coding: utf-8

"""
Monitor adapter rank dynamics during GLUE training.

This script tracks two low-rank spectrum metrics on the effective adapter update
matrix DeltaW for each monitored module:
  - stable rank: ||DeltaW||_F^2 / ||DeltaW||_2^2
  - entropy effective rank: exp(H(p)), where p_i = sigma_i / sum_j sigma_j

For LoRA/UniLoRA-style methods we avoid doing a full SVD on the large DeltaW
matrix and instead recover the non-zero singular values from the small low-rank
core implied by the A/B factors.
"""

import argparse
import json
import os

import torch

import run_unilora_variants_glue as base
import run_unilora_variants_glue_monitor_gradient as monitor_gradient


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=["roberta-base", "roberta-large"])
    parser.add_argument("--task", type=str, required=True, choices=base.GLUE_TASKS)
    parser.add_argument(
        "--variant",
        type=str,
        default="unilora",
        choices=[
            "lora",
            "unilora",
            "unilora_count_sketch",
            "unilora_sign",
            "unilora_nonorm",
            "unilora_fastfood",
            "unilora_gs",
            "unilora_block_routing",
            "unilora_stage_ratio",
            "unilora_learnable",
            "unilora_learnable_column",
            "unilora_isometric_control",
            "direct_unilora",
            "unilora_layer_wise",
            "unilora_learnable_layer",
        ],
    )
    parser.add_argument("--isometry_alpha", type=float, default=0.0, help="Control parameter for unilora_isometric_control.")
    parser.add_argument("--head_lr", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out_dir", "--output_dir", dest="out_dir", type=str, default="results_variants_monitor_rank")
    parser.add_argument("--batch_size", type=int, default=32, help="Per-device batch size for both train and eval dataloaders.")

    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--theta_d_length", type=int, default=23040)
    parser.add_argument("--v", type=int, default=3, help="Number of Count-Sketch rows (num_sketches).")
    parser.add_argument("--init_theta_d_bound", type=float, default=0.02)
    parser.add_argument("--theta_d_lr", type=float, default=5e-3)
    parser.add_argument("--alpha_lr", type=float, default=None, help="LR for unilora_layer_alpha_* parameters; defaults to theta_d_lr / 50.")
    parser.add_argument("--alpha_freeze_ratio", type=float, default=0.1, help="Fraction of total steps to freeze alpha params at the start.")
    parser.add_argument("--alpha_init", type=float, default=1.0, help="Initial bounded alpha value for unilora_learnable_layer.")
    parser.add_argument("--alpha_min", type=float, default=0.5, help="Lower bound for alpha in unilora_learnable_layer.")
    parser.add_argument("--alpha_max", type=float, default=1.5, help="Upper bound for alpha in unilora_learnable_layer.")
    parser.add_argument("--unilora_dropout", type=float, default=0.0)
    parser.add_argument("--init_logits_std", type=float, default=0.1)
    parser.add_argument("--init_logits_bias", type=float, default=2.0)
    parser.add_argument("--gumbel_tau", type=float, default=1.0)
    parser.add_argument("--num_blocks", type=int, default=8, help="Number of blocks for unilora_block_routing")
    parser.add_argument(
        "--stage_theta_d_ratios",
        type=float,
        nargs=3,
        default=[0.2, 0.3, 0.5],
        metavar=("FRONT", "MIDDLE", "BACK"),
        help="Theta_d ratios for front/middle/back stages in unilora_stage_ratio.",
    )
    parser.add_argument("--num_epochs", type=int, default=None, help="Override default number of epochs")

    parser.add_argument("--monitor_every", type=int, default=10, help="Collect rank statistics every N optimizer steps.")
    parser.add_argument("--monitor_top_modules", type=int, default=5, help="Number of top modules to store per metric in JSONL records.")
    parser.add_argument(
        "--monitor_scope",
        type=str,
        default="adapter",
        choices=["adapter"],
        help="Rank monitor currently tracks effective adapter delta matrices.",
    )
    return parser.parse_args()


def _get_active_adapters(module):
    if hasattr(module, "active_adapters"):
        return list(module.active_adapters)
    if hasattr(module, "_active_adapter"):
        return [module._active_adapter]
    return []


def _extract_factor_matrices(module, adapter):
    if hasattr(module, "_get_lora_matrices"):
        try:
            A, B = module._get_lora_matrices(adapter, cast_to_fp32=True)
            return A.detach().float().cpu(), B.detach().float().cpu()
        except TypeError:
            A, B = module._get_lora_matrices(adapter)
            return A.detach().float().cpu(), B.detach().float().cpu()
        except Exception:
            return None

    if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
        if adapter in getattr(module, "lora_A", {}) and adapter in getattr(module, "lora_B", {}):
            A = module.lora_A[adapter].weight.detach().float().cpu()
            B = module.lora_B[adapter].weight.detach().float().cpu()
            return A, B

    if hasattr(module, "lora_embedding_A") and hasattr(module, "lora_embedding_B"):
        if adapter in getattr(module, "lora_embedding_A", {}) and adapter in getattr(module, "lora_embedding_B", {}):
            A = module.lora_embedding_A[adapter].detach().float().cpu()
            B = module.lora_embedding_B[adapter].detach().float().cpu()
            return A, B

    return None


def _nonzero_singular_values_from_factors(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    if A.numel() == 0 or B.numel() == 0:
        return torch.empty(0, dtype=torch.float32)

    # DeltaW = B @ A, where A is r x in_features and B is out_features x r.
    qb, rb = torch.linalg.qr(B, mode="reduced")
    qa, ra = torch.linalg.qr(A.t(), mode="reduced")
    del qb, qa
    core = rb @ ra.t()
    singular_values = torch.linalg.svdvals(core).clamp_min(0.0)
    return singular_values


def _rank_stats_from_singular_values(singular_values: torch.Tensor) -> dict[str, float]:
    eps = 1e-12
    if singular_values.numel() == 0:
        return {
            "fro_norm": 0.0,
            "spectral_norm": 0.0,
            "stable_rank": 0.0,
            "entropy_effective_rank": 0.0,
            "rank_upper_bound": 0.0,
        }

    fro_sq = torch.sum(singular_values.square())
    spectral_sq = singular_values[0].square() if singular_values.numel() > 0 else torch.tensor(0.0)
    if fro_sq.item() <= eps or spectral_sq.item() <= eps:
        return {
            "fro_norm": float(torch.sqrt(fro_sq).item()),
            "spectral_norm": float(torch.sqrt(spectral_sq).item()),
            "stable_rank": 0.0,
            "entropy_effective_rank": 0.0,
            "rank_upper_bound": float((singular_values > eps).sum().item()),
        }

    probs = singular_values / singular_values.sum().clamp_min(eps)
    probs = probs[probs > eps]
    entropy = -(probs * torch.log(probs)).sum()
    return {
        "fro_norm": float(torch.sqrt(fro_sq).item()),
        "spectral_norm": float(torch.sqrt(spectral_sq).item()),
        "stable_rank": float((fro_sq / spectral_sq.clamp_min(eps)).item()),
        "entropy_effective_rank": float(torch.exp(entropy).item()),
        "rank_upper_bound": float((singular_values > eps).sum().item()),
    }


def collect_rank_stats(model):
    per_module = {}
    for name, module in model.named_modules():
        adapters = _get_active_adapters(module)
        if not adapters:
            continue
        for adapter in adapters:
            factors = _extract_factor_matrices(module, adapter)
            if factors is None:
                continue
            A, B = factors
            try:
                singular_values = _nonzero_singular_values_from_factors(A, B)
            except RuntimeError:
                continue
            stats = _rank_stats_from_singular_values(singular_values)
            key = f"{name}/{adapter}" if adapter is not None else name
            per_module[key] = stats
    return per_module


def summarize_rank_stats(per_module: dict[str, dict[str, float]], topk: int):
    if not per_module:
        return None

    stable_vals = [info["stable_rank"] for info in per_module.values()]
    entropy_vals = [info["entropy_effective_rank"] for info in per_module.values()]
    fro_weights = [max(info["fro_norm"] ** 2, 0.0) for info in per_module.values()]
    total_weight = sum(fro_weights)

    if total_weight > 0.0:
        stable_weighted = sum(v * w for v, w in zip(stable_vals, fro_weights)) / total_weight
        entropy_weighted = sum(v * w for v, w in zip(entropy_vals, fro_weights)) / total_weight
    else:
        stable_weighted = 0.0
        entropy_weighted = 0.0

    sorted_stable = sorted(per_module.items(), key=lambda item: item[1]["stable_rank"], reverse=True)
    sorted_entropy = sorted(per_module.items(), key=lambda item: item[1]["entropy_effective_rank"], reverse=True)
    topk = max(0, int(topk))

    return {
        "module_count": len(per_module),
        "stable_rank_mean": float(sum(stable_vals) / len(stable_vals)),
        "stable_rank_max": float(max(stable_vals)),
        "stable_rank_min": float(min(stable_vals)),
        "stable_rank_weighted_mean": float(stable_weighted),
        "entropy_effective_rank_mean": float(sum(entropy_vals) / len(entropy_vals)),
        "entropy_effective_rank_max": float(max(entropy_vals)),
        "entropy_effective_rank_min": float(min(entropy_vals)),
        "entropy_effective_rank_weighted_mean": float(entropy_weighted),
        "top_stable_rank_modules": [
            {"module": name, "stable_rank": info["stable_rank"], "entropy_effective_rank": info["entropy_effective_rank"]}
            for name, info in sorted_stable[:topk]
        ],
        "top_entropy_rank_modules": [
            {"module": name, "stable_rank": info["stable_rank"], "entropy_effective_rank": info["entropy_effective_rank"]}
            for name, info in sorted_entropy[:topk]
        ],
    }


def main():
    args = parse_args()
    base.set_seed(args.seed)

    model_name = args.model_name
    task = args.task
    variant = args.variant
    batch_size = args.batch_size
    max_length = base.MAX_LENGTH[model_name]
    num_epochs = args.num_epochs if args.num_epochs is not None else base.EPOCHS[model_name][task]
    warmup_ratio = 0.06

    device = "cuda" if base.torch.cuda.is_available() else "cpu"
    metric_name = base.TASK_TO_METRIC[task]

    current_init_bound = args.init_theta_d_bound
    theta_d_lr = args.theta_d_lr
    alpha_lr = args.alpha_lr if args.alpha_lr is not None else (theta_d_lr / 50.0)
    if variant == "unilora_nonorm":
        count, scale = base.estimate_nonorm_scale(model_name, args.rank, args.theta_d_length)
        if count is not None:
            current_init_bound = current_init_bound * scale
            theta_d_lr = theta_d_lr * scale
            print(
                f">>> Variant {variant} detected. "
                f"Estimated count={count:.2f}, scale={scale:.4f}. "
                f"Scaled init_theta_d_bound={current_init_bound:.6f}, theta_d_lr={theta_d_lr:.6f}."
            )

    tokenizer = base.AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    datasets = base.load_dataset("nyu-mll/glue", task)
    s1_key, s2_key = base.TASK_TO_KEYS[task]

    def tokenize_fn(examples):
        if s2_key is None:
            return tokenizer(examples[s1_key], truncation=True, padding="max_length", max_length=max_length)
        return tokenizer(examples[s1_key], examples[s2_key], truncation=True, padding="max_length", max_length=max_length)

    remove_cols = [col for col in ["idx", s1_key, s2_key] if col and col in datasets["train"].column_names]
    datasets = datasets.map(tokenize_fn, batched=True, remove_columns=remove_cols)
    if "label" in datasets["train"].column_names:
        datasets = datasets.rename_column("label", "labels")

    def collate_fn(examples):
        return tokenizer.pad(examples, return_tensors="pt")

    train_loader = base.DataLoader(
        datasets["train"], shuffle=True, batch_size=batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    eval_loader = base.DataLoader(
        datasets["validation"], shuffle=False, batch_size=batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=True
    )

    num_labels = 1 if task == "stsb" else 2
    base_model = base.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, return_dict=True)
    peft_config = monitor_gradient.build_peft_config(args, current_init_bound)
    model = base.get_peft_model(base_model, peft_config)
    model.to(device)

    head_params, theta_d_params, alpha_params = monitor_gradient.partition_trainable_params(model)

    print("=" * 80)
    print(f"Run Variant: {variant.upper()}")
    print(f"  model_name = {model_name} | task = {task} | seed = {args.seed}")
    print(f"  head_lr = {args.head_lr} | theta_d_lr = {theta_d_lr} | alpha_lr = {alpha_lr if alpha_params else 'N/A'}")
    print(f"  monitor_every = {args.monitor_every} | monitor_top_modules = {args.monitor_top_modules}")
    print("=" * 80)
    base.print_trainable_param_summary(
        model=model,
        variant=variant,
        theta_d_params=theta_d_params,
        alpha_params=alpha_params,
        head_params=head_params,
    )

    optimizer_groups = []
    if head_params:
        optimizer_groups.append({"params": head_params, "lr": args.head_lr, "weight_decay": 0.01})
    if theta_d_params:
        optimizer_groups.append({"params": theta_d_params, "lr": theta_d_lr, "weight_decay": 0.01})
    if alpha_params:
        optimizer_groups.append({"params": alpha_params, "lr": alpha_lr, "weight_decay": 0.0})
    optimizer = base.AdamW(optimizer_groups)

    total_steps = len(train_loader) * num_epochs
    scheduler = base.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )
    alpha_freeze_steps = int(args.alpha_freeze_ratio * total_steps) if alpha_params else 0
    if alpha_params and alpha_freeze_steps > 0:
        for param in alpha_params:
            param.requires_grad = False
        print(f"Freezing alpha parameters for first {alpha_freeze_steps}/{total_steps} steps.")

    run_stem = f"{variant}_{task}_{model_name}_lr{args.head_lr}_seed{args.seed}"
    log_dir = os.path.join(args.out_dir, "runs", run_stem)
    writer = base.SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging to: {log_dir}")

    os.makedirs(args.out_dir, exist_ok=True)
    monitor_jsonl_path = os.path.join(args.out_dir, f"{run_stem}_rank_monitor.jsonl")

    best_score = -1e18
    best_metric = None
    history = []
    global_step = 0
    rank_module_keys = None

    with open(monitor_jsonl_path, "w") as monitor_file:
        for epoch in range(num_epochs):
            model.train()
            pbar = base.tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
            epoch_loss = 0.0
            epoch_stable_ranks = []
            epoch_entropy_ranks = []

            for batch in pbar:
                if alpha_params and alpha_freeze_steps > 0 and global_step == alpha_freeze_steps:
                    for param in alpha_params:
                        param.requires_grad = True
                    print(f"Unfroze alpha parameters at step {global_step}.")

                batch = {k: v.to(device) for k, v in batch.items()}
                loss = model(**batch).loss
                loss.backward()

                optimizer.step()
                scheduler.step()

                rank_summary = None
                if args.monitor_every > 0 and global_step % args.monitor_every == 0:
                    per_module = collect_rank_stats(model)
                    if per_module:
                        if rank_module_keys is None:
                            rank_module_keys = sorted(per_module.keys())
                        rank_summary = summarize_rank_stats(per_module, args.monitor_top_modules)
                        if rank_summary is not None:
                            writer.add_scalar("Monitor/StableRankMean", rank_summary["stable_rank_mean"], global_step)
                            writer.add_scalar("Monitor/StableRankWeightedMean", rank_summary["stable_rank_weighted_mean"], global_step)
                            writer.add_scalar("Monitor/StableRankMax", rank_summary["stable_rank_max"], global_step)
                            writer.add_scalar(
                                "Monitor/EntropyEffectiveRankMean",
                                rank_summary["entropy_effective_rank_mean"],
                                global_step,
                            )
                            writer.add_scalar(
                                "Monitor/EntropyEffectiveRankWeightedMean",
                                rank_summary["entropy_effective_rank_weighted_mean"],
                                global_step,
                            )
                            writer.add_scalar(
                                "Monitor/EntropyEffectiveRankMax",
                                rank_summary["entropy_effective_rank_max"],
                                global_step,
                            )
                            writer.add_scalar("Monitor/ModuleCount", rank_summary["module_count"], global_step)

                            record = {
                                "epoch": epoch,
                                "global_step": global_step,
                                "train_loss": float(loss.item()),
                                **rank_summary,
                            }
                            monitor_file.write(json.dumps(record) + "\n")
                            epoch_stable_ranks.extend(info["stable_rank"] for info in per_module.values())
                            epoch_entropy_ranks.extend(info["entropy_effective_rank"] for info in per_module.values())

                optimizer.zero_grad()

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                writer.add_scalar("Train/Loss", loss.item(), global_step)
                epoch_loss += loss.item()
                global_step += 1

            avg_epoch_loss = epoch_loss / len(train_loader)
            writer.add_scalar("Train/Epoch_Loss", avg_epoch_loss, epoch)
            if epoch_stable_ranks:
                writer.add_histogram("Monitor/StableRankEpoch", torch.tensor(epoch_stable_ranks), epoch)
            if epoch_entropy_ranks:
                writer.add_histogram("Monitor/EntropyEffectiveRankEpoch", torch.tensor(epoch_entropy_ranks), epoch)

            model.eval()
            metric = base.evaluate.load("glue", task)
            eval_loss = 0.0
            for batch in eval_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with base.torch.no_grad():
                    outputs = model(**batch)
                    logits = outputs.logits
                    eval_loss += outputs.loss.item()
                if task == "stsb":
                    metric.add_batch(predictions=logits.squeeze(-1).cpu().numpy(), references=batch["labels"].cpu().numpy())
                else:
                    metric.add_batch(predictions=logits.argmax(dim=-1).cpu().numpy(), references=batch["labels"].cpu().numpy())

            avg_eval_loss = eval_loss / len(eval_loader)
            eval_results = metric.compute()
            score = eval_results[metric_name]
            print(f"Epoch {epoch} | train_loss: {avg_epoch_loss:.4f} | val_loss: {avg_eval_loss:.4f} | {metric_name}: {score:.4f} | {eval_results}")

            writer.add_scalar("Eval/Loss", avg_eval_loss, epoch)
            for key, value in eval_results.items():
                writer.add_scalar(f"Eval/{key}", value, epoch)

            history.append({
                "epoch": epoch,
                "train_loss": avg_epoch_loss,
                "val_loss": avg_eval_loss,
                "score": score,
                "metrics": eval_results,
            })
            if score > best_score:
                best_score = score
                best_metric = eval_results

    writer.close()

    out_path = os.path.join(args.out_dir, f"{run_stem}.json")
    with open(out_path, "w") as result_file:
        json.dump(
            {
                "variant": variant,
                "best_score": best_score,
                "best_metric": best_metric,
                "history": history,
                "rank_module_count": 0 if rank_module_keys is None else len(rank_module_keys),
                "rank_module_keys": [] if rank_module_keys is None else rank_module_keys,
                "monitor_jsonl_path": monitor_jsonl_path,
                "args": vars(args),
            },
            result_file,
            indent=2,
        )
    print(f"Best score: {best_score} saved to {out_path}")


if __name__ == "__main__":
    main()

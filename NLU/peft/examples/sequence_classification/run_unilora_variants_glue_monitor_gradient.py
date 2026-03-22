#!/usr/bin/env python
# coding: utf-8

"""
Monitor adapter gradients during GLUE training.

This script estimates gradient signal/noise statistics from a rolling window
of mini-batch gradients collected during training:
  - signal norm: ||g_bar||^2
  - noise norm: E ||g_b - g_bar||^2
  - SNR: ||g_bar||^2 / E ||g_b - g_bar||^2
  - effective rank of the centered gradient covariance

The estimate is window-based, so it is an online approximation of the
full-batch statistics described in the experiment note.
"""

import argparse
import json
import math
import os
from collections import deque

import torch

import run_unilora_variants_glue as base


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
    parser.add_argument("--out_dir", "--output_dir", dest="out_dir", type=str, default="results_variants_monitor_gradient")
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

    parser.add_argument("--monitor_every", type=int, default=10, help="Collect gradient statistics every N optimizer steps.")
    parser.add_argument("--monitor_window_size", type=int, default=32, help="Rolling window size used to estimate gradient mean/covariance.")
    parser.add_argument(
        "--monitor_scope",
        type=str,
        default="adapter",
        choices=["adapter", "all_trainable"],
        help="Whether to monitor adapter-only gradients or all trainable gradients.",
    )

    return parser.parse_args()


def build_peft_config(args, current_init_bound):
    if args.variant == "lora":
        return base.LoraConfig(
            task_type="SEQ_CLS",
            r=args.rank,
            lora_alpha=args.rank,
            lora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    if args.variant == "unilora_count_sketch":
        return base.UniLoRACountSketchConfig(
            task_type="SEQ_CLS", peft_type=base.PeftType.UNILORA_COUNT_SKETCH,
            r=args.rank, theta_d_length=args.theta_d_length,
            num_sketches=args.v,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    if args.variant == "unilora_nonorm":
        return base.UniLoRANonormConfig(
            task_type="SEQ_CLS", peft_type=base.PeftType.UNILORA_NONORM,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    if args.variant == "unilora_sign":
        return base.UniLoRASignConfig(
            task_type="SEQ_CLS", peft_type=base.PeftType.UNILORA_SIGN,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    if args.variant == "unilora_fastfood":
        return base.UniLoRAFastFoodConfig(
            task_type="SEQ_CLS", peft_type=base.PeftType.UNILORA_FASTFOOD,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    if args.variant == "unilora_gs":
        return base.UniLoRAGSConfig(
            task_type="SEQ_CLS", peft_type=base.PeftType.UNILORA_GS,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            init_logits_std=args.init_logits_std,
            init_logits_bias=args.init_logits_bias,
            gumbel_tau=args.gumbel_tau,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    if args.variant == "unilora_block_routing":
        return base.UniLoRABlockRoutingConfig(
            task_type="SEQ_CLS", peft_type=base.PeftType.UNILORA_BLOCK_ROUTING,
            r=args.rank, theta_d_length=args.theta_d_length,
            num_blocks=args.num_blocks,
            router_tau=args.gumbel_tau,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    if args.variant == "unilora_stage_ratio":
        return base.UniLoRAStageRatioConfig(
            task_type="SEQ_CLS", peft_type=base.PeftType.UNILORA_STAGE_RATIO,
            r=args.rank, theta_d_length=args.theta_d_length,
            stage_theta_d_ratios=args.stage_theta_d_ratios,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    if args.variant == "unilora_learnable":
        return base.UniLoRALearnableConfig(
            task_type="SEQ_CLS", peft_type=base.PeftType.UNILORA_LEARNABLE,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    if args.variant == "unilora_learnable_column":
        return base.UniLoRALearnableColumnConfig(
            task_type="SEQ_CLS", peft_type=base.PeftType.UNILORA_LEARNABLE_COLUMN,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    if args.variant == "unilora_isometric_control":
        return base.UniLoRAIsometricControlConfig(
            task_type="SEQ_CLS", peft_type=base.PeftType.UNILORA_ISOMETRIC_CONTROL,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            isometry_alpha=args.isometry_alpha,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    if args.variant == "direct_unilora":
        return base.DirectUniLoRAConfig(
            task_type="SEQ_CLS", peft_type=base.PeftType.DIRECT_UNILORA,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    if args.variant == "unilora_layer_wise":
        return base.UniLoRALayerWiseConfig(
            task_type="SEQ_CLS", peft_type=base.PeftType.UNILORA_LAYER_WISE,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    if args.variant == "unilora_learnable_layer":
        return base.UniLoRALearnableLayerConfig(
            task_type="SEQ_CLS", peft_type=base.PeftType.UNILORA_LEARNABLE_LAYER,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            alpha_init=args.alpha_init,
            alpha_min=args.alpha_min,
            alpha_max=args.alpha_max,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    return base.UniLoRAConfig(
        task_type="SEQ_CLS", peft_type=base.PeftType.UNILORA,
        r=args.rank, theta_d_length=args.theta_d_length,
        proj_seed=args.seed, init_theta_d_bound=current_init_bound,
        unilora_dropout=args.unilora_dropout,
        target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
        modules_to_save=["classifier"],
    )


def partition_trainable_params(model):
    head_params, theta_d_params, alpha_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            if any(term in name for term in ["theta_d", "unilora_layer_alpha"]):
                param.requires_grad = True
            else:
                continue

        if "unilora_layer_alpha" in name:
            alpha_params.append(param)
        elif name.endswith("theta_d") or "theta_d." in name:
            theta_d_params.append(param)
        else:
            head_params.append(param)
    return head_params, theta_d_params, alpha_params


def select_monitor_params(model, variant, scope):
    if scope == "all_trainable":
        selected = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    elif variant == "lora":
        selected = [
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad and any(term in name for term in ["lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"])
        ]
    else:
        selected = [
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad and any(term in name for term in ["theta_d", "unilora_scales", "unilora_layer_alpha"])
        ]

    if not selected:
        selected = [(name, param) for name, param in model.named_parameters() if param.requires_grad]

    names = [name for name, _ in selected]
    params = [param for _, param in selected]
    return params, names


def flatten_gradient(params):
    grads = [param.grad.detach().float().cpu().reshape(-1) for param in params if param.grad is not None]
    if not grads:
        return None
    return torch.cat(grads)


class RollingGradientMonitor:
    def __init__(self, window_size):
        self.buffer = deque(maxlen=window_size)

    def update(self, grad_vec):
        self.buffer.append(grad_vec)

    def stats(self):
        if len(self.buffer) < 2:
            return None

        grad_matrix = torch.stack(list(self.buffer), dim=0)
        mean_grad = grad_matrix.mean(dim=0)
        centered = grad_matrix - mean_grad

        signal_norm_sq = mean_grad.pow(2).sum()
        noise_norm_sq = centered.pow(2).sum(dim=1).mean()
        current_noise_norm_sq = centered[-1].pow(2).sum()
        current_grad_norm = grad_matrix[-1].norm()

        gram = centered @ centered.t()
        eigvals = torch.linalg.eigvalsh(gram).clamp_min(0.0)
        eig_sum = eigvals.sum()
        if eig_sum.item() > 0.0:
            probs = eigvals / eig_sum
            probs = probs[probs > 0]
            entropy = -(probs * torch.log(probs)).sum()
            effective_rank = torch.exp(entropy)
        else:
            effective_rank = torch.tensor(0.0)

        snr = signal_norm_sq / torch.clamp(noise_norm_sq, min=1e-12)
        return {
            "window_size": len(self.buffer),
            "current_grad_norm": current_grad_norm.item(),
            "signal_norm_sq": signal_norm_sq.item(),
            "noise_norm_sq": noise_norm_sq.item(),
            "current_noise_norm_sq": current_noise_norm_sq.item(),
            "snr": snr.item(),
            "effective_rank": effective_rank.item(),
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

    train_loader = base.DataLoader(datasets["train"], shuffle=True, batch_size=batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    eval_loader = base.DataLoader(datasets["validation"], shuffle=False, batch_size=batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    num_labels = 1 if task == "stsb" else 2
    base_model = base.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, return_dict=True)
    peft_config = build_peft_config(args, current_init_bound)
    model = base.get_peft_model(base_model, peft_config)
    model.to(device)

    head_params, theta_d_params, alpha_params = partition_trainable_params(model)
    monitor_params, monitor_param_names = select_monitor_params(model, variant, args.monitor_scope)

    print("=" * 80)
    print(f"Run Variant: {variant.upper()}")
    print(f"  model_name = {model_name} | task = {task} | seed = {args.seed}")
    print(f"  head_lr = {args.head_lr} | theta_d_lr = {theta_d_lr} | alpha_lr = {alpha_lr if alpha_params else 'N/A'}")
    print(f"  monitor_scope = {args.monitor_scope} | monitor_every = {args.monitor_every} | monitor_window_size = {args.monitor_window_size}")
    print("=" * 80)
    base.print_trainable_param_summary(
        model=model,
        variant=variant,
        theta_d_params=theta_d_params,
        alpha_params=alpha_params,
        head_params=head_params,
    )
    print(f"Monitoring {len(monitor_params)} parameter tensors across {sum(p.numel() for p in monitor_params)} scalars.")

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

    log_dir = os.path.join(args.out_dir, "runs", f"{variant}_{task}_{model_name}_lr{args.head_lr}_seed{args.seed}")
    writer = base.SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging to: {log_dir}")

    os.makedirs(args.out_dir, exist_ok=True)
    monitor_jsonl_path = os.path.join(
        args.out_dir,
        f"{variant}_{task}_{model_name}_lr{args.head_lr}_seed{args.seed}_gradient_monitor.jsonl",
    )

    best_score = -1e18
    best_metric = None
    history = []
    global_step = 0
    rolling_monitor = RollingGradientMonitor(args.monitor_window_size)

    with open(monitor_jsonl_path, "w") as monitor_file:
        for epoch in range(num_epochs):
            model.train()
            pbar = base.tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
            epoch_loss = 0.0
            for batch in pbar:
                if alpha_params and alpha_freeze_steps > 0 and global_step == alpha_freeze_steps:
                    for param in alpha_params:
                        param.requires_grad = True
                    print(f"Unfroze alpha parameters at step {global_step}.")

                batch = {k: v.to(device) for k, v in batch.items()}
                loss = model(**batch).loss
                loss.backward()

                if args.monitor_every > 0 and global_step % args.monitor_every == 0:
                    grad_vec = flatten_gradient(monitor_params)
                    if grad_vec is not None:
                        rolling_monitor.update(grad_vec)
                        stats = rolling_monitor.stats()
                        if stats is not None:
                            writer.add_scalar("Monitor/GradNorm", stats["current_grad_norm"], global_step)
                            writer.add_scalar("Monitor/SignalNormSq", stats["signal_norm_sq"], global_step)
                            writer.add_scalar("Monitor/NoiseNormSq", stats["noise_norm_sq"], global_step)
                            writer.add_scalar("Monitor/CurrentNoiseNormSq", stats["current_noise_norm_sq"], global_step)
                            writer.add_scalar("Monitor/SNR", stats["snr"], global_step)
                            writer.add_scalar("Monitor/EffectiveRank", stats["effective_rank"], global_step)
                            writer.add_scalar("Monitor/WindowSize", stats["window_size"], global_step)

                            record = {
                                "epoch": epoch,
                                "global_step": global_step,
                                "train_loss": float(loss.item()),
                                **stats,
                            }
                            monitor_file.write(json.dumps(record) + "\n")

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                writer.add_scalar("Train/Loss", loss.item(), global_step)
                epoch_loss += loss.item()
                global_step += 1

            avg_epoch_loss = epoch_loss / len(train_loader)
            writer.add_scalar("Train/Epoch_Loss", avg_epoch_loss, epoch)

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

    out_path = os.path.join(args.out_dir, f"{variant}_{task}_{model_name}_lr{args.head_lr}_seed{args.seed}.json")
    with open(out_path, "w") as result_file:
        json.dump({
            "variant": variant,
            "best_score": best_score,
            "best_metric": best_metric,
            "history": history,
            "monitor_param_count": len(monitor_params),
            "monitor_scalar_count": sum(param.numel() for param in monitor_params),
            "monitor_param_names": monitor_param_names,
            "monitor_jsonl_path": monitor_jsonl_path,
            "args": vars(args),
        }, result_file, indent=2)
    print(f"Best score: {best_score} saved to {out_path}")


if __name__ == "__main__":
    main()

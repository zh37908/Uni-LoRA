#!/usr/bin/env python
# coding: utf-8

"""
Few-label GLUE training script for UniLoRA variants.

This script reuses the main variant training pipeline and only changes
the training split size by sampling a fixed fraction of the original
training set before tokenization.
"""

import argparse
import json
import os

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
    parser.add_argument("--out_dir", "--output_dir", dest="out_dir", type=str, default="results_variants_few_label")
    parser.add_argument("--batch_size", type=int, default=32, help="Per-device batch size for both train and eval dataloaders.")
    parser.add_argument(
        "--train_subset_ratio",
        type=float,
        required=True,
        help="Fraction of the original training set to keep, e.g. 0.1 or 0.5.",
    )

    # UniLoRA common hyperparams
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

    return parser.parse_args()


def build_few_label_train_split(train_dataset, task, train_subset_ratio, seed):
    if not 0.0 < train_subset_ratio <= 1.0:
        raise ValueError(f"train_subset_ratio must be in (0, 1], got {train_subset_ratio}")

    original_size = len(train_dataset)
    target_size = max(1, int(original_size * train_subset_ratio))

    if target_size >= original_size:
        return train_dataset, original_size, original_size / original_size

    if task != "stsb" and "label" in train_dataset.column_names:
        sampled_train = train_dataset.train_test_split(
            train_size=target_size,
            seed=seed,
            stratify_by_column="label",
        )["train"]
    else:
        sampled_train = train_dataset.shuffle(seed=seed).select(range(target_size))

    actual_ratio = len(sampled_train) / float(original_size)
    return sampled_train, original_size, actual_ratio


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
        else:
            if current_init_bound == 0.02:
                current_init_bound = 0.005
                print(f">>> Variant {variant} detected. Lowering init_theta_d_bound to {current_init_bound} for stability.")
            print(f">>> Variant {variant} detected. Using theta_d_lr = {theta_d_lr}.")

    tokenizer = base.AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    datasets = base.load_dataset("nyu-mll/glue", task)
    sampled_train, original_train_size, actual_subset_ratio = build_few_label_train_split(
        datasets["train"],
        task,
        args.train_subset_ratio,
        args.seed,
    )
    datasets["train"] = sampled_train
    print(
        f"Using {len(sampled_train)}/{original_train_size} training examples "
        f"(requested ratio={args.train_subset_ratio:.3f}, actual ratio={actual_subset_ratio:.3f})."
    )

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

    if variant == "lora":
        peft_config = base.LoraConfig(
            task_type="SEQ_CLS",
            r=args.rank,
            lora_alpha=args.rank,
            lora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_count_sketch":
        peft_config = base.UniLoRACountSketchConfig(
            task_type="SEQ_CLS", peft_type=base.PeftType.UNILORA_COUNT_SKETCH,
            r=args.rank, theta_d_length=args.theta_d_length,
            num_sketches=args.v,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_nonorm":
        peft_config = base.UniLoRANonormConfig(
            task_type="SEQ_CLS", peft_type=base.PeftType.UNILORA_NONORM,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_sign":
        peft_config = base.UniLoRASignConfig(
            task_type="SEQ_CLS", peft_type=base.PeftType.UNILORA_SIGN,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_fastfood":
        peft_config = base.UniLoRAFastFoodConfig(
            task_type="SEQ_CLS", peft_type=base.PeftType.UNILORA_FASTFOOD,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_gs":
        peft_config = base.UniLoRAGSConfig(
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
    elif variant == "unilora_block_routing":
        peft_config = base.UniLoRABlockRoutingConfig(
            task_type="SEQ_CLS", peft_type=base.PeftType.UNILORA_BLOCK_ROUTING,
            r=args.rank, theta_d_length=args.theta_d_length,
            num_blocks=args.num_blocks,
            router_tau=args.gumbel_tau,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_stage_ratio":
        peft_config = base.UniLoRAStageRatioConfig(
            task_type="SEQ_CLS", peft_type=base.PeftType.UNILORA_STAGE_RATIO,
            r=args.rank, theta_d_length=args.theta_d_length,
            stage_theta_d_ratios=args.stage_theta_d_ratios,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_learnable":
        peft_config = base.UniLoRALearnableConfig(
            task_type="SEQ_CLS", peft_type=base.PeftType.UNILORA_LEARNABLE,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_learnable_column":
        peft_config = base.UniLoRALearnableColumnConfig(
            task_type="SEQ_CLS", peft_type=base.PeftType.UNILORA_LEARNABLE_COLUMN,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_isometric_control":
        peft_config = base.UniLoRAIsometricControlConfig(
            task_type="SEQ_CLS", peft_type=base.PeftType.UNILORA_ISOMETRIC_CONTROL,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            isometry_alpha=args.isometry_alpha,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "direct_unilora":
        peft_config = base.DirectUniLoRAConfig(
            task_type="SEQ_CLS", peft_type=base.PeftType.DIRECT_UNILORA,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_layer_wise":
        peft_config = base.UniLoRALayerWiseConfig(
            task_type="SEQ_CLS", peft_type=base.PeftType.UNILORA_LAYER_WISE,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_learnable_layer":
        peft_config = base.UniLoRALearnableLayerConfig(
            task_type="SEQ_CLS", peft_type=base.PeftType.UNILORA_LEARNABLE_LAYER,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            alpha_init=args.alpha_init, alpha_min=args.alpha_min, alpha_max=args.alpha_max,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    else:
        peft_config = base.UniLoRAConfig(
            task_type="SEQ_CLS", peft_type=base.PeftType.UNILORA,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )

    model = base.get_peft_model(base_model, peft_config)
    model.to(device)

    head_params, theta_d_params, alpha_params = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            if any(term in n for term in ["theta_d", "unilora_layer_alpha"]):
                p.requires_grad = True
            else:
                continue

        if "unilora_layer_alpha" in n:
            alpha_params.append(p)
        elif n.endswith("theta_d") or "theta_d." in n:
            theta_d_params.append(p)
        else:
            head_params.append(p)

    theta_d_lr_display = f"{theta_d_lr}" if theta_d_params else "N/A"
    alpha_lr_display = f"{alpha_lr}" if alpha_params else "N/A"
    print("=" * 80)
    print(f"Run Variant: {variant.upper()}")
    print(f"  model_name = {model_name} | task = {task} | seed = {args.seed}")
    print(f"  subset_ratio = {args.train_subset_ratio:.3f} | batch_size = {batch_size}")
    print(f"  head_lr = {args.head_lr} | theta_d_lr = {theta_d_lr_display} | alpha_lr = {alpha_lr_display}")
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
        for p in alpha_params:
            p.requires_grad = False
        print(f"Freezing alpha parameters for first {alpha_freeze_steps}/{total_steps} steps.")

    subset_tag = f"subset_{int(round(actual_subset_ratio * 100)):02d}"
    log_dir = os.path.join(
        args.out_dir,
        "runs",
        f"{variant}_{task}_{model_name}_{subset_tag}_lr{args.head_lr}_seed{args.seed}",
    )
    writer = base.SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging to: {log_dir}")

    best_score = -1e18
    best_metric = None
    history = []
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        pbar = base.tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        epoch_loss = 0
        for batch in pbar:
            if alpha_params and alpha_freeze_steps > 0 and global_step == alpha_freeze_steps:
                for p in alpha_params:
                    p.requires_grad = True
                print(f"Unfroze alpha parameters at step {global_step}.")

            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
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
        eval_loss = 0
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
        for k, v in eval_results.items():
            writer.add_scalar(f"Eval/{k}", v, epoch)

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

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{variant}_{task}_{model_name}_lr{args.head_lr}_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump({
            "variant": variant,
            "best_score": best_score,
            "best_metric": best_metric,
            "history": history,
            "original_train_size": original_train_size,
            "sampled_train_size": len(sampled_train),
            "train_subset_ratio_actual": actual_subset_ratio,
            "args": vars(args),
        }, f, indent=2)
    print(f"Best score: {best_score} saved to {out_path}")


if __name__ == "__main__":
    main()

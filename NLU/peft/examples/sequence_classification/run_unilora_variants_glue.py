#!/usr/bin/env python
# coding: utf-8

"""
Uni-LoRA Variants GLUE script (robust and adaptive)

Supports:
- lora
- unilora
- unilora_nonorm
"""

import os
import json
import argparse
import random
import math
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import evaluate
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from peft import get_peft_model
from peft import (
    LoraConfig,
    UniLoRAConfig,
    UniLoRASketchTuneConfig,
    UniLoRASketchDeltaConfig,
    UniLoRASharedSketchBankConfig,
    UniLoRASketchRoutedConfig,
    UniLoRAHessianAwareConfig,
    UniLoRACountSketchConfig,
    UniLoRASignConfig,
    UniLoRANonormConfig,
    UniLoRAFastFoodConfig,
    UniLoRAGSConfig,
    UniLoRABlockRoutingConfig,
    UniLoRAStageRatioConfig,
    UniLoRALearnableConfig,
    UniLoRALearnableColumnConfig,
    UniLoRAIsometricControlConfig,
    DirectUniLoRAConfig,
    UniLoRALayerWiseConfig,
    UniLoRALearnableLayerConfig,
    UniLoRASoftAssignConfig,
    PeftType,
)


# =========================
# Reproducibility
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# GLUE config
# =========================
GLUE_TASKS = ["cola", "sst2", "mrpc", "qnli", "rte", "stsb"]

TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "stsb": ("sentence1", "sentence2"),
}

MAX_LENGTH = {
    "roberta-base": 512,
    "roberta-large": 128,
}

EPOCHS = {
    "roberta-base": {
        "sst2": 60, "mrpc": 30, "cola": 80, "qnli": 25, "rte": 160, "stsb": 80,
    },
    "roberta-large": {
        "sst2": 20, "mrpc": 40, "cola": 40, "qnli": 20, "rte": 40, "stsb": 40,
    },
}

TASK_TO_METRIC = {
    "cola": "matthews_correlation",
    "sst2": "accuracy",
    "mrpc": "accuracy",
    "qnli": "accuracy",
    "rte": "accuracy",
    "stsb": "pearson",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=["roberta-base", "roberta-large"])
    parser.add_argument("--task", type=str, required=True, choices=GLUE_TASKS)
    parser.add_argument(
        "--variant",
        type=str,
        default="unilora",
        choices=[
            "lora",
            "unilora",
            "unilora_sketch_tune",
            "unilora_sketch_delta",
            "unilora_shared_sketch_bank",
            "unilora_sketch_routed",
            "unilora_count_sketch",
            "unilora_sign",
            "unilora_nonorm",
            "unilora_fastfood",
            "unilora_gs",
            "unilora_soft_assign",
            "unilora_block_routing",
            "unilora_stage_ratio",
            "unilora_learnable",
            "unilora_learnable_column",
            "unilora_isometric_control",
            "direct_unilora",
            "unilora_layer_wise",
            "unilora_learnable_layer",
            "unilora_hessian_aware",
        ],
    )
    parser.add_argument("--isometry_alpha", type=float, default=0.0, help="Control parameter for unilora_isometric_control (0.0: isometric, 1.0: non-isometric)")
    parser.add_argument("--head_lr", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out_dir", "--output_dir", dest="out_dir", type=str, default="results_variants")
    parser.add_argument("--batch_size", type=int, default=32, help="Per-device batch size for both train and eval dataloaders.")

    # UniLoRA common hyperparams
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--theta_d_length", type=int, default=23040)
    parser.add_argument("--sketch_bits", type=int, default=4, help="Codebook bit-width for unilora_sketch_tune.")
    parser.add_argument(
        "--sketch_groups_per_row",
        type=int,
        default=4,
        help="Number of codebook groups per output row for unilora_sketch_tune.",
    )
    parser.add_argument(
        "--sketch_bootstrap_method",
        type=str,
        default="uniform",
        choices=["uniform", "kmeans"],
        help="Dense-weight bootstrap method for unilora_sketch_tune.",
    )
    parser.add_argument(
        "--sketch_bootstrap_kmeans_iters",
        type=int,
        default=8,
        help="K-means iterations used when --sketch_bootstrap_method=kmeans.",
    )
    parser.add_argument("--sketch_num_banks", type=int, default=8, help="Number of shared sketch banks.")
    parser.add_argument("--sketch_num_experts", type=int, default=4, help="Number of shared sketch experts.")
    parser.add_argument("--sketch_router_tau", type=float, default=1.0, help="Router temperature for routed sketch variant.")
    parser.add_argument(
        "--sketch_router_mode",
        type=str,
        default="softmax",
        choices=["softmax", "gumbel"],
        help="Router mode for unilora_sketch_routed.",
    )
    parser.add_argument(
        "--sketch_router_gumbel_hard",
        action="store_true",
        help="Use hard straight-through Gumbel-Softmax for unilora_sketch_routed.",
    )
    parser.add_argument(
        "--sketch_router_soft_eval",
        action="store_true",
        help="Use soft router mixing during evaluation for unilora_sketch_routed.",
    )
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
    parser.add_argument("--num_candidates", type=int, default=4, help="Candidate theta_d entries per UniLoRA element for unilora_soft_assign")
    parser.add_argument(
        "--soft_assign_mode",
        type=str,
        default="softmax",
        choices=["softmax", "gumbel"],
        help="Assignment mode for unilora_soft_assign",
    )
    parser.add_argument("--soft_assign_temperature", type=float, default=1.0, help="Assignment temperature for unilora_soft_assign")
    parser.add_argument("--soft_assign_gumbel_hard", action="store_true", help="Use hard straight-through Gumbel-Softmax in unilora_soft_assign")
    parser.add_argument(
        "--soft_assign_soft_eval",
        action="store_true",
        help="Use deterministic soft evaluation instead of argmax hardening for unilora_soft_assign",
    )
    parser.add_argument(
        "--init_primary_bias",
        type=float,
        default=2.0,
        help="Bias on the primary candidate to initialize unilora_soft_assign near vanilla UniLoRA",
    )
    parser.add_argument("--num_blocks", type=int, default=8, help="Number of blocks for unilora_block_routing")
    parser.add_argument(
        "--stage_theta_d_ratios",
        type=float,
        nargs=3,
        default=[0.2, 0.3, 0.5],
        metavar=("FRONT", "MIDDLE", "BACK"),
        help="Theta_d ratios for front/middle/back stages in unilora_stage_ratio.",
    )
    parser.add_argument(
        "--hessian_aware_structure_update_interval",
        type=int,
        default=5,
        help="If > 0, run one Hessian-aware structure update every N epochs.",
    )
    parser.add_argument(
        "--hessian_aware_warmup_epochs",
        type=int,
        default=1,
        help="Do not run Hessian-aware structure updates before this epoch count is reached.",
    )
    parser.add_argument(
        "--hessian_aware_reassign_ratio",
        type=float,
        default=0.01,
        help="Top curvature fraction greedily reassigned in each Hessian-aware structure update.",
    )
    parser.add_argument(
        "--hessian_aware_candidate_pool_size",
        type=int,
        default=8,
        help="Number of value-near candidate buckets considered per reassigned position.",
    )
    parser.add_argument(
        "--hessian_aware_capacity_penalty",
        type=float,
        default=0.1,
        help="Penalty coefficient for overloaded buckets during Hessian-aware structure updates.",
    )
    parser.add_argument(
        "--hessian_aware_capacity_slack",
        type=float,
        default=2.0,
        help="Hard capacity multiplier relative to average bucket load for Hessian-aware structure updates.",
    )
    parser.add_argument(
        "--hessian_aware_curvature_ema_momentum",
        type=float,
        default=0.9,
        help="EMA momentum used for the Hessian/Fisher diagonal surrogate.",
    )
    parser.add_argument(
        "--hessian_aware_accept_tolerance",
        type=float,
        default=1e-6,
        help="Accept structure updates whose metric drop is no worse than this tolerance.",
    )
    parser.add_argument("--num_epochs", type=int, default=None, help="Override default number of epochs")

    return parser.parse_args()


def estimate_nonorm_scale(model_name: str, rank: int, theta_d_length: int):
    # Assumes target_modules are: query/key/value/output.dense/intermediate.dense
    # Per layer LoRA indices ≈ r*(h+h) * 4 + r*(h+4h) = 13*h*r
    model_specs = {
        "roberta-base": (12, 768),
        "roberta-large": (24, 1024),
    }
    if model_name not in model_specs:
        return None, 1.0
    num_layers, hidden_size = model_specs[model_name]
    lora_para_cnt = num_layers * 13 * hidden_size * rank
    if theta_d_length <= 0:
        return None, 1.0
    count = lora_para_cnt / float(theta_d_length)
    if count <= 0:
        return count, 1.0
    scale = 1.0 / math.sqrt(count)
    return count, scale


def summarize_projection_groups(model):
    group_to_modules = {}
    for _, module in model.named_modules():
        group_name = getattr(module, "projection_group_name", None)
        module_key = getattr(module, "projection_module_key", None)
        if not group_name or not module_key:
            continue
        group_to_modules.setdefault(group_name, set()).add(module_key)
    return {group_name: sorted(module_keys) for group_name, module_keys in sorted(group_to_modules.items())}


def print_trainable_param_summary(model, variant, theta_d_params, alpha_params=None, head_params=None):
    show_alpha = alpha_params is not None
    show_head = head_params is not None
    alpha_params = alpha_params or []
    head_params = head_params or []

    if variant == "unilora_layer_wise":
        group_to_modules = summarize_projection_groups(model)
        if group_to_modules:
            modules_per_group = ", ".join(
                f"{group_name}:{len(module_keys)}" for group_name, module_keys in group_to_modules.items()
            )
            print(f"Detected {len(group_to_modules)} transformer-layer local banks.")
            print(f"Modules per transformer-layer bank: {modules_per_group}")
        else:
            print(f"Detected {len(theta_d_params)} theta_d parameter tensors.")
    elif variant in {
        "unilora_sketch_tune",
        "unilora_sketch_delta",
        "unilora_shared_sketch_bank",
        "unilora_sketch_routed",
    }:
        print(f"Detected {len(theta_d_params)} sketch codebook parameter tensors.")
    else:
        print(f"Detected {len(theta_d_params)} shared vector bank parameters.")

    if show_alpha:
        print(f"Detected {len(alpha_params)} layer alpha parameters.")
    if show_head:
        print(f"Detected {len(head_params)} other trainable parameters (classifier/adapters).")


def get_hessian_aware_backend(model, variant):
    if variant != "unilora_hessian_aware":
        return None
    backend = getattr(model, "base_model", None)
    if backend is None:
        return None
    if not hasattr(backend, "update_structure") or not hasattr(backend, "accumulate_curvature_statistics"):
        return None
    return backend


def snapshot_hessian_aware_state(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if (
            "unilora_hessian_aware_theta_d" in key
            or "unilora_indices" in key
            or "unilora_scales" in key
        )
    }


def evaluate_glue_model(model, eval_loader, task, metric_name, device):
    model.eval()
    metric = evaluate.load("glue", task)
    eval_loss = 0.0
    for batch in eval_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
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
    return avg_eval_loss, eval_results, score


def should_accept_structure_update(baseline_score, baseline_loss, trial_score, trial_loss, tolerance):
    if trial_score > baseline_score + tolerance:
        return True
    return trial_score >= baseline_score - tolerance and trial_loss <= baseline_loss + tolerance


def main():
    args = parse_args()
    set_seed(args.seed)

    model_name = args.model_name
    task = args.task
    variant = args.variant

    batch_size = args.batch_size
    max_length = MAX_LENGTH[model_name]
    num_epochs = args.num_epochs if args.num_epochs is not None else EPOCHS[model_name][task]
    warmup_ratio = 0.06

    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric_name = TASK_TO_METRIC[task]

    # Variant-specific defaults for stability
    current_init_bound = args.init_theta_d_bound
    theta_d_lr = args.theta_d_lr
    alpha_lr = args.alpha_lr if args.alpha_lr is not None else (theta_d_lr / 50.0)

    if variant == "unilora_nonorm":
        # Since nonorm increases the scale of Delta weights significantly,
        # scale init bound and lr by 1/sqrt(count) to match unilora.
        count, scale = estimate_nonorm_scale(model_name, args.rank, args.theta_d_length)
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

    # Data
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    datasets = load_dataset("nyu-mll/glue", task)
    s1_key, s2_key = TASK_TO_KEYS[task]

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

    train_loader = DataLoader(datasets["train"], shuffle=True, batch_size=batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(datasets["validation"], shuffle=False, batch_size=batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    # Model
    num_labels = 1 if task == "stsb" else 2
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, return_dict=True)

    if variant == "lora":
        peft_config = LoraConfig(
            task_type="SEQ_CLS",
            r=args.rank,
            lora_alpha=args.rank,
            lora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_count_sketch":
        peft_config = UniLoRACountSketchConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_COUNT_SKETCH,
            r=args.rank, theta_d_length=args.theta_d_length,
            num_sketches=args.v,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_nonorm":
        peft_config = UniLoRANonormConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_NONORM,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_sign":
        peft_config = UniLoRASignConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_SIGN,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_fastfood":
        peft_config = UniLoRAFastFoodConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_FASTFOOD,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_gs":
        peft_config = UniLoRAGSConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_GS,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            init_logits_std=args.init_logits_std,
            init_logits_bias=args.init_logits_bias,
            gumbel_tau=args.gumbel_tau,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_soft_assign":
        peft_config = UniLoRASoftAssignConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_SOFT_ASSIGN,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            num_candidates=args.num_candidates,
            assignment_mode=args.soft_assign_mode,
            temperature=args.soft_assign_temperature,
            gumbel_hard=args.soft_assign_gumbel_hard,
            hard_eval=not args.soft_assign_soft_eval,
            init_logits_std=args.init_logits_std,
            init_primary_bias=args.init_primary_bias,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_block_routing":
        peft_config = UniLoRABlockRoutingConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_BLOCK_ROUTING,
            r=args.rank, theta_d_length=args.theta_d_length,
            num_blocks=args.num_blocks,
            router_tau=args.gumbel_tau,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_stage_ratio":
        peft_config = UniLoRAStageRatioConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_STAGE_RATIO,
            r=args.rank, theta_d_length=args.theta_d_length,
            stage_theta_d_ratios=args.stage_theta_d_ratios,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_sketch_tune":
        peft_config = UniLoRASketchTuneConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_SKETCH_TUNE,
            bits=args.sketch_bits,
            groups_per_row=args.sketch_groups_per_row,
            bootstrap_method=args.sketch_bootstrap_method,
            bootstrap_kmeans_iters=args.sketch_bootstrap_kmeans_iters,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_sketch_delta":
        peft_config = UniLoRASketchDeltaConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_SKETCH_DELTA,
            r=args.rank,
            proj_seed=args.seed,
            bits=args.sketch_bits,
            groups_per_row=args.sketch_groups_per_row,
            init_codebook_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_shared_sketch_bank":
        peft_config = UniLoRASharedSketchBankConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_SHARED_SKETCH_BANK,
            r=args.rank,
            proj_seed=args.seed,
            bits=args.sketch_bits,
            groups_per_row=args.sketch_groups_per_row,
            num_banks=args.sketch_num_banks,
            init_bank_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_sketch_routed":
        peft_config = UniLoRASketchRoutedConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_SKETCH_ROUTED,
            r=args.rank,
            proj_seed=args.seed,
            bits=args.sketch_bits,
            groups_per_row=args.sketch_groups_per_row,
            num_banks=args.sketch_num_banks,
            num_experts=args.sketch_num_experts,
            router_tau=args.sketch_router_tau,
            router_mode=args.sketch_router_mode,
            router_gumbel_hard=args.sketch_router_gumbel_hard,
            router_hard_eval=not args.sketch_router_soft_eval,
            init_expert_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_learnable":
        peft_config = UniLoRALearnableConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_LEARNABLE,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_learnable_column":
        peft_config = UniLoRALearnableColumnConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_LEARNABLE_COLUMN,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_isometric_control":
        peft_config = UniLoRAIsometricControlConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_ISOMETRIC_CONTROL,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            isometry_alpha=args.isometry_alpha,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "direct_unilora":
        peft_config = DirectUniLoRAConfig(
            task_type="SEQ_CLS", peft_type=PeftType.DIRECT_UNILORA,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_layer_wise":
        peft_config = UniLoRALayerWiseConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_LAYER_WISE,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_hessian_aware":
        peft_config = UniLoRAHessianAwareConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_HESSIAN_AWARE,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            curvature_ema_momentum=args.hessian_aware_curvature_ema_momentum,
            structure_reassign_ratio=args.hessian_aware_reassign_ratio,
            candidate_pool_size=args.hessian_aware_candidate_pool_size,
            capacity_penalty=args.hessian_aware_capacity_penalty,
            capacity_slack=args.hessian_aware_capacity_slack,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_learnable_layer":
        peft_config = UniLoRALearnableLayerConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_LEARNABLE_LAYER,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            alpha_init=args.alpha_init, alpha_min=args.alpha_min, alpha_max=args.alpha_max,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    else:
        peft_config = UniLoRAConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )

    model = get_peft_model(base_model, peft_config)
    model.to(device)
    hessian_aware_backend = get_hessian_aware_backend(model, variant)
    if hessian_aware_backend is not None:
        hessian_aware_backend.enable_curvature_capture(True)

    # Adaptive Parameter Grouping
    head_params, theta_d_params, alpha_params = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            # Force requires_grad if it's a known UniLoRA variant parameter
            if any(term in n for term in ["theta_d", "unilora_layer_alpha"]):
                p.requires_grad = True
            else:
                continue
        
        if "unilora_layer_alpha" in n:
            alpha_params.append(p)
        elif (
            n.endswith("theta_d")
            or "theta_d." in n
            or "unilora_sketch_tune_quant_grid" in n
            or "unilora_sketch_delta_" in n
            or "unilora_shared_sketch_bank_" in n
            or "unilora_sketch_routed_" in n
        ):
            theta_d_params.append(p)
        else:
            head_params.append(p)

    theta_d_lr_display = f"{theta_d_lr}" if theta_d_params else "N/A"
    alpha_lr_display = f"{alpha_lr}" if alpha_params else "N/A"
    print("=" * 80)
    print(f"Run Variant: {variant.upper()}")
    print(f"  model_name = {model_name} | task = {task} | seed = {args.seed}")
    print(f"  head_lr = {args.head_lr} | theta_d_lr = {theta_d_lr_display} | alpha_lr = {alpha_lr_display}")
    print("=" * 80)

    print_trainable_param_summary(
        model=model,
        variant=variant,
        theta_d_params=theta_d_params,
        alpha_params=alpha_params,
        head_params=head_params,
    )
    if hessian_aware_backend is not None:
        print(
            "Hessian-aware structure config: "
            f"interval={args.hessian_aware_structure_update_interval}, "
            f"warmup={args.hessian_aware_warmup_epochs}, "
            f"reassign_ratio={args.hessian_aware_reassign_ratio}, "
            f"candidate_pool_size={args.hessian_aware_candidate_pool_size}, "
            f"capacity_penalty={args.hessian_aware_capacity_penalty}, "
            f"capacity_slack={args.hessian_aware_capacity_slack}, "
            f"curvature_ema_momentum={args.hessian_aware_curvature_ema_momentum}"
        )
        print(f"Initial structure stats: {hessian_aware_backend.get_structure_stats()}")

    optimizer_groups = []
    if head_params:
        optimizer_groups.append({"params": head_params, "lr": args.head_lr, "weight_decay": 0.01})
    if theta_d_params:
        optimizer_groups.append({"params": theta_d_params, "lr": theta_d_lr, "weight_decay": 0.01})
    if alpha_params:
        optimizer_groups.append({"params": alpha_params, "lr": alpha_lr, "weight_decay": 0.0})

    optimizer = AdamW(optimizer_groups)

    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(warmup_ratio * total_steps), num_training_steps=total_steps)
    alpha_freeze_steps = int(args.alpha_freeze_ratio * total_steps) if alpha_params else 0
    if alpha_params and alpha_freeze_steps > 0:
        for p in alpha_params:
            p.requires_grad = False
        print(f"Freezing alpha parameters for first {alpha_freeze_steps}/{total_steps} steps.")

    # TensorBoard initialization
    log_dir = os.path.join(args.out_dir, "runs", f"{variant}_{task}_{model_name}_lr{args.head_lr}_seed{args.seed}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging to: {log_dir}")

    # Train / Eval
    best_score = -1e18
    best_metric = None
    history = []
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        epoch_loss = 0
        for batch in pbar:
            if alpha_params and alpha_freeze_steps > 0 and global_step == alpha_freeze_steps:
                for p in alpha_params:
                    p.requires_grad = True
                print(f"Unfroze alpha parameters at step {global_step}.")

            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            if hessian_aware_backend is not None:
                hessian_aware_backend.accumulate_curvature_statistics(
                    adapter_name="default",
                    ema_momentum=args.hessian_aware_curvature_ema_momentum,
                )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            writer.add_scalar("Train/Loss", loss.item(), global_step)
            epoch_loss += loss.item()
            global_step += 1
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        writer.add_scalar("Train/Epoch_Loss", avg_epoch_loss, epoch)

        avg_eval_loss, eval_results, score = evaluate_glue_model(model, eval_loader, task, metric_name, device)

        structure_update_info = None
        should_update_structure = (
            hessian_aware_backend is not None
            and args.hessian_aware_structure_update_interval > 0
            and (epoch + 1) >= args.hessian_aware_warmup_epochs
            and ((epoch + 1) % args.hessian_aware_structure_update_interval == 0)
            and (epoch + 1) < num_epochs
        )
        if should_update_structure:
            baseline_score = score
            baseline_eval_loss = avg_eval_loss
            structure_snapshot = snapshot_hessian_aware_state(model)
            update_info = hessian_aware_backend.update_structure(
                adapter_name="default",
                candidate_pool_size=args.hessian_aware_candidate_pool_size,
                reassign_ratio=args.hessian_aware_reassign_ratio,
                capacity_penalty=args.hessian_aware_capacity_penalty,
                capacity_slack=args.hessian_aware_capacity_slack,
            )
            trial_eval_loss, trial_eval_results, trial_score = evaluate_glue_model(model, eval_loader, task, metric_name, device)
            accepted = should_accept_structure_update(
                baseline_score=baseline_score,
                baseline_loss=baseline_eval_loss,
                trial_score=trial_score,
                trial_loss=trial_eval_loss,
                tolerance=args.hessian_aware_accept_tolerance,
            )
            if accepted:
                avg_eval_loss = trial_eval_loss
                eval_results = trial_eval_results
                score = trial_score
                optimizer.state.clear()
                print(
                    f"Accepted Hessian-aware structure update after epoch {epoch}: "
                    f"changed_ratio={update_info['changed_ratio']:.4f}, "
                    f"{metric_name}={score:.4f}, val_loss={avg_eval_loss:.4f}"
                )
            else:
                model.load_state_dict(structure_snapshot, strict=False)
                print(
                    f"Rejected Hessian-aware structure update after epoch {epoch}: "
                    f"changed_ratio={update_info['changed_ratio']:.4f}, "
                    f"trial_{metric_name}={trial_score:.4f}, trial_val_loss={trial_eval_loss:.4f}"
                )
            structure_update_info = {
                "accepted": accepted,
                "baseline_score": baseline_score,
                "baseline_val_loss": baseline_eval_loss,
                "trial_score": trial_score,
                "trial_val_loss": trial_eval_loss,
                "update": update_info,
            }

        print(f"Epoch {epoch} | train_loss: {avg_epoch_loss:.4f} | val_loss: {avg_eval_loss:.4f} | {metric_name}: {score:.4f} | {eval_results}")

        # Log metrics to TensorBoard
        writer.add_scalar("Eval/Loss", avg_eval_loss, epoch)
        for k, v in eval_results.items():
            writer.add_scalar(f"Eval/{k}", v, epoch)

        history.append({
            "epoch": epoch,
            "train_loss": avg_epoch_loss,
            "val_loss": avg_eval_loss,
            "score": score,
            "metrics": eval_results,
            "structure_update": structure_update_info,
        })

        if score > best_score:
            best_score = score
            best_metric = eval_results

    writer.close()

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{variant}_{task}_{model_name}_lr{args.head_lr}_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump({
            "variant": variant, 
            "best_score": best_score, 
            "best_metric": best_metric, 
            "history": history,
            "args": vars(args)
        }, f, indent=2)
    print(f"Best score: {best_score} saved to {out_path}")

if __name__ == "__main__":
    main()

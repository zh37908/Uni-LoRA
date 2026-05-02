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
import re
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import evaluate
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from transformers.pytorch_utils import Conv1D

from peft import get_peft_model
from peft import (
    LoraConfig,
    UniLoRAConfig,
    UniLoRAAromaConfig,
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
    UniLoRAGoRAConfig,
    UniLoRAGeLoRAConfig,
    DirectUniLoRAConfig,
    UniLoRALayerWiseConfig,
    UniLoRALearnableLayerConfig,
    UniLoRARoSAConfig,
    UniLoRARoSAStageConfig,
    UniLoRARoSAStageSnipConfig,
    UniLoRARoSACompressionConfig,
    UniLoRARoSADiscreteConfig,
    UniLoRARoSAGlobalConfig,
    UniLoRAMultiHashingConfig,
    UniLoRASwapConfig,
    UniLoRALocalSwapConfig,
    UniLoRAMultiStructuredConfig,
    UniLoRAMultiStructuredGlobalConfig,
    UniLoRASoftAssignConfig,
    UniLoRASoftWeightSharingConfig,
    UniLoRADeepKConfig,
    GeoUniLoRAConfig,
    IGUUniLoRAConfig,
    UniLoRAIGUConfig,
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
            "unilora_aroma",
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
            "unilora_soft_weight_sharing",
            "unilora_deepk",
            "unilora_block_routing",
            "unilora_stage_ratio",
            "unilora_learnable",
            "unilora_learnable_column",
            "unilora_isometric_control",
            "unilora_gora",
            "unilora_gelora",
            "direct_unilora",
            "unilora_layer_wise",
            "unilora_learnable_layer",
            "unilora_hessian_aware",
            "unilora_rosa",
            "unilora_rosa_stage",
            "unilora_rosa_stage_snip",
            "unilora_rosa_discrete",
            "unilora_rosa_global",
            "unilora_rosa_compression",
            "unilora_multi_hashing",
            "unilora_swap",
            "unilora_local_swap",
            "unilora_multi_structured",
            "unilora_multi_structured_global",
            "geo_unilora",
            "igu_unilora",
            "unilora_igu",
        ],
    )
    parser.add_argument("--isometry_alpha", type=float, default=0.0, help="Control parameter for unilora_isometric_control (0.0: isometric, 1.0: non-isometric)")
    parser.add_argument("--head_lr", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out_dir", "--output_dir", dest="out_dir", type=str, default="results_variants")
    parser.add_argument("--batch_size", type=int, default=32, help="Per-device batch size for both train and eval dataloaders.")
    parser.add_argument("--warmup_ratio", type=float, default=0.06, help="Warmup ratio for the main optimizer scheduler.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay used for head/theta_d optimizer groups.")
    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="linear",
        choices=["linear", "cosine"],
        help="Learning-rate scheduler used for the main optimizer.",
    )

    # UniLoRA common hyperparams
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--theta_d_length", type=int, default=23040)
    parser.add_argument(
        "--sparse_theta_d_length",
        type=int,
        default=None,
        help="Required for unilora_rosa_compression: sparse-bank theta_d length.",
    )
    parser.add_argument(
        "--rosa_sparse_theta_d_length",
        type=int,
        default=None,
        help="Optional sparse-bank theta_d length for unilora_rosa_discrete; defaults to --theta_d_length.",
    )
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
    parser.add_argument("--aroma_t_in", type=int, default=100, help="Fixed optimizer-step interval for UniLoRA-AROMA merge-and-reinit.")
    parser.add_argument(
        "--aroma_reset_optimizer_on_merge",
        action="store_true",
        default=True,
        help="Clear optimizer state after each UniLoRA-AROMA merge-and-reinit event.",
    )
    parser.add_argument(
        "--aroma_keep_optimizer_on_merge",
        action="store_false",
        dest="aroma_reset_optimizer_on_merge",
        help="Keep optimizer state after UniLoRA-AROMA merge-and-reinit.",
    )
    parser.add_argument("--rosa_sparse_lr", type=float, default=None, help="LR for UniLoRA-RoSA sparse compensation; defaults to theta_d_lr.")
    parser.add_argument(
        "--rosa_decay_sparse_lr_after_activation",
        action="store_true",
        help=(
            "Keep the UniLoRA-RoSA sparse LR at its base value until the sparse mask is activated, "
            "then decay it over the remaining training steps using --scheduler_type."
        ),
    )
    parser.add_argument("--rosa_density", type=float, default=0.01, help="Density of the UniLoRA-RoSA sparse compensation mask.")
    parser.add_argument(
        "--rosa_stage_ratio",
        type=float,
        default=0.5,
        help="For unilora_rosa_stage(_snip): fraction of total training epochs completed before starting sparse-mask selection.",
    )
    parser.add_argument(
        "--rosa_stage_warmup_steps",
        type=int,
        default=None,
        help="For unilora_rosa_stage(_snip): explicit optimizer-step warmup before starting sparse-mask selection; overrides --rosa_stage_ratio.",
    )
    parser.add_argument("--rosa_warmup_steps", type=int, default=64, help="Low-rank-only warmup steps before collecting RoSA sparse mask gradients.")
    parser.add_argument(
        "--rosa_mask_steps",
        type=int,
        default=1,
        help="Number of optimizer steps used to collect RoSA mask scores.",
    )
    parser.add_argument("--swap_dead_bucket_count", type=int, default=8, help="Number of low-importance buckets used per UniLoRA-Swap round.")
    parser.add_argument("--swap_split_ratio", type=float, default=0.5, help="Fraction of assignments moved from an overloaded bucket to a freed bucket in UniLoRA-Swap.")
    parser.add_argument("--swap_interval_steps", type=int, default=0, help="Run UniLoRA-Swap every N optimizer steps; 0 disables step-based swap.")
    parser.add_argument("--swap_start_after_steps", type=int, default=0, help="Do not run UniLoRA-Swap before this optimizer step.")
    parser.add_argument("--swap_interval_epochs", type=int, default=0, help="Run UniLoRA-Swap every N epochs; 0 disables epoch-based swap.")
    parser.add_argument("--swap_start_after_epochs", type=int, default=0, help="Do not run UniLoRA-Swap before this epoch.")
    parser.add_argument("--swap_reset_optimizer_state", action="store_true", help="Reset Adam moments for affected buckets after each UniLoRA-Swap round.")
    parser.add_argument("--local_swap_grad_ema_momentum", type=float, default=0.9, help="EMA momentum for UniLoRA-LocalSwap position gradients.")
    parser.add_argument("--local_swap_warmup_steps", type=int, default=0, help="Do not trigger UniLoRA-LocalSwap before this many optimizer steps.")
    parser.add_argument("--local_swap_bad_bucket_frac", type=float, default=0.1, help="Fraction of most conflicting buckets considered per UniLoRA-LocalSwap round.")
    parser.add_argument("--local_swap_candidates_per_bucket", type=int, default=2, help="Number of candidate positions examined per bad bucket in UniLoRA-LocalSwap.")
    parser.add_argument("--local_swap_target_bucket_samples", type=int, default=16, help="Number of target buckets sampled per UniLoRA-LocalSwap candidate.")
    parser.add_argument("--local_swap_min_delta", type=float, default=1e-3, help="Minimum total ratio improvement required to accept a UniLoRA-LocalSwap update.")
    parser.add_argument("--local_swap_max_target_drop", type=float, default=0.01, help="Maximum allowed target-bucket ratio drop in UniLoRA-LocalSwap.")
    parser.add_argument("--local_swap_min_bucket_size", type=int, default=2, help="Skip buckets smaller than this during UniLoRA-LocalSwap.")
    parser.add_argument("--local_swap_update_ratio", type=float, default=0.01, help="Maximum fraction of A/B positions that may change buckets in one UniLoRA-LocalSwap round.")
    parser.add_argument("--local_swap_interval_steps", type=int, default=0, help="Run UniLoRA-LocalSwap every N optimizer steps; 0 disables step-based updates.")
    parser.add_argument("--local_swap_start_after_steps", type=int, default=0, help="Do not run UniLoRA-LocalSwap before this optimizer step.")
    parser.add_argument("--local_swap_interval_epochs", type=int, default=5, help="Run UniLoRA-LocalSwap every N epochs; 0 disables epoch-based updates.")
    parser.add_argument("--local_swap_start_after_epochs", type=int, default=0, help="Do not run UniLoRA-LocalSwap before this epoch.")
    parser.add_argument("--local_swap_reset_optimizer_state", action="store_true", help="Reset Adam moments for affected buckets after each UniLoRA-LocalSwap round.")
    parser.add_argument("--gora_importance_type", type=str, default="union_mean", choices=[
        "union_mean",
        "union_frobenius_norm",
        "union_nuc_norm",
        "grad_mean",
        "grad_frobenius_norm",
    ], help="Importance metric used for GoRA rank allocation.")
    parser.add_argument("--gora_min_rank", type=int, default=None, help="Minimum allocated rank for GoRA.")
    parser.add_argument("--gora_max_rank", type=int, default=None, help="Maximum allocated rank for GoRA.")
    parser.add_argument("--gora_allocate_strategy", type=str, default="moderate", choices=["radical", "moderate", "conserved"], help="Rounding strategy for GoRA rank allocation.")
    parser.add_argument("--gora_features_func", type=str, default="none", choices=["none", "sqrt", "log1p"], help="Feature transform for GoRA allocation.")
    parser.add_argument("--gora_softmax_importance", action="store_true", help="Apply softmax to GoRA importance scores.")
    parser.add_argument("--gora_temperature", type=float, default=1.0, help="Softmax temperature for GoRA importance scores.")
    parser.add_argument("--gora_gradient_est_steps", type=int, default=8, help="Number of batches for GoRA gradient estimation.")
    parser.add_argument("--gelora_rank_offset", type=int, default=1, help="Offset added to GeLoRA rank after delta-ID.")
    # Geo-UniLoRA
    parser.add_argument("--geo_calibration_steps", type=int, default=16, help="Batches for Geo-UniLoRA activation statistics.")
    parser.add_argument(
        "--geo_total_budget",
        type=int,
        default=None,
        help="Sum over modules of (r_shared + r_innov). Default: num_target_modules * rank.",
    )
    parser.add_argument(
        "--geo_shared_ratio",
        type=float,
        default=0.5,
        help="Target fraction of geo_total_budget for shared-branch rank mass (sum_g |G_g| * c_g).",
    )
    parser.add_argument("--geo_shared_theta_d_length", type=int, default=None, help="Shared bank length; defaults to --theta_d_length.")
    parser.add_argument(
        "--geo_innovation_theta_d_length",
        type=int,
        default=None,
        help="Innovation bank length; defaults to --theta_d_length.",
    )
    parser.add_argument("--geo_gamma", type=float, default=0.25, help="Scale for geometry demand d_hat from ID estimate.")
    parser.add_argument("--geo_alpha", type=float, default=0.7, help="Shared compression in group shared-dimension formula.")
    parser.add_argument("--geo_eps", type=float, default=1e-6, help="Numerical epsilon for Geo-UniLoRA allocation.")
    parser.add_argument("--geo_tau", type=float, default=1.5, help="Temperature for innovation soft allocation.")
    parser.add_argument("--geo_r_min", type=int, default=0, help="Minimum innovation rank per module.")
    parser.add_argument("--geo_num_groups", type=int, default=8, help="Number of module groups for shared manifolds.")
    parser.add_argument(
        "--geo_grouping",
        type=str,
        default="layer_block",
        choices=["layer_block", "spectral"],
        help="How to group modules before shared-dimension allocation.",
    )
    parser.add_argument(
        "--geo_id_estimator",
        type=str,
        default="prank",
        choices=["prank", "erank"],
        help="Intrinsic complexity estimate from activation covariance.",
    )
    parser.add_argument("--geo_lambda_in", type=float, default=0.0, help="L2 penalty weight on innovation theta_d banks.")
    # IGU-UniLoRA
    parser.add_argument("--igu_calibration_steps", type=int, default=16, help="Batches for IGU-UniLoRA proxy scoring.")
    parser.add_argument("--igu_total_budget", type=int, default=None, help="Total rank mass; default=num_target_modules * rank.")
    parser.add_argument("--igu_shared_ratio", type=float, default=0.7, help="Shared branch ratio in IGU-UniLoRA.")
    parser.add_argument("--igu_beta1", type=float, default=0.85, help="EMA momentum for proxy importance in IGU-UniLoRA.")
    parser.add_argument("--igu_beta2", type=float, default=0.85, help="EMA momentum for proxy uncertainty in IGU-UniLoRA.")
    parser.add_argument("--igu_tau", type=float, default=1.5, help="Temperature for residual rank allocation in IGU-UniLoRA.")
    parser.add_argument("--igu_r_min", type=int, default=1, help="Minimum residual rank per module in IGU-UniLoRA.")
    parser.add_argument(
        "--igu_score_mode",
        type=str,
        default="proxy_ipt",
        choices=["proxy_ipt"],
        help="Scoring mode for IGU-UniLoRA; v1 supports proxy_ipt only.",
    )
    parser.add_argument(
        "--igu_score_agg",
        type=str,
        default="mean",
        choices=["mean", "fro", "max"],
        help="Aggregation mode over proxy importance tensor.",
    )
    parser.add_argument("--igu_eps", type=float, default=1e-6, help="Numerical epsilon for IGU-UniLoRA scoring/allocation.")
    parser.add_argument("--igu_shared_theta_d_length", type=int, default=None, help="Shared bank length; defaults to --theta_d_length.")
    parser.add_argument(
        "--igu_innovation_theta_d_length",
        type=int,
        default=None,
        help="Innovation bank length; defaults to --theta_d_length.",
    )
    parser.add_argument("--igu_lambda_in", type=float, default=0.0, help="L2 penalty on IGU-UniLoRA innovation bank.")
    parser.add_argument("--igu_target_rank", type=int, default=2, help="Final average active rank per module in UniLoRA-IGU.")
    parser.add_argument("--igu_init_warmup", type=int, default=100, help="Warmup steps before UniLoRA-IGU starts rank masking.")
    parser.add_argument("--igu_final_warmup", type=int, default=100, help="Final fine-tuning steps with fixed UniLoRA-IGU rank masks.")
    parser.add_argument("--igu_mask_interval", type=int, default=50, help="Apply UniLoRA-IGU rank masking every N optimizer steps.")
    parser.add_argument(
        "--igu_reset_optimizer_on_mask",
        action="store_true",
        help="Clear optimizer state right after each UniLoRA-IGU rank-mask update.",
    )
    parser.add_argument(
        "--rosa_reset_optimizer_on_mask",
        action="store_true",
        help="Clear optimizer state right after UniLoRA-RoSA sparse mask activation.",
    )
    parser.add_argument("--alpha_lr", type=float, default=None, help="LR for unilora_layer_alpha_* parameters; defaults to theta_d_lr / 50.")
    parser.add_argument("--alpha_freeze_ratio", type=float, default=0.1, help="Fraction of total steps to freeze alpha params at the start.")
    parser.add_argument("--alpha_init", type=float, default=1.0, help="Initial bounded alpha value for unilora_learnable_layer.")
    parser.add_argument("--alpha_min", type=float, default=0.5, help="Lower bound for alpha in unilora_learnable_layer.")
    parser.add_argument("--alpha_max", type=float, default=1.5, help="Upper bound for alpha in unilora_learnable_layer.")
    parser.add_argument("--unilora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--multi_hashing_num_components",
        type=int,
        default=4,
        help="Number of independently initialized (P_i, theta_d_i) pairs for unilora_multi_hashing.",
    )
    parser.add_argument(
        "--multi_hashing_init_p_bound",
        type=float,
        default=None,
        help="Uniform init half-width around 1 / num_components for each P_i in unilora_multi_hashing.",
    )
    parser.add_argument("--multi_structured_num_hash_pairs", type=int, default=4, help="M in sum-of-products M_hat.")
    parser.add_argument(
        "--multi_structured_target_trainable_params",
        type=int,
        default=None,
        help="Optional trainable-parameter budget for unilora_multi_structured.",
    )
    parser.add_argument(
        "--multi_structured_layerwise_learnable_scale",
        action="store_true",
        help="Enable per-layer learnable scale for unilora_multi_structured.",
    )
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
    parser.add_argument("--sws_num_components", type=int, default=16, help="Mixture component count for unilora_soft_weight_sharing")
    parser.add_argument("--sws_tau", type=float, default=1e-4, help="Soft weight-sharing loss coefficient for unilora_soft_weight_sharing")
    parser.add_argument(
        "--sws_grouping",
        type=str,
        default="global",
        choices=["global", "per_layer", "ab_split"],
        help="Grouping strategy for unilora_soft_weight_sharing",
    )
    parser.add_argument("--sws_sigma_floor", type=float, default=1e-4, help="Sigma floor for mixture stability in unilora_soft_weight_sharing")
    parser.add_argument("--sws_warmup_ratio", type=float, default=0.1, help="Warmup ratio for ramping sws_tau")
    parser.add_argument(
        "--sws_no_zero_component",
        action="store_true",
        help="Disable fixed zero mixture component in unilora_soft_weight_sharing",
    )
    parser.add_argument(
        "--sws_assign_stage",
        type=str,
        default="end",
        choices=["none", "end"],
        help="When to harden assignment for unilora_soft_weight_sharing",
    )
    parser.add_argument("--sws_merge_threshold", type=float, default=0.0, help="Reserved merge threshold for unilora_soft_weight_sharing")
    parser.add_argument(
        "--result_suffix",
        type=str,
        default=None,
        help=(
            "Optional suffix appended to the saved JSON filename stem to avoid sweep overwrites. "
            "If unset and variant is unilora_soft_weight_sharing, a suffix is auto-generated from "
            "sws_num_components, sws_tau, and sws_grouping."
        ),
    )
    parser.add_argument("--deepk_num_clusters_a", type=int, default=16, help="Cluster count for A-column DeepK regularization.")
    parser.add_argument("--deepk_num_clusters_b", type=int, default=16, help="Cluster count for B-row DeepK regularization.")
    parser.add_argument("--deepk_tau", type=float, default=1e-4, help="DeepK regularization coefficient.")
    parser.add_argument(
        "--deepk_f_update_interval",
        type=int,
        default=100,
        help="Refresh spectral auxiliary assignments every N steps for unilora_deepk.",
    )
    parser.add_argument("--deepk_warmup_ratio", type=float, default=0.1, help="Warmup ratio for ramping DeepK regularization.")
    parser.add_argument(
        "--deepk_assign_stage",
        type=str,
        default="none",
        choices=["none", "end"],
        help="When to finalize hard assignment for unilora_deepk.",
    )
    parser.add_argument(
        "--deepk_svd_rank_cap",
        type=int,
        default=0,
        help="Optional cap for spectral rank in unilora_deepk (<=0 disables cap).",
    )
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

    args = parser.parse_args()
    if args.rank is None:
        args.rank = 1 if args.variant == "unilora_aroma" else 4
    return args


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


def get_unilora_rosa_backend(model, variant):
    if variant not in {
        "unilora_rosa",
        "unilora_rosa_stage",
        "unilora_rosa_stage_snip",
        "unilora_rosa_discrete",
        "unilora_rosa_global",
        "unilora_rosa_compression",
    }:
        return None
    backend = getattr(model, "base_model", None)
    if backend is None:
        return None
    required_methods = (
        "enable_gradient_capture",
        "accumulate_gradient_statistics",
        "should_collect_gradients",
        "should_generate_masks",
        "generate_sparse_masks",
        "get_sparse_structure_stats",
    )
    if not all(hasattr(backend, method) for method in required_methods):
        return None
    return backend


def get_unilora_aroma_backend(model, variant):
    if variant != "unilora_aroma":
        return None
    backend = getattr(model, "base_model", None)
    if backend is None:
        return None
    required_methods = ("merge_and_reinit",)
    if not all(hasattr(backend, method) for method in required_methods):
        return None
    return backend


def get_unilora_igu_backend(model, variant):
    if variant != "unilora_igu":
        return None
    backend = getattr(model, "base_model", None)
    if backend is None:
        return None
    required_methods = (
        "enable_gradient_capture",
        "set_total_step",
        "should_update_importance",
        "accumulate_rank_statistics",
        "update_and_mask",
        "get_rank_structure_stats",
        "set_weight_coeffs",
        "compute_orth_regu",
    )
    if not all(hasattr(backend, method) for method in required_methods):
        return None
    return backend


def sample_igu_weight_coeff():
    values = [w / 20.0 for w in range(1, 21)]
    return random.choice(values)


def get_unilora_swap_backend(model, variant):
    if variant != "unilora_swap":
        return None
    backend = getattr(model, "base_model", None)
    if backend is None:
        return None
    required_methods = ("perform_swap", "refresh_unilora_scales")
    if not all(hasattr(backend, method) for method in required_methods):
        return None
    return backend


def get_unilora_local_swap_backend(model, variant):
    if variant != "unilora_local_swap":
        return None
    backend = getattr(model, "base_model", None)
    if backend is None:
        return None
    required_methods = (
        "enable_gradient_capture",
        "accumulate_local_swap_statistics",
        "perform_local_swap",
    )
    if not all(hasattr(backend, method) for method in required_methods):
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


def build_result_json_stem(variant: str, task: str, model_name: str, head_lr, seed: int, args) -> str:
    stem = f"{variant}_{task}_{model_name}_lr{head_lr}_seed{seed}"
    suffix = getattr(args, "result_suffix", None)
    if suffix is None and variant == "unilora_soft_weight_sharing":
        tau_s = str(args.sws_tau).replace(".", "p").replace("-", "neg")
        suffix = f"K{args.sws_num_components}_tau{tau_s}_{args.sws_grouping}"
    if suffix is None and variant == "geo_unilora":
        suffix = (
            f"g{getattr(args, 'geo_num_groups', 8)}_sr{str(getattr(args, 'geo_shared_ratio', 0.5)).replace('.', 'p')}"
                f"_id{getattr(args, 'geo_id_estimator', 'prank')}_cal{getattr(args, 'geo_calibration_steps', 16)}"
        )
    if suffix is None and variant == "igu_unilora":
        suffix = (
            f"sr{str(getattr(args, 'igu_shared_ratio', 0.7)).replace('.', 'p')}"
            f"_b{getattr(args, 'igu_beta1', 0.85)}_{getattr(args, 'igu_beta2', 0.85)}"
            f"_cal{getattr(args, 'igu_calibration_steps', 16)}"
        )
    if suffix is None and variant == "unilora_igu":
        suffix = (
            f"tr{getattr(args, 'igu_target_rank', 2)}"
            f"_b{getattr(args, 'igu_beta1', 0.85)}_{getattr(args, 'igu_beta2', 0.85)}"
            f"_w{getattr(args, 'igu_init_warmup', 100)}_{getattr(args, 'igu_final_warmup', 100)}"
            f"_m{getattr(args, 'igu_mask_interval', 50)}"
        )
    if suffix is None and variant == "unilora_aroma":
        suffix = f"t{getattr(args, 'aroma_t_in', 100)}_r{getattr(args, 'rank', 1)}"
    if suffix:
        stem = f"{stem}_{suffix}"
    return stem


def should_accept_structure_update(baseline_score, baseline_loss, trial_score, trial_loss, tolerance):
    if trial_score > baseline_score + tolerance:
        return True
    return trial_score >= baseline_score - tolerance and trial_loss <= baseline_loss + tolerance


def _compute_gora_importance(weight, grad, importance_type: str) -> float:
    weight = weight.float()
    grad = grad.float()
    if importance_type == "union_mean":
        return torch.mean(torch.abs(weight * grad)).item()
    if importance_type == "union_frobenius_norm":
        return torch.linalg.matrix_norm(weight * grad).item()
    if importance_type == "union_nuc_norm":
        return torch.linalg.matrix_norm(weight * grad, ord="nuc").item()
    if importance_type == "grad_mean":
        return torch.mean(torch.abs(grad)).item()
    if importance_type == "grad_frobenius_norm":
        return torch.linalg.matrix_norm(grad).item()
    raise ValueError(f"Unsupported GoRA importance type: {importance_type}")


def _resolve_gora_feature_func(name: str):
    if name == "sqrt":
        return math.sqrt
    if name == "log1p":
        return math.log1p
    return lambda x: x


def _select_gora_targets(model, target_modules):
    targets = []
    if target_modules == "all-linear":
        for module_name, module in model.named_modules():
            if isinstance(module, (nn.Linear, Conv1D)):
                if module_name.endswith(("classifier", "score")):
                    continue
                targets.append((module_name, module))
        return targets

    for module_name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, Conv1D)):
            continue
        if any(module_name.endswith(suffix) for suffix in target_modules):
            targets.append((module_name, module))
    return targets


def compute_gora_rank_map(base_model, train_loader, args, target_modules, device: str):
    targets = _select_gora_targets(base_model, target_modules)
    if not targets:
        raise ValueError("No GoRA target modules found for rank allocation.")

    prev_requires_grad = {name: param.requires_grad for name, param in base_model.named_parameters()}
    for _, param in base_model.named_parameters():
        param.requires_grad = False
    for _, module in targets:
        module.weight.requires_grad = True

    base_model.to(device)
    base_model.train()

    grad_sums = {name: torch.zeros_like(module.weight, device="cpu") for name, module in targets}
    weight_cache = {name: module.weight.detach().cpu() for name, module in targets}
    steps = 0

    for batch in train_loader:
        if steps >= args.gora_gradient_est_steps:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = base_model(**batch)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        loss.backward()

        for name, module in targets:
            grad = module.weight.grad
            if grad is not None:
                grad_sums[name].add_(grad.detach().cpu())

        base_model.zero_grad(set_to_none=True)
        steps += 1

    for name, param in base_model.named_parameters():
        param.requires_grad = prev_requires_grad.get(name, param.requires_grad)

    if steps == 0:
        raise ValueError("GoRA gradient estimation collected 0 steps.")

    importances = {}
    features = {}
    for name, module in targets:
        grad = grad_sums[name] / float(steps)
        weight = weight_cache[name]
        importance = _compute_gora_importance(weight, grad, args.gora_importance_type)
        importances[name] = importance
        if isinstance(module, nn.Linear):
            feature_dim = module.in_features + module.out_features
        else:
            weight_shape = module.weight.shape
            feature_dim = int(weight_shape[0] + weight_shape[1])
        features[name] = feature_dim

    importances_tensor = torch.tensor(list(importances.values()), dtype=torch.float32)
    if args.gora_softmax_importance:
        min_val = importances_tensor.min()
        max_val = importances_tensor.max()
        denom = (max_val - min_val).clamp_min(1e-12)
        scaled = (importances_tensor - min_val) / denom / args.gora_temperature
        normalized = torch.softmax(scaled, dim=0)
    else:
        normalized = importances_tensor / importances_tensor.sum().clamp_min(1e-12)

    allocate_func = {
        "radical": math.ceil,
        "moderate": round,
        "conserved": math.floor,
    }.get(args.gora_allocate_strategy, round)
    feature_func = _resolve_gora_feature_func(args.gora_features_func)
    smooth_total_budget = sum(feature_func(value) * args.rank for value in features.values())

    rank_map = {}
    for name, normalized_importance in zip(importances.keys(), normalized.tolist()):
        feature_value = feature_func(features[name])
        if feature_value <= 0:
            rank = args.rank
        else:
            smooth_trainable = allocate_func(smooth_total_budget * normalized_importance)
            rank = int(smooth_trainable // feature_value)
        rank = max(rank, 1)
        if args.gora_min_rank is not None:
            rank = max(rank, args.gora_min_rank)
        if args.gora_max_rank is not None:
            rank = min(rank, args.gora_max_rank)
        rank_map[name] = int(rank)

    return rank_map, importances, steps


def _pool_hidden_states(hidden_states: torch.Tensor, attention_mask) -> torch.Tensor:
    if attention_mask is None:
        return hidden_states.mean(dim=1)
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (hidden_states * mask).sum(dim=1) / denom


def _estimate_intrinsic_dimension_twonn_single(representations: torch.Tensor) -> float:
    if representations.size(0) < 3:
        return float(representations.size(1))
    distances = torch.cdist(representations.float(), representations.float(), p=2)
    distances.fill_diagonal_(float("inf"))
    nearest, _ = torch.topk(distances, k=2, largest=False)
    r1 = nearest[:, 0].clamp_min(1e-12)
    r2 = nearest[:, 1].clamp_min(1e-12)
    mu = (r2 / r1).clamp_min(1.0 + 1e-12)
    log_mu = torch.log(mu)
    denom = log_mu.sum().clamp_min(1e-12)
    return float(log_mu.numel() / denom)


def _estimate_intrinsic_dimension_twonn(representations: torch.Tensor) -> float:
    sample_count = representations.size(0)
    if sample_count < 3:
        return float(representations.size(1))

    min_subset_size = min(sample_count, 128)
    subset_size = sample_count
    permutation = torch.randperm(sample_count)
    shuffled = representations[permutation]

    estimates = []
    while subset_size >= min_subset_size:
        subset = shuffled[:subset_size]
        estimates.append((subset_size, _estimate_intrinsic_dimension_twonn_single(subset)))
        if subset_size == min_subset_size:
            break
        next_size = max(subset_size // 2, min_subset_size)
        if next_size == subset_size:
            break
        subset_size = next_size

    if len(estimates) == 1:
        return estimates[0][1]

    values = [value for _, value in estimates]
    best_pair_idx = min(range(1, len(values)), key=lambda idx: abs(values[idx] - values[idx - 1]))
    return float((values[best_pair_idx] + values[best_pair_idx - 1]) / 2.0)


def _extract_layer_idx(module_name: str):
    patterns = [
        r"\.layer\.(\d+)\.",
        r"\.layers\.(\d+)\.",
        r"\.h\.(\d+)\.",
        r"\.block\.(\d+)\.",
    ]
    for pattern in patterns:
        match = re.search(pattern, module_name)
        if match:
            return int(match.group(1))
    return None


def compute_gelora_rank_map(base_model, train_loader, args, target_modules, device: str):
    targets = _select_gora_targets(base_model, target_modules)
    if not targets:
        raise ValueError("No GeLoRA target modules found for rank allocation.")

    was_training = base_model.training
    base_model.to(device)
    base_model.eval()

    total_samples = 0
    layer_chunks = None

    with torch.no_grad():
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = base_model(**batch, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states
            attention_mask = batch.get("attention_mask")
            pooled_states = [_pool_hidden_states(hs, attention_mask) for hs in hidden_states]

            if layer_chunks is None:
                layer_chunks = [[] for _ in pooled_states]

            for idx, pooled in enumerate(pooled_states):
                layer_chunks[idx].append(pooled.detach().cpu())

            total_samples += pooled_states[0].size(0)

    if not layer_chunks:
        raise ValueError("GeLoRA ID estimation collected 0 samples.")

    if was_training:
        base_model.train()

    layer_reprs = [torch.cat(chunks, dim=0) for chunks in layer_chunks]
    intrinsic_dims = [_estimate_intrinsic_dimension_twonn(reprs) for reprs in layer_reprs]

    layer_ranks = []
    for idx in range(len(intrinsic_dims) - 1):
        diff = intrinsic_dims[idx + 1] - intrinsic_dims[idx]
        rank_value = max(diff, 0.0) + float(args.gelora_rank_offset)
        rank = max(int(math.ceil(rank_value)), 1)
        layer_ranks.append(rank)

    rank_map = {}
    for name, _module in targets:
        layer_idx = _extract_layer_idx(name)
        if layer_idx is None or layer_idx >= len(layer_ranks):
            rank = args.rank
        else:
            rank = layer_ranks[layer_idx]
        rank_map[name] = int(rank)

    return rank_map, intrinsic_dims, total_samples


def _geo_sigma_from_H(H: torch.Tensor) -> torch.Tensor:
    """H: [N, d] — returns PSD covariance S = H^T H / N."""
    H = H.float()
    n = H.shape[0]
    if n < 1:
        raise ValueError("Geo-UniLoRA: empty activation matrix.")
    return (H.T @ H) / float(n)


def _geo_prank_from_sigma(S: torch.Tensor) -> float:
    tr_s = torch.trace(S)
    tr_s2 = torch.sum(S * S)
    if tr_s2 < 1e-18:
        return 1.0
    return float((tr_s * tr_s / tr_s2).item())


def _geo_erank_from_sigma(S: torch.Tensor) -> float:
    w = torch.linalg.eigvalsh(S)
    w = torch.clamp(w, min=0.0)
    s = w.sum()
    if s < 1e-18:
        return 1.0
    p = w / s
    p = p[p > 1e-18]
    if p.numel() == 0:
        return 1.0
    return float(torch.exp(-(p * torch.log(p)).sum()).item())


def _geo_linear_cka_h(H1: torch.Tensor, H2: torch.Tensor) -> float:
    """
    Linear CKA between two row-aligned activation batches (same N, possibly different d).
    Replaces Frobenius cosine on covariances, which requires matching matrix shapes.
    CKA(H1, H2) = ||H2^T H1||_F^2 / (||H1^T H1||_F ||H2^T H2||_F).
    """
    X = H1.float()
    Y = H2.float()
    n = min(X.shape[0], Y.shape[0])
    if n < 1:
        return 0.0
    X = X[:n]
    Y = Y[:n]
    XtX = X.T @ X
    YtY = Y.T @ Y
    YtX = Y.T @ X
    num = torch.sum(YtX * YtX)
    den = torch.linalg.norm(XtX, ord="fro") * torch.linalg.norm(YtY, ord="fro") + 1e-18
    return float((num / den).item())


def _geo_group_layer_block(names: list[str], num_groups: int) -> dict[str, int]:
    names = sorted(names, key=lambda nm: (_extract_layer_idx(nm) if _extract_layer_idx(nm) is not None else -1, nm))
    n = len(names)
    if n == 0:
        return {}
    gcount = max(1, min(num_groups, n))
    out: dict[str, int] = {}
    for i, name in enumerate(names):
        gid = min(gcount - 1, int(i * gcount / n))
        out[name] = int(gid)
    return out


def _geo_group_spectral(names: list[str], S_mat: torch.Tensor, num_groups: int) -> dict[str, int]:
    try:
        from sklearn.cluster import SpectralClustering
    except ImportError:
        return _geo_group_layer_block(names, num_groups)

    m = len(names)
    if m <= 1 or num_groups <= 1:
        return {names[0]: 0} if names else {}
    sim = S_mat.cpu().numpy()
    sim = (sim + sim.T) / 2.0
    np.fill_diagonal(sim, 1.0)
    labels = SpectralClustering(
        n_clusters=min(num_groups, m),
        affinity="precomputed",
        random_state=0,
    ).fit_predict(sim)
    return {names[i]: int(labels[i]) for i in range(m)}


def _geo_integer_budget_fix(values: list[int], target: int, floor_min: int = 1) -> list[int]:
    """Adjust non-negative integers so sum equals target, each >= floor_min when possible."""
    if not values:
        return values
    n = len(values)
    target = max(target, n * floor_min)
    out = [max(floor_min, int(v)) for v in values]
    s = sum(out)
    if s == target:
        return out
    if s > target:
        # decrease from largest
        while s > target and max(out) > floor_min:
            j = max(range(n), key=lambda i: out[i])
            if out[j] > floor_min:
                out[j] -= 1
                s -= 1
            else:
                break
        while s > target:
            j = max(range(n), key=lambda i: out[i])
            if out[j] > 0:
                out[j] -= 1
                s -= 1
            else:
                break
        return out
    # s < target: increase smallest
    while s < target:
        j = min(range(n), key=lambda i: out[i])
        out[j] += 1
        s += 1
    return out


def compute_geo_unilora_plan(
    base_model,
    train_loader,
    args,
    target_modules,
    device: str,
):
    """
    Returns:
        group_map, shared_rank_map, innovation_rank_map, stats dict
    """
    targets = _select_gora_targets(base_model, target_modules)
    if not targets:
        raise ValueError("No Geo-UniLoRA target modules found.")

    storage_in: dict[str, list[torch.Tensor]] = {name: [] for name, _ in targets}
    storage_out: dict[str, list[torch.Tensor]] = {name: [] for name, _ in targets}

    def _hook(name: str):
        def hook(module, inp, out):
            x = inp[0] if inp else None
            if isinstance(x, torch.Tensor):
                if x.dim() == 3:
                    h_in = x.detach().float().mean(dim=1)
                else:
                    h_in = x.detach().float()
                storage_in[name].append(h_in.cpu())

            if isinstance(out, torch.Tensor):
                if out.dim() == 3:
                    h_out = out.detach().float().mean(dim=1)
                else:
                    h_out = out.detach().float()
                storage_out[name].append(h_out.cpu())

        return hook

    handles = [module.register_forward_hook(_hook(name)) for name, module in targets]
    base_model.eval()
    base_model.to(device)
    steps = 0
    with torch.no_grad():
        for batch in train_loader:
            if steps >= args.geo_calibration_steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            base_model(**batch)
            steps += 1
    for h in handles:
        h.remove()

    names = [name for name, _ in targets]
    H_sim_map: dict[str, torch.Tensor] = {}
    d_hat: dict[str, int] = {}

    for name in names:
        if not storage_in[name]:
            raise ValueError(f"Geo-UniLoRA: no activations collected for {name}.")
        H_in = torch.cat(storage_in[name], dim=0)
        sigma_in = _geo_sigma_from_H(H_in)
        if args.geo_id_estimator == "erank":
            id_in = _geo_erank_from_sigma(sigma_in)
        else:
            id_in = _geo_prank_from_sigma(sigma_in)

        id_out = id_in
        if storage_out[name]:
            H_out = torch.cat(storage_out[name], dim=0)
            sigma_out = _geo_sigma_from_H(H_out)
            if args.geo_id_estimator == "erank":
                id_out = _geo_erank_from_sigma(sigma_out)
            else:
                id_out = _geo_prank_from_sigma(sigma_out)
            H_sim_map[name] = H_out
        else:
            H_sim_map[name] = H_in

        d_hat[name] = max(1, int(math.ceil(float(args.geo_gamma) * float(max(id_in, id_out)))))

    m = len(names)
    S_mat = torch.zeros(m, m, dtype=torch.float64)
    for i in range(m):
        for j in range(m):
            S_mat[i, j] = _geo_linear_cka_h(H_sim_map[names[i]], H_sim_map[names[j]])

    if args.geo_grouping == "spectral":
        group_map = _geo_group_spectral(names, S_mat, args.geo_num_groups)
    else:
        group_map = _geo_group_layer_block(names, args.geo_num_groups)

    unique_groups = sorted(set(group_map.values()))
    # Raw shared dimension per group (c_g)
    c_raw: dict[int, float] = {}
    eps = float(args.geo_eps)
    alpha = float(args.geo_alpha)
    for g in unique_groups:
        mods = [nm for nm in names if group_map[nm] == g]
        num = 0.0
        den = 0.0
        for mi in mods:
            i = names.index(mi)
            for mj in mods:
                j = names.index(mj)
                num += float(S_mat[i, j].item()) * float(min(d_hat[mi], d_hat[mj]))
                den += float(S_mat[i, j].item())
        c_g = alpha * num / (den + eps) if den > 0 else 1.0
        c_g = min(float(c_g), float(min(d_hat[nm] for nm in mods)))
        c_raw[g] = max(1.0, float(c_g))

    total_budget = args.geo_total_budget
    r_min = int(args.geo_r_min)
    default_shared_rank = max(1, int(args.rank) // 2)
    default_innov_rank = max(r_min, int(args.rank) - default_shared_rank)

    # Target shared vs innovation rank mass: sum_g c_g = B_sh, sum_m r_in = B_res.
    if total_budget is None:
        B_sh = len(unique_groups) * default_shared_rank
        B_res = m * default_innov_rank
        total_budget = B_sh + B_res
    else:
        total_budget = int(total_budget)
        B_sh = int(round(float(total_budget) * float(args.geo_shared_ratio)))
        B_sh = max(len(unique_groups), B_sh)
        B_res = total_budget - B_sh
        min_innov_total = m * r_min
        if B_res < min_innov_total:
            B_res = min_innov_total
            B_sh = total_budget - B_res
        if B_sh < len(unique_groups):
            B_sh = len(unique_groups)
            B_res = total_budget - B_sh

    group_sizes = {g: len([nm for nm in names if group_map[nm] == g]) for g in unique_groups}
    raw_shared_sum = sum(c_raw[g] for g in unique_groups)
    scale_sh = B_sh / raw_shared_sum if raw_shared_sum > 1e-12 else 1.0

    c_g_int: dict[int, int] = {}
    for g in unique_groups:
        c_g_int[g] = max(1, int(math.floor(c_raw[g] * scale_sh)))

    def _shared_mass(cdict: dict[int, int]) -> int:
        return sum(cdict[g] for g in unique_groups)

    # Adjust c_g_int so sum_g c_g = B_sh
    while _shared_mass(c_g_int) > B_sh:
        gmax = max(unique_groups, key=lambda gg: c_g_int[gg])
        if c_g_int[gmax] > 1:
            c_g_int[gmax] -= 1
        else:
            break
    while _shared_mass(c_g_int) < B_sh:
        gmin = min(unique_groups, key=lambda gg: c_g_int[gg])
        c_g_int[gmin] += 1

    shared_rank_map = {name: int(c_g_int[group_map[name]]) for name in names}

    # Innovation allocation: sum_m r_in,m = B_res
    u = {name: max(0.0, float(d_hat[name]) - float(c_g_int[group_map[name]])) for name in names}
    zw = sum((u[nm] + eps) ** float(args.geo_tau) for nm in names)
    innov_vals = []
    if zw < 1e-18:
        innov_vals = [r_min] * m
    else:
        extra = B_res - m * r_min
        extra = max(0, extra)
        for nm in names:
            w = (u[nm] + eps) ** float(args.geo_tau) / zw
            innov_vals.append(r_min + int(math.floor(extra * w)))
    innov_fixed = _geo_integer_budget_fix(innov_vals, B_res, floor_min=r_min)
    innovation_rank_map = {names[i]: int(innov_fixed[i]) for i in range(m)}

    stats = {
        "calibration_steps": steps,
        "num_modules": m,
        "geo_total_budget": int(total_budget),
        "B_sh_target": int(B_sh),
        "B_sh_actual": int(_shared_mass(c_g_int)),
        "B_res_target": int(B_res),
        "B_res_actual": int(sum(innovation_rank_map.values())),
        "grouping": args.geo_grouping,
        "num_groups": len(unique_groups),
        "d_hat": d_hat,
        "group_map": {k: int(v) for k, v in group_map.items()},
        "c_g": {str(k): int(v) for k, v in c_g_int.items()},
        "id_estimator": args.geo_id_estimator,
    }
    return group_map, shared_rank_map, innovation_rank_map, stats


def compute_igu_unilora_plan(
    base_model,
    train_loader,
    args,
    target_modules,
    device: str,
):
    targets = _select_gora_targets(base_model, target_modules)
    if not targets:
        raise ValueError("No IGU-UniLoRA target modules found.")

    prev_requires_grad = {name: param.requires_grad for name, param in base_model.named_parameters()}
    for _, param in base_model.named_parameters():
        param.requires_grad = False
    for _, module in targets:
        module.weight.requires_grad = True

    base_model.to(device)
    base_model.train()

    beta1 = float(args.igu_beta1)
    beta2 = float(args.igu_beta2)
    eps = float(args.igu_eps)
    score_agg = str(args.igu_score_agg)

    ema_ipt = {name: 0.0 for name, _ in targets}
    ema_unc = {name: 0.0 for name, _ in targets}
    raw_latest = {name: 0.0 for name, _ in targets}
    steps = 0

    def _aggregate_proxy(t: torch.Tensor, mode: str) -> float:
        if mode == "fro":
            return float(torch.linalg.matrix_norm(t).item())
        if mode == "max":
            return float(torch.max(torch.abs(t)).item())
        return float(torch.mean(torch.abs(t)).item())

    for batch in train_loader:
        if steps >= int(args.igu_calibration_steps):
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = base_model(**batch)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        loss.backward()

        with torch.no_grad():
            for name, module in targets:
                grad = module.weight.grad
                if grad is None:
                    continue
                proxy = module.weight.detach() * grad.detach()
                raw_score = _aggregate_proxy(proxy, score_agg)
                raw_latest[name] = raw_score
                ema_ipt[name] = beta1 * ema_ipt[name] + (1.0 - beta1) * raw_score
                ema_unc[name] = beta2 * ema_unc[name] + (1.0 - beta2) * abs(raw_score - ema_ipt[name])

        base_model.zero_grad(set_to_none=True)
        steps += 1

    for name, param in base_model.named_parameters():
        param.requires_grad = prev_requires_grad.get(name, param.requires_grad)

    if steps == 0:
        raise ValueError("IGU-UniLoRA calibration collected 0 steps.")

    names = [name for name, _ in targets]
    module_scores = {name: float(ema_ipt[name] / (ema_unc[name] + eps)) for name in names}
    module_scores = {k: (v if math.isfinite(v) else 0.0) for k, v in module_scores.items()}

    m = len(names)
    total_budget = args.igu_total_budget
    if total_budget is None:
        total_budget = m * int(args.rank)
    total_budget = int(total_budget)

    r_min = max(0, int(args.igu_r_min))
    B_sh = int(round(float(total_budget) * float(args.igu_shared_ratio)))
    B_sh = max(m, min(B_sh, total_budget))
    B_res = total_budget - B_sh

    min_res_total = m * r_min
    if B_res < min_res_total:
        B_res = min_res_total
        B_sh = total_budget - B_res
    if B_sh < m:
        B_sh = m
        B_res = total_budget - B_sh

    shared_seed = [B_sh // m] * m
    for i in range(B_sh % m):
        shared_seed[i] += 1
    shared_rank_map = {names[i]: int(shared_seed[i]) for i in range(m)}

    w_den = 0.0
    tau = float(args.igu_tau)
    for nm in names:
        w_den += (module_scores[nm] + eps) ** tau
    if w_den < 1e-18:
        innovation_vals = [r_min] * m
    else:
        extra = max(0, B_res - m * r_min)
        innovation_vals = []
        for nm in names:
            w = ((module_scores[nm] + eps) ** tau) / w_den
            innovation_vals.append(r_min + int(math.floor(extra * w)))
    innovation_vals = _geo_integer_budget_fix(innovation_vals, B_res, floor_min=r_min)
    innovation_rank_map = {names[i]: int(innovation_vals[i]) for i in range(m)}

    group_map = {name: 0 for name in names}
    stats = {
        "calibration_steps": steps,
        "num_modules": m,
        "igu_total_budget": int(total_budget),
        "B_sh_actual": int(sum(shared_rank_map.values())),
        "B_res_actual": int(sum(innovation_rank_map.values())),
        "score_mode": args.igu_score_mode,
        "score_agg": score_agg,
        "beta1": beta1,
        "beta2": beta2,
        "module_scores": module_scores,
        "module_raw_latest": raw_latest,
    }
    return group_map, shared_rank_map, innovation_rank_map, stats


def main():
    args = parse_args()
    set_seed(args.seed)

    model_name = args.model_name
    task = args.task
    variant = args.variant

    batch_size = args.batch_size
    max_length = MAX_LENGTH[model_name]
    num_epochs = args.num_epochs if args.num_epochs is not None else EPOCHS[model_name][task]
    warmup_ratio = args.warmup_ratio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric_name = TASK_TO_METRIC[task]

    # Variant-specific defaults for stability
    current_init_bound = args.init_theta_d_bound
    theta_d_lr = args.theta_d_lr
    rosa_sparse_lr = args.rosa_sparse_lr if args.rosa_sparse_lr is not None else theta_d_lr
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

    target_modules = ["query", "key", "value", "output.dense", "intermediate.dense"]
    gelora_target_modules = ["query", "key", "value", "attention.output.dense"]
    gora_rank_map = None
    if variant == "unilora_gora":
        gora_rank_map, gora_importances, gora_steps = compute_gora_rank_map(
            base_model=base_model,
            train_loader=train_loader,
            args=args,
            target_modules=target_modules,
            device=device,
        )
        args.gora_rank_map = gora_rank_map
        ranks = list(gora_rank_map.values())
        print(
            "GoRA rank allocation: "
            f"steps={gora_steps}, modules={len(ranks)}, "
            f"min={min(ranks)}, max={max(ranks)}, mean={sum(ranks)/len(ranks):.2f}"
        )
    gelora_rank_map = None
    if variant == "unilora_gelora":
        gelora_rank_map, gelora_dims, gelora_samples = compute_gelora_rank_map(
            base_model=base_model,
            train_loader=train_loader,
            args=args,
            target_modules=gelora_target_modules,
            device=device,
        )
        args.gelora_rank_map = gelora_rank_map
        args.gelora_intrinsic_dims = gelora_dims
        ranks = list(gelora_rank_map.values())
        print(
            "GeLoRA rank allocation: "
            f"samples={gelora_samples}, modules={len(ranks)}, "
            f"min={min(ranks)}, max={max(ranks)}, mean={sum(ranks)/len(ranks):.2f}"
        )

    geo_group_map = None
    geo_shared_rank_map = None
    geo_innovation_rank_map = None
    geo_plan_stats = None
    igu_group_map = None
    igu_shared_rank_map = None
    igu_innovation_rank_map = None
    igu_plan_stats = None
    if variant == "geo_unilora":
        geo_group_map, geo_shared_rank_map, geo_innovation_rank_map, geo_plan_stats = compute_geo_unilora_plan(
            base_model=base_model,
            train_loader=train_loader,
            args=args,
            target_modules=target_modules,
            device=device,
        )
        args.geo_plan_stats = geo_plan_stats
        sr = list(geo_shared_rank_map.values())
        ir = list(geo_innovation_rank_map.values())
        print(
            "Geo-UniLoRA plan: "
            f"modules={len(sr)}, shared_ranks min/max/mean={min(sr)}/{max(sr)}/{sum(sr)/len(sr):.2f}, "
            f"innov_ranks min/max/mean={min(ir)}/{max(ir)}/{sum(ir)/len(ir):.2f}, "
            f"B_sh={geo_plan_stats.get('B_sh_actual')}, B_res={geo_plan_stats.get('B_res_actual')}"
        )
    if variant == "igu_unilora":
        igu_group_map, igu_shared_rank_map, igu_innovation_rank_map, igu_plan_stats = compute_igu_unilora_plan(
            base_model=base_model,
            train_loader=train_loader,
            args=args,
            target_modules=target_modules,
            device=device,
        )
        args.igu_plan_stats = igu_plan_stats
        sr = list(igu_shared_rank_map.values())
        ir = list(igu_innovation_rank_map.values())
        print(
            "IGU-UniLoRA plan: "
            f"modules={len(sr)}, shared_ranks min/max/mean={min(sr)}/{max(sr)}/{sum(sr)/len(sr):.2f}, "
            f"innov_ranks min/max/mean={min(ir)}/{max(ir)}/{sum(ir)/len(ir):.2f}, "
            f"B_sh={igu_plan_stats.get('B_sh_actual')}, B_res={igu_plan_stats.get('B_res_actual')}"
        )

    if variant == "lora":
        peft_config = LoraConfig(
            task_type="SEQ_CLS",
            r=args.rank,
            lora_alpha=args.rank,
            lora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_aroma":
        peft_config = UniLoRAAromaConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_AROMA,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            aroma_t_in=args.aroma_t_in,
            aroma_reset_optimizer_on_merge=args.aroma_reset_optimizer_on_merge,
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
    elif variant == "unilora_soft_weight_sharing":
        peft_config = UniLoRASoftWeightSharingConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_SOFT_WEIGHT_SHARING,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            num_candidates=args.num_candidates,
            assignment_mode=args.soft_assign_mode,
            temperature=args.soft_assign_temperature,
            gumbel_hard=args.soft_assign_gumbel_hard,
            hard_eval=not args.soft_assign_soft_eval,
            init_logits_std=args.init_logits_std,
            init_primary_bias=args.init_primary_bias,
            num_components=args.sws_num_components,
            sharing_tau=args.sws_tau,
            sharing_grouping=args.sws_grouping,
            sharing_zero_component=not args.sws_no_zero_component,
            sharing_sigma_floor=args.sws_sigma_floor,
            sharing_warmup_ratio=args.sws_warmup_ratio,
            sharing_assign_stage=args.sws_assign_stage,
            sharing_merge_threshold=args.sws_merge_threshold,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_deepk":
        peft_config = UniLoRADeepKConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_DEEPK,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            num_candidates=args.num_candidates,
            assignment_mode=args.soft_assign_mode,
            temperature=args.soft_assign_temperature,
            gumbel_hard=args.soft_assign_gumbel_hard,
            hard_eval=not args.soft_assign_soft_eval,
            init_logits_std=args.init_logits_std,
            init_primary_bias=args.init_primary_bias,
            deepk_num_clusters_a=args.deepk_num_clusters_a,
            deepk_num_clusters_b=args.deepk_num_clusters_b,
            deepk_tau=args.deepk_tau,
            deepk_f_update_interval=args.deepk_f_update_interval,
            deepk_warmup_ratio=args.deepk_warmup_ratio,
            deepk_assign_stage=args.deepk_assign_stage,
            deepk_svd_rank_cap=args.deepk_svd_rank_cap,
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
            target_modules=target_modules,
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_gora":
        gora_features_func = None if args.gora_features_func == "none" else args.gora_features_func
        peft_config = UniLoRAGoRAConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_GORA,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            gora_rank_map=gora_rank_map,
            gora_importance_type=args.gora_importance_type,
            gora_min_rank=args.gora_min_rank,
            gora_max_rank=args.gora_max_rank,
            gora_allocate_strategy=args.gora_allocate_strategy,
            gora_features_func=gora_features_func,
            gora_softmax_importance=args.gora_softmax_importance,
            gora_temperature=args.gora_temperature,
            gora_gradient_est_steps=args.gora_gradient_est_steps,
            target_modules=target_modules,
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_gelora":
        peft_config = UniLoRAGeLoRAConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_GELORA,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            gelora_rank_map=gelora_rank_map,
            target_modules=gelora_target_modules,
            modules_to_save=["classifier"],
        )
    elif variant == "geo_unilora":
        shared_len = args.geo_shared_theta_d_length or args.theta_d_length
        innov_len = args.geo_innovation_theta_d_length or args.theta_d_length
        peft_config = GeoUniLoRAConfig(
            task_type="SEQ_CLS", peft_type=PeftType.GEO_UNILORA,
            r=args.rank,
            proj_seed=args.seed,
            shared_theta_d_length=shared_len,
            innovation_theta_d_length=innov_len,
            init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            geo_group_map=geo_group_map,
            geo_shared_rank_map=geo_shared_rank_map,
            geo_innovation_rank_map=geo_innovation_rank_map,
            target_modules=target_modules,
            modules_to_save=["classifier"],
        )
    elif variant == "igu_unilora":
        shared_len = args.igu_shared_theta_d_length or args.theta_d_length
        innov_len = args.igu_innovation_theta_d_length or args.theta_d_length
        peft_config = IGUUniLoRAConfig(
            task_type="SEQ_CLS", peft_type=PeftType.IGU_UNILORA,
            r=args.rank,
            proj_seed=args.seed,
            shared_theta_d_length=shared_len,
            innovation_theta_d_length=innov_len,
            init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            geo_group_map=igu_group_map,
            geo_shared_rank_map=igu_shared_rank_map,
            geo_innovation_rank_map=igu_innovation_rank_map,
            target_modules=target_modules,
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_igu":
        peft_config = UniLoRAIGUConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_IGU,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            igu_target_rank=args.igu_target_rank,
            igu_init_warmup=args.igu_init_warmup,
            igu_final_warmup=args.igu_final_warmup,
            igu_mask_interval=args.igu_mask_interval,
            igu_beta1=args.igu_beta1,
            igu_beta2=args.igu_beta2,
            igu_eps=args.igu_eps,
            igu_r_min=args.igu_r_min,
            igu_reset_optimizer_on_mask=args.igu_reset_optimizer_on_mask,
            target_modules=target_modules,
            modules_to_save=["classifier"],
        )
    elif variant == "direct_unilora":
        peft_config = DirectUniLoRAConfig(
            task_type="SEQ_CLS", peft_type=PeftType.DIRECT_UNILORA,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=target_modules,
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
    elif variant == "unilora_rosa":
        peft_config = UniLoRARoSAConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_ROSA,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            rosa_density=args.rosa_density,
            rosa_warmup_steps=args.rosa_warmup_steps,
            rosa_mask_steps=args.rosa_mask_steps,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_rosa_stage":
        peft_config = UniLoRARoSAStageConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_ROSA_STAGE,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            rosa_density=args.rosa_density,
            rosa_stage_ratio=args.rosa_stage_ratio,
            rosa_mask_steps=args.rosa_mask_steps,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_rosa_stage_snip":
        peft_config = UniLoRARoSAStageSnipConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_ROSA_STAGE_SNIP,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            rosa_density=args.rosa_density,
            rosa_stage_ratio=args.rosa_stage_ratio,
            rosa_mask_steps=args.rosa_mask_steps,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_rosa_compression":
        if args.sparse_theta_d_length is None:
            raise ValueError("--sparse_theta_d_length must be specified for unilora_rosa_compression.")
        peft_config = UniLoRARoSACompressionConfig(
            task_type="SEQ_CLS",
            peft_type=PeftType.UNILORA_ROSA_COMPRESSION,
            r=args.rank,
            theta_d_length=args.theta_d_length,
            sparse_theta_d_length=args.sparse_theta_d_length,
            proj_seed=args.seed,
            init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            rosa_density=args.rosa_density,
            rosa_warmup_steps=args.rosa_warmup_steps,
            rosa_mask_steps=args.rosa_mask_steps,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_rosa_discrete":
        peft_config = UniLoRARoSADiscreteConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_ROSA_DISCRETE,
            r=args.rank, theta_d_length=args.theta_d_length,
            sparse_theta_d_length=args.rosa_sparse_theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            rosa_density=args.rosa_density,
            rosa_warmup_steps=args.rosa_warmup_steps,
            rosa_mask_steps=args.rosa_mask_steps,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_rosa_global":
        peft_config = UniLoRARoSAGlobalConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_ROSA_GLOBAL,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            rosa_density=args.rosa_density,
            rosa_warmup_steps=args.rosa_warmup_steps,
            rosa_mask_steps=args.rosa_mask_steps,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_swap":
        peft_config = UniLoRASwapConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_SWAP,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            swap_dead_bucket_count=args.swap_dead_bucket_count,
            swap_split_ratio=args.swap_split_ratio,
            swap_interval_steps=args.swap_interval_steps,
            swap_start_after_steps=args.swap_start_after_steps,
            swap_interval_epochs=args.swap_interval_epochs,
            swap_start_after_epochs=args.swap_start_after_epochs,
            swap_reset_optimizer_state=args.swap_reset_optimizer_state,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_local_swap":
        peft_config = UniLoRALocalSwapConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_LOCAL_SWAP,
            r=args.rank, theta_d_length=args.theta_d_length,
            proj_seed=args.seed, init_theta_d_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            local_swap_grad_ema_momentum=args.local_swap_grad_ema_momentum,
            local_swap_warmup_steps=args.local_swap_warmup_steps,
            local_swap_bad_bucket_frac=args.local_swap_bad_bucket_frac,
            local_swap_candidates_per_bucket=args.local_swap_candidates_per_bucket,
            local_swap_target_bucket_samples=args.local_swap_target_bucket_samples,
            local_swap_min_delta=args.local_swap_min_delta,
            local_swap_max_target_drop=args.local_swap_max_target_drop,
            local_swap_min_bucket_size=args.local_swap_min_bucket_size,
            local_swap_update_ratio=args.local_swap_update_ratio,
            local_swap_interval_steps=args.local_swap_interval_steps,
            local_swap_start_after_steps=args.local_swap_start_after_steps,
            local_swap_interval_epochs=args.local_swap_interval_epochs,
            local_swap_start_after_epochs=args.local_swap_start_after_epochs,
            local_swap_reset_optimizer_state=args.local_swap_reset_optimizer_state,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_multi_structured":
        peft_config = UniLoRAMultiStructuredConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_MULTI_STRUCTURED,
            r=args.rank,
            proj_seed=args.seed,
            num_hash_pairs=args.multi_structured_num_hash_pairs,
            target_trainable_params=args.multi_structured_target_trainable_params,
            layerwise_learnable_scale=args.multi_structured_layerwise_learnable_scale,
            init_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_multi_structured_global":
        peft_config = UniLoRAMultiStructuredGlobalConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_MULTI_STRUCTURED_GLOBAL,
            r=args.rank,
            proj_seed=args.seed,
            num_hash_pairs=args.multi_structured_num_hash_pairs,
            target_trainable_params=args.multi_structured_target_trainable_params,
            layerwise_learnable_scale=args.multi_structured_layerwise_learnable_scale,
            init_bound=current_init_bound,
            unilora_dropout=args.unilora_dropout,
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )
    elif variant == "unilora_multi_hashing":
        peft_config = UniLoRAMultiHashingConfig(
            task_type="SEQ_CLS", peft_type=PeftType.UNILORA_MULTI_HASHING,
            r=args.rank,
            theta_d_length=args.theta_d_length,
            num_hash_components=args.multi_hashing_num_components,
            proj_seed=args.seed,
            init_theta_d_bound=current_init_bound,
            init_p_bound=args.multi_hashing_init_p_bound,
            unilora_dropout=args.unilora_dropout,
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
    unilora_aroma_backend = get_unilora_aroma_backend(model, variant)
    unilora_rosa_backend = get_unilora_rosa_backend(model, variant)
    unilora_igu_backend = get_unilora_igu_backend(model, variant)
    unilora_swap_backend = get_unilora_swap_backend(model, variant)
    unilora_local_swap_backend = get_unilora_local_swap_backend(model, variant)
    if hessian_aware_backend is not None:
        hessian_aware_backend.enable_curvature_capture(True)
    if unilora_rosa_backend is not None:
        unilora_rosa_backend.enable_gradient_capture(False)
    if unilora_igu_backend is not None:
        unilora_igu_backend.enable_gradient_capture(True)
    if unilora_local_swap_backend is not None:
        unilora_local_swap_backend.enable_gradient_capture(True)

    # Adaptive Parameter Grouping
    head_params, theta_d_params, sparse_params, alpha_params = [], [], [], []
    for n, p in model.named_parameters():
        if "unilora_rosa_sparse_theta_D" in n or "unilora_rosa_sparse_theta_d" in n:
            sparse_params.append(p)
            continue
        if "unilora_rosa_discrete_sparse_theta_d" in n:
            sparse_params.append(p)
            continue
        if not p.requires_grad:
            # Force requires_grad if it's a known UniLoRA variant parameter
            if any(term in n for term in ["theta_d", "unilora_layer_alpha", "geo_ul_"]):
                p.requires_grad = True
            else:
                continue
        
        if "unilora_layer_alpha" in n:
            alpha_params.append(p)
        elif (
            "geo_ul_shared_theta_d" in n
            or "geo_ul_innovation_theta_d" in n
        ):
            theta_d_params.append(p)
        elif (
            n.endswith("theta_d")
            or "theta_d." in n
            or "unilora_soft_assign_logits" in n
            or "unilora_soft_weight_sharing_" in n
            or "unilora_multi_structured_left" in n
            or "unilora_multi_structured_right" in n
            or "unilora_multi_structured_layer_scale" in n
            or "unilora_multi_structured_global_" in n
            or "unilora_multi_hashing_" in n
            or "unilora_sketch_tune_quant_grid" in n
            or "unilora_sketch_delta_" in n
            or "unilora_shared_sketch_bank_" in n
            or "unilora_sketch_routed_" in n
        ):
            theta_d_params.append(p)
        else:
            head_params.append(p)

    theta_d_lr_display = f"{theta_d_lr}" if theta_d_params else "N/A"
    rosa_sparse_lr_display = f"{rosa_sparse_lr}" if sparse_params else "N/A"
    alpha_lr_display = f"{alpha_lr}" if alpha_params else "N/A"
    print("=" * 80)
    print(f"Run Variant: {variant.upper()}")
    print(f"  model_name = {model_name} | task = {task} | seed = {args.seed}")
    print(
        f"  head_lr = {args.head_lr} | theta_d_lr = {theta_d_lr_display} "
        f"| sparse_lr = {rosa_sparse_lr_display} | alpha_lr = {alpha_lr_display}"
    )
    print("=" * 80)

    print_trainable_param_summary(
        model=model,
        variant=variant,
        theta_d_params=theta_d_params + sparse_params,
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
    if unilora_rosa_backend is not None:
        if variant in {"unilora_rosa_stage", "unilora_rosa_stage_snip"}:
            score_mode = "snip |W*g|" if variant == "unilora_rosa_stage_snip" else "max-abs gradient"
            print(
                "UniLoRA-RoSA sparse config: "
                f"density={args.rosa_density}, "
                f"stage_ratio={args.rosa_stage_ratio}, "
                f"stage_warmup_steps={args.rosa_stage_warmup_steps}, "
                f"mask_steps={args.rosa_mask_steps}, "
                f"score_mode={score_mode}, "
                f"reset_optimizer_on_mask={args.rosa_reset_optimizer_on_mask}"
            )
        else:
            print(
                "UniLoRA-RoSA sparse config: "
                f"density={args.rosa_density}, "
                f"warmup_steps={args.rosa_warmup_steps}, "
                f"mask_steps={args.rosa_mask_steps}, "
                f"reset_optimizer_on_mask={args.rosa_reset_optimizer_on_mask}"
            )
        print(f"Initial sparse stats: {unilora_rosa_backend.get_sparse_structure_stats()}")
    if unilora_aroma_backend is not None:
        print(
            "UniLoRA-AROMA config: "
            f"rank={args.rank}, "
            f"T_in={args.aroma_t_in}, "
            f"reset_optimizer_on_merge={args.aroma_reset_optimizer_on_merge}"
        )
    if unilora_igu_backend is not None:
        print(
            "UniLoRA-IGU rank config: "
            f"target_rank={args.igu_target_rank}, "
            f"warmup={args.igu_init_warmup}, "
            f"final_warmup={args.igu_final_warmup}, "
            f"mask_interval={args.igu_mask_interval}, "
            f"r_min={args.igu_r_min}, "
            f"reset_optimizer_on_mask={args.igu_reset_optimizer_on_mask}"
        )
        print(f"Initial rank stats: {unilora_igu_backend.get_rank_structure_stats()}")
    if unilora_swap_backend is not None:
        print(
            "UniLoRA-Swap config: "
            f"dead_bucket_count={args.swap_dead_bucket_count}, "
            f"split_ratio={args.swap_split_ratio}, "
            f"step_interval={args.swap_interval_steps}, "
            f"step_start={args.swap_start_after_steps}, "
            f"epoch_interval={args.swap_interval_epochs}, "
            f"epoch_start={args.swap_start_after_epochs}, "
            f"reset_optimizer_state={args.swap_reset_optimizer_state}"
        )
    if unilora_local_swap_backend is not None:
        print(
            "UniLoRA-LocalSwap config: "
            f"grad_ema_momentum={args.local_swap_grad_ema_momentum}, "
            f"warmup_steps={args.local_swap_warmup_steps}, "
            f"bad_bucket_frac={args.local_swap_bad_bucket_frac}, "
            f"candidates_per_bucket={args.local_swap_candidates_per_bucket}, "
            f"target_bucket_samples={args.local_swap_target_bucket_samples}, "
            f"min_delta={args.local_swap_min_delta}, "
            f"max_target_drop={args.local_swap_max_target_drop}, "
            f"min_bucket_size={args.local_swap_min_bucket_size}, "
            f"update_ratio={args.local_swap_update_ratio}, "
            f"step_interval={args.local_swap_interval_steps}, "
            f"step_start={args.local_swap_start_after_steps}, "
            f"epoch_interval={args.local_swap_interval_epochs}, "
            f"epoch_start={args.local_swap_start_after_epochs}, "
            f"reset_optimizer_state={args.local_swap_reset_optimizer_state}"
        )
    if variant == "unilora_gora":
        gora_features_func = None if args.gora_features_func == "none" else args.gora_features_func
        print(
            "UniLoRA-GoRA config: "
            f"importance_type={args.gora_importance_type}, "
            f"min_rank={args.gora_min_rank}, "
            f"max_rank={args.gora_max_rank}, "
            f"allocate_strategy={args.gora_allocate_strategy}, "
            f"features_func={gora_features_func}, "
            f"softmax_importance={args.gora_softmax_importance}, "
            f"temperature={args.gora_temperature}, "
            f"gradient_est_steps={args.gora_gradient_est_steps}"
        )

    optimizer_groups = []
    if head_params:
        optimizer_groups.append({"params": head_params, "lr": args.head_lr, "weight_decay": args.weight_decay})
    if theta_d_params:
        optimizer_groups.append({"params": theta_d_params, "lr": theta_d_lr, "weight_decay": args.weight_decay})
    sparse_group_indices = []
    if sparse_params:
        sparse_group_indices.append(len(optimizer_groups))
        optimizer_groups.append({"params": sparse_params, "lr": rosa_sparse_lr, "weight_decay": 0.0})
    if alpha_params:
        optimizer_groups.append({"params": alpha_params, "lr": alpha_lr, "weight_decay": 0.0})

    optimizer = AdamW(optimizer_groups)

    total_steps = len(train_loader) * num_epochs
    rosa_stage_info = None
    if unilora_igu_backend is not None:
        unilora_igu_backend.set_total_step(total_steps)
    if unilora_rosa_backend is not None and hasattr(unilora_rosa_backend, "set_training_schedule"):
        rosa_stage_info = unilora_rosa_backend.set_training_schedule(
            total_steps=total_steps,
            steps_per_epoch=len(train_loader),
            total_epochs=num_epochs,
            stage_start_step_override=(
                args.rosa_stage_warmup_steps
                if variant in {"unilora_rosa_stage", "unilora_rosa_stage_snip"}
                else None
            ),
            adapter_name="default",
        )
        if variant in {"unilora_rosa_stage", "unilora_rosa_stage_snip"}:
            schedule_prefix = (
                "UniLoRA-RoSA-Stage-SNIP schedule"
                if variant == "unilora_rosa_stage_snip"
                else "UniLoRA-RoSA-Stage schedule"
            )
            if bool(rosa_stage_info.get("using_warmup_steps", 0)):
                schedule_prefix += f" (warmup_steps={args.rosa_stage_warmup_steps})"
            print(
                f"{schedule_prefix}: "
                f"stage_ratio={rosa_stage_info['stage_ratio']}, "
                f"stage_progress_epochs={rosa_stage_info['stage_progress_epochs']:.4f}, "
                f"stage_start_step={rosa_stage_info['stage_start_step']}, "
                f"mask_steps={rosa_stage_info['mask_steps']}, "
                f"score_mode={rosa_stage_info.get('score_mode', 'max_abs_grad')}"
            )
    num_warmup_steps = int(warmup_ratio * total_steps)
    if args.scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
        )
    sparse_lr_activation_step = 0
    if sparse_group_indices:
        if variant in {"unilora_rosa_stage", "unilora_rosa_stage_snip"} and rosa_stage_info is not None:
            sparse_lr_activation_step = int(rosa_stage_info["stage_start_step"]) + int(rosa_stage_info["mask_steps"])
        elif variant in {"unilora_rosa", "unilora_rosa_discrete", "unilora_rosa_global", "unilora_rosa_compression"}:
            sparse_lr_activation_step = int(args.rosa_warmup_steps) + int(args.rosa_mask_steps)
        sparse_lr_activation_step = max(0, min(total_steps, sparse_lr_activation_step))
        if args.rosa_decay_sparse_lr_after_activation:
            print(
                "UniLoRA-RoSA sparse LR schedule: "
                f"base_lr={rosa_sparse_lr}, activation_step={sparse_lr_activation_step}, "
                f"decay_steps={max(1, total_steps - sparse_lr_activation_step)}, "
                f"scheduler_type={args.scheduler_type}"
            )

    def get_sparse_lr_for_step(step_after_update: int) -> float:
        if not args.rosa_decay_sparse_lr_after_activation:
            return float(rosa_sparse_lr)
        if step_after_update <= sparse_lr_activation_step:
            return float(rosa_sparse_lr)
        decay_steps = max(1, total_steps - sparse_lr_activation_step)
        progress = (float(step_after_update) - float(sparse_lr_activation_step)) / float(decay_steps)
        progress = max(0.0, min(1.0, progress))
        if args.scheduler_type == "cosine":
            lr_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            lr_factor = 1.0 - progress
        return float(rosa_sparse_lr) * max(0.0, lr_factor)

    alpha_freeze_steps = int(args.alpha_freeze_ratio * total_steps) if alpha_params else 0
    if alpha_params and alpha_freeze_steps > 0:
        for p in alpha_params:
            p.requires_grad = False
        print(f"Freezing alpha parameters for first {alpha_freeze_steps}/{total_steps} steps.")

    # TensorBoard + result file stem (include sweep suffix for unilora_soft_weight_sharing, etc.)
    run_stem = build_result_json_stem(variant, task, model_name, args.head_lr, args.seed, args)
    log_dir = os.path.join(args.out_dir, "runs", run_stem)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging to: {log_dir}")

    # Train / Eval
    best_score = -1e18
    best_metric = None
    history = []
    global_step = 0
    sws_warmup_steps = int(args.sws_warmup_ratio * total_steps) if variant == "unilora_soft_weight_sharing" else 0
    deepk_warmup_steps = int(args.deepk_warmup_ratio * total_steps) if variant == "unilora_deepk" else 0
    swap_event_history = []
    local_swap_event_history = []

    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        epoch_loss = 0
        for batch in pbar:
            if alpha_params and alpha_freeze_steps > 0 and global_step == alpha_freeze_steps:
                for p in alpha_params:
                    p.requires_grad = True
                print(f"Unfroze alpha parameters at step {global_step}.")

            rosa_collecting = (
                unilora_rosa_backend is not None
                and unilora_rosa_backend.should_collect_gradients(global_step, adapter_name="default")
            )
            if unilora_rosa_backend is not None:
                unilora_rosa_backend.enable_gradient_capture(rosa_collecting)

            batch = {k: v.to(device) for k, v in batch.items()}
            if unilora_igu_backend is not None:
                unilora_igu_backend.set_weight_coeffs(sample_igu_weight_coeff(), adapter_name="default")
            task_loss = model(**batch).loss
            loss = task_loss
            if variant in {"geo_unilora", "igu_unilora"}:
                lambda_in = float(args.geo_lambda_in if variant == "geo_unilora" else args.igu_lambda_in)
            else:
                lambda_in = 0.0
            if lambda_in > 0.0:
                innov_reg = task_loss.new_zeros(())
                for n, p in model.named_parameters():
                    if "geo_ul_innovation_theta_d" in n:
                        innov_reg = innov_reg + (p ** 2).sum()
                loss = loss + lambda_in * innov_reg
            sws_loss_value = None
            sws_tau_current = None
            if variant == "unilora_soft_weight_sharing" and hasattr(model.base_model, "compute_soft_weight_sharing_loss"):
                sws_info = model.base_model.compute_soft_weight_sharing_loss(adapter_name="default")
                if sws_warmup_steps > 0:
                    ramp = min(1.0, float(global_step + 1) / float(sws_warmup_steps))
                else:
                    ramp = 1.0
                sws_loss_value = sws_info["loss"] * ramp
                sws_tau_current = float(sws_info["tau"].item()) * ramp
                loss = loss + sws_loss_value
            deepk_loss_value = None
            deepk_tau_current = None
            deepk_reg_a = None
            deepk_reg_b = None
            deepk_reg_total = None
            if variant == "unilora_deepk" and hasattr(model.base_model, "compute_deepk_loss"):
                deepk_info = model.base_model.compute_deepk_loss(adapter_name="default", global_step=global_step)
                if deepk_warmup_steps > 0:
                    deepk_ramp = min(1.0, float(global_step + 1) / float(deepk_warmup_steps))
                else:
                    deepk_ramp = 1.0
                deepk_loss_value = deepk_info["loss"] * deepk_ramp
                deepk_tau_current = float(deepk_info["tau"].item()) * deepk_ramp
                deepk_reg_total = deepk_info["reg_total"]
                deepk_reg_a = deepk_info["reg_a"]
                deepk_reg_b = deepk_info["reg_b"]
                loss = loss + deepk_loss_value
            igu_orth_reg = None
            if unilora_igu_backend is not None:
                igu_orth_reg = unilora_igu_backend.compute_orth_regu(adapter_name="default")
                loss = loss + igu_orth_reg
            loss.backward()
            if unilora_igu_backend is not None:
                unilora_igu_backend.set_weight_coeffs(1.0, adapter_name="default")
            if rosa_collecting:
                unilora_rosa_backend.accumulate_gradient_statistics(adapter_name="default")
            if hessian_aware_backend is not None:
                hessian_aware_backend.accumulate_curvature_statistics(
                    adapter_name="default",
                    ema_momentum=args.hessian_aware_curvature_ema_momentum,
                )
            if (
                unilora_igu_backend is not None
                and unilora_igu_backend.should_update_importance(global_step + 1, adapter_name="default")
            ):
                unilora_igu_backend.accumulate_rank_statistics(
                    adapter_name="default",
                    beta1=args.igu_beta1,
                    beta2=args.igu_beta2,
                )
            if unilora_local_swap_backend is not None:
                unilora_local_swap_backend.accumulate_local_swap_statistics(
                    adapter_name="default",
                    ema_momentum=args.local_swap_grad_ema_momentum,
                )
            optimizer.step()
            scheduler.step()
            current_sparse_lr = None
            for sparse_group_idx in sparse_group_indices:
                current_sparse_lr = get_sparse_lr_for_step(global_step + 1)
                optimizer.param_groups[sparse_group_idx]["lr"] = current_sparse_lr
            if (
                unilora_rosa_backend is not None
                and unilora_rosa_backend.should_generate_masks(global_step + 1, adapter_name="default")
            ):
                mask_info = unilora_rosa_backend.generate_sparse_masks(adapter_name="default")
                if args.rosa_reset_optimizer_on_mask:
                    optimizer.state.clear()
                print(
                    "Activated UniLoRA-RoSA sparse compensation: "
                    f"selected_ratio={mask_info['selected_ratio']:.4f}, "
                    f"selected_positions={mask_info['selected_positions']}, "
                    f"density={mask_info['selected_density']:.4f}"
                )
                writer.add_scalar("RoSA/Selected_Ratio", mask_info["selected_ratio"], global_step)
            if unilora_igu_backend is not None:
                igu_mask_info = unilora_igu_backend.update_and_mask(global_step + 1, adapter_name="default")
                writer.add_scalar("IGU/Target_Rank", igu_mask_info["target_rank"], global_step)
                writer.add_scalar("IGU/Active_Rank", igu_mask_info["active_rank_after"], global_step)
                if igu_mask_info.get("mask_applied", False):
                    if igu_mask_info.get("reset_optimizer", False):
                        optimizer.state.clear()
                    print(
                        "Applied UniLoRA-IGU rank update: "
                        f"step={global_step + 1}, "
                        f"masked={igu_mask_info['masked_ranks']}, "
                        f"target_rank={igu_mask_info['target_rank']}, "
                        f"active_rank={igu_mask_info['active_rank_after']}"
                    )
                    writer.add_scalar("IGU/Masked_Ranks", igu_mask_info["masked_ranks"], global_step)
            if (
                unilora_swap_backend is not None
                and args.swap_interval_steps > 0
                and (global_step + 1) >= args.swap_start_after_steps
                and ((global_step + 1) % args.swap_interval_steps == 0)
            ):
                swap_info = unilora_swap_backend.perform_swap(optimizer=optimizer, adapter_name="default")
                if swap_info.get("swapped", False):
                    print(
                        "Applied UniLoRA-Swap step update: "
                        f"step={global_step + 1}, "
                        f"num_pairs={swap_info['num_pairs']}, "
                        f"sink_bucket={swap_info['sink_bucket']}, "
                        f"freed_buckets={swap_info['num_freed_buckets']}"
                    )
                    writer.add_scalar("Swap/Num_Pairs", swap_info["num_pairs"], global_step)
                    writer.add_scalar("Swap/Count_Max_After", swap_info["count_max_after"], global_step)
                swap_event_history.append(
                    {
                        "trigger": "step",
                        "step": global_step + 1,
                        "epoch": epoch,
                        "info": swap_info,
                    }
                )
            if (
                unilora_local_swap_backend is not None
                and args.local_swap_interval_steps > 0
                and (global_step + 1) >= max(args.local_swap_warmup_steps, args.local_swap_start_after_steps)
                and ((global_step + 1) % args.local_swap_interval_steps == 0)
            ):
                local_swap_info = unilora_local_swap_backend.perform_local_swap(optimizer=optimizer, adapter_name="default")
                if local_swap_info.get("swapped", False):
                    print(
                        "Applied UniLoRA-LocalSwap step update: "
                        f"step={global_step + 1}, "
                        f"num_swaps={local_swap_info['num_swaps']}, "
                        f"changed_ratio={local_swap_info['changed_ratio']:.6f}, "
                        f"mean_delta={local_swap_info['mean_delta']:.6f}"
                    )
                    writer.add_scalar("LocalSwap/Num_Swaps", local_swap_info["num_swaps"], global_step)
                    writer.add_scalar("LocalSwap/Changed_Positions", local_swap_info["changed_positions"], global_step)
                    writer.add_scalar("LocalSwap/Changed_Ratio", local_swap_info["changed_ratio"], global_step)
                    writer.add_scalar("LocalSwap/Mean_Delta", local_swap_info["mean_delta"], global_step)
                    writer.add_scalar("LocalSwap/Count_Max_After", local_swap_info["count_max_after"], global_step)
                local_swap_event_history.append(
                    {
                        "trigger": "step",
                        "step": global_step + 1,
                        "epoch": epoch,
                        "info": local_swap_info,
                    }
                )
            if (
                unilora_aroma_backend is not None
                and args.aroma_t_in > 0
                and ((global_step + 1) % args.aroma_t_in == 0)
            ):
                aroma_info = unilora_aroma_backend.merge_and_reinit(global_step=global_step + 1, adapter_name="default")
                if args.aroma_reset_optimizer_on_merge:
                    optimizer.state.clear()
                print(
                    "Applied UniLoRA-AROMA merge-and-reinit: "
                    f"step={aroma_info['step']}, "
                    f"merged_modules={aroma_info['merged_modules']}, "
                    f"reinit_seed={aroma_info['reinit_seed']}"
                )
                writer.add_scalar("AROMA/Merged_Modules", aroma_info["merged_modules"], global_step)
                writer.add_scalar("AROMA/Reinit_Seed", aroma_info["reinit_seed"], global_step)
            optimizer.zero_grad()
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            writer.add_scalar("Train/Loss", loss.item(), global_step)
            writer.add_scalar("Train/Task_Loss", task_loss.item(), global_step)
            if current_sparse_lr is not None:
                writer.add_scalar("RoSA/Sparse_LR", current_sparse_lr, global_step)
            if igu_orth_reg is not None:
                writer.add_scalar("IGU/Orth_Reg", igu_orth_reg.item(), global_step)
            if sws_loss_value is not None:
                writer.add_scalar("SWS/Loss", sws_loss_value.item(), global_step)
                writer.add_scalar("SWS/Tau", sws_tau_current, global_step)
            if deepk_loss_value is not None:
                writer.add_scalar("DeepK/Loss", deepk_loss_value.item(), global_step)
                writer.add_scalar("DeepK/Tau", deepk_tau_current, global_step)
                writer.add_scalar("DeepK/Reg_Total", deepk_reg_total.item(), global_step)
                writer.add_scalar("DeepK/Reg_A", deepk_reg_a.item(), global_step)
                writer.add_scalar("DeepK/Reg_B", deepk_reg_b.item(), global_step)
            epoch_loss += loss.item()
            global_step += 1
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        writer.add_scalar("Train/Epoch_Loss", avg_epoch_loss, epoch)

        epoch_swap_info = None
        epoch_local_swap_info = None
        if (
            unilora_swap_backend is not None
            and args.swap_interval_epochs > 0
            and (epoch + 1) >= args.swap_start_after_epochs
            and ((epoch + 1) % args.swap_interval_epochs == 0)
            and (epoch + 1) < num_epochs
        ):
            epoch_swap_info = unilora_swap_backend.perform_swap(optimizer=optimizer, adapter_name="default")
            if epoch_swap_info.get("swapped", False):
                print(
                    "Applied UniLoRA-Swap epoch update: "
                    f"epoch={epoch + 1}, "
                    f"num_pairs={epoch_swap_info['num_pairs']}, "
                    f"sink_bucket={epoch_swap_info['sink_bucket']}, "
                    f"freed_buckets={epoch_swap_info['num_freed_buckets']}"
                )
                writer.add_scalar("Swap/Epoch_Num_Pairs", epoch_swap_info["num_pairs"], epoch)
                writer.add_scalar("Swap/Epoch_Count_Max_After", epoch_swap_info["count_max_after"], epoch)
            swap_event_history.append(
                {
                    "trigger": "epoch",
                    "step": global_step,
                    "epoch": epoch + 1,
                    "info": epoch_swap_info,
                }
            )
        if (
            unilora_local_swap_backend is not None
            and args.local_swap_interval_epochs > 0
            and (epoch + 1) >= args.local_swap_start_after_epochs
            and ((epoch + 1) % args.local_swap_interval_epochs == 0)
            and (epoch + 1) < num_epochs
        ):
            epoch_local_swap_info = unilora_local_swap_backend.perform_local_swap(optimizer=optimizer, adapter_name="default")
            if epoch_local_swap_info.get("swapped", False):
                print(
                    "Applied UniLoRA-LocalSwap epoch update: "
                    f"epoch={epoch + 1}, "
                    f"num_swaps={epoch_local_swap_info['num_swaps']}, "
                    f"changed_ratio={epoch_local_swap_info['changed_ratio']:.6f}, "
                    f"mean_delta={epoch_local_swap_info['mean_delta']:.6f}"
                )
                writer.add_scalar("LocalSwap/Epoch_Num_Swaps", epoch_local_swap_info["num_swaps"], epoch)
                writer.add_scalar("LocalSwap/Epoch_Changed_Positions", epoch_local_swap_info["changed_positions"], epoch)
                writer.add_scalar("LocalSwap/Epoch_Changed_Ratio", epoch_local_swap_info["changed_ratio"], epoch)
                writer.add_scalar("LocalSwap/Epoch_Mean_Delta", epoch_local_swap_info["mean_delta"], epoch)
                writer.add_scalar("LocalSwap/Epoch_Count_Max_After", epoch_local_swap_info["count_max_after"], epoch)
            local_swap_event_history.append(
                {
                    "trigger": "epoch",
                    "step": global_step,
                    "epoch": epoch + 1,
                    "info": epoch_local_swap_info,
                }
            )

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
            "swap_update": epoch_swap_info,
            "local_swap_update": epoch_local_swap_info,
            "structure_update": structure_update_info,
        })

        if score > best_score:
            best_score = score
            best_metric = eval_results

    sws_finalize_info = None
    sws_stats = None
    deepk_stats = None
    deepk_finalize_info = None
    if variant == "unilora_soft_weight_sharing" and hasattr(model.base_model, "get_soft_weight_sharing_stats"):
        sws_stats = model.base_model.get_soft_weight_sharing_stats(adapter_name="default")
        print(f"SWS stats before finalize: {sws_stats}")
        if args.sws_assign_stage == "end" and hasattr(model.base_model, "finalize_soft_weight_sharing"):
            sws_finalize_info = model.base_model.finalize_soft_weight_sharing(adapter_name="default")
            print(f"SWS hard assignment finalized: {sws_finalize_info}")
    if variant == "unilora_deepk" and hasattr(model.base_model, "get_deepk_stats"):
        deepk_stats = model.base_model.get_deepk_stats(adapter_name="default")
        print(f"DeepK stats: {deepk_stats}")
        if args.deepk_assign_stage == "end" and hasattr(model.base_model, "finalize_deepk_assignment"):
            deepk_finalize_info = model.base_model.finalize_deepk_assignment(adapter_name="default")
            print(f"DeepK hard assignment finalized: {deepk_finalize_info}")

    writer.close()

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{run_stem}.json")
    with open(out_path, "w") as f:
        json.dump({
            "variant": variant, 
            "best_score": best_score, 
            "best_metric": best_metric, 
            "history": history,
            "sws_stats": sws_stats,
            "sws_finalize_info": sws_finalize_info,
            "deepk_stats": deepk_stats,
            "deepk_finalize_info": deepk_finalize_info,
            "swap_event_history": swap_event_history,
            "local_swap_event_history": local_swap_event_history,
            "geo_plan_stats": geo_plan_stats if variant == "geo_unilora" else None,
            "igu_plan_stats": igu_plan_stats if variant == "igu_unilora" else None,
            "args": vars(args)
        }, f, indent=2)
    print(f"Best score: {best_score} saved to {out_path}")

if __name__ == "__main__":
    main()

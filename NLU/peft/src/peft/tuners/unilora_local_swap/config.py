from dataclasses import dataclass, field
from typing import List, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class UniLoRALocalSwapConfig(PeftConfig):
    """
    UniLoRA with local swap-based bucket reassignment.

    The model keeps the standard one-hot UniLoRA bucket parameterization, while
    maintaining an EMA of position-level gradients and occasionally swapping a
    small number of assignments between conflicting buckets.
    """

    r: int = field(default=4, metadata={"help": "The rank of incremental matrices."})
    proj_seed: int = field(default=42, metadata={"help": "Random seed for initializing the projection matrix."})
    theta_d_length: int = field(default=256, metadata={"help": "Total theta_d length."})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with UniLoRA-LocalSwap. "
                "This can also be the wildcard 'all-linear'."
            )
        },
    )
    unilora_dropout: float = field(default=0.0, metadata={"help": "UniLoRA dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)."},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for UniLoRA-LocalSwap adapters."})
    modules_to_save: Optional[List[str]] = field(default=None)
    init_theta_d_bound: float = field(default=0.02, metadata={"help": "Uniform init bound for theta_d."})
    local_swap_grad_ema_momentum: float = field(
        default=0.9,
        metadata={"help": "EMA momentum used for position-level local swap gradient statistics."},
    )
    local_swap_warmup_steps: int = field(
        default=0,
        metadata={"help": "Do not run local swap updates before this optimizer step count."},
    )
    local_swap_bad_bucket_frac: float = field(
        default=0.1,
        metadata={"help": "Fraction of most conflicting buckets considered in each local swap round."},
    )
    local_swap_candidates_per_bucket: int = field(
        default=2,
        metadata={"help": "Maximum number of candidate positions examined per bad bucket."},
    )
    local_swap_target_bucket_samples: int = field(
        default=16,
        metadata={"help": "Number of target buckets sampled per candidate position."},
    )
    local_swap_min_delta: float = field(
        default=1e-3,
        metadata={"help": "Minimum total ratio improvement required to accept a local swap."},
    )
    local_swap_max_target_drop: float = field(
        default=0.01,
        metadata={"help": "Maximum allowed alignment-ratio drop of the target bucket after a swap."},
    )
    local_swap_min_bucket_size: int = field(
        default=2,
        metadata={"help": "Buckets smaller than this are skipped during local swap search."},
    )
    local_swap_update_ratio: float = field(
        default=0.01,
        metadata={"help": "Maximum fraction of A/B positions that may change buckets in one local swap round."},
    )
    local_swap_interval_steps: int = field(
        default=0,
        metadata={"help": "Trigger one local swap round every N optimizer steps. Set to 0 to disable step-based swap."},
    )
    local_swap_start_after_steps: int = field(
        default=0,
        metadata={"help": "Do not run local swap updates before this optimizer step."},
    )
    local_swap_interval_epochs: int = field(
        default=5,
        metadata={"help": "Trigger one local swap round every N epochs. Set to 0 to disable epoch-based swap."},
    )
    local_swap_start_after_epochs: int = field(
        default=0,
        metadata={"help": "Do not run local swap updates before this epoch."},
    )
    local_swap_reset_optimizer_state: bool = field(
        default=True,
        metadata={"help": "Whether to reset Adam moments for buckets whose theta_d values were refit after a swap."},
    )
    layers_to_transform: Optional[Union[List[int], int]] = field(default=None)
    layers_pattern: Optional[Union[List[str], str]] = field(default=None)

    def __post_init__(self):
        self.peft_type = PeftType.UNILORA_LOCAL_SWAP
        self.target_modules = set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules

        if self.r <= 0:
            raise ValueError("`r` must be a positive integer for UniLoRA-LocalSwap.")
        if self.theta_d_length <= 1:
            raise ValueError("`theta_d_length` must be greater than 1 for UniLoRA-LocalSwap.")
        if not 0.0 <= self.local_swap_bad_bucket_frac <= 1.0:
            raise ValueError("`local_swap_bad_bucket_frac` must be in [0, 1].")
        if self.local_swap_candidates_per_bucket <= 0:
            raise ValueError("`local_swap_candidates_per_bucket` must be positive.")
        if self.local_swap_target_bucket_samples <= 0:
            raise ValueError("`local_swap_target_bucket_samples` must be positive.")
        if not 0.0 <= self.local_swap_grad_ema_momentum < 1.0:
            raise ValueError("`local_swap_grad_ema_momentum` must be in [0, 1).")
        if self.local_swap_min_bucket_size <= 1:
            raise ValueError("`local_swap_min_bucket_size` must be greater than 1.")
        if not 0.0 < self.local_swap_update_ratio <= 1.0:
            raise ValueError("`local_swap_update_ratio` must be in the interval (0, 1].")
        if self.local_swap_warmup_steps < 0:
            raise ValueError("`local_swap_warmup_steps` must be non-negative.")
        if self.local_swap_interval_steps < 0 or self.local_swap_start_after_steps < 0:
            raise ValueError("`local_swap_interval_steps` and `local_swap_start_after_steps` must be non-negative.")
        if self.local_swap_interval_epochs < 0 or self.local_swap_start_after_epochs < 0:
            raise ValueError("`local_swap_interval_epochs` and `local_swap_start_after_epochs` must be non-negative.")

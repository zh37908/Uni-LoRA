from dataclasses import dataclass, field
from typing import List, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class UniLoRASwapConfig(PeftConfig):
    """
    UniLoRA swap variant.

    The model starts from the same random bucket assignment as UniLoRA, then
    periodically rewires a small subset of theta_d buckets according to the
    current importance score |theta_d| * |exp_avg|.
    """

    r: int = field(default=4, metadata={"help": "The rank of incremental matrices."})
    proj_seed: int = field(default=42, metadata={"help": "Random seed for global index assignment."})
    theta_d_length: int = field(default=256, metadata={"help": "Total theta_d length."})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with UniLoRA-Swap. "
                "This can also be the wildcard 'all-linear'."
            )
        },
    )
    unilora_dropout: float = field(default=0.0, metadata={"help": "UniLoRA dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)."},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for UniLoRA-Swap adapters."})
    modules_to_save: Optional[List[str]] = field(default=None)
    init_theta_d_bound: float = field(default=0.02, metadata={"help": "Uniform init bound for theta_d."})
    swap_dead_bucket_count: int = field(
        default=8,
        metadata={"help": "Number of low-importance buckets used in each swap round."},
    )
    swap_split_ratio: float = field(
        default=0.5,
        metadata={"help": "Fraction of assignments moved from an overloaded bucket to a freed bucket."},
    )
    swap_interval_steps: int = field(
        default=0,
        metadata={"help": "Trigger one swap round every N optimizer steps. Set to 0 to disable step-based swap."},
    )
    swap_start_after_steps: int = field(
        default=0,
        metadata={"help": "Do not run swaps before this optimizer step."},
    )
    swap_interval_epochs: int = field(
        default=0,
        metadata={"help": "Trigger one swap round every N epochs. Set to 0 to disable epoch-based swap."},
    )
    swap_start_after_epochs: int = field(
        default=0,
        metadata={"help": "Do not run swaps before this epoch."},
    )
    swap_reset_optimizer_state: bool = field(
        default=True,
        metadata={"help": "Whether to reset Adam moments for source and destination buckets after each split."},
    )
    layers_to_transform: Optional[Union[List[int], int]] = field(default=None)
    layers_pattern: Optional[Union[List[str], str]] = field(default=None)

    def __post_init__(self):
        self.peft_type = PeftType.UNILORA_SWAP
        self.target_modules = set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules

        if self.r <= 0:
            raise ValueError("`r` must be a positive integer for UniLoRA-Swap.")
        if self.theta_d_length <= 1:
            raise ValueError("`theta_d_length` must be greater than 1 for UniLoRA-Swap.")
        if self.swap_dead_bucket_count <= 1:
            raise ValueError("`swap_dead_bucket_count` must be greater than 1 for UniLoRA-Swap.")
        if not 0.0 < self.swap_split_ratio < 1.0:
            raise ValueError("`swap_split_ratio` must be in the open interval (0, 1).")
        if self.swap_interval_steps < 0 or self.swap_start_after_steps < 0:
            raise ValueError("`swap_interval_steps` and `swap_start_after_steps` must be non-negative.")
        if self.swap_interval_epochs < 0 or self.swap_start_after_epochs < 0:
            raise ValueError("`swap_interval_epochs` and `swap_start_after_epochs` must be non-negative.")

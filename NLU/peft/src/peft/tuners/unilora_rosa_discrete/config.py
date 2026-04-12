from dataclasses import dataclass, field
from typing import List, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class UniLoRARoSADiscreteConfig(PeftConfig):
    """
    UniLoRA-RoSA with separate compression banks for low-rank and sparse parts.

    The LoRA A/B entries are compressed through one shared low-dimensional bank,
    while the sparse compensation matrix S is compressed through another bank.
    """

    r: int = field(default=4, metadata={"help": "The rank of incremental matrices."})
    proj_seed: int = field(default=42, metadata={"help": "Random seed for LoRA-bank projection/index generation."})
    theta_d_length: int = field(default=256, metadata={"help": "Total theta_d length for the low-rank branch."})
    sparse_theta_d_length: Optional[int] = field(
        default=None,
        metadata={"help": "Total theta_d length for the sparse branch; defaults to theta_d_length."},
    )
    sparse_proj_seed: Optional[int] = field(
        default=None,
        metadata={"help": "Random seed for sparse-bank projection/index generation; defaults to proj_seed + 1."},
    )
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with UniLoRA-RoSA-Discrete. "
                "This can also be the wildcard 'all-linear'."
            )
        },
    )
    unilora_dropout: float = field(default=0.0, metadata={"help": "UniLoRA dropout."})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)."},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for UniLoRA-RoSA-Discrete adapters."})
    modules_to_save: Optional[List[str]] = field(default=None)
    init_theta_d_bound: float = field(default=0.02, metadata={"help": "Uniform init bound for both theta_d banks."})
    rosa_density: float = field(
        default=0.01,
        metadata={"help": "Target density for the sparse compensation matrix S."},
    )
    rosa_warmup_steps: int = field(
        default=64,
        metadata={"help": "Number of low-rank-only warmup steps before collecting sparse gradients."},
    )
    rosa_mask_steps: int = field(
        default=1,
        metadata={"help": "Number of optimizer steps used to accumulate sparse gradient scores."},
    )
    layers_to_transform: Optional[Union[List[int], int]] = field(default=None)
    layers_pattern: Optional[Union[List[str], str]] = field(default=None)

    def __post_init__(self):
        self.peft_type = PeftType.UNILORA_ROSA_DISCRETE
        self.target_modules = set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules

        if self.sparse_theta_d_length is None:
            self.sparse_theta_d_length = self.theta_d_length
        if self.sparse_proj_seed is None:
            self.sparse_proj_seed = int(self.proj_seed) + 1

        if self.r <= 0:
            raise ValueError("`r` must be a positive integer for UniLoRA-RoSA-Discrete.")
        if self.theta_d_length <= 0:
            raise ValueError("`theta_d_length` must be positive for UniLoRA-RoSA-Discrete.")
        if self.sparse_theta_d_length <= 0:
            raise ValueError("`sparse_theta_d_length` must be positive for UniLoRA-RoSA-Discrete.")
        if not 0.0 <= self.rosa_density <= 1.0:
            raise ValueError("`rosa_density` must be in [0, 1].")
        if self.rosa_warmup_steps < 0:
            raise ValueError("`rosa_warmup_steps` must be non-negative.")
        if self.rosa_mask_steps < 0:
            raise ValueError("`rosa_mask_steps` must be non-negative.")
        if self.rosa_density > 0.0 and self.rosa_mask_steps == 0:
            raise ValueError("`rosa_mask_steps` must be positive when `rosa_density` is greater than zero.")

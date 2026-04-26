from dataclasses import dataclass, field
from typing import List, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class UniLoRARoSACompressionConfig(PeftConfig):
    """
    UniLoRA + RoSA:
    - dense UniLoRA uses theta_d (full bank) with indices + scaling
    - RoSA sparse residual uses a separate compressed bank (sparse_theta_d_length)
      but still selects offsets via per-offset top-k mask.
    """

    r: int = field(default=4, metadata={"help": "The rank of incremental matrices."})
    proj_seed: int = field(
        default=42,
        metadata={"help": "Random seed for initializing the projection index mapping."},
    )

    theta_d_length: int = field(
        default=256,
        metadata={"help": "Length of each theta_d component bank and the effective theta_D bank."},
    )
    sparse_theta_d_length: int = field(
        default=-1,
        metadata={"help": "Length of the compressed RoSA sparse bank. Must be explicitly set (>0)."},
    )

    target_modules: Optional[Union[List[str], str]] = field(default=None)

    unilora_dropout: float = field(default=0.0, metadata={"help": "UniLoRA dropout."})
    fan_in_fan_out: bool = field(default=False)
    bias: str = field(default="none", metadata={"help": "Bias type for UniLoRA adapters."})
    modules_to_save: Optional[List[str]] = field(default=None)
    init_theta_d_bound: float = field(default=0.02, metadata={"help": "Uniform init bound for theta_d."})

    rosa_density: float = field(default=0.01, metadata={"help": "Target density for the RoSA sparse mask."})
    rosa_warmup_steps: int = field(
        default=64,
        metadata={"help": "Warmup steps before collecting gradients for mask generation."},
    )
    rosa_mask_steps: int = field(
        default=1,
        metadata={"help": "Optimizer steps used to accumulate max-abs gradients for RoSA mask."},
    )

    layers_to_transform: Optional[Union[List[int], int]] = field(default=None)
    layers_pattern: Optional[Union[List[str], str]] = field(default=None)

    def __post_init__(self):
        self.peft_type = PeftType.UNILORA_ROSA_COMPRESSION

        if self.r <= 0:
            raise ValueError("`r` must be a positive integer for UniLoRA-RoSA-Compression.")
        if self.theta_d_length <= 0:
            raise ValueError("`theta_d_length` must be positive for UniLoRA-RoSA-Compression.")
        if self.sparse_theta_d_length is None or self.sparse_theta_d_length <= 0:
            raise ValueError("`sparse_theta_d_length` must be explicitly set and > 0 for UniLoRA-RoSA-Compression.")

        if not 0.0 <= self.rosa_density <= 1.0:
            raise ValueError("`rosa_density` must be in [0, 1].")
        if self.rosa_warmup_steps < 0:
            raise ValueError("`rosa_warmup_steps` must be non-negative.")
        if self.rosa_mask_steps < 0:
            raise ValueError("`rosa_mask_steps` must be non-negative.")
        if self.rosa_density > 0.0 and self.rosa_mask_steps == 0:
            raise ValueError("`rosa_mask_steps` must be positive when `rosa_density` is greater than zero.")


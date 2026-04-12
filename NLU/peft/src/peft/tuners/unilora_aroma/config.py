from dataclasses import dataclass, field
from typing import List, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class UniLoRAAromaConfig(PeftConfig):
    """
    UniLoRA-AROMA: combine UniLoRA's shared `theta_d` bank with AROMA-style
    periodic merge-and-reinit training.

    The effective LoRA factors A/B are still reconstructed from a shared 1-D
    parameter bank, while training periodically merges the current low-rank
    update into the base weight and restarts the bank/index assignment.
    """

    r: int = field(default=1, metadata={"help": "Rank of the low-rank update. Defaults to 1 to match AROMA."})
    proj_seed: int = field(default=42, metadata={"help": "Base random seed for theta_d projection/index generation."})
    theta_d_length: int = field(default=256, metadata={"help": "Length of the shared theta_d bank."})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of module names to replace with UniLoRA-AROMA. "
                "This can also be the wildcard 'all-linear'."
            )
        },
    )
    unilora_dropout: float = field(default=0.0, metadata={"help": "Dropout on the UniLoRA-AROMA adapter path."})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set to True if the target layer stores weights like (fan_in, fan_out)."},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for UniLoRA-AROMA adapters."})
    modules_to_save: Optional[List[str]] = field(default=None)
    init_theta_d_bound: float = field(default=0.02, metadata={"help": "Uniform init bound for theta_d."})
    aroma_t_in: int = field(
        default=100,
        metadata={"help": "Fixed optimizer-step interval for AROMA-style merge-and-reinit."},
    )
    aroma_reset_optimizer_on_merge: bool = field(
        default=True,
        metadata={"help": "Whether the training loop should clear optimizer state after each merge-and-reinit."},
    )
    layers_to_transform: Optional[Union[List[int], int]] = field(default=None)
    layers_pattern: Optional[Union[List[str], str]] = field(default=None)

    def __post_init__(self):
        self.peft_type = PeftType.UNILORA_AROMA
        self.target_modules = set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules

        if self.r <= 0:
            raise ValueError("`r` must be a positive integer for UniLoRA-AROMA.")
        if self.theta_d_length <= 0:
            raise ValueError("`theta_d_length` must be positive for UniLoRA-AROMA.")
        if self.aroma_t_in <= 0:
            raise ValueError("`aroma_t_in` must be a positive integer for UniLoRA-AROMA.")

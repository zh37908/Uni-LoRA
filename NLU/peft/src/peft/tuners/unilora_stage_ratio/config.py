from dataclasses import dataclass, field
from typing import List, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class UniLoRAStageRatioConfig(PeftConfig):
    """
    UniLoRA stage-ratio variant.

    Keep one global theta_D, but split it into three stage-specific segments
    (front/middle/back layers) with different theta_d ratios.
    """

    r: int = field(default=4, metadata={"help": "The rank of incremental matrices."})
    proj_seed: int = field(default=42, metadata={"help": "Random seed for global index assignment."})
    theta_d_length: int = field(default=256, metadata={"help": "Total theta_d length."})
    stage_theta_d_ratios: List[float] = field(
        default_factory=lambda: [0.2, 0.3, 0.5],
        metadata={"help": "Theta_d ratios for front/middle/back stages."},
    )
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of module names to replace."
                " Can also be 'all-linear'."
            )
        },
    )
    unilora_dropout: float = field(default=0.0, metadata={"help": "UniLoRA dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer stores weights like (fan_in, fan_out)."},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for UniLoRA adapters."})
    modules_to_save: Optional[List[str]] = field(default=None)
    init_theta_d_bound: float = field(default=0.02, metadata={"help": "Uniform init bound for theta_d."})
    layers_to_transform: Optional[Union[List[int], int]] = field(default=None)
    layers_pattern: Optional[Union[List[str], str]] = field(default=None)

    def __post_init__(self):
        self.peft_type = PeftType.UNILORA_STAGE_RATIO
        self.target_modules = set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules

        if len(self.stage_theta_d_ratios) != 3:
            raise ValueError("`stage_theta_d_ratios` must contain exactly 3 values: [front, middle, back].")
        if any(r <= 0 for r in self.stage_theta_d_ratios):
            raise ValueError("All values in `stage_theta_d_ratios` must be > 0.")

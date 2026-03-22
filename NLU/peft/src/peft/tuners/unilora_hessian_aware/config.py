from dataclasses import dataclass, field
from typing import List, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class UniLoRAHessianAwareConfig(PeftConfig):
    """
    UniLoRA with Hessian-aware structure updates.

    This variant keeps the standard global Uni-LoRA parameterization but augments
    it with curvature statistics that can be used to periodically reassign the
    implicit projection matrix.
    """

    r: int = field(default=4, metadata={"help": "The rank of incremental matrices."})
    proj_seed: int = field(default=42, metadata={"help": "Random seed for initializing the projection matrix."})
    theta_d_length: int = field(default=256, metadata={"help": "Total theta_d length."})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with LoRA. "
                "This can also be the wildcard 'all-linear'."
            )
        },
    )
    unilora_dropout: float = field(default=0.0, metadata={"help": "UniLoRA dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)."},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for UniLoRA adapters."})
    modules_to_save: Optional[List[str]] = field(default=None)
    init_theta_d_bound: float = field(default=0.02, metadata={"help": "Uniform init bound for theta_d."})
    curvature_ema_momentum: float = field(
        default=0.9,
        metadata={"help": "EMA momentum used for Hessian/Fisher diagonal surrogates."},
    )
    structure_reassign_ratio: float = field(
        default=0.01,
        metadata={"help": "Fraction of high-curvature positions greedily reassigned during one structure update."},
    )
    candidate_pool_size: int = field(
        default=8,
        metadata={"help": "Number of candidate buckets considered per reassigned position."},
    )
    capacity_penalty: float = field(
        default=0.1,
        metadata={"help": "Penalty coefficient for overloaded buckets during structure update."},
    )
    capacity_slack: float = field(
        default=2.0,
        metadata={"help": "Hard capacity multiplier relative to the average bucket load."},
    )
    layers_to_transform: Optional[Union[List[int], int]] = field(default=None)
    layers_pattern: Optional[Union[List[str], str]] = field(default=None)

    def __post_init__(self):
        self.peft_type = PeftType.UNILORA_HESSIAN_AWARE
        self.target_modules = set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules

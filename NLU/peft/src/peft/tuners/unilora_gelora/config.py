from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class UniLoRAGeLoRAConfig(PeftConfig):
    """
    UniLoRA-GeLoRA: UniLoRA with GeLoRA-style pre-allocation of per-module ranks.

    This config keeps the UniLoRA parameterization (shared theta_d + indices), but
    allows different modules to use different ranks based on an externally computed
    rank map (e.g., from intrinsic-dimension estimates).
    """

    r: int = field(default=4, metadata={"help": "Default rank used when no per-module rank is provided."})
    proj_seed: int = field(default=42, metadata={"help": "Random seed for initializing the projection matrix."})
    theta_d_length: int = field(default=256, metadata={"help": "Total theta_d length."})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with UniLoRA-GeLoRA. "
                "This can also be the wildcard 'all-linear'."
            )
        },
    )
    unilora_dropout: float = field(default=0.0, metadata={"help": "UniLoRA dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)."},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for UniLoRA-GeLoRA adapters."})
    modules_to_save: Optional[List[str]] = field(default=None)
    init_theta_d_bound: float = field(default=0.02, metadata={"help": "Uniform init bound for theta_d."})

    # GeLoRA-style rank allocation metadata (computed before adapter injection).
    gelora_rank_map: Optional[Dict[str, int]] = field(
        default=None,
        metadata={"help": "Optional mapping from module name to allocated rank."},
    )

    layers_to_transform: Optional[Union[List[int], int]] = field(default=None)
    layers_pattern: Optional[Union[List[str], str]] = field(default=None)

    def __post_init__(self):
        self.peft_type = PeftType.UNILORA_GELORA
        self.target_modules = set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules

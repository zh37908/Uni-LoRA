from dataclasses import dataclass, field
from typing import List, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class UniLoRAMultiHashingConfig(PeftConfig):
    """
    UniLoRA multi-hashing variant:
    reconstruct the shared bank via theta_D = sum_i P_i * theta_d_i.
    """

    r: int = field(default=4, metadata={"help": "The rank of incremental matrices."})
    proj_seed: int = field(
        default=42,
        metadata={"help": "Random seed for initializing the global index mapping."},
    )
    theta_d_length: int = field(
        default=256,
        metadata={"help": "Length of each theta_d component bank and the effective theta_D bank."},
    )
    num_hash_components: int = field(
        default=4,
        metadata={"help": "Number of independently initialized (P_i, theta_d_i) components."},
    )
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={"help": "Target module names or regex to inject adapters."},
    )
    unilora_dropout: float = field(default=0.0, metadata={"help": "UniLoRA dropout."})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set True if the target layer stores weight as (fan_in, fan_out)."},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for UniLoRA adapters."})
    modules_to_save: Optional[List[str]] = field(default=None)
    init_theta_d_bound: float = field(
        default=0.02,
        metadata={"help": "Uniform init bound for each theta_d_i component bank."},
    )
    init_p_bound: Optional[float] = field(
        default=None,
        metadata={"help": "Uniform init half-width around 1 / num_hash_components for each P_i component."},
    )
    layers_to_transform: Optional[Union[List[int], int]] = field(default=None)
    layers_pattern: Optional[Union[List[str], str]] = field(default=None)

    def __post_init__(self):
        self.peft_type = PeftType.UNILORA_MULTI_HASHING
        self.target_modules = set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules

        if self.init_p_bound is None:
            self.init_p_bound = self.init_theta_d_bound

        if self.r <= 0:
            raise ValueError("`r` must be positive.")
        if self.theta_d_length <= 0:
            raise ValueError("`theta_d_length` must be positive.")
        if self.num_hash_components <= 0:
            raise ValueError("`num_hash_components` must be positive.")
        if self.init_p_bound < 0:
            raise ValueError("`init_p_bound` must be non-negative.")

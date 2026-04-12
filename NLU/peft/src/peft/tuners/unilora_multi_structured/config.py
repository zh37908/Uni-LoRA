from dataclasses import dataclass, field
from typing import List, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class UniLoRAMultiStructuredConfig(PeftConfig):
    """
    UniLoRA multi-structured variant:
    reconstruct A/B values via a global M_hat parameterization using sum-of-products.
    """

    r: int = field(default=4, metadata={"help": "The rank of incremental matrices."})
    proj_seed: int = field(default=42, metadata={"help": "Random seed for deterministic index mapping."})
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

    # M_hat sum-of-products controls
    num_hash_pairs: int = field(
        default=4,
        metadata={"help": "Number of sum-of-products pairs M used to reconstruct each value."},
    )
    target_trainable_params: Optional[int] = field(
        default=None,
        metadata={"help": "Optional trainable-parameter budget. If set, it overrides num_hash_pairs."},
    )
    init_bound: float = field(
        default=0.02,
        metadata={"help": "Uniform init bound for shared sum-of-products banks."},
    )
    layerwise_learnable_scale: bool = field(
        default=True,
        metadata={"help": "Enable one learnable scalar per adapter layer."},
    )
    layers_to_transform: Optional[Union[List[int], int]] = field(default=None)
    layers_pattern: Optional[Union[List[str], str]] = field(default=None)

    def __post_init__(self):
        self.peft_type = PeftType.UNILORA_MULTI_STRUCTURED
        self.target_modules = set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules

        if self.r <= 0:
            raise ValueError("`r` must be positive.")
        if self.num_hash_pairs <= 0:
            raise ValueError("`num_hash_pairs` must be positive.")
        if self.target_trainable_params is not None and self.target_trainable_params <= 0:
            raise ValueError("`target_trainable_params` must be positive when provided.")

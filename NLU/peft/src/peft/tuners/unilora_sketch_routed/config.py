from dataclasses import dataclass, field
from typing import List, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class UniLoRASketchRoutedConfig(PeftConfig):
    """
    UniLoRA variant with multiple shared sketch experts and a learnable per-layer
    router that mixes them before decoding LoRA delta matrices.
    """

    r: int = field(default=4, metadata={"help": "The rank of incremental matrices."})
    proj_seed: int = field(default=42, metadata={"help": "Random seed for routed sketch assignments."})
    bits: int = field(default=4, metadata={"help": "Codebook bit-width for each shared sketch expert."})
    groups_per_row: int = field(
        default=4,
        metadata={"help": "Number of groups along the input dimension for each sketch-parameterized matrix row."},
    )
    num_banks: int = field(default=8, metadata={"help": "Number of banks per sketch expert."})
    num_experts: int = field(default=4, metadata={"help": "Number of shared sketch experts."})
    router_tau: float = field(default=1.0, metadata={"help": "Temperature used by the layer-wise sketch router."})
    router_mode: str = field(
        default="softmax",
        metadata={"help": "Router mode used during training: 'softmax' or 'gumbel'."},
    )
    router_gumbel_hard: bool = field(
        default=False,
        metadata={"help": "Use hard straight-through routing when router_mode='gumbel'."},
    )
    router_hard_eval: bool = field(
        default=True,
        metadata={"help": "Use argmax routing at evaluation time instead of soft mixing."},
    )
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace. "
                "This can also be a wildcard 'all-linear'."
            )
        },
    )
    unilora_dropout: float = field(default=0.0, metadata={"help": "Dropout applied before the routed sketch delta branch."})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the target stores weight like (fan_in, fan_out)."},
    )
    bias: str = field(default="none", metadata={"help": "Bias type. Can be 'none', 'all' or 'unilora_only'."})
    modules_to_save: Optional[List[str]] = field(default=None)
    init_expert_bound: float = field(
        default=0.02,
        metadata={"help": "Initialize sketch experts with U(-bound, bound)."},
    )
    layers_to_transform: Optional[Union[List[int], int]] = field(default=None)
    layers_pattern: Optional[Union[List[str], str]] = field(default=None)

    def __post_init__(self):
        self.peft_type = PeftType.UNILORA_SKETCH_ROUTED
        self.target_modules = set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules

        if self.r <= 0:
            raise ValueError("`r` should be a positive integer value")
        if self.bits <= 0:
            raise ValueError("`bits` should be a positive integer value")
        if self.groups_per_row <= 0:
            raise ValueError("`groups_per_row` should be a positive integer value")
        if self.num_banks <= 0:
            raise ValueError("`num_banks` should be a positive integer value")
        if self.num_experts <= 0:
            raise ValueError("`num_experts` should be a positive integer value")
        if self.router_tau <= 0:
            raise ValueError("`router_tau` should be > 0")
        if self.router_mode not in {"softmax", "gumbel"}:
            raise ValueError("`router_mode` must be one of {'softmax', 'gumbel'}")
        if self.router_gumbel_hard and self.router_mode != "gumbel":
            raise ValueError("`router_gumbel_hard=True` requires `router_mode='gumbel'`")

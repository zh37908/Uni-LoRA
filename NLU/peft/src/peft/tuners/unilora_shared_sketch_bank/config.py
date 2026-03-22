from dataclasses import dataclass, field
from typing import List, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class UniLoRASharedSketchBankConfig(PeftConfig):
    """
    UniLoRA variant where all layers decode their LoRA delta matrices from one
    globally shared sketch bank.
    """

    r: int = field(default=4, metadata={"help": "The rank of incremental matrices."})
    proj_seed: int = field(default=42, metadata={"help": "Random seed for shared sketch assignments."})
    bits: int = field(default=4, metadata={"help": "Codebook bit-width for the shared sketch bank."})
    groups_per_row: int = field(
        default=4,
        metadata={"help": "Number of groups along the input dimension for each sketch-parameterized matrix row."},
    )
    num_banks: int = field(default=8, metadata={"help": "Number of shared sketch codebooks."})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace. "
                "This can also be a wildcard 'all-linear'."
            )
        },
    )
    unilora_dropout: float = field(default=0.0, metadata={"help": "Dropout applied before the shared sketch delta branch."})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the target stores weight like (fan_in, fan_out)."},
    )
    bias: str = field(default="none", metadata={"help": "Bias type. Can be 'none', 'all' or 'unilora_only'."})
    modules_to_save: Optional[List[str]] = field(default=None)
    init_bank_bound: float = field(
        default=0.02,
        metadata={"help": "Initialize the shared sketch bank with U(-bound, bound)."},
    )
    layers_to_transform: Optional[Union[List[int], int]] = field(default=None)
    layers_pattern: Optional[Union[List[str], str]] = field(default=None)

    def __post_init__(self):
        self.peft_type = PeftType.UNILORA_SHARED_SKETCH_BANK
        self.target_modules = set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules

        if self.r <= 0:
            raise ValueError("`r` should be a positive integer value")
        if self.bits <= 0:
            raise ValueError("`bits` should be a positive integer value")
        if self.groups_per_row <= 0:
            raise ValueError("`groups_per_row` should be a positive integer value")
        if self.num_banks <= 0:
            raise ValueError("`num_banks` should be a positive integer value")

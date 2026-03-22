from dataclasses import dataclass, field
from typing import List, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class UniLoRASketchTuneConfig(PeftConfig):
    """
    SketchTune-style UniLoRA variant that replaces each target linear weight matrix
    with a fixed-code, trainable-codebook sketch.

    Each adapted layer stores:
    - a trainable per-row/group codebook (`quant_grid`)
    - fixed discrete assignments (`weight_codes`)

    During training and inference, the effective dense weight is reconstructed from
    this sketch instead of adding a LoRA delta to the frozen base weight.
    """

    bits: int = field(default=4, metadata={"help": "Number of bits used for the discrete codebook indices."})
    groups_per_row: int = field(
        default=4,
        metadata={"help": "Number of codebook groups per output row."},
    )
    bootstrap_method: str = field(
        default="uniform",
        metadata={"help": "Bootstrap method for initializing codebooks from dense weights: 'uniform' or 'kmeans'."},
    )
    bootstrap_kmeans_iters: int = field(
        default=8,
        metadata={"help": "Number of Lloyd iterations when bootstrap_method='kmeans'."},
    )
    unilora_dropout: float = field(
        default=0.0,
        metadata={"help": "Optional dropout applied to the input before the sketched linear projection."},
    )
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with SketchTune-style "
                "compressed linear layers. This can also be a wildcard 'all-linear'."
            )
        },
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the target stores weight like (fan_in, fan_out)."},
    )
    bias: str = field(
        default="none",
        metadata={"help": "Bias type. Can be 'none', 'all' or 'unilora_only'."},
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": (
                "Additional modules to train and save, e.g. a classification head on top of the adapted encoder."
            )
        },
    )
    layers_to_transform: Optional[Union[List[int], int]] = field(default=None)
    layers_pattern: Optional[Union[List[str], str]] = field(default=None)

    def __post_init__(self):
        self.peft_type = PeftType.UNILORA_SKETCH_TUNE
        self.target_modules = set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        if self.bits <= 0:
            raise ValueError(f"`bits` must be > 0, got {self.bits}.")
        if self.groups_per_row <= 0:
            raise ValueError(f"`groups_per_row` must be > 0, got {self.groups_per_row}.")
        if self.bootstrap_method not in {"uniform", "kmeans"}:
            raise ValueError(
                f"`bootstrap_method` must be one of ['uniform', 'kmeans'], got {self.bootstrap_method}."
            )

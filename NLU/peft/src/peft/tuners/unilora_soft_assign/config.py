from dataclasses import dataclass, field
from typing import List, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class UniLoRASoftAssignConfig(PeftConfig):
    """
    UniLoRA-SoftAssign: each UniLoRA A/B entry softly chooses among a small
    candidate set from the shared theta_d bank instead of selecting from the
    full bank.
    """

    r: int = field(default=4, metadata={"help": "The rank of incremental matrices."})
    proj_seed: int = field(
        default=42,
        metadata={"help": "Random seed for initializing the projection matrix."},
    )
    theta_d_length: int = field(
        default=256,
        metadata={"help": "The length of the shared theta_d vector bank."},
    )
    num_candidates: int = field(
        default=4,
        metadata={"help": "Number of candidate theta_d entries per UniLoRA element."},
    )
    assignment_mode: str = field(
        default="softmax",
        metadata={"help": "Assignment mode used during training: 'softmax' or 'gumbel'."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for soft assignment / Gumbel-Softmax."},
    )
    gumbel_hard: bool = field(
        default=False,
        metadata={"help": "Use hard straight-through Gumbel-Softmax during training."},
    )
    hard_eval: bool = field(
        default=True,
        metadata={"help": "Use argmax hardening during evaluation instead of deterministic softmax."},
    )
    init_logits_std: float = field(
        default=0.1,
        metadata={"help": "Std used to initialize candidate assignment logits."},
    )
    init_primary_bias: float = field(
        default=2.0,
        metadata={"help": "Bias applied to the primary candidate to start near vanilla UniLoRA."},
    )
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace. "
                "This can also be a wildcard 'all-linear' which matches all linear/Conv1D layers except the output layer."
            )
        },
    )
    unilora_dropout: float = field(default=0.0, metadata={"help": "UniLoRA dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: str = field(
        default="none",
        metadata={"help": "Bias type for UniLoRA-SoftAssign. Can be 'none', 'all' or 'unilora_only'"},
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": (
                "List of modules apart from UniLoRA-SoftAssign layers to be set as trainable and saved in the final checkpoint."
            )
        },
    )
    init_theta_d_bound: float = field(
        default=0.02,
        metadata={"help": "Initialize theta_d with a uniform distribution in [-bound, bound]."},
    )
    layers_to_transform: Optional[Union[List[int], int]] = field(
        default=None,
        metadata={
            "help": (
                "Layer indexes to transform. This only works when target_modules is a list of str."
            )
        },
    )
    layers_pattern: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={"help": "Layer pattern name used only if layers_to_transform is not None."},
    )

    def __post_init__(self):
        self.peft_type = PeftType.UNILORA_SOFT_ASSIGN
        self.target_modules = set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules

        if self.num_candidates < 1:
            raise ValueError("`num_candidates` should be a positive integer value")
        if self.temperature <= 0.0:
            raise ValueError("`temperature` should be > 0")
        if self.assignment_mode not in {"softmax", "gumbel"}:
            raise ValueError("`assignment_mode` must be one of {'softmax', 'gumbel'}")
        if self.gumbel_hard and self.assignment_mode != "gumbel":
            raise ValueError("`gumbel_hard=True` requires `assignment_mode='gumbel'`")

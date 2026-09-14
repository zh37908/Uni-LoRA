from dataclasses import dataclass, field
from typing import Optional, Union

from peft.tuners.lora import LoraConfig
from peft.utils import PeftType


@dataclass
class LoraRoSAConfig(LoraConfig):
    """LoRA with a RoSA-style sparse branch on frozen base-weight coordinates."""

    rosa_density: float = field(default=0.01, metadata={"help": "Sparse branch density over target base weights."})
    rosa_sparse_budget: Optional[int] = field(
        default=None,
        metadata={"help": "Exact total sparse branch budget. If set, overrides rosa_density."},
    )
    rosa_warmup_steps: int = field(
        default=64,
        metadata={"help": "Number of LoRA-only optimizer steps before sparse-mask scoring starts."},
    )
    rosa_mask_steps: int = field(
        default=1,
        metadata={"help": "Number of optimizer steps used to collect sparse-mask scores."},
    )
    rosa_score_mode: str = field(
        default="grad",
        metadata={"help": "Sparse mask scoring mode: grad, snip, or random."},
    )
    rosa_grad_acc_mode: str = field(
        default="max_abs",
        metadata={"help": "Gradient aggregation for mask scoring: max_abs, mean_abs, or mean_squared."},
    )
    rosa_seed: int = field(default=0, metadata={"help": "Seed used for random sparse-mask selection."})

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.LORA_ROSA
        self._validate_rosa_fields()

    def _validate_rosa_fields(self):
        if not 0.0 <= float(self.rosa_density) <= 1.0:
            raise ValueError("`rosa_density` must be in [0, 1].")
        if self.rosa_sparse_budget is not None and self.rosa_sparse_budget < 0:
            raise ValueError("`rosa_sparse_budget` must be non-negative.")
        if self.rosa_warmup_steps < 0:
            raise ValueError("`rosa_warmup_steps` must be non-negative.")
        if self.rosa_mask_steps < 0:
            raise ValueError("`rosa_mask_steps` must be non-negative.")
        if self.rosa_score_mode not in {"grad", "snip", "random"}:
            raise ValueError("`rosa_score_mode` must be one of: grad, snip, random.")
        if self.rosa_grad_acc_mode not in {"max_abs", "mean_abs", "mean_squared"}:
            raise ValueError("`rosa_grad_acc_mode` must be one of: max_abs, mean_abs, mean_squared.")
        if self.rosa_score_mode != "random" and self.rosa_density > 0.0 and self.rosa_mask_steps == 0:
            raise ValueError("`rosa_mask_steps` must be positive when score-based sparse selection is enabled.")


@dataclass
class LoraRoSASnipConfig(LoraRoSAConfig):
    """LoRA-RoSA with SNIP |W * grad| sparse-mask scoring."""

    rosa_score_mode: str = field(default="snip", metadata={"help": "Fixed to snip for this variant."})

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.LORA_ROSA_SNIP


@dataclass
class LoraRoSARandomConfig(LoraRoSAConfig):
    """LoRA-RoSA with random sparse-mask selection at the same sparse budget."""

    rosa_score_mode: str = field(default="random", metadata={"help": "Fixed to random for this variant."})

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.LORA_ROSA_RANDOM

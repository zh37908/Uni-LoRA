from dataclasses import dataclass, field

from peft.tuners.unilora_soft_assign.config import UniLoRASoftAssignConfig
from peft.utils import PeftType


@dataclass
class UniLoRASoftWeightSharingConfig(UniLoRASoftAssignConfig):
    """
    UniLoRA Soft-Weight-Sharing:
    - keeps UniLoRA-SoftAssign candidate routing
    - adds a learnable Gaussian-mixture sharing prior over realized A/B values
    """

    num_components: int = field(default=16, metadata={"help": "Number of Gaussian mixture components for soft sharing."})
    sharing_tau: float = field(default=1e-4, metadata={"help": "Weight of soft-sharing loss term."})
    sharing_grouping: str = field(
        default="global",
        metadata={"help": "Grouping strategy for soft-sharing loss. One of {'global','per_layer','ab_split'}."},
    )
    sharing_zero_component: bool = field(
        default=True,
        metadata={"help": "Whether to fix the first component mean at zero to encourage sparsity."},
    )
    sharing_sigma_floor: float = field(
        default=1e-4,
        metadata={"help": "Lower bound for Gaussian std to stabilize training."},
    )
    sharing_warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "Fraction of total steps used to ramp sharing loss from 0 to sharing_tau."},
    )
    sharing_assign_stage: str = field(
        default="end",
        metadata={"help": "When to finalize hard sharing assignment. One of {'none','end'}."},
    )
    sharing_merge_threshold: float = field(
        default=0.0,
        metadata={"help": "Reserved merge threshold for future component merging; 0 disables merge."},
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.UNILORA_SOFT_WEIGHT_SHARING
        if self.num_components < 2:
            raise ValueError("`num_components` should be >= 2")
        if self.sharing_tau < 0.0:
            raise ValueError("`sharing_tau` should be >= 0")
        if self.sharing_sigma_floor <= 0.0:
            raise ValueError("`sharing_sigma_floor` should be > 0")
        if not (0.0 <= self.sharing_warmup_ratio < 1.0):
            raise ValueError("`sharing_warmup_ratio` should be in [0, 1)")
        if self.sharing_grouping not in {"global", "per_layer", "ab_split"}:
            raise ValueError("`sharing_grouping` must be one of {'global','per_layer','ab_split'}")
        if self.sharing_assign_stage not in {"none", "end"}:
            raise ValueError("`sharing_assign_stage` must be one of {'none', 'end'}")

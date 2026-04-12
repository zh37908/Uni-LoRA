from dataclasses import dataclass, field

from peft.tuners.unilora_soft_assign.config import UniLoRASoftAssignConfig
from peft.utils import PeftType


@dataclass
class UniLoRADeepKConfig(UniLoRASoftAssignConfig):
    """
    UniLoRA-DeepK (layer-wise):
    - keeps UniLoRA-SoftAssign parameterization
    - adds Deep k-Means style spectral regularization on:
      * A columns
      * B rows
    """

    deepk_num_clusters_a: int = field(default=16, metadata={"help": "Cluster count for A-column DeepK regularization."})
    deepk_num_clusters_b: int = field(default=16, metadata={"help": "Cluster count for B-row DeepK regularization."})
    deepk_tau: float = field(default=1e-4, metadata={"help": "DeepK regularization coefficient."})
    deepk_f_update_interval: int = field(
        default=100,
        metadata={"help": "Refresh cached spectral assignment every N optimization steps."},
    )
    deepk_warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "Warmup ratio for ramping DeepK loss from 0 to full weight."},
    )
    deepk_assign_stage: str = field(
        default="none",
        metadata={"help": "When to run hard assignment finalize. One of {'none', 'end'}."},
    )
    deepk_svd_rank_cap: int = field(
        default=0,
        metadata={"help": "Optional cap for spectral rank used in F update. <=0 means disabled."},
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.UNILORA_DEEPK
        if self.deepk_num_clusters_a < 1:
            raise ValueError("`deepk_num_clusters_a` should be >= 1")
        if self.deepk_num_clusters_b < 1:
            raise ValueError("`deepk_num_clusters_b` should be >= 1")
        if self.deepk_tau < 0.0:
            raise ValueError("`deepk_tau` should be >= 0")
        if self.deepk_f_update_interval < 1:
            raise ValueError("`deepk_f_update_interval` should be >= 1")
        if not (0.0 <= self.deepk_warmup_ratio < 1.0):
            raise ValueError("`deepk_warmup_ratio` should be in [0, 1)")
        if self.deepk_assign_stage not in {"none", "end"}:
            raise ValueError("`deepk_assign_stage` must be one of {'none', 'end'}")

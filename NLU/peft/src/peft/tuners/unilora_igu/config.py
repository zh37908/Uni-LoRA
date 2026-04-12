from dataclasses import dataclass, field
from typing import List, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class UniLoRAIGUConfig(PeftConfig):
    """
    UniLoRA-IGU: apply IGU-style iterative rank masking on UniLoRA's compressed projection.

    The underlying shared bank `theta_d` is still trained as in UniLoRA. The effective dense
    projection `theta_D = P * theta_d` is interpreted through the reconstructed LoRA matrices,
    and IGU-style pruning masks whole rank slices instead of individual projection indices.
    """

    r: int = field(default=4, metadata={"help": "Initial over-parameterized rank per module."})
    proj_seed: int = field(default=42, metadata={"help": "Random seed for projection/index generation."})
    theta_d_length: int = field(default=256, metadata={"help": "Length of the shared theta_d bank."})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of module names to replace with UniLoRA-IGU. "
                "This can also be the wildcard 'all-linear'."
            )
        },
    )
    unilora_dropout: float = field(default=0.0, metadata={"help": "Dropout on the adapter path."})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set to True if the target layer stores weights like (fan_in, fan_out)."},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for UniLoRA-IGU adapters."})
    modules_to_save: Optional[List[str]] = field(default=None)
    init_theta_d_bound: float = field(default=0.02, metadata={"help": "Uniform init bound for theta_d."})

    igu_target_rank: int = field(default=2, metadata={"help": "Final target average rank per module."})
    igu_init_warmup: int = field(default=100, metadata={"help": "Warmup steps before any IGU rank pruning."})
    igu_final_warmup: int = field(default=100, metadata={"help": "Final fine-tuning steps with fixed rank masks."})
    igu_mask_interval: int = field(default=50, metadata={"help": "Step interval between two rank-mask updates."})
    igu_beta1: float = field(default=0.85, metadata={"help": "EMA momentum for importance scores."})
    igu_beta2: float = field(default=0.85, metadata={"help": "EMA momentum for uncertainty scores."})
    igu_eps: float = field(default=1e-6, metadata={"help": "Numerical epsilon for SNR scoring."})
    igu_r_min: int = field(default=1, metadata={"help": "Minimum active rank preserved per module."})
    igu_reset_optimizer_on_mask: bool = field(
        default=False,
        metadata={"help": "Whether to clear optimizer state after each rank-mask update."},
    )

    layers_to_transform: Optional[Union[List[int], int]] = field(default=None)
    layers_pattern: Optional[Union[List[str], str]] = field(default=None)

    def __post_init__(self):
        self.peft_type = PeftType.UNILORA_IGU
        self.target_modules = set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules

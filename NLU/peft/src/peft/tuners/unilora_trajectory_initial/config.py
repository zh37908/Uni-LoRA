from dataclasses import dataclass, field
from typing import List, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class UniLoRATrajectoryInitialConfig(PeftConfig):
    """
    UniLoRA trajectory-initial variant.

    Keep one global theta_d, but cluster target layers by block signatures extracted
    from pretrained weights. Layers assigned to the same cluster share the same
    theta_d bucket during index initialization.
    """

    r: int = field(default=4, metadata={"help": "The rank of incremental matrices."})
    proj_seed: int = field(default=42, metadata={"help": "Random seed for global index assignment."})
    theta_d_length: int = field(default=256, metadata={"help": "Total theta_d length."})
    num_buckets: int = field(default=4, metadata={"help": "Number of clustered theta_d buckets."})
    block_rows: int = field(default=4, metadata={"help": "Block rows for layer signature extraction."})
    block_cols: int = field(default=4, metadata={"help": "Block cols for layer signature extraction."})
    kmeans_iters: int = field(default=15, metadata={"help": "Number of k-means refinement iterations."})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of module names to replace."
                " Can also be 'all-linear'."
            )
        },
    )
    unilora_dropout: float = field(default=0.0, metadata={"help": "UniLoRA dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer stores weights like (fan_in, fan_out)."},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for UniLoRA adapters."})
    modules_to_save: Optional[List[str]] = field(default=None)
    init_theta_d_bound: float = field(default=0.02, metadata={"help": "Uniform init bound for theta_d."})
    layers_to_transform: Optional[Union[List[int], int]] = field(default=None)
    layers_pattern: Optional[Union[List[str], str]] = field(default=None)

    def __post_init__(self):
        self.peft_type = PeftType.UNILORA_TRAJECTORY_INITIAL
        self.target_modules = set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules

        if self.theta_d_length <= 0:
            raise ValueError("`theta_d_length` must be > 0.")
        if self.num_buckets <= 0:
            raise ValueError("`num_buckets` must be > 0.")
        if self.block_rows <= 0 or self.block_cols <= 0:
            raise ValueError("`block_rows` and `block_cols` must be > 0.")
        if self.kmeans_iters <= 0:
            raise ValueError("`kmeans_iters` must be > 0.")

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class GeoUniLoRAConfig(PeftConfig):
    """
    Geo-UniLoRA: grouped shared bank + per-module innovation bank, dual-branch UniLoRA.

    Each module applies DeltaW = B_sh @ A_sh + B_in @ A_in with indices into shared/innovation
    theta_d banks (implicit projection), matching the repository's UniLoRA style.
    """

    r: int = field(default=4, metadata={"help": "Default rank when rank maps are missing."})
    proj_seed: int = field(default=42, metadata={"help": "Random seed for index generation."})
    shared_theta_d_length: int = field(default=256, metadata={"help": "Length of per-group shared theta_d bank."})
    innovation_theta_d_length: int = field(
        default=256, metadata={"help": "Length of per-module innovation theta_d bank."}
    )
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with Geo-UniLoRA. "
                "This can also be the wildcard 'all-linear'."
            )
        },
    )
    unilora_dropout: float = field(default=0.0, metadata={"help": "Dropout on adapter path."})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)."},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for Geo-UniLoRA adapters."})
    modules_to_save: Optional[List[str]] = field(default=None)
    init_theta_d_bound: float = field(default=0.02, metadata={"help": "Uniform init bound for theta_d banks."})

    # Geometry / allocation (computed before adapter injection, typically in training script).
    geo_group_map: Optional[Dict[str, int]] = field(
        default=None,
        metadata={"help": "Mapping from full module name to group id (int)."},
    )
    geo_shared_rank_map: Optional[Dict[str, int]] = field(
        default=None,
        metadata={"help": "Per-module shared-branch rank r_sh."},
    )
    geo_innovation_rank_map: Optional[Dict[str, int]] = field(
        default=None,
        metadata={"help": "Per-module innovation-branch rank r_in."},
    )

    layers_to_transform: Optional[Union[List[int], int]] = field(default=None)
    layers_pattern: Optional[Union[List[str], str]] = field(default=None)

    def __post_init__(self):
        self.peft_type = PeftType.GEO_UNILORA
        self.target_modules = set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules

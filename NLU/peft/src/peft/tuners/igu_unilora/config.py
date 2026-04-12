from dataclasses import dataclass

from peft.utils import PeftType

from ..geo_unilora.config import GeoUniLoRAConfig


@dataclass
class IGUUniLoRAConfig(GeoUniLoRAConfig):
    """
    IGU-inspired UniLoRA config.

    v1 reuses Geo-UniLoRA dual-bank parameterization, while rank maps are provided
    by an IGU-style calibration planner in the training script.
    """

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.IGU_UNILORA

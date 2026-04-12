from peft.utils import register_peft_method

from .config import UniLoRASoftWeightSharingConfig
from .layer import Linear, UniLoRASoftWeightSharingLayer
from .model import UniLoRASoftWeightSharingModel

__all__ = [
    "UniLoRASoftWeightSharingConfig",
    "UniLoRASoftWeightSharingLayer",
    "Linear",
    "UniLoRASoftWeightSharingModel",
]

register_peft_method(
    name="unilora_soft_weight_sharing",
    config_cls=UniLoRASoftWeightSharingConfig,
    model_cls=UniLoRASoftWeightSharingModel,
)

from peft.utils import register_peft_method

from .config import UniLoRARoSAConfig, UniLoRARoSASnipConfig
from .layer import Linear, UniLoRARoSALayer
from .model import UniLoRARoSAModel, UniLoRARoSASnipModel

__all__ = [
    "UniLoRARoSAConfig",
    "UniLoRARoSASnipConfig",
    "UniLoRARoSALayer",
    "Linear",
    "UniLoRARoSAModel",
    "UniLoRARoSASnipModel",
]

register_peft_method(
    name="unilora_rosa",
    config_cls=UniLoRARoSAConfig,
    model_cls=UniLoRARoSAModel,
)

register_peft_method(
    name="unilora_rosa_snip",
    config_cls=UniLoRARoSASnipConfig,
    model_cls=UniLoRARoSASnipModel,
)

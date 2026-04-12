from peft.utils import register_peft_method

from .config import UniLoRARoSAConfig
from .layer import Linear, UniLoRARoSALayer
from .model import UniLoRARoSAModel

__all__ = [
    "UniLoRARoSAConfig",
    "UniLoRARoSALayer",
    "Linear",
    "UniLoRARoSAModel",
]

register_peft_method(
    name="unilora_rosa",
    config_cls=UniLoRARoSAConfig,
    model_cls=UniLoRARoSAModel,
)

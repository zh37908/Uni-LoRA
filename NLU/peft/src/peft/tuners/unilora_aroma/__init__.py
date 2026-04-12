from peft.utils import register_peft_method

from .config import UniLoRAAromaConfig
from .layer import Linear, UniLoRAAromaLayer
from .model import UniLoRAAromaModel

__all__ = [
    "UniLoRAAromaConfig",
    "UniLoRAAromaLayer",
    "Linear",
    "UniLoRAAromaModel",
]

register_peft_method(
    name="unilora_aroma",
    config_cls=UniLoRAAromaConfig,
    model_cls=UniLoRAAromaModel,
)

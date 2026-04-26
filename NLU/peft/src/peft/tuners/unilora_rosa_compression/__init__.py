from peft.utils import register_peft_method

from .config import UniLoRARoSACompressionConfig
from .layer import UniLoRARoSACompressionLayer, Linear
from .model import UniLoRARoSACompressionModel

__all__ = [
    "UniLoRARoSACompressionConfig",
    "UniLoRARoSACompressionLayer",
    "Linear",
    "UniLoRARoSACompressionModel",
]

register_peft_method(
    name="unilora_rosa_compression",
    config_cls=UniLoRARoSACompressionConfig,
    model_cls=UniLoRARoSACompressionModel,
)


from peft.utils import register_peft_method

from .config import UniLoRARoSAGlobalConfig
from .layer import Linear, UniLoRARoSAGlobalLayer
from .model import UniLoRARoSAGlobalModel

__all__ = [
    "UniLoRARoSAGlobalConfig",
    "UniLoRARoSAGlobalLayer",
    "Linear",
    "UniLoRARoSAGlobalModel",
]

register_peft_method(
    name="unilora_rosa_global",
    config_cls=UniLoRARoSAGlobalConfig,
    model_cls=UniLoRARoSAGlobalModel,
)

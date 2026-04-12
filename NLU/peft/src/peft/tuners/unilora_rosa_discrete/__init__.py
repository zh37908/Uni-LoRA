from peft.utils import register_peft_method

from .config import UniLoRARoSADiscreteConfig
from .layer import Linear, UniLoRARoSADiscreteLayer
from .model import UniLoRARoSADiscreteModel

__all__ = [
    "UniLoRARoSADiscreteConfig",
    "UniLoRARoSADiscreteLayer",
    "Linear",
    "UniLoRARoSADiscreteModel",
]

register_peft_method(
    name="unilora_rosa_discrete",
    config_cls=UniLoRARoSADiscreteConfig,
    model_cls=UniLoRARoSADiscreteModel,
)

from peft.utils import register_peft_method

from .config import UniLoRAIGUConfig
from .layer import Linear, UniLoRAIGULayer
from .model import UniLoRAIGUModel

__all__ = [
    "UniLoRAIGUConfig",
    "UniLoRAIGULayer",
    "Linear",
    "UniLoRAIGUModel",
]

register_peft_method(
    name="unilora_igu",
    config_cls=UniLoRAIGUConfig,
    model_cls=UniLoRAIGUModel,
)

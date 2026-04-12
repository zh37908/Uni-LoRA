from peft.utils import register_peft_method

from .config import IGUUniLoRAConfig
from .layer import IGUUniLoRALayer, Linear
from .model import IGUUniLoRAModel

__all__ = [
    "IGUUniLoRAConfig",
    "IGUUniLoRALayer",
    "Linear",
    "IGUUniLoRAModel",
]

register_peft_method(
    name="igu_unilora",
    config_cls=IGUUniLoRAConfig,
    model_cls=IGUUniLoRAModel,
)


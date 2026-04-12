from peft.utils import register_peft_method

from .config import UniLoRAGeLoRAConfig
from .layer import Linear, UniLoRAGeLoRALayer
from .model import UniLoRAGeLoRAModel

__all__ = [
    "UniLoRAGeLoRAConfig",
    "UniLoRAGeLoRALayer",
    "Linear",
    "UniLoRAGeLoRAModel",
]

register_peft_method(
    name="unilora_gelora",
    config_cls=UniLoRAGeLoRAConfig,
    model_cls=UniLoRAGeLoRAModel,
)

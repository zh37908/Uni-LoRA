from peft.utils import register_peft_method

from .config import UniLoRAGoRAConfig
from .layer import Linear, UniLoRAGoRALayer
from .model import UniLoRAGoRAModel

__all__ = [
    "UniLoRAGoRAConfig",
    "UniLoRAGoRALayer",
    "Linear",
    "UniLoRAGoRAModel",
]

register_peft_method(
    name="unilora_gora",
    config_cls=UniLoRAGoRAConfig,
    model_cls=UniLoRAGoRAModel,
)

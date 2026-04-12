from peft.utils import register_peft_method

from .config import UniLoRASwapConfig
from .layer import Linear, UniLoRALayer
from .model import UniLoRASwapModel

__all__ = [
    "UniLoRASwapConfig",
    "UniLoRALayer",
    "Linear",
    "UniLoRASwapModel",
]

register_peft_method(
    name="unilora_swap",
    config_cls=UniLoRASwapConfig,
    model_cls=UniLoRASwapModel,
)

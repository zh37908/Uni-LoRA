from peft.utils import register_peft_method

from .config import UniLoRALocalSwapConfig
from .layer import Linear, UniLoRALocalSwapLayer
from .model import UniLoRALocalSwapModel

__all__ = [
    "UniLoRALocalSwapConfig",
    "UniLoRALocalSwapLayer",
    "Linear",
    "UniLoRALocalSwapModel",
]

register_peft_method(
    name="unilora_local_swap",
    config_cls=UniLoRALocalSwapConfig,
    model_cls=UniLoRALocalSwapModel,
)

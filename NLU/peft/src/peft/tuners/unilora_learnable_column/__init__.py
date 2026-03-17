from peft.utils import register_peft_method

from .config import UniLoRALearnableColumnConfig
from .layer import Linear, UniLoRALayer
from .model import UniLoRALearnableColumnModel

__all__ = [
    "UniLoRALearnableColumnConfig",
    "Linear",
    "UniLoRALayer",
    "UniLoRALearnableColumnModel",
]

register_peft_method(
    name="unilora_learnable_column",
    config_cls=UniLoRALearnableColumnConfig,
    model_cls=UniLoRALearnableColumnModel,
)

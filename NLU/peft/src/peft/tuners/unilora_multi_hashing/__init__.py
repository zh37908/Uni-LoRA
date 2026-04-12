from peft.utils import register_peft_method

from .config import UniLoRAMultiHashingConfig
from .layer import Linear, UniLoRAMultiHashingLayer
from .model import UniLoRAMultiHashingModel

__all__ = [
    "UniLoRAMultiHashingConfig",
    "UniLoRAMultiHashingLayer",
    "Linear",
    "UniLoRAMultiHashingModel",
]

register_peft_method(
    name="unilora_multi_hashing",
    config_cls=UniLoRAMultiHashingConfig,
    model_cls=UniLoRAMultiHashingModel,
)

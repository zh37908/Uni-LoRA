from peft.utils import register_peft_method

from .config import UniLoRAMultiStructuredGlobalConfig
from .layer import Linear, UniLoRAMultiStructuredGlobalLayer
from .model import UniLoRAMultiStructuredGlobalModel

__all__ = [
    "UniLoRAMultiStructuredGlobalConfig",
    "UniLoRAMultiStructuredGlobalLayer",
    "Linear",
    "UniLoRAMultiStructuredGlobalModel",
]

register_peft_method(
    name="unilora_multi_structured_global",
    config_cls=UniLoRAMultiStructuredGlobalConfig,
    model_cls=UniLoRAMultiStructuredGlobalModel,
)

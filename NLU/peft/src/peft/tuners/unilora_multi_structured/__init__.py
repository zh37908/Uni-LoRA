from peft.utils import register_peft_method

from .config import UniLoRAMultiStructuredConfig
from .layer import Linear, UniLoRAMultiStructuredLayer
from .model import UniLoRAMultiStructuredModel

__all__ = [
    "UniLoRAMultiStructuredConfig",
    "UniLoRAMultiStructuredLayer",
    "Linear",
    "UniLoRAMultiStructuredModel",
]

register_peft_method(
    name="unilora_multi_structured",
    config_cls=UniLoRAMultiStructuredConfig,
    model_cls=UniLoRAMultiStructuredModel,
)

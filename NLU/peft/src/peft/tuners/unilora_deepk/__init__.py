from peft.utils import register_peft_method

from .config import UniLoRADeepKConfig
from .layer import Linear, UniLoRADeepKLayer
from .model import UniLoRADeepKModel

__all__ = ["UniLoRADeepKConfig", "UniLoRADeepKLayer", "Linear", "UniLoRADeepKModel"]

register_peft_method(
    name="unilora_deepk",
    config_cls=UniLoRADeepKConfig,
    model_cls=UniLoRADeepKModel,
)

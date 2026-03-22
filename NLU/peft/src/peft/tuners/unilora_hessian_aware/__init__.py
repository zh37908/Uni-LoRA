from peft.utils import register_peft_method

from .config import UniLoRAHessianAwareConfig
from .layer import Linear, UniLoRAHessianAwareLayer
from .model import UniLoRAHessianAwareModel

__all__ = [
    "UniLoRAHessianAwareConfig",
    "UniLoRAHessianAwareLayer",
    "Linear",
    "UniLoRAHessianAwareModel",
]

register_peft_method(
    name="unilora_hessian_aware",
    config_cls=UniLoRAHessianAwareConfig,
    model_cls=UniLoRAHessianAwareModel,
)

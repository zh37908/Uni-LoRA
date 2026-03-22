from peft.utils import register_peft_method

from .config import UniLoRASketchRoutedConfig
from .layer import Linear, UniLoRASketchRoutedLayer
from .model import UniLoRASketchRoutedModel

__all__ = [
    "UniLoRASketchRoutedConfig",
    "UniLoRASketchRoutedLayer",
    "Linear",
    "UniLoRASketchRoutedModel",
]

register_peft_method(
    name="unilora_sketch_routed",
    config_cls=UniLoRASketchRoutedConfig,
    model_cls=UniLoRASketchRoutedModel,
)

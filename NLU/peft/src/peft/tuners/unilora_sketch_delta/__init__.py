from peft.utils import register_peft_method

from .config import UniLoRASketchDeltaConfig
from .layer import Linear, UniLoRASketchDeltaLayer
from .model import UniLoRASketchDeltaModel

__all__ = [
    "UniLoRASketchDeltaConfig",
    "UniLoRASketchDeltaLayer",
    "Linear",
    "UniLoRASketchDeltaModel",
]

register_peft_method(
    name="unilora_sketch_delta",
    config_cls=UniLoRASketchDeltaConfig,
    model_cls=UniLoRASketchDeltaModel,
)

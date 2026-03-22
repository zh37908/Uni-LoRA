from peft.utils import register_peft_method

from .config import UniLoRASketchTuneConfig
from .layer import Linear, UniLoRASketchTuneLayer
from .model import UniLoRASketchTuneModel

__all__ = [
    "UniLoRASketchTuneConfig",
    "UniLoRASketchTuneLayer",
    "Linear",
    "UniLoRASketchTuneModel",
]

register_peft_method(
    name="unilora_sketch_tune",
    config_cls=UniLoRASketchTuneConfig,
    model_cls=UniLoRASketchTuneModel,
)

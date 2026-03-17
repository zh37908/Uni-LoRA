from peft.utils import register_peft_method

from .config import UniLoRACountSketchConfig
from .layer import Linear, UniLoRACountSketchLayer
from .model import UniLoRACountSketchModel

__all__ = ["UniLoRACountSketchConfig", "UniLoRACountSketchLayer", "Linear", "UniLoRACountSketchModel"]

register_peft_method(
    name="unilora_count_sketch", config_cls=UniLoRACountSketchConfig, model_cls=UniLoRACountSketchModel
)

from peft.utils import register_peft_method

from .config import DirectUniLoRAConfig
from .layer import Linear, DirectUniLoRALayer
from .model import DirectUniLoRAModel

__all__ = ["DirectUniLoRAConfig", "DirectUniLoRALayer", "Linear", "DirectUniLoRAModel"]
register_peft_method(name="direct_unilora", config_cls=DirectUniLoRAConfig, model_cls=DirectUniLoRAModel)

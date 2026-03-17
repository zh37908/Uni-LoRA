from peft.utils import register_peft_method

from .config import UniLoRAGSConfig
from .layer import Linear, UniLoRGSLayer
from .model import UniLoRAGSModel

__all__ = ["UniLoRAGSConfig", "UniLoRGSLayer", "Linear", "UniLoRAGSModel"]
register_peft_method(name="unilora_gs", config_cls=UniLoRAGSConfig, model_cls=UniLoRAGSModel)

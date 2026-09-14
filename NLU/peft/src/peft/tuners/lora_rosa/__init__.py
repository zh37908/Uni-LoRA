from peft.utils import register_peft_method

from .config import LoraRoSAConfig, LoraRoSARandomConfig, LoraRoSASnipConfig
from .layer import Linear, LoraRoSALayer
from .model import LoraRoSAModel, LoraRoSARandomModel, LoraRoSASnipModel

__all__ = [
    "LoraRoSAConfig",
    "LoraRoSASnipConfig",
    "LoraRoSARandomConfig",
    "LoraRoSALayer",
    "Linear",
    "LoraRoSAModel",
    "LoraRoSASnipModel",
    "LoraRoSARandomModel",
]

register_peft_method(
    name="lora_rosa",
    config_cls=LoraRoSAConfig,
    model_cls=LoraRoSAModel,
)

register_peft_method(
    name="lora_rosa_snip",
    config_cls=LoraRoSASnipConfig,
    model_cls=LoraRoSASnipModel,
)

register_peft_method(
    name="lora_rosa_random",
    config_cls=LoraRoSARandomConfig,
    model_cls=LoraRoSARandomModel,
)

from peft.utils import register_peft_method

from .config import UniLoRABlockRoutingConfig
from .layer import Linear, UniLoRABlockRoutingLayer
from .model import UniLoRABlockRoutingModel

__all__ = ["UniLoRABlockRoutingConfig", "UniLoRABlockRoutingLayer", "Linear", "UniLoRABlockRoutingModel"]
register_peft_method(name="unilora_block_routing", config_cls=UniLoRABlockRoutingConfig, model_cls=UniLoRABlockRoutingModel)

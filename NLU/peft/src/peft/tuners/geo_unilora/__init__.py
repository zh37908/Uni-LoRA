from peft.utils import register_peft_method

from .config import GeoUniLoRAConfig
from .layer import GeoUniLoRALayer, Linear
from .model import GeoUniLoRAModel

__all__ = [
    "GeoUniLoRAConfig",
    "GeoUniLoRALayer",
    "Linear",
    "GeoUniLoRAModel",
]

register_peft_method(
    name="geo_unilora",
    config_cls=GeoUniLoRAConfig,
    model_cls=GeoUniLoRAModel,
)

from peft.utils import register_peft_method

from .config import UniLoRASoftAssignConfig
from .layer import Linear, UniLoRASoftAssignLayer
from .model import UniLoRASoftAssignModel

__all__ = ["UniLoRASoftAssignConfig", "UniLoRASoftAssignLayer", "Linear", "UniLoRASoftAssignModel"]
register_peft_method(
    name="unilora_soft_assign",
    config_cls=UniLoRASoftAssignConfig,
    model_cls=UniLoRASoftAssignModel,
)

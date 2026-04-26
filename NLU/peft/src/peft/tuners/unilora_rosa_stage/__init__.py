from peft.utils import register_peft_method

from ..unilora_rosa.layer import Linear, UniLoRARoSALayer
from .config import UniLoRARoSAStageConfig
from .model import UniLoRARoSAStageModel

__all__ = [
    "UniLoRARoSAStageConfig",
    "UniLoRARoSALayer",
    "Linear",
    "UniLoRARoSAStageModel",
]

register_peft_method(
    name="unilora_rosa_stage",
    config_cls=UniLoRARoSAStageConfig,
    model_cls=UniLoRARoSAStageModel,
)

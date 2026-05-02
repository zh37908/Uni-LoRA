from peft.utils import register_peft_method

from ..unilora_rosa.layer import Linear, UniLoRARoSALayer
from .config import UniLoRARoSAStageConfig, UniLoRARoSAStageSnipConfig
from .model import UniLoRARoSAStageModel, UniLoRARoSAStageSnipModel

__all__ = [
    "UniLoRARoSAStageConfig",
    "UniLoRARoSAStageSnipConfig",
    "UniLoRARoSALayer",
    "Linear",
    "UniLoRARoSAStageModel",
    "UniLoRARoSAStageSnipModel",
]

register_peft_method(
    name="unilora_rosa_stage",
    config_cls=UniLoRARoSAStageConfig,
    model_cls=UniLoRARoSAStageModel,
)

register_peft_method(
    name="unilora_rosa_stage_snip",
    config_cls=UniLoRARoSAStageSnipConfig,
    model_cls=UniLoRARoSAStageSnipModel,
)

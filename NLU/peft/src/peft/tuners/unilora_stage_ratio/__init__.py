from peft.utils import register_peft_method

from .config import UniLoRAStageRatioConfig
from .layer import Linear, UniLoRALayer
from .model import UniLoRAStageRatioModel

__all__ = ["UniLoRAStageRatioConfig", "UniLoRALayer", "Linear", "UniLoRAStageRatioModel"]

register_peft_method(
    name="unilora_stage_ratio",
    config_cls=UniLoRAStageRatioConfig,
    model_cls=UniLoRAStageRatioModel,
)

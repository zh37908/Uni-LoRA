from peft.utils import register_peft_method

from .config import UniLoRATrajectoryInitialConfig
from .layer import Linear, UniLoRALayer
from .model import UniLoRATrajectoryInitialModel

__all__ = ["UniLoRATrajectoryInitialConfig", "UniLoRALayer", "Linear", "UniLoRATrajectoryInitialModel"]

register_peft_method(
    name="unilora_trajectory_initial",
    config_cls=UniLoRATrajectoryInitialConfig,
    model_cls=UniLoRATrajectoryInitialModel,
)

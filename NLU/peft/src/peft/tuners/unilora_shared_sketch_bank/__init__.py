from peft.utils import register_peft_method

from .config import UniLoRASharedSketchBankConfig
from .layer import Linear, UniLoRASharedSketchBankLayer
from .model import UniLoRASharedSketchBankModel

__all__ = [
    "UniLoRASharedSketchBankConfig",
    "UniLoRASharedSketchBankLayer",
    "Linear",
    "UniLoRASharedSketchBankModel",
]

register_peft_method(
    name="unilora_shared_sketch_bank",
    config_cls=UniLoRASharedSketchBankConfig,
    model_cls=UniLoRASharedSketchBankModel,
)

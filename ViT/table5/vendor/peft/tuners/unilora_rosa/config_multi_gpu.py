from dataclasses import dataclass

from peft.utils import PeftType

from .config import UniLoRARoSASnipConfig


@dataclass
class UniLoRARoSASnipMultiGpuConfig(UniLoRARoSASnipConfig):
    """
    UniLoRA-RoSA-SNIP variant whose forward path avoids whole-vector transfers
    under `device_map=auto` model parallelism.
    """

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.UNILORA_ROSA_SNIP_MULTI_GPU

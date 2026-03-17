# Copyright 2024-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import List, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class UniLoRAIsometricControlConfig(PeftConfig):
    """
    Configuration for UniLoRA with Isometry Control.

    Args:
        isometry_alpha (`float`):
            Control parameter for isometry deviation. 
            0.0: Fully isometric (P^T P = I).
            1.0: Non-isometric (P^T P = diag(counts)).
            Default is 0.0.
    """

    r: int = field(default=4, metadata={"help": "The rank of incremental matrices."})
    proj_seed: int = field(
        default=42,
        metadata={"help": "Random seed for initializing the projection matrix."}
    )
    isometry_alpha: float = field(
        default=0.0,
        metadata={"help": "Control parameter for isometry deviation. 0.0: Fully isometric, 1.0: Non-isometric."}
    )
    theta_d_length: int = field(
        default=256,
        metadata={
            "help": "The length of the vectors in the vector bank. The length of the vectors should be divisible by "
            "the hidden dimension of the model."
        },
    )
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with LoRA."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'."
                "This can also be a wildcard 'all-linear' which matches all linear/Conv1D layers except the output layer."
            )
        },
    )
    unilora_dropout: float = field(default=0.0, metadata={"help": "UniLoRA dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for UniLoRA. Can be 'none', 'all' or 'unilora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of modules apart from UniLoRA layers to be set as trainable and saved."}
    )
    init_theta_d_bound: float = field(
        default=0.02,
        metadata={"help": "Initialization bound for the vector bank."},
    )
    layers_to_transform: Optional[Union[List[int], int]] = field(default=None)
    layers_pattern: Optional[Union[List[str], str]] = field(default=None)

    def __post_init__(self):
        self.peft_type = PeftType.UNILORA_ISOMETRIC_CONTROL
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )

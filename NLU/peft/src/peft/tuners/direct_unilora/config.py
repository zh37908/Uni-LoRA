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
class DirectUniLoRAConfig(PeftConfig):
    """
    Configuration for Direct-UniLoRA: directly projects theta_d to a full delta weight.
    """

    # Keep r for CLI compatibility even though direct variant doesn't use low-rank AB
    r: int = field(default=4, metadata={"help": "Unused for direct_unilora, kept for CLI compatibility."})
    proj_seed: int = field(default=42, metadata={"help": "Random seed for initializing the projection matrix."})
    theta_d_length: int = field(
        default=256,
        metadata={
            "help": "Length of the vector bank used by the projection mapping."
        },
    )
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with Direct-UniLoRA."
                "This can also be a wildcard 'all-linear' which matches all linear/Conv1D layers except the output layer."
            )
        },
    )
    unilora_dropout: float = field(default=0.0, metadata={"help": "Direct-UniLoRA dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for Direct-UniLoRA. Can be 'none', 'all' or 'unilora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": (
                "List of modules apart from Direct-UniLoRA layers to be set as trainable and saved in the final checkpoint."
            )
        },
    )
    init_theta_d_bound: float = field(
        default=0.02,
        metadata={
            "help": (
                "The vector bank is initialized with a uniform distribution between -init_theta_d_bound and"
                " init_theta_d_bound."
            )
        },
    )
    layers_to_transform: Optional[Union[List[int], int]] = field(
        default=None,
        metadata={
            "help": (
                "Layer indexes to transform. This only works when target_modules is a list of str."
            )
        },
    )
    layers_pattern: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "Layer pattern name used only if layers_to_transform is not None."
            )
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.DIRECT_UNILORA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )

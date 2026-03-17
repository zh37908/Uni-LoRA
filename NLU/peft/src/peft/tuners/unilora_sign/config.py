# Copyright 2026-present
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
class UniLoRASignConfig(PeftConfig):
    r: int = field(default=4, metadata={"help": "The rank of incremental matrices."})
    proj_seed: int = field(default=42, metadata={"help": "Random seed for initializing the projection matrix."})
    theta_d_length: int = field(
        default=256,
        metadata={"help": "The length of the shared vector bank."},
    )
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with LoRA."
                " This can also be a wildcard 'all-linear' which matches all linear/Conv1D layers except output."
            )
        },
    )
    unilora_dropout: float = field(default=0.0, metadata={"help": "UniLoRA dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the target stores weight like (fan_in, fan_out)."},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for UniLoRA. Can be 'none', 'all' or 'unilora_only'"})
    modules_to_save: Optional[List[str]] = field(default=None)
    init_theta_d_bound: float = field(
        default=0.02,
        metadata={"help": "Initialize theta_d with U(-b, b)."},
    )
    layers_to_transform: Optional[Union[List[int], int]] = field(default=None)
    layers_pattern: Optional[Union[List[str], str]] = field(default=None)

    def __post_init__(self):
        self.peft_type = PeftType.UNILORA_SIGN
        self.target_modules = set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules

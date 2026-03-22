# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all

# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
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
import enum
from typing import Optional


class PeftType(str, enum.Enum):
    """
    Enum class for the different types of adapters in PEFT.

    Supported PEFT types:
    - PROMPT_TUNING
    - MULTITASK_PROMPT_TUNING
    - P_TUNING
    - PREFIX_TUNING
    - LORA
    - ADALORA
    - BOFT
    - ADAPTION_PROMPT
    - IA3
    - LOHA
    - LOKR
    - OFT
    - XLORA
    - POLY
    - LN_TUNING
    - VERA
    - FOURIERFT
    - HRA
    """

    PROMPT_TUNING = "PROMPT_TUNING"
    MULTITASK_PROMPT_TUNING = "MULTITASK_PROMPT_TUNING"
    P_TUNING = "P_TUNING"
    PREFIX_TUNING = "PREFIX_TUNING"
    LORA = "LORA"
    ADALORA = "ADALORA"
    BOFT = "BOFT"
    ADAPTION_PROMPT = "ADAPTION_PROMPT"
    IA3 = "IA3"
    LOHA = "LOHA"
    LOKR = "LOKR"
    OFT = "OFT"
    POLY = "POLY"
    LN_TUNING = "LN_TUNING"
    VERA = "VERA"
    FOURIERFT = "FOURIERFT"
    XLORA = "XLORA"
    HRA = "HRA"
    UNILORA = "UNILORA"
    UNILORA_NONORM = "UNILORA_NONORM"


class TaskType(str, enum.Enum):
    """
    Enum class for the different types of tasks supported by PEFT.

    Overview of the supported task types:
    - SEQ_CLS: Text classification.
    - SEQ_2_SEQ_LM: Sequence-to-sequence language modeling.
    - CAUSAL_LM: Causal language modeling.
    - TOKEN_CLS: Token classification.
    - QUESTION_ANS: Question answering.
    - FEATURE_EXTRACTION: Feature extraction. Provides the hidden states which can be used as embeddings or features
      for downstream tasks.
    """

    SEQ_CLS = "SEQ_CLS"
    SEQ_2_SEQ_LM = "SEQ_2_SEQ_LM"
    CAUSAL_LM = "CAUSAL_LM"
    TOKEN_CLS = "TOKEN_CLS"
    QUESTION_ANS = "QUESTION_ANS"
    FEATURE_EXTRACTION = "FEATURE_EXTRACTION"


def register_peft_method(*, name: str, config_cls, model_cls, prefix: Optional[str] = None, is_mixed_compatible=False) -> None:
    """
    Register a PEFT method in the legacy instruction_tuning mapping tables.

    This local PEFT fork predates the newer dynamic registration API, so we only
    update the config/tuner mappings that exist in this tree.
    """
    del prefix
    del is_mixed_compatible

    from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING, PEFT_TYPE_TO_TUNER_MAPPING

    if name.endswith("_"):
        raise ValueError(f"Please pass the name of the PEFT method without '_' suffix, got {name}.")

    if not name.islower():
        raise ValueError(f"The name of the PEFT method should be in lower case letters, got {name}.")

    peft_type_name = name.upper()
    if peft_type_name not in list(PeftType):
        raise ValueError(f"Unknown PEFT type {peft_type_name}, please add it to peft.utils.peft_types.PeftType.")

    peft_type = getattr(PeftType, peft_type_name)
    PEFT_TYPE_TO_CONFIG_MAPPING[peft_type] = config_cls
    PEFT_TYPE_TO_TUNER_MAPPING[peft_type] = model_cls

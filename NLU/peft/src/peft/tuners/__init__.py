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

from .adalora import AdaLoraConfig, AdaLoraModel
from .adaption_prompt import AdaptionPromptConfig, AdaptionPromptModel
from .boft import BOFTConfig, BOFTModel
from .bone import BoneConfig, BoneModel
from .c3a import C3AConfig, C3AModel
from .cpt import CPTConfig, CPTEmbedding
from .delora import DeloraConfig, DeloraModel
from .fourierft import FourierFTConfig, FourierFTModel
from .gralora import GraloraConfig, GraloraModel
from .hra import HRAConfig, HRAModel
from .ia3 import IA3Config, IA3Model
from .ln_tuning import LNTuningConfig, LNTuningModel
from .loha import LoHaConfig, LoHaModel
from .lokr import LoKrConfig, LoKrModel
from .lora import (
    ArrowConfig,
    BdLoraConfig,
    EvaConfig,
    LoftQConfig,
    LoraConfig,
    LoraModel,
    LoraRuntimeConfig,
    create_arrow_model,
    get_eva_state_dict,
    initialize_lora_eva_weights,
)
from .miss import MissConfig, MissModel
from .mixed import MixedModel
from .multitask_prompt_tuning import MultitaskPromptEmbedding, MultitaskPromptTuningConfig, MultitaskPromptTuningInit
from .oft import OFTConfig, OFTModel
from .osf import OSFConfig, OSFModel
from .p_tuning import PromptEncoder, PromptEncoderConfig, PromptEncoderReparameterizationType
from .poly import PolyConfig, PolyModel
from .prefix_tuning import PrefixEncoder, PrefixTuningConfig
from .prompt_tuning import PromptEmbedding, PromptTuningConfig, PromptTuningInit
from .randlora import RandLoraConfig, RandLoraModel
from .road import RoadConfig, RoadModel
from .shira import ShiraConfig, ShiraModel
from .trainable_tokens import TrainableTokensConfig, TrainableTokensModel
from .vblora import VBLoRAConfig, VBLoRAModel
from .unilora import UniLoRAConfig, UniLoRAModel
from .unilora_aroma import UniLoRAAromaConfig, UniLoRAAromaModel
from .unilora_sketch_tune import UniLoRASketchTuneConfig, UniLoRASketchTuneModel
from .unilora_sketch_delta import UniLoRASketchDeltaConfig, UniLoRASketchDeltaModel
from .unilora_shared_sketch_bank import UniLoRASharedSketchBankConfig, UniLoRASharedSketchBankModel
from .unilora_sketch_routed import UniLoRASketchRoutedConfig, UniLoRASketchRoutedModel
from .unilora_count_sketch import UniLoRACountSketchConfig, UniLoRACountSketchModel
from .unilora_sign import UniLoRASignConfig, UniLoRASignModel
from .unilora_nonorm import UniLoRANonormConfig, UniLoRANonormModel
from .unilora_fastfood import UniLoRAFastFoodConfig, UniLoRAFastFoodModel
from .unilora_learnable import UniLoRALearnableConfig, UniLoRALearnableModel
from .unilora_learnable_column import UniLoRALearnableColumnConfig, UniLoRALearnableColumnModel
from .unilora_isometric_control import UniLoRAIsometricControlConfig, UniLoRAIsometricControlModel
from .unilora_gs import UniLoRAGSConfig, UniLoRAGSModel
from .unilora_gora import UniLoRAGoRAConfig, UniLoRAGoRAModel
from .unilora_gelora import UniLoRAGeLoRAConfig, UniLoRAGeLoRAModel
from .geo_unilora import GeoUniLoRAConfig, GeoUniLoRAModel
from .igu_unilora import IGUUniLoRAConfig, IGUUniLoRAModel
from .unilora_igu import UniLoRAIGUConfig, UniLoRAIGUModel
from .unilora_soft_assign import UniLoRASoftAssignConfig, UniLoRASoftAssignModel
from .unilora_swap import UniLoRASwapConfig, UniLoRASwapModel
from .unilora_local_swap import UniLoRALocalSwapConfig, UniLoRALocalSwapModel
from .unilora_soft_weight_sharing import UniLoRASoftWeightSharingConfig, UniLoRASoftWeightSharingModel
from .unilora_deepk import UniLoRADeepKConfig, UniLoRADeepKModel
from .direct_unilora import DirectUniLoRAConfig, DirectUniLoRAModel
from .unilora_block_routing import UniLoRABlockRoutingConfig, UniLoRABlockRoutingModel
from .unilora_stage_ratio import UniLoRAStageRatioConfig, UniLoRAStageRatioModel
from .unilora_trajectory_initial import UniLoRATrajectoryInitialConfig, UniLoRATrajectoryInitialModel
from .unilora_layer_wise import UniLoRALayerWiseConfig, UniLoRALayerWiseModel
from .unilora_learnable_layer import UniLoRALearnableLayerConfig, UniLoRALearnableLayerModel
from .unilora_hessian_aware import UniLoRAHessianAwareConfig, UniLoRAHessianAwareModel
from .unilora_rosa import UniLoRARoSAConfig, UniLoRARoSAModel
from .unilora_rosa_discrete import UniLoRARoSADiscreteConfig, UniLoRARoSADiscreteModel
from .unilora_rosa_global import UniLoRARoSAGlobalConfig, UniLoRARoSAGlobalModel
from .unilora_multi_hashing import UniLoRAMultiHashingConfig, UniLoRAMultiHashingModel
from .unilora_multi_structured import UniLoRAMultiStructuredConfig, UniLoRAMultiStructuredModel
from .unilora_multi_structured_global import (
    UniLoRAMultiStructuredGlobalConfig,
    UniLoRAMultiStructuredGlobalModel,
)
from .vera import VeraConfig, VeraModel
from .waveft import WaveFTConfig, WaveFTModel
from .xlora import XLoraConfig, XLoraModel


__all__ = [
    "AdaLoraConfig",
    "AdaLoraModel",
    "AdaptionPromptConfig",
    "AdaptionPromptModel",
    "ArrowConfig",
    "BOFTConfig",
    "BOFTModel",
    "BdLoraConfig",
    "BoneConfig",
    "BoneModel",
    "C3AConfig",
    "C3AModel",
    "CPTConfig",
    "CPTEmbedding",
    "DeloraConfig",
    "DeloraModel",
    "EvaConfig",
    "FourierFTConfig",
    "FourierFTModel",
    "GraloraConfig",
    "GraloraModel",
    "HRAConfig",
    "HRAModel",
    "IA3Config",
    "IA3Model",
    "LNTuningConfig",
    "LNTuningModel",
    "LoHaConfig",
    "LoHaModel",
    "LoKrConfig",
    "LoKrModel",
    "LoftQConfig",
    "LoraConfig",
    "LoraModel",
    "LoraRuntimeConfig",
    "MissConfig",
    "MissModel",
    "MixedModel",
    "MultitaskPromptEmbedding",
    "MultitaskPromptTuningConfig",
    "MultitaskPromptTuningInit",
    "OFTConfig",
    "OFTModel",
    "OSFConfig",
    "OSFModel",
    "PolyConfig",
    "PolyModel",
    "PrefixEncoder",
    "PrefixTuningConfig",
    "PromptEmbedding",
    "PromptEncoder",
    "PromptEncoderConfig",
    "PromptEncoderReparameterizationType",
    "PromptTuningConfig",
    "PromptTuningInit",
    "RandLoraConfig",
    "RandLoraModel",
    "RoadConfig",
    "RoadModel",
    "ShiraConfig",
    "ShiraModel",
    "TrainableTokensConfig",
    "TrainableTokensModel",
    "VBLoRAConfig",
    "VBLoRAModel",
    "VeraConfig",
    "VeraModel",
    "WaveFTConfig",
    "WaveFTModel",
    "XLoraConfig",
    "XLoraModel",
    "UniLoRAConfig",
    "UniLoRAAromaConfig",
    "UniLoRAModel",
    "UniLoRAAromaModel",
    "UniLoRASketchTuneConfig",
    "UniLoRASketchTuneModel",
    "UniLoRASketchDeltaConfig",
    "UniLoRASketchDeltaModel",
    "UniLoRASharedSketchBankConfig",
    "UniLoRASharedSketchBankModel",
    "UniLoRASketchRoutedConfig",
    "UniLoRASketchRoutedModel",
    "UniLoRACountSketchConfig",
    "UniLoRACountSketchModel",
    "UniLoRASignConfig",
    "UniLoRASignModel",
    "UniLoRANonormConfig",
    "UniLoRANonormModel",
    "UniLoRAFastFoodConfig",
    "UniLoRAFastFoodModel",
    "UniLoRALearnableConfig",
    "UniLoRALearnableModel",
    "UniLoRALearnableColumnConfig",
    "UniLoRALearnableColumnModel",
    "UniLoRAIsometricControlConfig",
    "UniLoRAIsometricControlModel",
    "UniLoRAGoRAConfig",
    "UniLoRAGoRAModel",
    "UniLoRAGeLoRAConfig",
    "UniLoRAGeLoRAModel",
    "GeoUniLoRAConfig",
    "GeoUniLoRAModel",
    "IGUUniLoRAConfig",
    "IGUUniLoRAModel",
    "UniLoRAIGUConfig",
    "UniLoRAIGUModel",
    "UniLoRAGSConfig",
    "UniLoRAGSModel",
    "UniLoRASoftAssignConfig",
    "UniLoRASoftAssignModel",
    "UniLoRASwapConfig",
    "UniLoRASwapModel",
    "UniLoRALocalSwapConfig",
    "UniLoRALocalSwapModel",
    "UniLoRASoftWeightSharingConfig",
    "UniLoRASoftWeightSharingModel",
    "UniLoRADeepKConfig",
    "UniLoRADeepKModel",
    "DirectUniLoRAConfig",
    "DirectUniLoRAModel",
    "UniLoRABlockRoutingConfig",
    "UniLoRABlockRoutingModel",
    "UniLoRAStageRatioConfig",
    "UniLoRAStageRatioModel",
    "UniLoRATrajectoryInitialConfig",
    "UniLoRATrajectoryInitialModel",
    "UniLoRALayerWiseConfig",
    "UniLoRALayerWiseModel",
    "UniLoRALearnableLayerConfig",
    "UniLoRALearnableLayerModel",
    "UniLoRAHessianAwareConfig",
    "UniLoRAHessianAwareModel",
    "UniLoRARoSAConfig",
    "UniLoRARoSAModel",
    "UniLoRARoSADiscreteConfig",
    "UniLoRARoSADiscreteModel",
    "UniLoRARoSAGlobalConfig",
    "UniLoRARoSAGlobalModel",
    "UniLoRAMultiHashingConfig",
    "UniLoRAMultiHashingModel",
    "UniLoRAMultiStructuredConfig",
    "UniLoRAMultiStructuredModel",
    "UniLoRAMultiStructuredGlobalConfig",
    "UniLoRAMultiStructuredGlobalModel",
    "create_arrow_model",
    "get_eva_state_dict",
    "initialize_lora_eva_weights",
]

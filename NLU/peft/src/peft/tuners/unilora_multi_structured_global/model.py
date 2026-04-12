from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

from .._buffer_dict import BufferDict
from .config import UniLoRAMultiStructuredGlobalConfig
from .layer import Linear, UniLoRAMultiStructuredGlobalLayer


class UniLoRAMultiStructuredGlobalModel(BaseTuner):
    """
    UniLoRA variant that conceptually tiles all LoRA A/B elements into a single global matrix and
    parameterizes that global matrix with a structured multi-hash sum-of-products form.
    """

    prefix: str = "unilora_multi_structured_global_"
    tuner_layer_cls = UniLoRAMultiStructuredGlobalLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

        cfg = config[adapter_name]
        modules = [m for m in model.modules() if isinstance(m, UniLoRAMultiStructuredGlobalLayer)]
        total_ab_params = sum(
            (m.r[adapter_name] * m.in_features) + (m.out_features * m.r[adapter_name]) for m in modules
        )

        global_matrix_dim = int(math.ceil(math.sqrt(max(total_ab_params, 1))))
        if cfg.target_trainable_params is not None:
            num_hash_pairs = max(1, int(math.ceil(cfg.target_trainable_params / float(2 * global_matrix_dim))))
        else:
            num_hash_pairs = cfg.num_hash_pairs

        self._init_shared_global_factors(
            adapter_name,
            global_matrix_dim=global_matrix_dim,
            num_hash_pairs=num_hash_pairs,
            init_bound=cfg.init_bound,
        )
        self._assign_global_positions(model, adapter_name, global_matrix_dim=global_matrix_dim)

    def _init_shared_global_factors(
        self, adapter_name: str, global_matrix_dim: int, num_hash_pairs: int, init_bound: float
    ) -> None:
        if adapter_name in self.unilora_multi_structured_global_u:
            return

        u = torch.empty(num_hash_pairs, global_matrix_dim)
        v = torch.empty(num_hash_pairs, global_matrix_dim)
        torch.nn.init.uniform_(u, -init_bound, init_bound)
        torch.nn.init.uniform_(v, -init_bound, init_bound)
        self.unilora_multi_structured_global_u[adapter_name] = u
        self.unilora_multi_structured_global_v[adapter_name] = v
        self.unilora_multi_structured_global_meta[adapter_name] = torch.tensor(
            [global_matrix_dim, num_hash_pairs], dtype=torch.long
        )

    def _assign_global_positions(self, model: nn.Module, adapter_name: str, global_matrix_dim: int) -> None:
        modules = [m for m in model.modules() if isinstance(m, UniLoRAMultiStructuredGlobalLayer)]
        pointer = 0
        position_ids = []

        for module in modules:
            a_numel = module.global_linear_indices_A[adapter_name].numel()
            b_numel = module.global_linear_indices_B[adapter_name].numel()
            module.global_matrix_dim[adapter_name] = global_matrix_dim

            pos_a = torch.arange(pointer, pointer + a_numel, dtype=torch.long)
            pointer += a_numel
            pos_b = torch.arange(pointer, pointer + b_numel, dtype=torch.long)
            pointer += b_numel

            target_device = module.get_base_layer().weight.device
            module.global_linear_indices_A[adapter_name] = pos_a.view_as(module.global_linear_indices_A[adapter_name]).to(
                device=target_device, dtype=torch.long
            )
            module.global_linear_indices_B[adapter_name] = pos_b.view_as(module.global_linear_indices_B[adapter_name]).to(
                device=target_device, dtype=torch.long
            )
            position_ids.extend(pos_a.tolist())
            position_ids.extend(pos_b.tolist())

        if not position_ids:
            return

        flat_ids = torch.tensor(position_ids, dtype=torch.long)
        counts = torch.bincount(flat_ids, minlength=global_matrix_dim * global_matrix_dim).float()
        inv_sqrt = torch.zeros_like(counts)
        non_zero = counts > 0
        inv_sqrt[non_zero] = 1.0 / torch.sqrt(counts[non_zero])

        pointer = 0
        for module in modules:
            a_numel = module.global_linear_indices_A[adapter_name].numel()
            b_numel = module.global_linear_indices_B[adapter_name].numel()
            scale_a = inv_sqrt[flat_ids[pointer : pointer + a_numel]].view_as(module.global_linear_indices_A[adapter_name])
            pointer += a_numel
            scale_b = inv_sqrt[flat_ids[pointer : pointer + b_numel]].view_as(module.global_linear_indices_B[adapter_name])
            pointer += b_numel
            module.update_norm(adapter_name, scale_a, scale_b)

    def _pre_injection_hook(
        self, model: nn.Module, config: UniLoRAMultiStructuredGlobalConfig, adapter_name: str
    ) -> None:
        self.unilora_multi_structured_global_u = nn.ParameterDict({})
        self.unilora_multi_structured_global_v = nn.ParameterDict({})
        self.unilora_multi_structured_global_meta = BufferDict({}, persistent=True)

    def _create_and_replace(
        self,
        unilora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "fan_in_fan_out": unilora_config.fan_in_fan_out,
            "bias": bias,
        }

        global_matrix_dim = (
            int(self.unilora_multi_structured_global_meta[adapter_name][0].item())
            if adapter_name in self.unilora_multi_structured_global_meta
            else 1
        )
        if isinstance(target, Linear):
            target.update_layer(
                adapter_name=adapter_name,
                unilora_multi_structured_global_u=self.unilora_multi_structured_global_u,
                unilora_multi_structured_global_v=self.unilora_multi_structured_global_v,
                r=unilora_config.r,
                global_matrix_dim=global_matrix_dim,
                unilora_dropout=unilora_config.unilora_dropout,
                layerwise_learnable_scale=unilora_config.layerwise_learnable_scale,
            )
        else:
            new_module = self._create_new_module(
                unilora_config=unilora_config,
                unilora_multi_structured_global_u=self.unilora_multi_structured_global_u,
                unilora_multi_structured_global_v=self.unilora_multi_structured_global_v,
                adapter_name=adapter_name,
                target=target,
                global_matrix_dim=global_matrix_dim,
                **kwargs,
            )
            if adapter_name not in self.active_adapter:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(
        unilora_config,
        unilora_multi_structured_global_u,
        unilora_multi_structured_global_v,
        adapter_name,
        target,
        global_matrix_dim,
        **kwargs,
    ):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = unilora_config.fan_in_fan_out = False
        elif isinstance(target_base_layer, Conv1D):
            kwargs["is_target_conv_1d_layer"] = True
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = unilora_config.fan_in_fan_out = True
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only `torch.nn.Linear` and `transformers.pytorch_utils.Conv1D`."
            )

        return Linear(
            base_layer=target,
            unilora_multi_structured_global_u=unilora_multi_structured_global_u,
            unilora_multi_structured_global_v=unilora_multi_structured_global_v,
            adapter_name=adapter_name,
            r=unilora_config.r,
            global_matrix_dim=global_matrix_dim,
            unilora_dropout=unilora_config.unilora_dropout,
            layerwise_learnable_scale=unilora_config.layerwise_learnable_scale,
            **kwargs,
        )

    def get_nb_savable_parameters(self, adapter="default") -> tuple[int, int]:
        structured_params = 0
        other_params = 0
        for name, param in self.named_parameters():
            if (
                "unilora_multi_structured_global_u" in name
                or "unilora_multi_structured_global_v" in name
                or "unilora_multi_structured_global_layer_scale" in name
            ):
                structured_params += param.numel()

        for name, buffer in self.named_buffers():
            if "global_linear_indices" in name or "global_scales" in name or "unilora_multi_structured_global_meta" in name:
                other_params += buffer.numel()

        return structured_params, other_params

    def print_savable_parameters(self) -> None:
        structured_params, other_params = self.get_nb_savable_parameters()
        print(
            f"UniLoRA-multi-structured-global params to-be-saved (float32-equivalent): {structured_params:,d} "
            f"|| total params to-be-saved: {(structured_params + other_params):,d}"
        )

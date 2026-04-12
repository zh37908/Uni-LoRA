from __future__ import annotations

import warnings

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

from .config import UniLoRAAromaConfig
from .layer import Linear, UniLoRAAromaLayer


class UniLoRAAromaModel(BaseTuner):
    """
    UniLoRA-AROMA model: train a UniLoRA theta_d bank, periodically merge the
    current update into the base model, then reinitialize theta_d and remap A/B.
    """

    prefix: str = "unilora_aroma_"
    tuner_layer_cls = UniLoRAAromaLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)
        cfg: UniLoRAAromaConfig = config[adapter_name]
        self._assign_global_structure(adapter_name=adapter_name, proj_seed=cfg.proj_seed)

    def _iter_unilora_modules(self):
        return [module for module in self.model.modules() if isinstance(module, UniLoRAAromaLayer)]

    def generate_index(self, total_length: int, theta_d_length: int, proj_seed: int) -> torch.Tensor:
        if total_length <= 0:
            return torch.empty(0, dtype=torch.long)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(proj_seed))

        base_count = total_length // theta_d_length
        remaining = total_length % theta_d_length
        data = torch.arange(theta_d_length, dtype=torch.long).repeat_interleave(base_count)
        if remaining > 0:
            extras = torch.randperm(theta_d_length, generator=generator)[:remaining]
            data = torch.cat([data, extras], dim=0)
        shuffle = torch.randperm(data.numel(), generator=generator)
        return data[shuffle]

    def _init_unilora_theta_d(self, config: UniLoRAAromaConfig, adapter_name: str) -> None:
        if adapter_name in self.unilora_aroma_theta_d:
            return
        unilora_theta_d = torch.zeros(config.theta_d_length)
        torch.nn.init.uniform_(unilora_theta_d, -config.init_theta_d_bound, config.init_theta_d_bound)
        self.unilora_aroma_theta_d[adapter_name] = unilora_theta_d

    def _reset_theta_d(self, adapter_name: str) -> None:
        cfg: UniLoRAAromaConfig = self.peft_config[adapter_name]
        theta = self.unilora_aroma_theta_d[adapter_name]
        with torch.no_grad():
            torch.nn.init.uniform_(theta, -cfg.init_theta_d_bound, cfg.init_theta_d_bound)

    def _assign_global_structure(self, adapter_name: str, proj_seed: int) -> None:
        modules = self._iter_unilora_modules()
        total_params = 0
        for module in modules:
            total_params += module.unilora_indices_A[adapter_name].numel()
            total_params += module.unilora_indices_B[adapter_name].numel()

        theta_d_length = self.peft_config[adapter_name].theta_d_length
        all_elements = self.generate_index(total_params, theta_d_length, proj_seed)
        pointer = 0

        for module in modules:
            num_a = module.unilora_indices_A[adapter_name].numel()
            chunk_a = all_elements[pointer : pointer + num_a]
            indices_a = chunk_a.view_as(module.unilora_indices_A[adapter_name]).clone()
            pointer += num_a

            num_b = module.unilora_indices_B[adapter_name].numel()
            chunk_b = all_elements[pointer : pointer + num_b]
            indices_b = chunk_b.view_as(module.unilora_indices_B[adapter_name]).clone()
            pointer += num_b

            module.unilora_indices_A[adapter_name] = indices_a.to(
                device=module.get_base_layer().weight.device, dtype=torch.long
            )
            module.unilora_indices_B[adapter_name] = indices_b.to(
                device=module.get_base_layer().weight.device, dtype=torch.long
            )

        if pointer != len(all_elements):
            raise RuntimeError("UniLoRA-AROMA index assignment is inconsistent.")

        counts = torch.bincount(all_elements, minlength=theta_d_length)
        inv_sqrt_counts = torch.zeros(theta_d_length, dtype=torch.float32)
        non_zero = counts > 0
        inv_sqrt_counts[non_zero] = 1.0 / torch.sqrt(counts[non_zero].float())

        for module in modules:
            scale_a = inv_sqrt_counts[module.unilora_indices_A[adapter_name].detach().cpu().long()]
            scale_b = inv_sqrt_counts[module.unilora_indices_B[adapter_name].detach().cpu().long()]
            module.update_norm(adapter_name, scale_a, scale_b)

    def _compute_reinit_seed(self, adapter_name: str, global_step: int | None = None) -> int:
        cfg: UniLoRAAromaConfig = self.peft_config[adapter_name]
        step = 0 if global_step is None else int(global_step)
        return int(cfg.proj_seed + 104729 * (step + 1))

    @torch.no_grad()
    def merge_and_reinit(self, global_step: int | None = None, adapter_name: str = "default") -> dict[str, int]:
        modules = self._iter_unilora_modules()
        merged_modules = 0
        for module in modules:
            if adapter_name not in module.unilora_indices_A:
                continue
            if not isinstance(module, Linear):
                continue
            module.merge_current_adapter_into_base(adapter_name)
            merged_modules += 1

        self._reset_theta_d(adapter_name)
        self._assign_global_structure(adapter_name, proj_seed=self._compute_reinit_seed(adapter_name, global_step))
        theta = self.unilora_aroma_theta_d[adapter_name]
        if theta.grad is not None:
            theta.grad = None

        return {
            "step": 0 if global_step is None else int(global_step),
            "merged_modules": int(merged_modules),
            "reinit_seed": int(self._compute_reinit_seed(adapter_name, global_step)),
        }

    def get_nb_savable_parameters(self, adapter="default") -> tuple[int, int]:
        theta_d_params = 0
        other_params = 0
        for name, param in self.named_parameters():
            if "unilora_aroma_theta_d" in name:
                theta_d_params += param.numel()
            elif "unilora_indices" in name:
                other_params += param.numel()
            elif "unilora_scales" in name:
                other_params += param.numel()

        for name, buffer in self.named_buffers():
            if "unilora_indices" in name or "unilora_scales" in name:
                other_params += buffer.numel()

        return theta_d_params, other_params

    def print_savable_parameters(self) -> None:
        theta_d_params, other_params = self.get_nb_savable_parameters()
        print(
            f"UniLoRA-AROMA params to-be-saved (float32-equivalent): {theta_d_params:,d} "
            f"|| total params to-be-saved: {(theta_d_params + other_params):,d}"
        )

    def _pre_injection_hook(self, model: nn.Module, config: UniLoRAAromaConfig, adapter_name: str) -> None:
        self.unilora_aroma_theta_d = nn.ParameterDict({})

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
        self._init_unilora_theta_d(unilora_config, adapter_name)

        if isinstance(target, Linear):
            target.update_layer(
                adapter_name=adapter_name,
                unilora_aroma_theta_d=self.unilora_aroma_theta_d,
                r=unilora_config.r,
                theta_d_length=unilora_config.theta_d_length,
                unilora_dropout=unilora_config.unilora_dropout,
            )
        else:
            new_module = self._create_new_module(
                unilora_config=unilora_config,
                unilora_aroma_theta_d=self.unilora_aroma_theta_d,
                adapter_name=adapter_name,
                target=target,
                **kwargs,
            )
            if adapter_name not in self.active_adapter:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(unilora_config, unilora_aroma_theta_d, adapter_name, target, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = unilora_config.fan_in_fan_out = False
        elif isinstance(target_base_layer, Conv1D):
            kwargs["is_target_conv_1d_layer"] = True
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                    "Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = unilora_config.fan_in_fan_out = True
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `transformers.pytorch_utils.Conv1D`."
            )
        new_module = Linear(
            base_layer=target,
            unilora_aroma_theta_d=unilora_aroma_theta_d,
            adapter_name=adapter_name,
            r=unilora_config.r,
            theta_d_length=unilora_config.theta_d_length,
            unilora_dropout=unilora_config.unilora_dropout,
            **kwargs,
        )
        return new_module

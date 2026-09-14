from __future__ import annotations

import math
import os
import warnings
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING, ModulesToSaveWrapper, _get_submodules

from .config import LoraRoSAConfig, LoraRoSARandomConfig, LoraRoSASnipConfig
from .layer import Linear, LoraRoSALayer


class LoraRoSAModel(BaseTuner):
    prefix: str = "lora_rosa_"
    tuner_layer_cls = LoraRoSALayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)
        peft_config = self.peft_config[adapter_name]
        modules = self._iter_lora_rosa_modules()
        budgets = self._allocate_sparse_budgets(modules, peft_config)

        total_budget = int(sum(budgets))
        sparse_device = modules[0].get_base_layer().weight.device if modules else None
        sparse_dtype = modules[0].get_base_layer().weight.dtype if modules else torch.float32
        self.lora_rosa_sparse_values[adapter_name] = nn.Parameter(
            torch.zeros(total_budget, device=sparse_device, dtype=sparse_dtype),
            requires_grad=False,
        )

        offset = 0
        for module, budget in zip(modules, budgets):
            module.assign_sparse_value_offsets(adapter_name, offset, int(budget))
            offset += int(budget)
            module.set_sparse_requires_grad(adapter_name, False)
        self._lora_rosa_sparse_active[adapter_name] = False

    def _iter_lora_rosa_modules(self) -> list[LoraRoSALayer]:
        return [module for module in self.model.modules() if isinstance(module, LoraRoSALayer)]

    def _named_lora_rosa_modules(self) -> dict[str, LoraRoSALayer]:
        return {
            name: module
            for name, module in self.model.named_modules()
            if isinstance(module, LoraRoSALayer)
        }

    @staticmethod
    def _allocate_sparse_budgets(modules: list[LoraRoSALayer], config: LoraRoSAConfig) -> list[int]:
        sizes = [module.get_base_layer().weight.numel() for module in modules]
        total_positions = int(sum(sizes))
        if total_positions <= 0:
            return [0 for _ in sizes]

        if config.rosa_sparse_budget is None:
            return [int(math.ceil(size * float(config.rosa_density))) for size in sizes]

        target_budget = max(0, min(int(config.rosa_sparse_budget), total_positions))
        raw = [target_budget * (float(size) / float(total_positions)) for size in sizes]
        budgets = [min(size, int(math.floor(value))) for size, value in zip(sizes, raw)]
        remaining = target_budget - sum(budgets)
        order = sorted(range(len(raw)), key=lambda idx: raw[idx] - math.floor(raw[idx]), reverse=True)
        for idx in order:
            if remaining <= 0:
                break
            if budgets[idx] < sizes[idx]:
                budgets[idx] += 1
                remaining -= 1
        return budgets

    def _pre_injection_hook(self, model: nn.Module, config: LoraRoSAConfig, adapter_name: str) -> None:
        self.lora_rosa_sparse_values = nn.ParameterDict({})
        self._lora_rosa_sparse_active = {}

    def _check_new_adapter_config(self, config: LoraRoSAConfig) -> None:
        if (len(self.peft_config) > 1) and (config.bias != "none"):
            raise ValueError("LoRA-RoSA supports only one bias-enabled adapter. Set bias='none' for multiple adapters.")

    def _create_and_replace(
        self,
        lora_rosa_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        if isinstance(target, Linear):
            target.update_layer(
                adapter_name=adapter_name,
                lora_rosa_sparse_values=self.lora_rosa_sparse_values,
                r=lora_rosa_config.r,
                lora_alpha=lora_rosa_config.lora_alpha,
                lora_dropout=lora_rosa_config.lora_dropout,
                init_lora_weights=lora_rosa_config.init_lora_weights,
            )
            return

        new_module = self._create_new_module(
            lora_rosa_config=lora_rosa_config,
            lora_rosa_sparse_values=self.lora_rosa_sparse_values,
            adapter_name=adapter_name,
            target=target,
        )
        if adapter_name not in self.active_adapter:
            new_module.requires_grad_(False)
        self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(lora_rosa_config, lora_rosa_sparse_values, adapter_name, target, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if not isinstance(target_base_layer, nn.Linear):
            raise ValueError(
                f"Target module {target} is not supported by LoRA-RoSA. Currently only torch.nn.Linear is supported."
            )

        return Linear(
            base_layer=target,
            lora_rosa_sparse_values=lora_rosa_sparse_values,
            adapter_name=adapter_name,
            r=lora_rosa_config.r,
            lora_alpha=lora_rosa_config.lora_alpha,
            lora_dropout=lora_rosa_config.lora_dropout,
            init_lora_weights=lora_rosa_config.init_lora_weights,
        )

    @staticmethod
    def _replace_module(parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        if hasattr(child, "base_layer"):
            child = child.base_layer
        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias
        new_module.to(child.weight.device)

    def enable_gradient_capture(self, enabled: bool = True, mode: str | None = None) -> None:
        for module in self._iter_lora_rosa_modules():
            module.set_capture_gradient(enabled)

    def clear_gradient_statistics(self, adapter_name: str = "default") -> None:
        for module in self._iter_lora_rosa_modules():
            module.clear_gradient_statistics(adapter_name)

    def accumulate_gradient_statistics(self, adapter_name: str = "default") -> dict[str, int]:
        config: LoraRoSAConfig = self.peft_config[adapter_name]
        updated_modules = 0
        for module in self._iter_lora_rosa_modules():
            updated_modules += int(
                module.accumulate_gradient_statistics(
                    adapter_name,
                    config.rosa_score_mode,
                    config.rosa_grad_acc_mode,
                )
                > 0
            )
        return {"updated_modules": updated_modules, "updated_tensors": updated_modules}

    def has_sparse_masks(self, adapter_name: str = "default") -> bool:
        return any(module.has_sparse_mask(adapter_name) for module in self._iter_lora_rosa_modules())

    def _sync_sparse_state_from_masks(self, adapter_name: str = "default") -> bool:
        active = False
        for module in self._iter_lora_rosa_modules():
            active = module.sync_sparse_active_from_mask(adapter_name) or active
        self.enable_gradient_capture(False)
        self.set_sparse_requires_grad(adapter_name, active)
        self._lora_rosa_sparse_active[adapter_name] = active
        return active

    def export_sparse_masks(self, path: str | os.PathLike, adapter_name: str = "default") -> dict[str, int]:
        modules = {}
        selected_positions = 0
        for name, module in self._named_lora_rosa_modules().items():
            indices = module.lora_rosa_selected_indices[adapter_name].detach().cpu().long()
            modules[name] = {
                "weight_shape": tuple(module.get_base_layer().weight.shape),
                "sparse_budget": int(module.sparse_budget.get(adapter_name, 0)),
                "selected_indices": indices,
            }
            selected_positions += int(indices.numel())

        payload = {
            "format_version": 1,
            "adapter_name": adapter_name,
            "modules": modules,
        }
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(output_path.name + ".tmp")
        torch.save(payload, temporary_path)
        os.replace(temporary_path, output_path)
        return {"modules": len(modules), "selected_positions": selected_positions}

    def load_sparse_masks(
        self,
        path: str | os.PathLike,
        adapter_name: str = "default",
        strict: bool = True,
    ) -> dict[str, int]:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict) or payload.get("format_version") != 1:
            raise ValueError(f"Unsupported LoRA-RoSA mask file: {path}")

        saved_modules = payload.get("modules")
        if not isinstance(saved_modules, dict):
            raise ValueError(f"LoRA-RoSA mask file has no module mapping: {path}")
        current_modules = self._named_lora_rosa_modules()
        if strict and set(saved_modules) != set(current_modules):
            missing = sorted(set(current_modules) - set(saved_modules))
            unexpected = sorted(set(saved_modules) - set(current_modules))
            raise ValueError(
                "LoRA-RoSA mask modules do not match the current model; "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}."
            )

        loaded_modules = 0
        selected_positions = 0
        for name, module in current_modules.items():
            entry = saved_modules.get(name)
            if entry is None:
                continue
            expected_shape = tuple(module.get_base_layer().weight.shape)
            if tuple(entry.get("weight_shape", ())) != expected_shape:
                raise ValueError(
                    f"LoRA-RoSA mask shape mismatch for {name}: "
                    f"saved={entry.get('weight_shape')}, current={expected_shape}."
                )
            indices = torch.as_tensor(entry.get("selected_indices"), dtype=torch.long)
            budget = int(module.sparse_budget.get(adapter_name, 0))
            if indices.numel() != budget:
                raise ValueError(
                    f"LoRA-RoSA mask budget mismatch for {name}: "
                    f"saved={indices.numel()}, configured={budget}."
                )
            num_positions = module.get_base_layer().weight.numel()
            if indices.numel() and (indices.min().item() < 0 or indices.max().item() >= num_positions):
                raise ValueError(f"LoRA-RoSA mask indices are out of range for {name}.")
            device = module.get_base_layer().weight.device
            module.lora_rosa_selected_indices[adapter_name] = indices.to(device=device)
            loaded_modules += 1
            selected_positions += int(indices.numel())

        if not self._sync_sparse_state_from_masks(adapter_name):
            raise ValueError(f"LoRA-RoSA mask file selected no parameters: {path}")
        return {"modules": loaded_modules, "selected_positions": selected_positions}

    def should_collect_gradients(self, global_step: int, adapter_name: str = "default") -> bool:
        config: LoraRoSAConfig = self.peft_config[adapter_name]
        if config.rosa_score_mode == "random" or self.has_sparse_masks(adapter_name):
            return False
        if self.lora_rosa_sparse_values[adapter_name].numel() == 0:
            return False
        return config.rosa_warmup_steps <= global_step < (config.rosa_warmup_steps + config.rosa_mask_steps)

    def should_generate_masks(self, next_global_step: int, adapter_name: str = "default") -> bool:
        config: LoraRoSAConfig = self.peft_config[adapter_name]
        if self.has_sparse_masks(adapter_name):
            return False
        if self.lora_rosa_sparse_values[adapter_name].numel() == 0:
            return False
        return next_global_step >= (config.rosa_warmup_steps + config.rosa_mask_steps)

    @torch.no_grad()
    def generate_sparse_masks(self, adapter_name: str = "default") -> dict[str, float]:
        config: LoraRoSAConfig = self.peft_config[adapter_name]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(getattr(config, "rosa_seed", 0)))

        total_positions = 0
        selected_positions = 0
        for module in self._iter_lora_rosa_modules():
            info = module.generate_sparse_mask(adapter_name, config.rosa_score_mode, generator=generator)
            total_positions += int(info["total_positions"])
            selected_positions += int(info["selected_positions"])

        self._sync_sparse_state_from_masks(adapter_name)
        return {
            "total_positions": int(total_positions),
            "selected_positions": int(selected_positions),
            "selected_density": 0.0 if total_positions == 0 else float(selected_positions) / float(total_positions),
            "selected_ratio": 0.0 if total_positions == 0 else float(selected_positions) / float(total_positions),
        }

    def get_sparse_structure_stats(self, adapter_name: str = "default") -> dict[str, float]:
        total_positions = 0
        selected_positions = 0
        sparse_budget = 0
        for module in self._iter_lora_rosa_modules():
            total_positions += module.get_base_layer().weight.numel()
            sparse_budget += int(module.sparse_budget.get(adapter_name, 0))
            if adapter_name in module.lora_rosa_selected_indices:
                selected_positions += int(module.lora_rosa_selected_indices[adapter_name].numel())
        return {
            "total_positions": int(total_positions),
            "sparse_budget": int(sparse_budget),
            "selected_positions": int(selected_positions),
            "selected_density": 0.0 if total_positions == 0 else float(selected_positions) / float(total_positions),
        }

    def set_sparse_requires_grad(self, adapter_name: str = "default", requires_grad: bool = True) -> None:
        if adapter_name in self.lora_rosa_sparse_values:
            self.lora_rosa_sparse_values[adapter_name].requires_grad_(requires_grad)
        for module in self._iter_lora_rosa_modules():
            module.set_sparse_requires_grad(adapter_name, requires_grad)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        trainable_markers = ("lora_A", "lora_B")
        for name, param in model.named_parameters():
            if any(marker in name for marker in trainable_markers):
                param.requires_grad = True
            else:
                param.requires_grad = False

        for active_adapter in self.active_adapters:
            bias = self.peft_config[active_adapter].bias
            if bias == "none":
                continue
            if bias == "all":
                for name, param in model.named_parameters():
                    if "bias" in name:
                        param.requires_grad = True
            elif bias == "lora_only":
                for module in model.modules():
                    if isinstance(module, LoraRoSALayer):
                        base_bias = getattr(module.get_base_layer(), "bias", None)
                        if base_bias is not None:
                            base_bias.requires_grad = True
            else:
                raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")

    def _set_adapter_layers(self, enabled: bool = True) -> None:
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)

    def enable_adapter_layers(self) -> None:
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self) -> None:
        for active_adapter in self.active_adapters:
            bias = self.peft_config[active_adapter].bias
            if bias != "none":
                warnings.warn(
                    f"Disabling adapter layers with bias='{bias}' does not reproduce the base model exactly."
                )
        self._set_adapter_layers(enabled=False)
        for active_adapter in self.active_adapters:
            self.set_sparse_requires_grad(active_adapter, False)

    def set_adapter(self, adapter_name: str | list[str], inference_mode: bool = False) -> None:
        for module in self.model.modules():
            if isinstance(module, LoraRoSALayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when merged. Unmerging first.")
                    module.unmerge()
                module.set_adapter(adapter_name, inference_mode=inference_mode)
        self.active_adapter = adapter_name

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`.")
            peft_config.target_modules = set(TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]])
        return peft_config

    @staticmethod
    def _check_target_module_exists(lora_rosa_config, key):
        return check_target_module_exists(lora_rosa_config, key)

    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ):
        key_list = [key for key, _ in self.model.named_modules() if "lora_rosa_" not in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, LoraRoSALayer):
                if merge:
                    target.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                self._replace_module(parent, target_name, target.get_base_layer(), target)
            elif isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])
        return self.model

    def merge_and_unload(
        self, progressbar: bool = False, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
    ) -> torch.nn.Module:
        return self._unload_and_optionally_merge(
            progressbar=progressbar, safe_merge=safe_merge, adapter_names=adapter_names
        )

    def unload(self):
        return self._unload_and_optionally_merge(merge=False)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
            config_dict[key] = config
        return config_dict

    def get_nb_savable_parameters(self, adapter="default") -> tuple[int, int]:
        trainable = 0
        buffers = 0
        for name, param in self.named_parameters():
            if "lora_A" in name or "lora_B" in name or "lora_rosa_sparse_values" in name:
                trainable += param.numel()
        for name, buffer in self.named_buffers():
            if "lora_rosa_selected_indices" in name or "lora_rosa_value_offsets" in name:
                buffers += buffer.numel()
        return trainable, buffers


class LoraRoSASnipModel(LoraRoSAModel):
    prefix: str = "lora_rosa_snip_"


class LoraRoSARandomModel(LoraRoSAModel):
    prefix: str = "lora_rosa_random_"

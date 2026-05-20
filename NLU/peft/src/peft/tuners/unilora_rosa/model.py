from __future__ import annotations

import math
import warnings
from dataclasses import asdict
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING, ModulesToSaveWrapper, _get_submodules

from .._buffer_dict import BufferDict
from .config import UniLoRARoSAConfig, UniLoRARoSASnipConfig
from .layer import Linear, UniLoRARoSALayer


class UniLoRARoSAModel(BaseTuner):
    """
    UniLoRA with a RoSA-style sparse compensation vector.
    """

    prefix: str = "unilora_rosa_"
    tuner_layer_cls = UniLoRARoSALayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

        lora_para_cnt = 0
        for _, module in model.named_modules():
            if isinstance(module, UniLoRARoSALayer):
                lora_para_cnt += module.unilora_indices_A[adapter_name].numel()
                lora_para_cnt += module.unilora_indices_B[adapter_name].numel()

        peft_config = self.peft_config[adapter_name]
        theta_d_length = peft_config.theta_d_length
        proj_seed = peft_config.proj_seed
        sparse_device = None
        sparse_dtype = None
        for _, module in model.named_modules():
            if isinstance(module, UniLoRARoSALayer):
                sparse_device = module.get_base_layer().weight.device
                sparse_dtype = module.unilora_rosa_theta_d[adapter_name].dtype
                break
        self._init_sparse_theta_D(lora_para_cnt, adapter_name, device=sparse_device, dtype=sparse_dtype)
        all_elements = self.generate_index(lora_para_cnt, theta_d_length, proj_seed)
        pointer = 0

        for _, module in model.named_modules():
            if isinstance(module, UniLoRARoSALayer):
                param_numel = module.unilora_indices_A[adapter_name].numel()
                chunk = all_elements[pointer : pointer + param_numel]
                target_device = module.get_base_layer().weight.device
                offset_chunk = torch.arange(pointer, pointer + param_numel, dtype=torch.long)
                module.unilora_indices_A[adapter_name] = chunk.view_as(module.unilora_indices_A[adapter_name]).clone().to(
                    device=target_device, dtype=torch.long
                )
                module.unilora_theta_D_offsets_A[adapter_name] = offset_chunk.view_as(
                    module.unilora_indices_A[adapter_name]
                ).clone().to(device=target_device, dtype=torch.long)
                pointer += param_numel

                param_numel = module.unilora_indices_B[adapter_name].numel()
                chunk = all_elements[pointer : pointer + param_numel]
                offset_chunk = torch.arange(pointer, pointer + param_numel, dtype=torch.long)
                module.unilora_indices_B[adapter_name] = chunk.view_as(module.unilora_indices_B[adapter_name]).clone().to(
                    device=target_device, dtype=torch.long
                )
                module.unilora_theta_D_offsets_B[adapter_name] = offset_chunk.view_as(
                    module.unilora_indices_B[adapter_name]
                ).clone().to(device=target_device, dtype=torch.long)
                pointer += param_numel

        assert pointer == len(all_elements)

        counts = torch.bincount(all_elements, minlength=theta_d_length)
        inv_sqrt_counts = torch.zeros(theta_d_length, dtype=torch.float32)
        non_zero = counts > 0
        inv_sqrt_counts[non_zero] = 1.0 / torch.sqrt(counts[non_zero].float())

        for _, module in model.named_modules():
            if isinstance(module, UniLoRARoSALayer):
                scale_a = inv_sqrt_counts[module.unilora_indices_A[adapter_name].detach().cpu().long()]
                scale_b = inv_sqrt_counts[module.unilora_indices_B[adapter_name].detach().cpu().long()]
                module.update_norm(adapter_name, scale_a, scale_b)
                module.set_sparse_requires_grad(adapter_name, False)

    def _iter_unilora_modules(self):
        return [module for module in self.model.modules() if isinstance(module, UniLoRARoSALayer)]

    def has_sparse_masks(self, adapter_name: str = "default") -> bool:
        if adapter_name not in self.unilora_rosa_sparse_mask:
            return False
        return bool(self.unilora_rosa_sparse_mask[adapter_name].any().item())

    def get_sparse_parameter_stats(self, adapter_name: str = "default") -> dict[str, float]:
        if adapter_name not in self.unilora_rosa_sparse_theta_D:
            return {
                "sparse_nonzero": 0,
                "sparse_absmax": 0.0,
                "selected_sparse_nonzero": 0,
                "selected_sparse_absmax": 0.0,
            }

        sparse_theta = self.unilora_rosa_sparse_theta_D[adapter_name].detach()
        sparse_abs = sparse_theta.abs()
        sparse_nonzero = int((sparse_theta != 0).sum().item())
        sparse_absmax = float(sparse_abs.max().item()) if sparse_abs.numel() > 0 else 0.0

        selected_sparse = sparse_theta
        if adapter_name in self.unilora_rosa_sparse_mask:
            mask = self.unilora_rosa_sparse_mask[adapter_name].to(device=sparse_theta.device, dtype=torch.bool)
            if mask.numel() == sparse_theta.numel() and mask.any():
                selected_sparse = sparse_theta[mask]

        selected_abs = selected_sparse.abs()
        selected_nonzero = int((selected_sparse != 0).sum().item())
        selected_absmax = float(selected_abs.max().item()) if selected_abs.numel() > 0 else 0.0
        return {
            "sparse_nonzero": sparse_nonzero,
            "sparse_absmax": sparse_absmax,
            "selected_sparse_nonzero": selected_nonzero,
            "selected_sparse_absmax": selected_absmax,
        }

    def enable_gradient_capture(self, enabled: bool = True, mode: str = "grad") -> None:
        for module in self._iter_unilora_modules():
            module.set_capture_gradient(enabled, mode=mode)

    def set_sparse_requires_grad(self, adapter_name: str = "default", requires_grad: bool = True) -> None:
        if adapter_name in self.unilora_rosa_sparse_theta_D:
            self.unilora_rosa_sparse_theta_D[adapter_name].requires_grad_(requires_grad)
        for module in self._iter_unilora_modules():
            module.set_sparse_requires_grad(adapter_name, requires_grad)

    def clear_gradient_statistics(self, adapter_name: str = "default") -> None:
        if adapter_name in self.unilora_rosa_grad_accum:
            self.unilora_rosa_grad_accum[adapter_name].zero_()
        for module in self._iter_unilora_modules():
            module.clear_cached_gradients(adapter_name)

    def accumulate_gradient_statistics(self, adapter_name: str = "default") -> dict[str, int]:
        updated_modules = 0
        updated_tensors = 0
        for module in self._iter_unilora_modules():
            updated = module.accumulate_gradient_statistics(adapter_name)
            if updated > 0:
                updated_modules += 1
                updated_tensors += updated
        return {"updated_modules": updated_modules, "updated_tensors": updated_tensors}

    def should_collect_gradients(self, global_step: int, adapter_name: str = "default") -> bool:
        config: UniLoRARoSAConfig = self.peft_config[adapter_name]
        if config.rosa_density <= 0.0 or config.rosa_mask_steps <= 0 or self.has_sparse_masks(adapter_name):
            return False
        return config.rosa_warmup_steps <= global_step < (config.rosa_warmup_steps + config.rosa_mask_steps)

    def should_generate_masks(self, next_global_step: int, adapter_name: str = "default") -> bool:
        config: UniLoRARoSAConfig = self.peft_config[adapter_name]
        if config.rosa_density <= 0.0 or config.rosa_mask_steps <= 0 or self.has_sparse_masks(adapter_name):
            return False
        return next_global_step >= (config.rosa_warmup_steps + config.rosa_mask_steps)

    def get_sparse_structure_stats(self, adapter_name: str = "default") -> dict[str, float]:
        if adapter_name not in self.unilora_rosa_sparse_theta_D:
            return {"total_positions": 0, "selected_positions": 0, "selected_density": 0.0}

        total_positions = int(self.unilora_rosa_sparse_theta_D[adapter_name].numel())
        selected_positions = int(self.unilora_rosa_sparse_mask[adapter_name].sum().item())
        density = 0.0 if total_positions == 0 else float(selected_positions) / float(total_positions)
        return {
            "total_positions": int(total_positions),
            "selected_positions": int(selected_positions),
            "selected_density": density,
        }

    @torch.no_grad()
    def generate_sparse_masks(self, adapter_name: str = "default", density: float | None = None) -> dict[str, float]:
        config: UniLoRARoSAConfig = self.peft_config[adapter_name]
        density = config.rosa_density if density is None else density
        if adapter_name not in self.unilora_rosa_grad_accum:
            return {"skipped": True, "reason": "no_unilora_modules"}

        flat_scores = self.unilora_rosa_grad_accum[adapter_name].detach().clone()
        num_positions = int(flat_scores.numel())
        if num_positions == 0:
            return {"skipped": True, "reason": "empty_projection"}

        num_selected = int(math.ceil(num_positions * density))
        num_selected = max(0, min(num_positions, num_selected))
        sparse_mask = torch.zeros_like(flat_scores, dtype=torch.bool)
        if num_selected > 0:
            score_max = float(flat_scores.max().item()) if flat_scores.numel() > 0 else 0.0
            if score_max <= 0.0:
                raise RuntimeError(
                    "UniLoRA-RoSA-SNIP sparse mask generation found only zero saliency scores. "
                    "Gradient capture did not run correctly; aborting to avoid training a dead sparse adapter."
                )
            topk = torch.topk(flat_scores, k=num_selected, largest=True, sorted=False).indices
            sparse_mask[topk] = True

        self.unilora_rosa_sparse_mask[adapter_name] = sparse_mask.to(
            device=self.unilora_rosa_sparse_theta_D[adapter_name].device, dtype=torch.bool
        )

        with torch.no_grad():
            self.unilora_rosa_sparse_theta_D[adapter_name].mul_(
                self.unilora_rosa_sparse_mask[adapter_name].to(dtype=self.unilora_rosa_sparse_theta_D[adapter_name].dtype)
            )

        self.set_sparse_requires_grad(adapter_name, num_selected > 0)
        self._unilora_rosa_sparse_active[adapter_name] = num_selected > 0
        for module in self._iter_unilora_modules():
            module.set_sparse_active(adapter_name, num_selected > 0)
        self.enable_gradient_capture(False)
        self.clear_gradient_statistics(adapter_name)

        stats = self.get_sparse_structure_stats(adapter_name)
        stats.update(
            {
                "selected_positions": int(num_selected),
                "selected_ratio": 0.0 if num_positions == 0 else float(num_selected) / float(num_positions),
                "score_max": float(flat_scores.max().item()) if flat_scores.numel() > 0 else 0.0,
                "score_mean": float(flat_scores.mean().item()) if flat_scores.numel() > 0 else 0.0,
            }
        )
        return stats

    def generate_index(self, lora_para_cnt, theta_d_length, proj_seed):
        import numpy as np

        total_length = lora_para_cnt
        num_unique = theta_d_length
        base_count = total_length // num_unique
        remaining = total_length % num_unique
        rng = np.random.default_rng(proj_seed)
        data = np.repeat(np.arange(num_unique), base_count)
        if remaining > 0:
            extras = rng.choice(num_unique, size=remaining, replace=False)
            data = np.concatenate([data, extras])
        rng.shuffle(data)
        return torch.tensor(data)

    def _init_unilora_theta_d(self, config: UniLoRARoSAConfig, adapter_name: str) -> None:
        if adapter_name in self.unilora_rosa_theta_d:
            return
        unilora_theta_d = torch.zeros(config.theta_d_length)
        torch.nn.init.uniform_(unilora_theta_d, -config.init_theta_d_bound, config.init_theta_d_bound)
        self.unilora_rosa_theta_d[adapter_name] = unilora_theta_d

    def _init_sparse_theta_D(
        self,
        lora_para_cnt: int,
        adapter_name: str,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        if adapter_name in self.unilora_rosa_sparse_theta_D:
            return
        sparse_theta_D = torch.zeros(lora_para_cnt, device=device, dtype=dtype or torch.float32)
        self.unilora_rosa_sparse_theta_D[adapter_name] = sparse_theta_D
        self.unilora_rosa_sparse_mask[adapter_name] = torch.zeros(lora_para_cnt, device=device, dtype=torch.bool)
        self.unilora_rosa_grad_accum[adapter_name] = torch.zeros(lora_para_cnt, device=device, dtype=torch.float32)
        self._unilora_rosa_sparse_active[adapter_name] = False

    def _pre_injection_hook(self, model: nn.Module, config: UniLoRARoSAConfig, adapter_name: str) -> None:
        self.unilora_rosa_theta_d = nn.ParameterDict({})
        self.unilora_rosa_sparse_theta_D = nn.ParameterDict({})
        self.unilora_rosa_sparse_mask = BufferDict({}, persistent=True)
        # Name must not end with "score": PeftModel SEQ_CLS adds modules_to_save=["classifier","score"] and matches
        # submodule paths with endswith("score"), which would wrap this BufferDict in ModulesToSaveWrapper.
        self.unilora_rosa_grad_accum = BufferDict({}, persistent=False)
        self._unilora_rosa_sparse_active = {}

    def _check_new_adapter_config(self, config: UniLoRARoSAConfig) -> None:
        if (len(self.peft_config) > 1) and (config.bias != "none"):
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 adapter with bias. When using multiple adapters, "
                "set bias to 'none' for all adapters."
            )

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
                unilora_rosa_theta_d=self.unilora_rosa_theta_d,
                unilora_rosa_sparse_theta_D=self.unilora_rosa_sparse_theta_D,
                unilora_rosa_sparse_mask=self.unilora_rosa_sparse_mask,
                unilora_rosa_grad_accum=self.unilora_rosa_grad_accum,
                r=unilora_config.r,
                theta_d_length=unilora_config.theta_d_length,
                unilora_dropout=unilora_config.unilora_dropout,
            )
        else:
            new_module = self._create_new_module(
                unilora_config=unilora_config,
                unilora_rosa_theta_d=self.unilora_rosa_theta_d,
                unilora_rosa_sparse_theta_D=self.unilora_rosa_sparse_theta_D,
                unilora_rosa_sparse_mask=self.unilora_rosa_sparse_mask,
                unilora_rosa_grad_accum=self.unilora_rosa_grad_accum,
                adapter_name=adapter_name,
                target=target,
                **kwargs,
            )
            if adapter_name not in self.active_adapter:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _replace_module(parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)

        meta = torch.device("meta")
        # dispatch RoSA adapter tensors to the same device as the wrapped layer
        for name, module in new_module.named_modules():
            if "unilora_rosa_" in name:
                if not any(p.device == meta for p in module.parameters()):
                    module.to(child.weight.device)

    @staticmethod
    def _create_new_module(
        unilora_config,
        unilora_rosa_theta_d,
        unilora_rosa_sparse_theta_D,
        unilora_rosa_sparse_mask,
        unilora_rosa_grad_accum,
        adapter_name,
        target,
        **kwargs,
    ):
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
            unilora_rosa_theta_d=unilora_rosa_theta_d,
            unilora_rosa_sparse_theta_D=unilora_rosa_sparse_theta_D,
            unilora_rosa_sparse_mask=unilora_rosa_sparse_mask,
            unilora_rosa_grad_accum=unilora_rosa_grad_accum,
            adapter_name=adapter_name,
            r=unilora_config.r,
            theta_d_length=unilora_config.theta_d_length,
            unilora_dropout=unilora_config.unilora_dropout,
            **kwargs,
        )
        return new_module

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            if name == "model":  # prevent infinite recursion if class is not initialized
                raise
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
            config_dict[key] = config
        return config_dict

    @staticmethod
    def _check_target_module_exists(unilora_config, key):
        return check_target_module_exists(unilora_config, key)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        trainable_markers = ("unilora_rosa_theta_d", "unilora_rosa_sparse_theta_D")
        for n, p in model.named_parameters():
            if not any(marker in n for marker in trainable_markers):
                p.requires_grad = False

        for active_adapter in self.active_adapters:
            bias = self.peft_config[active_adapter].bias
            if bias == "none":
                continue
            if bias == "all":
                for n, p in model.named_parameters():
                    if "bias" in n:
                        p.requires_grad = True
            elif bias == "unilora_only":
                for module in model.modules():
                    if isinstance(module, UniLoRARoSALayer) and hasattr(module, "bias") and module.bias is not None:
                        module.bias.requires_grad = True
            else:
                raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")

    def _set_adapter_layers(self, enabled: bool = True) -> None:
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)

    def _sync_sparse_requires_grad_with_masks(self, inference_mode: bool = False) -> None:
        for active_adapter in self.active_adapters:
            active = self.has_sparse_masks(active_adapter)
            self._unilora_rosa_sparse_active[active_adapter] = active
            self.set_sparse_requires_grad(active_adapter, (not inference_mode) and active)
            for module in self._iter_unilora_modules():
                module.set_sparse_active(active_adapter, active)

    def enable_adapter_layers(self) -> None:
        self._set_adapter_layers(enabled=True)
        self._sync_sparse_requires_grad_with_masks()

    def disable_adapter_layers(self) -> None:
        for active_adapter in self.active_adapters:
            bias = self.peft_config[active_adapter].bias
            if bias != "none":
                warnings.warn(
                    f"Careful, disabling adapter layers with bias configured to be '{bias}' does not produce the "
                    "same output as the base model would without adaptation."
                )
        self._set_adapter_layers(enabled=False)
        for active_adapter in self.active_adapters:
            self.set_sparse_requires_grad(active_adapter, False)

    def set_adapter(self, adapter_name: str | list[str], inference_mode: bool = False) -> None:
        """Set the active adapter(s)."""
        for module in self.model.modules():
            if isinstance(module, UniLoRARoSALayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name, inference_mode=inference_mode)
        self.active_adapter = adapter_name
        self._sync_sparse_requires_grad_with_masks(inference_mode=inference_mode)

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config

    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ):
        key_list = [key for key, _ in self.model.named_modules() if "unilora_rosa_" not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue

            if isinstance(target, UniLoRARoSALayer):
                if merge:
                    target.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                self._replace_module(parent, target_name, target.get_base_layer(), target)
            elif isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

    def delete_adapter(self, adapter_name: str) -> None:
        if adapter_name not in list(self.peft_config.keys()):
            raise ValueError(f"Adapter {adapter_name} does not exist")
        del self.peft_config[adapter_name]
        for adapter_store in (
            self.unilora_rosa_sparse_mask,
            self.unilora_rosa_grad_accum,
        ):
            if adapter_name in adapter_store:
                del adapter_store[adapter_name]
        self._unilora_rosa_sparse_active.pop(adapter_name, None)

        key_list = [key for key, _ in self.model.named_modules() if "unilora_rosa_" not in key]
        new_adapter = None
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, UniLoRARoSALayer):
                target.delete_adapter(adapter_name)
                if new_adapter is None:
                    new_adapter = target.active_adapter[:]

        self.active_adapter = new_adapter or []

    def merge_and_unload(
        self, progressbar: bool = False, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
    ) -> torch.nn.Module:
        return self._unload_and_optionally_merge(
            progressbar=progressbar, safe_merge=safe_merge, adapter_names=adapter_names
        )

    def unload(self):
        return self._unload_and_optionally_merge(merge=False)

    def get_nb_savable_parameters(self, adapter="default") -> tuple[int, int]:
        theta_d_params = 0
        sparse_params = 0
        other_params = 0
        for name, param in self.named_parameters():
            if "unilora_rosa_theta_d" in name:
                theta_d_params += param.numel()
            elif "unilora_rosa_sparse_theta_D" in name:
                sparse_params += param.numel()
            elif "unilora_indices" in name or "unilora_scales" in name or "unilora_theta_D_offsets" in name:
                other_params += param.numel()

        for name, buffer in self.named_buffers():
            if "unilora_indices" in name or "unilora_scales" in name or "unilora_theta_D_offsets" in name:
                other_params += buffer.numel()
            elif "unilora_rosa_sparse_mask" in name:
                other_params += buffer.numel()

        return theta_d_params + sparse_params, other_params

    def print_savable_parameters(self) -> None:
        unilora_params, other_params = self.get_nb_savable_parameters()
        print(
            f"UniLoRA-RoSA params to-be-saved (float32-equivalent): {unilora_params:,d} "
            f"|| total params to-be-saved: {(unilora_params + other_params):,d}"
        )


class UniLoRARoSASnipModel(UniLoRARoSAModel):
    """
    UniLoRA-RoSA variant that selects sparse positions with SNIP |W_ij * g_ij|
    saliency.
    """

    prefix: str = "unilora_rosa_snip_"

    def accumulate_gradient_statistics(self, adapter_name: str = "default") -> dict[str, int]:
        updated_modules = 0
        updated_tensors = 0
        for module in self._iter_unilora_modules():
            updated = module.accumulate_snip_statistics(adapter_name)
            if updated > 0:
                updated_modules += 1
                updated_tensors += updated
        return {"updated_modules": updated_modules, "updated_tensors": updated_tensors}

    def should_collect_gradients(self, global_step: int, adapter_name: str = "default") -> bool:
        config: UniLoRARoSASnipConfig = self.peft_config[adapter_name]
        if config.rosa_density <= 0.0 or config.rosa_mask_steps <= 0 or self.has_sparse_masks(adapter_name):
            return False
        return config.rosa_warmup_steps <= global_step < (config.rosa_warmup_steps + config.rosa_mask_steps)

    def should_generate_masks(self, next_global_step: int, adapter_name: str = "default") -> bool:
        config: UniLoRARoSASnipConfig = self.peft_config[adapter_name]
        if config.rosa_density <= 0.0 or config.rosa_mask_steps <= 0 or self.has_sparse_masks(adapter_name):
            return False
        return next_global_step >= (config.rosa_warmup_steps + config.rosa_mask_steps)

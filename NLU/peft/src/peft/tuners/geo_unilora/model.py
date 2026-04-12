from __future__ import annotations

import hashlib
import warnings

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from .config import GeoUniLoRAConfig
from .layer import GeoUniLoRALayer, Linear


def _stable_int_seed(proj_seed: int, *parts: str) -> int:
    msg = f"{proj_seed}:" + ":".join(parts)
    h = hashlib.md5(msg.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % (2**31)


def _safe_module_token(module_name: str) -> str:
    # ParameterDict keys cannot contain ".", so use a deterministic hash token.
    return "m_" + hashlib.md5(module_name.encode("utf-8")).hexdigest()


class GeoUniLoRAModel(BaseTuner):
    """
    Geo-UniLoRA: per-group shared theta_d + per-module innovation theta_d, dual low-rank branch.
    """

    prefix: str = "geo_unilora_"
    tuner_layer_cls = GeoUniLoRALayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)
        cfg = config[adapter_name]
        proj_seed = cfg.proj_seed
        shared_len = cfg.shared_theta_d_length
        innov_len = cfg.innovation_theta_d_length

        # --- Per-group shared index assignment ---
        group_to_modules: dict[int, list[tuple[str, GeoUniLoRALayer]]] = {}
        for name, module in model.named_modules():
            if isinstance(module, GeoUniLoRALayer) and adapter_name in module.r_shared:
                gid = int(module.group_id[adapter_name])
                group_to_modules.setdefault(gid, []).append((name, module))

        for gid in sorted(group_to_modules.keys()):
            modules_sorted = sorted(group_to_modules[gid], key=lambda x: x[0])
            total_len = 0
            for _name, mod in modules_sorted:
                total_len += mod.unilora_indices_shared_A[adapter_name].numel()
                total_len += mod.unilora_indices_shared_B[adapter_name].numel()

            all_elements = self.generate_index(total_len, shared_len, _stable_int_seed(proj_seed, "shared", str(gid)))
            pointer = 0
            for _name, mod in modules_sorted:
                na = mod.unilora_indices_shared_A[adapter_name].numel()
                chunk_a = all_elements[pointer : pointer + na]
                mod.unilora_indices_shared_A[adapter_name] = chunk_a.view_as(
                    mod.unilora_indices_shared_A[adapter_name]
                ).clone()
                pointer += na

                nb = mod.unilora_indices_shared_B[adapter_name].numel()
                chunk_b = all_elements[pointer : pointer + nb]
                mod.unilora_indices_shared_B[adapter_name] = chunk_b.view_as(
                    mod.unilora_indices_shared_B[adapter_name]
                ).clone()
                pointer += nb

            if pointer != len(all_elements):
                raise RuntimeError("Geo-UniLoRA shared index assignment is inconsistent.")

            counts = torch.bincount(all_elements, minlength=shared_len)
            inv_sqrt = torch.zeros(shared_len, dtype=torch.float32)
            nz = counts > 0
            inv_sqrt[nz] = 1.0 / torch.sqrt(counts[nz].float())

            for _name, mod in modules_sorted:
                sa = inv_sqrt[mod.unilora_indices_shared_A[adapter_name].long()]
                sb = inv_sqrt[mod.unilora_indices_shared_B[adapter_name].long()]
                mod.update_norm_shared(adapter_name, sa, sb)

        # --- Per-module innovation index assignment ---
        innov_modules = [
            (n, m)
            for n, m in model.named_modules()
            if isinstance(m, GeoUniLoRALayer) and adapter_name in m.r_innov
        ]
        innov_modules.sort(key=lambda x: x[0])

        for name, mod in innov_modules:
            total_len = mod.unilora_indices_innov_A[adapter_name].numel() + mod.unilora_indices_innov_B[
                adapter_name
            ].numel()
            all_elements = self.generate_index(
                total_len, innov_len, _stable_int_seed(proj_seed, "innov", name)
            )
            pointer = 0
            na = mod.unilora_indices_innov_A[adapter_name].numel()
            mod.unilora_indices_innov_A[adapter_name] = all_elements[pointer : pointer + na].view_as(
                mod.unilora_indices_innov_A[adapter_name]
            ).clone()
            pointer += na
            nb = mod.unilora_indices_innov_B[adapter_name].numel()
            mod.unilora_indices_innov_B[adapter_name] = all_elements[pointer : pointer + nb].view_as(
                mod.unilora_indices_innov_B[adapter_name]
            ).clone()
            pointer += nb
            if pointer != len(all_elements):
                raise RuntimeError("Geo-UniLoRA innovation index assignment is inconsistent.")

            counts = torch.bincount(all_elements, minlength=innov_len)
            inv_sqrt = torch.zeros(innov_len, dtype=torch.float32)
            nz = counts > 0
            inv_sqrt[nz] = 1.0 / torch.sqrt(counts[nz].float())
            ia = inv_sqrt[mod.unilora_indices_innov_A[adapter_name].long()]
            ib = inv_sqrt[mod.unilora_indices_innov_B[adapter_name].long()]
            mod.update_norm_innov(adapter_name, ia, ib)

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

    def _shared_param_key(self, adapter_name: str, group_id: int) -> str:
        return f"{adapter_name}__g{int(group_id)}"

    def _innov_param_key(self, adapter_name: str, module_name: str) -> str:
        return f"{adapter_name}__{_safe_module_token(module_name)}"

    def _init_shared_bank(self, cfg: GeoUniLoRAConfig, adapter_name: str, group_id: int) -> None:
        key = self._shared_param_key(adapter_name, group_id)
        if key in self.geo_ul_shared_theta_d:
            return
        t = torch.zeros(cfg.shared_theta_d_length)
        torch.nn.init.uniform_(t, -cfg.init_theta_d_bound, cfg.init_theta_d_bound)
        self.geo_ul_shared_theta_d[key] = t

    def _init_innov_bank(self, cfg: GeoUniLoRAConfig, adapter_name: str, module_name: str) -> None:
        key = self._innov_param_key(adapter_name, module_name)
        if key in self.geo_ul_innovation_theta_d:
            return
        t = torch.zeros(cfg.innovation_theta_d_length)
        torch.nn.init.uniform_(t, -cfg.init_theta_d_bound, cfg.init_theta_d_bound)
        self.geo_ul_innovation_theta_d[key] = t

    def _pre_injection_hook(self, model: nn.Module, config: GeoUniLoRAConfig, adapter_name: str) -> None:
        self.geo_ul_shared_theta_d = nn.ParameterDict({})
        self.geo_ul_innovation_theta_d = nn.ParameterDict({})

    @staticmethod
    def _resolve_rank_from_map(rank_map: dict[str, int] | None, module_key: str, default_rank: int) -> int:
        if not rank_map:
            return default_rank
        if module_key in rank_map:
            return int(rank_map[module_key])
        for key, value in rank_map.items():
            if module_key.endswith(key):
                return int(value)
        return default_rank

    @staticmethod
    def _resolve_group_from_map(group_map: dict[str, int] | None, module_key: str) -> int:
        if not group_map:
            return 0
        if module_key in group_map:
            return int(group_map[module_key])
        for key, value in group_map.items():
            if module_key.endswith(key):
                return int(value)
        return 0

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

        group_map = getattr(unilora_config, "geo_group_map", None) or {}
        shared_map = getattr(unilora_config, "geo_shared_rank_map", None)
        innov_map = getattr(unilora_config, "geo_innovation_rank_map", None)
        default_r = unilora_config.r
        r_shared = self._resolve_rank_from_map(shared_map, current_key, max(1, default_r // 2))
        r_innov = self._resolve_rank_from_map(innov_map, current_key, max(1, default_r - max(1, default_r // 2)))
        if shared_map is None and innov_map is None:
            r_shared = max(1, default_r // 2)
            r_innov = max(1, default_r - r_shared)
        gid = self._resolve_group_from_map(group_map, current_key)
        innovation_module_key = _safe_module_token(current_key)

        self._init_shared_bank(unilora_config, adapter_name, gid)
        self._init_innov_bank(unilora_config, adapter_name, current_key)

        if isinstance(target, Linear):
            target.update_layer(
                adapter_name=adapter_name,
                geo_ul_shared_theta_d=self.geo_ul_shared_theta_d,
                geo_ul_innovation_theta_d=self.geo_ul_innovation_theta_d,
                group_id=gid,
                innovation_module_key=innovation_module_key,
                r_shared=r_shared,
                r_innov=r_innov,
                shared_theta_d_length=unilora_config.shared_theta_d_length,
                innovation_theta_d_length=unilora_config.innovation_theta_d_length,
                unilora_dropout=unilora_config.unilora_dropout,
            )
        else:
            new_module = self._create_new_module(
                unilora_config=unilora_config,
                geo_ul_shared_theta_d=self.geo_ul_shared_theta_d,
                geo_ul_innovation_theta_d=self.geo_ul_innovation_theta_d,
                adapter_name=adapter_name,
                target=target,
                group_id=gid,
                innovation_module_key=innovation_module_key,
                r_shared=r_shared,
                r_innov=r_innov,
                **kwargs,
            )
            if adapter_name not in self.active_adapter:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(
        unilora_config,
        geo_ul_shared_theta_d,
        geo_ul_innovation_theta_d,
        adapter_name,
        target,
        group_id: int,
        innovation_module_key: str,
        r_shared: int,
        r_innov: int,
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
            geo_ul_shared_theta_d=geo_ul_shared_theta_d,
            geo_ul_innovation_theta_d=geo_ul_innovation_theta_d,
            adapter_name=adapter_name,
            group_id=group_id,
            innovation_module_key=innovation_module_key,
            r_shared=r_shared,
            r_innov=r_innov,
            shared_theta_d_length=unilora_config.shared_theta_d_length,
            innovation_theta_d_length=unilora_config.innovation_theta_d_length,
            unilora_dropout=unilora_config.unilora_dropout,
            **kwargs,
        )
        return new_module

    def get_nb_savable_parameters(self, adapter="default") -> tuple[int, int]:
        theta_d_params = 0
        other_params = 0
        for name, param in self.named_parameters():
            if "geo_ul_shared_theta_d" in name or "geo_ul_innovation_theta_d" in name:
                theta_d_params += param.numel()
            elif "unilora_indices" in name:
                other_params += param.numel()
            elif "unilora_scales" in name:
                other_params += param.numel()

        return theta_d_params, other_params

    def print_savable_parameters(self) -> None:
        theta_d_params, other_params = self.get_nb_savable_parameters()
        print(
            f"Geo-UniLoRA params to-be-saved (float32-equivalent): {theta_d_params:,d} "
            f"|| total params to-be-saved: {(theta_d_params + other_params):,d}"
        )

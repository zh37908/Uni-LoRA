from __future__ import annotations

import torch
import torch.nn as nn

from peft.tuners.unilora_soft_assign.model import UniLoRASoftAssignModel

from .layer import UniLoRASoftWeightSharingLayer


class UniLoRASoftWeightSharingModel(UniLoRASoftAssignModel):
    prefix: str = "unilora_soft_weight_sharing_"
    tuner_layer_cls = UniLoRASoftWeightSharingLayer

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)
        cfg = config[adapter_name] if isinstance(config, dict) else config
        self.unilora_soft_weight_sharing_mu = nn.ParameterDict({})
        self.unilora_soft_weight_sharing_log_sigma = nn.ParameterDict({})
        self.unilora_soft_weight_sharing_logit_pi = nn.ParameterDict({})
        self.unilora_soft_weight_sharing_export_cache = {}
        self._init_soft_weight_sharing_params(cfg, adapter_name)
        self.soft_weight_sharing_config = cfg

    def _init_soft_weight_sharing_params(self, cfg, adapter_name: str) -> None:
        k = int(cfg.num_components)
        theta_d = self.unilora_soft_assign_theta_d[adapter_name].detach()
        lo, hi = theta_d.min().item(), theta_d.max().item()
        mu = torch.linspace(lo, hi, k)
        if cfg.sharing_zero_component:
            mu[0] = 0.0
        log_sigma = torch.full((k,), -2.3)  # sigma ~= 0.1
        logit_pi = torch.zeros(k)
        self.unilora_soft_weight_sharing_mu[adapter_name] = nn.Parameter(mu)
        self.unilora_soft_weight_sharing_log_sigma[adapter_name] = nn.Parameter(log_sigma)
        self.unilora_soft_weight_sharing_logit_pi[adapter_name] = nn.Parameter(logit_pi)

    def _collect_ab_values(self, adapter_name: str):
        grouped = []
        for _, module in self.model.named_modules():
            if not isinstance(module, UniLoRASoftWeightSharingLayer):
                continue
            if adapter_name not in module.unilora_soft_assign_logits_A:
                continue
            A, B = module._get_lora_matrices(adapter_name)
            grouped.append((A.reshape(-1), B.reshape(-1)))
        return grouped

    def _mixture_nll(self, values: torch.Tensor, adapter_name: str, sigma_floor: float, zero_component: bool) -> torch.Tensor:
        if values.numel() == 0:
            return values.new_tensor(0.0)
        mu = self.unilora_soft_weight_sharing_mu[adapter_name]
        log_sigma = self.unilora_soft_weight_sharing_log_sigma[adapter_name]
        logit_pi = self.unilora_soft_weight_sharing_logit_pi[adapter_name]
        if zero_component:
            mu = torch.cat([torch.zeros_like(mu[:1]), mu[1:]], dim=0)
        sigma = log_sigma.exp().clamp_min(sigma_floor)
        log_pi = torch.log_softmax(logit_pi, dim=0)
        centered = values.unsqueeze(-1) - mu.unsqueeze(0)
        comp_log_prob = -0.5 * ((centered / sigma.unsqueeze(0)) ** 2) - torch.log(sigma.unsqueeze(0)) - 0.5 * torch.log(
            values.new_tensor(2.0 * torch.pi)
        )
        log_prob = torch.logsumexp(log_pi.unsqueeze(0) + comp_log_prob, dim=-1)
        return -log_prob.mean()

    def compute_soft_weight_sharing_loss(self, adapter_name: str = "default") -> dict[str, torch.Tensor]:
        cfg = self.soft_weight_sharing_config
        grouped = self._collect_ab_values(adapter_name)
        if not grouped:
            zero = next(self.parameters()).new_tensor(0.0)
            return {"loss": zero, "nll": zero, "tau": zero, "num_values": zero}

        if cfg.sharing_grouping == "global":
            values = torch.cat([a for a, _ in grouped] + [b for _, b in grouped], dim=0)
            nll = self._mixture_nll(values, adapter_name, cfg.sharing_sigma_floor, cfg.sharing_zero_component)
        elif cfg.sharing_grouping == "ab_split":
            values_a = torch.cat([a for a, _ in grouped], dim=0)
            values_b = torch.cat([b for _, b in grouped], dim=0)
            nll = 0.5 * (
                self._mixture_nll(values_a, adapter_name, cfg.sharing_sigma_floor, cfg.sharing_zero_component)
                + self._mixture_nll(values_b, adapter_name, cfg.sharing_sigma_floor, cfg.sharing_zero_component)
            )
        else:
            losses = []
            for a, b in grouped:
                values = torch.cat([a, b], dim=0)
                losses.append(self._mixture_nll(values, adapter_name, cfg.sharing_sigma_floor, cfg.sharing_zero_component))
            nll = torch.stack(losses).mean()

        tau = torch.as_tensor(float(cfg.sharing_tau), device=nll.device, dtype=nll.dtype)
        return {"loss": tau * nll, "nll": nll, "tau": tau, "num_values": torch.as_tensor(sum(a.numel() + b.numel() for a, b in grouped), device=nll.device)}

    def get_soft_weight_sharing_stats(self, adapter_name: str = "default") -> dict[str, float]:
        with torch.no_grad():
            info = self.compute_soft_weight_sharing_loss(adapter_name=adapter_name)
            pi = torch.softmax(self.unilora_soft_weight_sharing_logit_pi[adapter_name], dim=0)
            effective_clusters = torch.exp(-(pi * torch.log(pi.clamp_min(1e-12))).sum())
            assigned_theta_indices = []
            for _, module in self.model.named_modules():
                if not isinstance(module, UniLoRASoftWeightSharingLayer):
                    continue
                if adapter_name not in module.unilora_soft_assign_logits_A:
                    continue
                winner_a = module.unilora_soft_assign_logits_A[adapter_name].argmax(dim=-1, keepdim=True)
                winner_b = module.unilora_soft_assign_logits_B[adapter_name].argmax(dim=-1, keepdim=True)
                idx_a = torch.gather(
                    module.unilora_soft_assign_candidate_indices_A[adapter_name],
                    dim=-1,
                    index=winner_a,
                ).reshape(-1)
                idx_b = torch.gather(
                    module.unilora_soft_assign_candidate_indices_B[adapter_name],
                    dim=-1,
                    index=winner_b,
                ).reshape(-1)
                assigned_theta_indices.append(idx_a)
                assigned_theta_indices.append(idx_b)
            if assigned_theta_indices:
                all_idx = torch.cat(assigned_theta_indices, dim=0)
                used_theta_entries = float(torch.unique(all_idx).numel())
                total_theta_entries = float(self.unilora_soft_assign_theta_d[adapter_name].numel())
                used_theta_ratio = used_theta_entries / max(total_theta_entries, 1.0)
            else:
                used_theta_entries = 0.0
                total_theta_entries = 0.0
                used_theta_ratio = 0.0
            return {
                "nll": float(info["nll"].item()),
                "tau": float(info["tau"].item()),
                "num_values": float(info["num_values"].item()),
                "effective_clusters": float(effective_clusters.item()),
                "used_theta_entries": used_theta_entries,
                "total_theta_entries": total_theta_entries,
                "used_theta_ratio": used_theta_ratio,
            }

    def _merge_mixture_components(self, adapter_name: str = "default") -> int:
        threshold = float(self.soft_weight_sharing_config.sharing_merge_threshold)
        if threshold <= 0.0:
            return 0
        with torch.no_grad():
            mu = self.unilora_soft_weight_sharing_mu[adapter_name]
            log_sigma = self.unilora_soft_weight_sharing_log_sigma[adapter_name]
            logit_pi = self.unilora_soft_weight_sharing_logit_pi[adapter_name]
            pi = torch.softmax(logit_pi, dim=0)

            merge_count = 0
            for i in range(mu.numel()):
                for j in range(i + 1, mu.numel()):
                    if abs(float(mu[i] - mu[j])) >= threshold:
                        continue
                    weight = (pi[i] + pi[j]).clamp_min(1e-12)
                    mu_new = (pi[i] * mu[i] + pi[j] * mu[j]) / weight
                    sigma_i = log_sigma[i].exp()
                    sigma_j = log_sigma[j].exp()
                    sigma_new = (pi[i] * sigma_i + pi[j] * sigma_j) / weight
                    if self.soft_weight_sharing_config.sharing_zero_component and i == 0:
                        mu_new = mu_new.new_tensor(0.0)
                    mu[i] = mu_new
                    log_sigma[i] = sigma_new.clamp_min(self.soft_weight_sharing_config.sharing_sigma_floor).log()
                    mu[j] = mu_new
                    log_sigma[j] = log_sigma[i]
                    # Collapse component j into i with very low mass.
                    logit_pi[j] = logit_pi.new_tensor(-20.0)
                    merge_count += 1
            return merge_count

    def export_shared_codebook_and_indices(self, adapter_name: str = "default") -> dict[str, torch.Tensor]:
        codebook = self.unilora_soft_weight_sharing_mu[adapter_name].detach().clone()
        if self.soft_weight_sharing_config.sharing_zero_component:
            codebook[0] = 0.0
        module_indices = {}
        for name, module in self.model.named_modules():
            if not isinstance(module, UniLoRASoftWeightSharingLayer):
                continue
            if adapter_name not in module.unilora_soft_assign_logits_A:
                continue
            idx_a = module.unilora_soft_assign_logits_A[adapter_name].argmax(dim=-1).detach().cpu().to(torch.int16)
            idx_b = module.unilora_soft_assign_logits_B[adapter_name].argmax(dim=-1).detach().cpu().to(torch.int16)
            module_indices[f"{name}.A"] = idx_a
            module_indices[f"{name}.B"] = idx_b

        exported = {
            "codebook": codebook.detach().cpu(),
            "pi": torch.softmax(self.unilora_soft_weight_sharing_logit_pi[adapter_name], dim=0).detach().cpu(),
        }
        exported.update(module_indices)
        self.unilora_soft_weight_sharing_export_cache[adapter_name] = exported
        return exported

    def finalize_soft_weight_sharing(self, adapter_name: str = "default") -> dict[str, float]:
        merged_components = self._merge_mixture_components(adapter_name=adapter_name)
        total_changed = 0
        total_numel = 0
        for _, module in self.model.named_modules():
            if not isinstance(module, UniLoRASoftWeightSharingLayer):
                continue
            if adapter_name not in module.unilora_soft_assign_logits_A:
                continue
            with torch.no_grad():
                logits_a = module.unilora_soft_assign_logits_A[adapter_name]
                logits_b = module.unilora_soft_assign_logits_B[adapter_name]
                prev_hard_a = logits_a.argmax(dim=-1, keepdim=True)
                prev_hard_b = logits_b.argmax(dim=-1, keepdim=True)
                prev_prob_a = torch.softmax(logits_a, dim=-1).gather(dim=-1, index=prev_hard_a).squeeze(-1)
                prev_prob_b = torch.softmax(logits_b, dim=-1).gather(dim=-1, index=prev_hard_b).squeeze(-1)
                idx_a = logits_a.argmax(dim=-1, keepdim=True)
                idx_b = logits_b.argmax(dim=-1, keepdim=True)
                hard_a = torch.zeros_like(logits_a).scatter_(-1, idx_a, 1.0)
                hard_b = torch.zeros_like(logits_b).scatter_(-1, idx_b, 1.0)
                # Count positions that were not already effectively hard.
                total_changed += (prev_prob_a < 1.0 - 1e-6).sum().item()
                total_changed += (prev_prob_b < 1.0 - 1e-6).sum().item()
                total_numel += logits_a[..., 0].numel() + logits_b[..., 0].numel()
                module.unilora_soft_assign_logits_A[adapter_name].copy_(hard_a)
                module.unilora_soft_assign_logits_B[adapter_name].copy_(hard_b)
                module.unilora_soft_assign_hard_eval[adapter_name] = True
        exported = self.export_shared_codebook_and_indices(adapter_name=adapter_name)
        changed_ratio = float(total_changed) / float(total_numel) if total_numel > 0 else 0.0
        return {
            "changed_positions": int(total_changed),
            "total_positions": int(total_numel),
            "changed_ratio": changed_ratio,
            "merged_components": int(merged_components),
            "export_entries": int(len(exported)),
        }

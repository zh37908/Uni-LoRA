from __future__ import annotations

import torch

from peft.tuners.unilora_soft_assign.model import UniLoRASoftAssignModel
from peft.tuners.unilora_soft_assign.layer import UniLoRASoftAssignLayer

from .layer import UniLoRADeepKLayer


class UniLoRADeepKModel(UniLoRASoftAssignModel):
    prefix: str = "unilora_deepk_"
    tuner_layer_cls = UniLoRADeepKLayer

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)
        cfg = config[adapter_name] if isinstance(config, dict) else config
        self.deepk_config = cfg
        self._deepk_f_cache: dict[tuple[str, str], torch.Tensor] = {}
        self._deepk_f_step: dict[tuple[str, str], int] = {}
        self._deepk_last_stats: dict[str, float] = {}
        self._deepk_export_cache: dict[str, dict[str, torch.Tensor]] = {}

    @staticmethod
    def _spectral_F(W: torch.Tensor, k: int, rank_cap: int = 0) -> torch.Tensor:
        # W: [d, N], F: [N, k], F^T F = I
        d, n = W.shape
        k_eff = max(1, min(k, d, n))
        if rank_cap > 0:
            k_eff = min(k_eff, rank_cap)
        W_detached = W.detach()
        try:
            _, _, vh = torch.linalg.svd(W_detached, full_matrices=False)
        except RuntimeError as exc:
            # Some CUDA/cuSOLVER builds occasionally fail on SVD for medium matrices.
            # Fallback to CPU SVD for robustness.
            if W_detached.is_cuda and "cusolver" in str(exc).lower():
                _, _, vh = torch.linalg.svd(W_detached.float().cpu(), full_matrices=False)
                vh = vh.to(device=W_detached.device, dtype=W_detached.dtype)
            else:
                raise
        return vh[:k_eff, :].transpose(0, 1).contiguous()

    def _should_refresh_F(self, cache_key: tuple[str, str], global_step: int | None) -> bool:
        if cache_key not in self._deepk_f_cache:
            return True
        if global_step is None:
            return False
        interval = int(self.deepk_config.deepk_f_update_interval)
        last_step = self._deepk_f_step.get(cache_key, -1)
        return (global_step - last_step) >= interval

    def _layerwise_terms(self, adapter_name: str):
        for module_name, module in self.model.named_modules():
            if not isinstance(module, UniLoRASoftAssignLayer):
                continue
            if adapter_name not in module.unilora_soft_assign_logits_A:
                continue
            A, B = module._get_lora_matrices(adapter_name)
            # A: [r, in_features] -> columns are samples
            yield module_name, "A", A
            # B: [out_features, r] -> rows are samples, convert to [r, out_features]
            yield module_name, "B", B.transpose(0, 1)

    def compute_deepk_loss(self, adapter_name: str = "default", global_step: int | None = None) -> dict[str, torch.Tensor]:
        terms = []
        terms_a = []
        terms_b = []
        for module_name, matrix_type, W in self._layerwise_terms(adapter_name=adapter_name):
            if W.numel() == 0:
                continue
            k = int(self.deepk_config.deepk_num_clusters_a if matrix_type == "A" else self.deepk_config.deepk_num_clusters_b)
            cache_key = (module_name, matrix_type)
            if self._should_refresh_F(cache_key=cache_key, global_step=global_step):
                F = self._spectral_F(W, k=k, rank_cap=int(self.deepk_config.deepk_svd_rank_cap))
                self._deepk_f_cache[cache_key] = F.detach()
                if global_step is not None:
                    self._deepk_f_step[cache_key] = int(global_step)
            else:
                F = self._deepk_f_cache[cache_key].to(device=W.device, dtype=W.dtype)

            term1 = (W * W).sum()
            term2 = (W @ F).pow(2).sum()
            reg = 0.5 * (term1 - term2) / max(W.shape[1], 1)
            terms.append(reg)
            if matrix_type == "A":
                terms_a.append(reg)
            else:
                terms_b.append(reg)

        if not terms:
            zero = next(self.parameters()).new_tensor(0.0)
            return {
                "loss": zero,
                "reg_total": zero,
                "reg_a": zero,
                "reg_b": zero,
                "tau": zero,
                "num_terms": zero,
            }

        reg_total = torch.stack(terms).mean()
        reg_a = torch.stack(terms_a).mean() if terms_a else reg_total.new_tensor(0.0)
        reg_b = torch.stack(terms_b).mean() if terms_b else reg_total.new_tensor(0.0)
        tau = torch.as_tensor(float(self.deepk_config.deepk_tau), device=reg_total.device, dtype=reg_total.dtype)
        loss = tau * reg_total
        self._deepk_last_stats = {
            "reg_total": float(reg_total.detach().item()),
            "reg_a": float(reg_a.detach().item()),
            "reg_b": float(reg_b.detach().item()),
            "tau": float(tau.detach().item()),
            "num_terms": float(len(terms)),
        }
        return {
            "loss": loss,
            "reg_total": reg_total,
            "reg_a": reg_a,
            "reg_b": reg_b,
            "tau": tau,
            "num_terms": torch.as_tensor(float(len(terms)), device=reg_total.device, dtype=reg_total.dtype),
        }

    def get_deepk_stats(self, adapter_name: str = "default") -> dict[str, float]:
        if self._deepk_last_stats:
            return dict(self._deepk_last_stats)
        with torch.no_grad():
            info = self.compute_deepk_loss(adapter_name=adapter_name, global_step=None)
        return {
            "reg_total": float(info["reg_total"].item()),
            "reg_a": float(info["reg_a"].item()),
            "reg_b": float(info["reg_b"].item()),
            "tau": float(info["tau"].item()),
            "num_terms": float(info["num_terms"].item()),
        }

    @staticmethod
    def _run_kmeans(samples: torch.Tensor, k: int, max_iter: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
        # samples: [N, d]
        n, d = samples.shape
        if n == 0:
            return samples.new_zeros((0, d)), samples.new_zeros((0,), dtype=torch.long)
        k_eff = max(1, min(k, n))
        if k_eff == n:
            centers = samples.clone()
            labels = torch.arange(n, device=samples.device, dtype=torch.long)
            return centers, labels

        # Deterministic init from random permutation in current RNG state.
        perm = torch.randperm(n, device=samples.device)
        centers = samples[perm[:k_eff]].clone()
        labels = torch.zeros(n, device=samples.device, dtype=torch.long)
        for _ in range(max_iter):
            dist = torch.cdist(samples, centers, p=2)
            new_labels = dist.argmin(dim=1)
            if torch.equal(new_labels, labels):
                break
            labels = new_labels
            for i in range(k_eff):
                mask = labels == i
                if mask.any():
                    centers[i] = samples[mask].mean(dim=0)
        return centers, labels

    def finalize_deepk_assignment(self, adapter_name: str = "default", max_iter: int = 20) -> dict[str, float]:
        total_positions = 0
        changed_positions = 0
        module_exports: dict[str, torch.Tensor] = {}

        for module_name, module in self.model.named_modules():
            if not isinstance(module, UniLoRASoftAssignLayer):
                continue
            if adapter_name not in module.unilora_soft_assign_logits_A:
                continue

            with torch.no_grad():
                A, B = module._get_lora_matrices(adapter_name)
                # A: [r, in] -> column samples [in, r]
                a_samples = A.transpose(0, 1).contiguous()
                centers_a, labels_a = self._run_kmeans(
                    a_samples,
                    k=int(self.deepk_config.deepk_num_clusters_a),
                    max_iter=max_iter,
                )
                A_q = centers_a[labels_a].transpose(0, 1).contiguous()

                # B: [out, r] -> row samples [out, r]
                b_samples = B.contiguous()
                centers_b, labels_b = self._run_kmeans(
                    b_samples,
                    k=int(self.deepk_config.deepk_num_clusters_b),
                    max_iter=max_iter,
                )
                B_q = centers_b[labels_b].contiguous()

                prev_prob_a = torch.softmax(module.unilora_soft_assign_logits_A[adapter_name], dim=-1).amax(dim=-1)
                prev_prob_b = torch.softmax(module.unilora_soft_assign_logits_B[adapter_name], dim=-1).amax(dim=-1)
                changed_positions += int((prev_prob_a < 1.0 - 1e-6).sum().item())
                changed_positions += int((prev_prob_b < 1.0 - 1e-6).sum().item())
                total_positions += int(prev_prob_a.numel() + prev_prob_b.numel())

                module.unilora_soft_assign_hard_eval[adapter_name] = True
                module.unilora_deepk_override_A[adapter_name] = A_q.to(
                    device=module.get_base_layer().weight.device,
                    dtype=module.get_base_layer().weight.dtype,
                )
                module.unilora_deepk_override_B[adapter_name] = B_q.to(
                    device=module.get_base_layer().weight.device,
                    dtype=module.get_base_layer().weight.dtype,
                )
                logits_a = module.unilora_soft_assign_logits_A[adapter_name]
                logits_b = module.unilora_soft_assign_logits_B[adapter_name]
                idx_a = logits_a.argmax(dim=-1, keepdim=True)
                idx_b = logits_b.argmax(dim=-1, keepdim=True)
                hard_a = torch.zeros_like(logits_a).scatter_(-1, idx_a, 1.0)
                hard_b = torch.zeros_like(logits_b).scatter_(-1, idx_b, 1.0)
                module.unilora_soft_assign_logits_A[adapter_name].copy_(hard_a)
                module.unilora_soft_assign_logits_B[adapter_name].copy_(hard_b)

                module_exports[f"{module_name}.A_codebook"] = centers_a.detach().cpu()
                module_exports[f"{module_name}.A_indices"] = labels_a.detach().cpu().to(torch.int16)
                module_exports[f"{module_name}.B_codebook"] = centers_b.detach().cpu()
                module_exports[f"{module_name}.B_indices"] = labels_b.detach().cpu().to(torch.int16)
                module_exports[f"{module_name}.A_quantized"] = A_q.detach().cpu()
                module_exports[f"{module_name}.B_quantized"] = B_q.detach().cpu()

        self._deepk_export_cache[adapter_name] = module_exports
        changed_ratio = float(changed_positions) / float(total_positions) if total_positions > 0 else 0.0
        return {
            "changed_positions": float(changed_positions),
            "total_positions": float(total_positions),
            "changed_ratio": changed_ratio,
            "export_entries": float(len(module_exports)),
        }

    def export_deepk_cache(self, adapter_name: str = "default") -> dict[str, torch.Tensor]:
        return dict(self._deepk_export_cache.get(adapter_name, {}))

from __future__ import annotations

from typing import Tuple

import torch

from .layer import Linear


class MultiGpuLinear(Linear):
    """
    RoSA-SNIP linear layer optimized for `device_map=auto` model parallelism.

    The original RoSA forward moves the whole shared theta vector to each layer's
    device before indexing. This variant gathers on the parameter's resident
    device first, then transfers only the entries needed by the current layer.
    """

    @staticmethod
    def _gather_shared_values(
        values: torch.Tensor,
        indices: torch.Tensor,
        target_device: torch.device,
        cast_to_fp32: bool = False,
    ) -> torch.Tensor:
        flat_indices = indices.reshape(-1).long()
        if values.device == target_device:
            gathered = values.index_select(0, flat_indices.to(device=target_device))
        else:
            gathered = values.index_select(0, flat_indices.to(device=values.device))
            gathered = gathered.to(device=target_device)
        if cast_to_fp32:
            gathered = gathered.float()
        return gathered.view_as(indices)

    @staticmethod
    def _gather_sparse_values(
        sparse_theta_D: torch.Tensor,
        offsets: torch.Tensor,
        target_device: torch.device,
        target_dtype: torch.dtype,
        cast_to_fp32: bool = False,
    ) -> torch.Tensor:
        if sparse_theta_D.device == target_device:
            gathered = sparse_theta_D.index_select(0, offsets.to(device=target_device, dtype=torch.long))
        else:
            gathered = sparse_theta_D.index_select(0, offsets.to(device=sparse_theta_D.device, dtype=torch.long))
            gathered = gathered.to(device=target_device)
        if cast_to_fp32:
            gathered = gathered.float()
        return gathered.to(dtype=target_dtype)

    def _get_lora_matrices(self, adapter: str, cast_to_fp32: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        indices_A = self.unilora_indices_A[adapter]
        indices_B = self.unilora_indices_B[adapter]
        target_device = indices_A.device

        theta_d = self.unilora_rosa_theta_d[adapter]
        scales_A = self.unilora_scales_A[adapter]
        scales_B = self.unilora_scales_B[adapter]

        if cast_to_fp32:
            scales_A = scales_A.float()
            scales_B = scales_B.float()

        base_A = self._gather_shared_values(theta_d, indices_A, target_device, cast_to_fp32) * scales_A
        base_B = self._gather_shared_values(theta_d, indices_B, target_device, cast_to_fp32) * scales_B

        if self.sparse_active.get(adapter, False):
            sparse_theta_D = self.unilora_rosa_sparse_theta_D[adapter]

            flat_indices_A = self.sparse_flat_indices_A.get(adapter)
            if flat_indices_A is not None and flat_indices_A.numel() > 0:
                A_flat = base_A.reshape(-1)
                sparse_values_A = self._gather_sparse_values(
                    sparse_theta_D,
                    self.sparse_theta_offsets_A_active[adapter],
                    target_device,
                    A_flat.dtype,
                    cast_to_fp32,
                )
                A_flat.index_add_(0, flat_indices_A.to(device=target_device, dtype=torch.long), sparse_values_A)
                A = A_flat.view_as(base_A)
            else:
                A = base_A

            flat_indices_B = self.sparse_flat_indices_B.get(adapter)
            if flat_indices_B is not None and flat_indices_B.numel() > 0:
                B_flat = base_B.reshape(-1)
                sparse_values_B = self._gather_sparse_values(
                    sparse_theta_D,
                    self.sparse_theta_offsets_B_active[adapter],
                    target_device,
                    B_flat.dtype,
                    cast_to_fp32,
                )
                B_flat.index_add_(0, flat_indices_B.to(device=target_device, dtype=torch.long), sparse_values_B)
                B = B_flat.view_as(base_B)
            else:
                B = base_B
        else:
            A = base_A
            B = base_B

        if self.capture_gradient_stats and self.training and not cast_to_fp32:
            self._register_gradient_capture_hook(adapter, "A", A)
            self._register_gradient_capture_hook(adapter, "B", B)

        return A, B

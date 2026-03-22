from __future__ import annotations

import torch


def select_code_dtype(codebook_size: int) -> torch.dtype:
    if codebook_size <= 2**8:
        return torch.uint8
    if codebook_size <= 2**15:
        return torch.int16
    if codebook_size <= 2**31:
        return torch.int32
    return torch.int64


def compute_group_setup(width: int, groups_per_row: int) -> tuple[int, int]:
    if groups_per_row <= 0:
        raise ValueError(f"`groups_per_row` must be positive, got {groups_per_row}.")
    group_size = (width + groups_per_row - 1) // groups_per_row
    padded_width = groups_per_row * group_size
    return group_size, padded_width


def generate_balanced_indices(total_length: int, num_buckets: int, seed: int) -> torch.Tensor:
    if total_length <= 0:
        return torch.empty(0, dtype=torch.long)
    if num_buckets <= 0:
        raise ValueError(f"`num_buckets` must be positive, got {num_buckets}.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    base_count = total_length // num_buckets
    remainder = total_length % num_buckets

    chunks = []
    if base_count > 0:
        chunks.append(torch.arange(num_buckets, dtype=torch.long).repeat_interleave(base_count))
    if remainder > 0:
        chunks.append(torch.randperm(num_buckets, generator=generator, dtype=torch.long)[:remainder])

    indices = torch.cat(chunks, dim=0)
    perm = torch.randperm(indices.numel(), generator=generator, dtype=torch.long)
    return indices[perm]


def decode_local_codebook(codebook: torch.Tensor, codes: torch.Tensor, width: int) -> torch.Tensor:
    values = torch.gather(codebook, dim=-1, index=codes.long())
    return values.reshape(codebook.shape[0], -1)[:, :width]


def decode_shared_bank(
    sketch_bank: torch.Tensor,
    bank_indices: torch.Tensor,
    codes: torch.Tensor,
    width: int,
) -> torch.Tensor:
    selected_banks = sketch_bank[bank_indices.long()]
    if selected_banks.dim() == codes.dim() + 1:
        values = torch.gather(selected_banks, dim=-1, index=codes.long().unsqueeze(-1)).squeeze(-1)
    else:
        values = torch.gather(selected_banks, dim=-1, index=codes.long())
    return values.reshape(codes.shape[0], -1)[:, :width]

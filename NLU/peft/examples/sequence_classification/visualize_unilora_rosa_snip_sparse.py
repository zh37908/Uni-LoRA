#!/usr/bin/env python
# pyright: reportMissingImports=false
"""Visualize UniLoRA-RoSA-SNIP sparse adapter activation positions.

The script reads a saved checkpoint/state_dict containing
`unilora_rosa_sparse_mask` and per-module `unilora_theta_D_offsets_{A,B}`.
By default it visualizes the dense rows/columns affected by sparse updates in
the original linear layer, which is closest to the RoSA mask plots in the paper.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


CHECKPOINT_PATTERNS = (
    "adapter_model.safetensors",
    "model.safetensors",
    "pytorch_model.bin",
    "adapter_model.bin",
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize UniLoRA-RoSA-SNIP sparse mask activation positions from a checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint file or directory. Directories are searched recursively for common state_dict files.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path.")
    parser.add_argument("--summary-csv", type=Path, default=None, help="Optional CSV summary path.")
    parser.add_argument("--adapter", default="default", help="Adapter name in the state_dict.")
    parser.add_argument(
        "--mode",
        choices=["dense_support", "low_rank"],
        default="dense_support",
        help=(
            "dense_support plots original-layer rows/columns affected by sparse A/B updates; "
            "low_rank plots sparse positions in the low-rank A/B tensors directly."
        ),
    )
    parser.add_argument(
        "--module-regex",
        default=None,
        help="Regex filter after stripping common state_dict prefixes, e.g. 'layer\\.0.*(query|key|value)'.",
    )
    parser.add_argument("--max-modules", type=int, default=12, help="Maximum modules to show in one figure.")
    parser.add_argument("--cols", type=int, default=4, help="Number of subplot columns.")
    parser.add_argument(
        "--pool",
        type=int,
        default=4,
        help="Max-pool kernel/stride for display. Use 1 to draw the raw mask.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Figure DPI.")
    return parser.parse_args()


def find_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    candidates: list[Path] = []
    for pattern in CHECKPOINT_PATTERNS:
        candidates.extend(p for p in path.rglob(pattern) if p.is_file())
    candidates = sorted(set(candidates), key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint/state_dict file found under {path}. "
            "The result directory you pass must contain .safetensors/.bin/.pt weights with RoSA masks."
        )
    return candidates[0]


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("Install safetensors to read .safetensors checkpoints.") from exc
        obj = load_file(str(path), device="cpu")
    else:
        obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict):
        for key in ("state_dict", "model", "module"):
            nested = obj.get(key)
            if isinstance(nested, dict) and any(torch.is_tensor(v) for v in nested.values()):
                obj = nested
                break

    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint object type: {type(obj)!r}")

    state_dict = {str(k): v.detach().cpu() for k, v in obj.items() if torch.is_tensor(v)}
    if not state_dict:
        raise ValueError(f"No tensors found in checkpoint: {path}")
    return state_dict


def strip_common_prefix(name: str) -> str:
    prefixes = (
        "base_model.model.",
        "base_model.",
        "model.",
        "module.",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                changed = True
    return name


def sort_key(name: str) -> tuple[int, int, str]:
    layer_match = re.search(r"(?:layer|layers)\.(\d+)", name)
    layer_id = int(layer_match.group(1)) if layer_match else 10**9
    target_priority = (
        ("attention.self.query", 0),
        ("self_attn.q_proj", 0),
        ("query", 0),
        ("attention.self.key", 1),
        ("self_attn.k_proj", 1),
        ("key", 1),
        ("attention.self.value", 2),
        ("self_attn.v_proj", 2),
        ("value", 2),
        ("attention.output.dense", 3),
        ("self_attn.o_proj", 3),
        ("intermediate.dense", 4),
        ("output.dense", 5),
    )
    priority = 100
    for token, value in target_priority:
        if token in name:
            priority = value
            break
    return (layer_id, priority, name)


def find_global_mask(state_dict: dict[str, torch.Tensor], adapter: str) -> tuple[str, torch.Tensor]:
    mask_suffix = f"unilora_rosa_sparse_mask.{adapter}"
    candidates = [(k, v) for k, v in state_dict.items() if k.endswith(mask_suffix) and v.ndim == 1]
    if not candidates:
        theta_suffix = f"unilora_rosa_sparse_theta_D.{adapter}"
        candidates = [(k, v.ne(0)) for k, v in state_dict.items() if k.endswith(theta_suffix) and v.ndim == 1]
    if not candidates:
        raise KeyError(
            f"Cannot find '{mask_suffix}' in the checkpoint. "
            "Make sure sparse masks were activated and saved."
        )

    key, tensor = max(candidates, key=lambda item: item[1].numel())
    return key, tensor.bool()


def collect_modules(state_dict: dict[str, torch.Tensor], adapter: str) -> dict[str, dict[str, torch.Tensor]]:
    suffix_a = f".unilora_theta_D_offsets_A.{adapter}"
    suffix_b = f".unilora_theta_D_offsets_B.{adapter}"
    modules: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in state_dict.items():
        if key.endswith(suffix_a):
            modules.setdefault(key[: -len(suffix_a)], {})["A"] = tensor.long()
        elif key.endswith(suffix_b):
            modules.setdefault(key[: -len(suffix_b)], {})["B"] = tensor.long()

    modules = {name: tensors for name, tensors in modules.items() if {"A", "B"} <= tensors.keys()}
    if not modules:
        raise KeyError(
            f"Cannot find per-module unilora_theta_D_offsets_A/B tensors for adapter '{adapter}'."
        )
    return modules


def pool_for_display(mask: torch.Tensor, pool: int) -> torch.Tensor:
    mask = mask.float()
    if pool <= 1:
        return mask
    # Max pooling mirrors the visualization style used in the RoSA paper figure.
    return F.max_pool2d(mask[None, None], kernel_size=pool, stride=pool, ceil_mode=True)[0, 0]


def build_dense_support(active_a: torch.Tensor, active_b: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    active_cols = active_a.any(dim=0)
    active_rows = active_b.any(dim=1)
    dense_support = active_rows[:, None] | active_cols[None, :]
    row_density = active_rows.float().mean().item() if active_rows.numel() else 0.0
    col_density = active_cols.float().mean().item() if active_cols.numel() else 0.0
    support_density = 1.0 - (1.0 - row_density) * (1.0 - col_density)
    return dense_support, {
        "active_rows": int(active_rows.sum().item()),
        "total_rows": int(active_rows.numel()),
        "active_cols": int(active_cols.sum().item()),
        "total_cols": int(active_cols.numel()),
        "row_density": row_density,
        "col_density": col_density,
        "support_density": support_density,
    }


def make_plot_items(
    modules: dict[str, dict[str, torch.Tensor]],
    sparse_mask: torch.Tensor,
    module_regex: str | None,
    mode: str,
    pool: int,
    max_modules: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    regex = re.compile(module_regex) if module_regex else None
    items: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []

    sorted_modules = sorted(modules.items(), key=lambda item: sort_key(strip_common_prefix(item[0])))
    for module_name, tensors in sorted_modules:
        label = strip_common_prefix(module_name)
        if regex and not regex.search(label):
            continue

        offsets_a = tensors["A"]
        offsets_b = tensors["B"]
        if offsets_a.max().item() >= sparse_mask.numel() or offsets_b.max().item() >= sparse_mask.numel():
            raise IndexError(
                f"Offsets in {label} exceed sparse mask length {sparse_mask.numel()}."
            )

        active_a = sparse_mask[offsets_a.reshape(-1)].reshape_as(offsets_a)
        active_b = sparse_mask[offsets_b.reshape(-1)].reshape_as(offsets_b)
        selected_a = int(active_a.sum().item())
        selected_b = int(active_b.sum().item())
        selected_total = selected_a + selected_b
        total_low_rank = int(active_a.numel() + active_b.numel())

        if mode == "dense_support":
            matrix, dense_stats = build_dense_support(active_a, active_b)
            display_matrix = pool_for_display(matrix, pool)
            title = (
                f"{label}\n"
                f"Empty Rows: {(1.0 - dense_stats['row_density']) * 100:.0f}%  "
                f"Empty Columns: {(1.0 - dense_stats['col_density']) * 100:.0f}%"
            )
            items.append({"title": title, "matrix": display_matrix})
            summary.append(
                {
                    "module": label,
                    "view": "dense_support",
                    "shape": f"{matrix.shape[0]}x{matrix.shape[1]}",
                    "selected_A": selected_a,
                    "selected_B": selected_b,
                    "selected_total": selected_total,
                    "low_rank_density": selected_total / total_low_rank if total_low_rank else 0.0,
                    **dense_stats,
                }
            )
        else:
            for side, active in (("A", active_a), ("B", active_b)):
                display_matrix = pool_for_display(active, pool)
                density = active.float().mean().item() if active.numel() else 0.0
                title = f"{label} | {side}\nActive: {density * 100:.2f}%"
                items.append({"title": title, "matrix": display_matrix})
                summary.append(
                    {
                        "module": label,
                        "view": side,
                        "shape": f"{active.shape[0]}x{active.shape[1]}",
                        "selected_A": selected_a,
                        "selected_B": selected_b,
                        "selected_total": selected_total,
                        "low_rank_density": selected_total / total_low_rank if total_low_rank else 0.0,
                    }
                )

        if len(items) >= max_modules:
            break

    if not items:
        raise ValueError("No modules matched. Try removing or relaxing --module-regex.")
    return items, summary


def save_figure(items: list[dict[str, Any]], output: Path, cols: int, dpi: int) -> None:
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]

    cols = max(1, cols)
    rows = int(math.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 4.2 * rows), squeeze=False)
    for ax in axes.reshape(-1):
        ax.axis("off")

    for ax, item in zip(axes.reshape(-1), items):
        matrix = item["matrix"].numpy()
        ax.imshow(matrix, cmap="Greys", interpolation="nearest", aspect="auto", vmin=0.0, vmax=1.0)
        ax.set_title(item["title"], fontsize=9)
        ax.set_xlabel("input / columns")
        ax.set_ylabel("output / rows")
        ax.axis("on")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_summary(summary: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in summary for key in row.keys()})
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def main() -> None:
    args = parse_args()
    checkpoint = find_checkpoint(args.checkpoint)
    state_dict = load_state_dict(checkpoint)
    mask_key, sparse_mask = find_global_mask(state_dict, args.adapter)
    modules = collect_modules(state_dict, args.adapter)

    output = args.output
    if output is None:
        output = checkpoint.parent / f"unilora_rosa_snip_sparse_{args.mode}.png"
    summary_csv = args.summary_csv
    if summary_csv is None:
        summary_csv = output.with_suffix(".csv")

    items, summary = make_plot_items(
        modules=modules,
        sparse_mask=sparse_mask,
        module_regex=args.module_regex,
        mode=args.mode,
        pool=args.pool,
        max_modules=args.max_modules,
    )
    save_figure(items, output, cols=args.cols, dpi=args.dpi)
    save_summary(summary, summary_csv)

    selected = int(sparse_mask.sum().item())
    print(f"Loaded checkpoint: {checkpoint}")
    print(f"Sparse mask key: {mask_key}")
    print(f"Selected positions: {selected}/{sparse_mask.numel()} ({selected / sparse_mask.numel():.6f})")
    print(f"Visualized modules/views: {len(items)}")
    print(f"Saved figure: {output}")
    print(f"Saved summary: {summary_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

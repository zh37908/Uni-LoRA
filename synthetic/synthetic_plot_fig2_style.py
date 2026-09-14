#!/usr/bin/env python
# coding: utf-8

"""Create a paper Fig. 2-style synthetic validation figure from CSV results."""

from __future__ import annotations

import argparse
import csv
import importlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ResultRow:
    d: int
    n: int
    alpha: float
    noise_std: float
    method: str
    risk_mean: float
    risk_std: float
    bias: float
    variance: float


METHOD_ORDER = ["lora", "unilora", "prolosa", "unilora_oracle"]
METHOD_LABELS = {
    "lora": "LoRA",
    "unilora": "Uni-LoRA",
    "prolosa": "ProLoSA",
    "unilora_oracle": "Uni-LoRA (oracle)",
}
METHOD_COLORS = {
    "lora": "#1f77b4",
    "unilora": "#ff7f0e",
    "prolosa": "#2ca02c",
    "unilora_oracle": "#9467bd",
}


def read_rows(path: Path) -> list[ResultRow]:
    rows: list[ResultRow] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            rows.append(
                ResultRow(
                    d=int(raw["d"]),
                    n=int(raw["n"]),
                    alpha=float(raw["alpha"]),
                    noise_std=float(raw["noise_std"]),
                    method=str(raw["method"]),
                    risk_mean=float(raw["risk_mean"]),
                    risk_std=float(raw["risk_std"]),
                    bias=float(raw["bias"]),
                    variance=float(raw["variance"]),
                )
            )
    if not rows:
        raise ValueError(f"No rows found in {path}.")
    return rows


def median_value(values: Iterable[float]) -> float:
    ordered = sorted(set(values))
    return ordered[len(ordered) // 2]


def ordered_methods(rows: Iterable[ResultRow]) -> list[str]:
    methods = {row.method for row in rows}
    ordered = [method for method in METHOD_ORDER if method in methods]
    ordered.extend(sorted(methods.difference(ordered)))
    return ordered


def nice_upper(value: float) -> float:
    if value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    fraction = value / (10**exponent)
    if fraction <= 1.5:
        nice = 1.5
    elif fraction <= 2:
        nice = 2
    elif fraction <= 3:
        nice = 3
    elif fraction <= 5:
        nice = 5
    else:
        nice = 10
    return nice * (10**exponent)


def make_ticks(max_value: float, count: int = 6) -> list[float]:
    upper = nice_upper(max_value)
    return [upper * i / (count - 1) for i in range(count)]


def select_figure_data(
    rows: list[ResultRow],
    d: int | None,
    noise_std: float | None,
    small_n: int | None,
    large_n: int | None,
) -> tuple[int, float, int, int, list[ResultRow], list[ResultRow]]:
    selected_d = d if d is not None else min({row.d for row in rows})
    d_rows = [row for row in rows if row.d == selected_d]
    if not d_rows:
        raise ValueError(f"No rows match d={selected_d}.")

    selected_noise = noise_std if noise_std is not None else median_value(row.noise_std for row in d_rows)
    noise_rows = [row for row in d_rows if math.isclose(row.noise_std, selected_noise)]
    if not noise_rows:
        raise ValueError(f"No rows match d={selected_d}, noise_std={selected_noise:g}.")

    sample_sizes = sorted({row.n for row in noise_rows})
    selected_small_n = small_n if small_n is not None else sample_sizes[0]
    selected_large_n = large_n if large_n is not None else sample_sizes[-1]
    risk_rows = [row for row in noise_rows if row.n in {selected_small_n, selected_large_n}]
    if not risk_rows:
        raise ValueError(
            f"No rows match n in {{{selected_small_n}, {selected_large_n}}} "
            f"with d={selected_d}, noise_std={selected_noise:g}."
        )

    biasvar_rows = [row for row in noise_rows if row.n == selected_large_n]
    if not biasvar_rows:
        raise ValueError(f"No bias/variance rows match n={selected_large_n}.")

    return selected_d, selected_noise, selected_small_n, selected_large_n, risk_rows, biasvar_rows


def plot_with_matplotlib(
    output_png: Path,
    output_pdf: Path | None,
    d: int,
    noise_std: float,
    small_n: int,
    large_n: int,
    risk_rows: list[ResultRow],
    biasvar_rows: list[ResultRow],
    dpi: int,
) -> None:
    plt = importlib.import_module("matplotlib.pyplot")

    methods = ordered_methods(risk_rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6))

    ax = axes[0]
    for n, linestyle in [(small_n, "-"), (large_n, "--")]:
        for method in methods:
            rows = sorted(
                [row for row in risk_rows if row.n == n and row.method == method],
                key=lambda row: row.alpha,
            )
            if not rows:
                continue
            ax.plot(
                [row.alpha for row in rows],
                [row.risk_mean for row in rows],
                marker="o",
                linewidth=2.0,
                markersize=4,
                linestyle=linestyle,
                color=METHOD_COLORS.get(method, "#4b5563"),
                label=f"{METHOD_LABELS.get(method, method)} (n={n})",
            )
    ax.set_title("Synthetic theory validation: risk vs. mismatch", fontsize=10)
    ax.set_xlabel("Subspace mismatch", fontsize=9)
    ax.set_ylabel("Population excess risk", fontsize=9)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=True)

    ax = axes[1]
    for method in methods:
        rows = sorted([row for row in biasvar_rows if row.method == method], key=lambda row: row.alpha)
        if not rows:
            continue
        color = METHOD_COLORS.get(method, "#4b5563")
        label = METHOD_LABELS.get(method, method)
        ax.plot(
            [row.alpha for row in rows],
            [row.bias for row in rows],
            marker="o",
            linewidth=2.0,
            markersize=4,
            color=color,
            label=f"{label} bias",
        )
        ax.plot(
            [row.alpha for row in rows],
            [row.variance for row in rows],
            marker="s",
            linewidth=2.0,
            markersize=3.5,
            linestyle="--",
            color=color,
            label=f"{label} variance",
        )
    ax.set_title(f"Bias-variance decomposition (n={large_n})", fontsize=10)
    ax.set_xlabel("Subspace mismatch", fontsize=9)
    ax.set_ylabel("Risk contribution", fontsize=9)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=True)

    fig.suptitle(f"d={d}, noise std={noise_std:g}", fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    if output_pdf is not None:
        fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def load_font(size: int, bold: bool = False) -> Any:
    image_font = importlib.import_module("PIL.ImageFont")
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return image_font.truetype(str(path), size=size)
    return image_font.load_default()


def draw_rotated_text(draw_image, xy: tuple[int, int], text: str, font, fill: str, angle: int = 90) -> None:
    image_module = importlib.import_module("PIL.Image")
    image_draw = importlib.import_module("PIL.ImageDraw")
    bbox = font.getbbox(text)
    width = bbox[2] - bbox[0] + 8
    height = bbox[3] - bbox[1] + 8
    text_image = image_module.new("RGBA", (width, height), (255, 255, 255, 0))
    text_draw = image_draw.Draw(text_image)
    text_draw.text((4, 4 - bbox[1]), text, font=font, fill=fill)
    rotated = text_image.rotate(angle, expand=True)
    draw_image.alpha_composite(rotated, (xy[0] - rotated.width // 2, xy[1] - rotated.height // 2))


def draw_dashed_line(draw, start: tuple[float, float], end: tuple[float, float], fill: str, width: int) -> None:
    x0, y0 = start
    x1, y1 = end
    length = math.hypot(x1 - x0, y1 - y0)
    if length <= 0:
        return
    dash = 12.0
    gap = 8.0
    distance = 0.0
    while distance < length:
        next_distance = min(distance + dash, length)
        sx = x0 + (x1 - x0) * distance / length
        sy = y0 + (y1 - y0) * distance / length
        ex = x0 + (x1 - x0) * next_distance / length
        ey = y0 + (y1 - y0) * next_distance / length
        draw.line((sx, sy, ex, ey), fill=fill, width=width)
        distance += dash + gap


def draw_polyline(
    draw,
    points: list[tuple[float, float]],
    fill: str,
    width: int,
    dashed: bool = False,
    marker: str = "circle",
) -> None:
    for start, end in zip(points, points[1:]):
        if dashed:
            draw_dashed_line(draw, start, end, fill=fill, width=width)
        else:
            draw.line((*start, *end), fill=fill, width=width)

    radius = 5
    for x, y in points:
        if marker == "square":
            draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=fill)
        else:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)


def draw_panel(
    draw,
    image,
    box: tuple[int, int, int, int],
    title: str,
    xlabel: str,
    ylabel: str,
    x_values: list[float],
    y_max: float,
    series: list[tuple[str, str, bool, str, list[tuple[float, float]]]],
    fonts: dict[str, object],
) -> None:
    left, top, right, bottom = box
    plot_w = right - left
    plot_h = bottom - top
    x_min, x_max = min(x_values), max(x_values)
    x_range = x_max - x_min if not math.isclose(x_min, x_max) else 1.0
    y_ticks = make_ticks(y_max)
    y_upper = y_ticks[-1]

    def sx(value: float) -> float:
        return left + (value - x_min) / x_range * plot_w

    def sy(value: float) -> float:
        return bottom - value / y_upper * plot_h

    grid_color = "#d9d9d9"
    axis_color = "#222222"
    text_color = "#222222"

    for value in y_ticks:
        y = sy(value)
        draw.line((left, y, right, y), fill=grid_color, width=1)
        draw.text((left - 12, y), f"{value:g}", anchor="rm", font=fonts["tick"], fill=text_color)

    for value in x_values:
        x = sx(value)
        draw.line((x, bottom, x, bottom + 7), fill=axis_color, width=2)
        draw.text((x, bottom + 18), f"{value:.2f}", anchor="mt", font=fonts["tick"], fill=text_color)

    draw.line((left, bottom, right, bottom), fill=axis_color, width=2)
    draw.line((left, top, left, bottom), fill=axis_color, width=2)
    draw.text(((left + right) // 2, top - 30), title, anchor="mm", font=fonts["title"], fill=text_color)
    draw.text(((left + right) // 2, bottom + 54), xlabel, anchor="mm", font=fonts["label"], fill=text_color)
    draw_rotated_text(image, (left - 62, (top + bottom) // 2), ylabel, fonts["label"], text_color)

    legend_x = left + 12
    legend_y = top + 12
    line_height = 22
    max_label_width = 0
    for label, _, _, _, _ in series:
        bbox = fonts["legend"].getbbox(label)
        max_label_width = max(max_label_width, bbox[2] - bbox[0])

    for _, color, dashed, marker, raw_points in series:
        points = [(sx(x), sy(y)) for x, y in raw_points]
        draw_polyline(draw, points, fill=color, width=3, dashed=dashed, marker=marker)

    draw.rounded_rectangle(
        (
            legend_x - 8,
            legend_y - 8,
            legend_x + max_label_width + 42,
            legend_y + line_height * len(series) + 2,
        ),
        radius=4,
        fill="#ffffff",
        outline="#dddddd",
    )

    for idx, (label, color, dashed, marker, _) in enumerate(series):
        y = legend_y + idx * line_height
        if dashed:
            draw_dashed_line(draw, (legend_x, y + 6), (legend_x + 24, y + 6), fill=color, width=3)
        else:
            draw.line((legend_x, y + 6, legend_x + 24, y + 6), fill=color, width=3)
        if marker == "square":
            draw.rectangle((legend_x + 8, y + 1, legend_x + 16, y + 9), fill=color)
        else:
            draw.ellipse((legend_x + 8, y + 1, legend_x + 16, y + 9), fill=color)
        draw.text((legend_x + 32, y), label, font=fonts["legend"], fill=text_color)


def plot_with_pillow(
    output_png: Path,
    output_pdf: Path | None,
    d: int,
    noise_std: float,
    small_n: int,
    large_n: int,
    risk_rows: list[ResultRow],
    biasvar_rows: list[ResultRow],
    dpi: int,
) -> None:
    image_module = importlib.import_module("PIL.Image")
    image_draw = importlib.import_module("PIL.ImageDraw")
    scale = max(1.0, dpi / 150)
    width = int(1500 * scale)
    height = int(540 * scale)
    image = image_module.new("RGBA", (width, height), "white")
    draw = image_draw.Draw(image)
    fonts = {
        "title": load_font(int(15 * scale), bold=True),
        "label": load_font(int(13 * scale)),
        "tick": load_font(int(10 * scale)),
        "legend": load_font(int(10 * scale)),
        "suptitle": load_font(int(11 * scale)),
    }

    methods = ordered_methods(risk_rows)
    alphas = sorted({row.alpha for row in risk_rows})
    left_series = []
    for n, dashed in [(small_n, False), (large_n, True)]:
        for method in methods:
            rows = sorted(
                [row for row in risk_rows if row.n == n and row.method == method],
                key=lambda row: row.alpha,
            )
            if not rows:
                continue
            label = f"{METHOD_LABELS.get(method, method)} (n={n})"
            left_series.append(
                (
                    label,
                    METHOD_COLORS.get(method, "#4b5563"),
                    dashed,
                    "circle",
                    [(row.alpha, row.risk_mean) for row in rows],
                )
            )

    right_series = []
    for method in methods:
        rows = sorted([row for row in biasvar_rows if row.method == method], key=lambda row: row.alpha)
        if not rows:
            continue
        label = METHOD_LABELS.get(method, method)
        color = METHOD_COLORS.get(method, "#4b5563")
        right_series.append((f"{label} bias", color, False, "circle", [(row.alpha, row.bias) for row in rows]))
        right_series.append(
            (f"{label} variance", color, True, "square", [(row.alpha, row.variance) for row in rows])
        )

    draw.text(
        (width // 2, int(24 * scale)),
        f"d={d}, noise std={noise_std:g}",
        anchor="mm",
        font=fonts["suptitle"],
        fill="#222222",
    )
    panel_top = int(90 * scale)
    panel_bottom = int(420 * scale)
    left_box = (int(105 * scale), panel_top, int(690 * scale), panel_bottom)
    right_box = (int(875 * scale), panel_top, int(1460 * scale), panel_bottom)

    left_max = max(y for _, _, _, _, points in left_series for _, y in points) * 1.05
    right_max = max(y for _, _, _, _, points in right_series for _, y in points) * 1.08
    draw_panel(
        draw,
        image,
        left_box,
        "Synthetic theory validation: risk vs. mismatch",
        "Subspace mismatch",
        "Population excess risk",
        alphas,
        left_max,
        left_series,
        fonts,
    )
    draw_panel(
        draw,
        image,
        right_box,
        f"Bias-variance decomposition (n={large_n})",
        "Subspace mismatch",
        "Risk contribution",
        alphas,
        right_max,
        right_series,
        fonts,
    )

    rgb_image = image.convert("RGB")
    rgb_image.save(output_png, dpi=(dpi, dpi))
    if output_pdf is not None:
        rgb_image.save(output_pdf, "PDF", resolution=dpi)


def plot_figure(
    rows: list[ResultRow],
    output_png: Path,
    output_pdf: Path | None,
    d: int | None,
    noise_std: float | None,
    small_n: int | None,
    large_n: int | None,
    dpi: int,
) -> None:
    selected_d, selected_noise, selected_small_n, selected_large_n, risk_rows, biasvar_rows = select_figure_data(
        rows=rows,
        d=d,
        noise_std=noise_std,
        small_n=small_n,
        large_n=large_n,
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    if output_pdf is not None:
        output_pdf.parent.mkdir(parents=True, exist_ok=True)

    try:
        plot_with_matplotlib(
            output_png=output_png,
            output_pdf=output_pdf,
            d=selected_d,
            noise_std=selected_noise,
            small_n=selected_small_n,
            large_n=selected_large_n,
            risk_rows=risk_rows,
            biasvar_rows=biasvar_rows,
            dpi=dpi,
        )
    except ImportError:
        plot_with_pillow(
            output_png=output_png,
            output_pdf=output_pdf,
            d=selected_d,
            noise_std=selected_noise,
            small_n=selected_small_n,
            large_n=selected_large_n,
            risk_rows=risk_rows,
            biasvar_rows=biasvar_rows,
            dpi=dpi,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results_synthetic_theory_snip_better"),
        help="Directory containing synthetic_theory_results.csv.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional explicit CSV path. Overrides --results-dir.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to <results-dir>/synthetic_fig2_style.png.",
    )
    parser.add_argument(
        "--pdf-output",
        type=Path,
        default=None,
        help="Optional output PDF path. Defaults to the PNG path with .pdf suffix.",
    )
    parser.add_argument("--d", type=int, default=None, help="Compressed dimension to plot.")
    parser.add_argument("--noise-std", type=float, default=None, help="Noise level to plot.")
    parser.add_argument("--small-n", type=int, default=None, help="Small-sample curve. Defaults to min n.")
    parser.add_argument("--large-n", type=int, default=None, help="Large-sample curve. Defaults to max n.")
    parser.add_argument("--dpi", type=int, default=300, help="PNG output resolution.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    csv_path = args.csv if args.csv is not None else args.results_dir / "synthetic_theory_results.csv"
    output_png = args.output if args.output is not None else args.results_dir / "synthetic_fig2_style.png"
    output_pdf = args.pdf_output if args.pdf_output is not None else output_png.with_suffix(".pdf")

    rows = read_rows(csv_path)
    plot_figure(
        rows=rows,
        output_png=output_png,
        output_pdf=output_pdf,
        d=args.d,
        noise_std=args.noise_std,
        small_n=args.small_n,
        large_n=args.large_n,
        dpi=args.dpi,
    )
    print(f"Wrote {output_png}")
    if output_pdf is not None:
        print(f"Wrote {output_pdf}")


if __name__ == "__main__":
    main()

import argparse
import json
import math
import os

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Plot Hessian history heatmaps from result JSON",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-json",
        type=str,
        required=True,
        help="Path to a result JSON produced by multi_hash_hessian_aware.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory used to save generated heatmaps",
    )
    parser.add_argument(
        "--max-snapshots",
        type=int,
        default=0,
        help="Maximum number of snapshots to plot, 0 means all",
    )
    parser.add_argument(
        "--only-epoch",
        type=int,
        default=None,
        help="If set, only plot the snapshot recorded at this epoch",
    )
    parser.add_argument(
        "--tick-label-limit",
        type=int,
        default=20,
        help="Show coordinate tick labels only when sample size is <= this limit",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Figure DPI",
    )
    return parser.parse_args()


def load_payload(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_output_dir(results_json, output_dir):
    if output_dir is not None:
        return output_dir

    base_name = os.path.splitext(os.path.basename(results_json))[0]
    parent = os.path.dirname(results_json) or "."
    return os.path.join(parent, base_name + "_hessian_heatmaps")


def collect_hessian_snapshots(payload):
    snapshots = payload.get("hessian_history", [])
    if snapshots:
        return snapshots

    fallback = []
    for item in payload.get("structure_history", []):
        snapshot = item.get("hessian_snapshot")
        if snapshot is not None:
            fallback.append(snapshot)
    return fallback


def shorten_float(value):
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1000 or magnitude < 1e-3:
        return "{:.2e}".format(value)
    return "{:.4f}".format(value)


def build_tick_labels(snapshot):
    labels = []
    for coordinate, flat_index in zip(snapshot.get("sample_coordinates", []), snapshot.get("sample_indices", [])):
        row = coordinate.get("row", "?")
        col = coordinate.get("col", "?")
        labels.append("({},{})\n#{}".format(row, col, flat_index))
    return labels


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def lerp_color(color_a, color_b, t):
    return tuple(int(round(color_a[i] + (color_b[i] - color_a[i]) * t)) for i in range(3))


def interpolate_stops(stops, value):
    if value <= stops[0][0]:
        return stops[0][1]
    if value >= stops[-1][0]:
        return stops[-1][1]

    for index in range(1, len(stops)):
        left_pos, left_color = stops[index - 1]
        right_pos, right_color = stops[index]
        if value <= right_pos:
            t = (value - left_pos) / max(right_pos - left_pos, 1e-12)
            return lerp_color(left_color, right_color, t)
    return stops[-1][1]


def signed_heat_color(value, vmax):
    if vmax <= 0.0:
        return (255, 255, 255)
    normalized = clamp(value / vmax, -1.0, 1.0)
    stops = [
        (-1.0, (49, 54, 149)),
        (-0.35, (116, 173, 209)),
        (0.0, (255, 255, 255)),
        (0.35, (244, 109, 67)),
        (1.0, (165, 0, 38)),
    ]
    return interpolate_stops(stops, normalized)


def unsigned_heat_color(value, vmax):
    if vmax <= 0.0:
        return (0, 0, 0)
    normalized = clamp(value / vmax, 0.0, 1.0)
    stops = [
        (0.0, (0, 0, 4)),
        (0.25, (59, 15, 112)),
        (0.5, (136, 34, 106)),
        (0.75, (229, 107, 40)),
        (1.0, (252, 253, 191)),
    ]
    return interpolate_stops(stops, normalized)


def matrix_dimensions(matrix):
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    return rows, cols


def max_abs_value(matrix):
    best = 0.0
    for row in matrix:
        for value in row:
            best = max(best, abs(float(value)))
    return best


def max_value(matrix):
    best = 0.0
    for row in matrix:
        for value in row:
            best = max(best, float(value))
    return best


def build_abs_matrix(matrix):
    return [[abs(float(value)) for value in row] for row in matrix]


def choose_cell_size(size):
    if size <= 8:
        return 34
    if size <= 16:
        return 24
    if size <= 24:
        return 18
    return 12


def text_block_height(font, line_count, line_spacing=4):
    bbox = font.getbbox("Ag")
    line_height = bbox[3] - bbox[1]
    return (line_height + line_spacing) * line_count


def render_heatmap_image(matrix, color_fn, vmax, cell_size):
    rows, cols = matrix_dimensions(matrix)
    image = Image.new("RGB", (max(cols, 1) * cell_size, max(rows, 1) * cell_size), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    for row in range(rows):
        for col in range(cols):
            value = float(matrix[row][col])
            color = color_fn(value, vmax)
            x0 = col * cell_size
            y0 = row * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            draw.rectangle([x0, y0, x1, y1], fill=color)
            draw.rectangle([x0, y0, x1, y1], outline=(215, 215, 215), width=1)
    return image


def draw_multiline(draw, xy, text, font, fill, line_spacing=4):
    x, y = xy
    for line in text.splitlines():
        draw.text((x, y), line, font=font, fill=fill)
        bbox = font.getbbox(line if line else " ")
        y += (bbox[3] - bbox[1]) + line_spacing


def plot_single_snapshot(snapshot, output_path, tick_label_limit, dpi):
    del dpi
    matrix = [[float(value) for value in row] for row in snapshot["hessian_matrix"]]
    abs_matrix = build_abs_matrix(matrix)
    size, _ = matrix_dimensions(matrix)
    cell_size = choose_cell_size(size)

    raw_vmax = max_abs_value(matrix)
    abs_vmax = max_value(abs_matrix)
    if raw_vmax == 0.0:
        raw_vmax = 1.0
    if abs_vmax == 0.0:
        abs_vmax = 1.0

    raw_image = render_heatmap_image(matrix, signed_heat_color, raw_vmax, cell_size)
    abs_image = render_heatmap_image(abs_matrix, unsigned_heat_color, abs_vmax, cell_size)

    title_font = ImageFont.load_default()
    body_font = ImageFont.load_default()
    small_font = ImageFont.load_default()

    metrics = snapshot.get("metrics", {})
    header_lines = [
        "Epoch {} | Layer {} | size={}".format(
            snapshot.get("epoch", "?"),
            snapshot.get("layer", "?"),
            snapshot.get("sample_size", "?"),
        ),
        "diag/offdiag={} | diag_mass={}".format(
            shorten_float(metrics.get("diag_to_offdiag_norm_ratio", 0.0)),
            shorten_float(metrics.get("diag_mass_ratio", 0.0)),
        ),
        "mean|diag|={} | mean|offdiag|={} | symmetry_err={}".format(
            shorten_float(metrics.get("mean_abs_diag", 0.0)),
            shorten_float(metrics.get("mean_abs_off_diag", 0.0)),
            shorten_float(metrics.get("symmetry_max_abs_error", 0.0)),
        ),
    ]

    label_lines = []
    tick_labels = build_tick_labels(snapshot)
    if size <= tick_label_limit and tick_labels:
        label_lines.append("Row/Col order:")
        for index, label in enumerate(tick_labels):
            flat_label = label.replace("\n", " ")
            label_lines.append("{}: {}".format(index, flat_label))

    top_margin = 20
    side_margin = 20
    middle_gap = 24
    bottom_margin = 20
    caption_height = text_block_height(body_font, 1)
    header_height = text_block_height(body_font, len(header_lines))
    label_width = 0
    if label_lines:
        label_width = 260
    canvas_width = side_margin * 2 + raw_image.width + abs_image.width + middle_gap + label_width
    canvas_height = top_margin + header_height + 16 + max(raw_image.height + caption_height, abs_image.height + caption_height) + bottom_margin
    if label_lines:
        canvas_height = max(canvas_height, top_margin + header_height + 16 + text_block_height(small_font, len(label_lines)))

    canvas = Image.new("RGB", (canvas_width, canvas_height), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)

    draw_multiline(draw, (side_margin, top_margin), "\n".join(header_lines), title_font, (20, 20, 20))

    heatmap_top = top_margin + header_height + 16
    raw_x = side_margin
    abs_x = raw_x + raw_image.width + middle_gap
    canvas.paste(raw_image, (raw_x, heatmap_top))
    canvas.paste(abs_image, (abs_x, heatmap_top))

    draw.text((raw_x, heatmap_top - 14), "Raw Hessian", font=body_font, fill=(20, 20, 20))
    draw.text((abs_x, heatmap_top - 14), "Absolute Hessian", font=body_font, fill=(20, 20, 20))
    draw.text(
        (raw_x, heatmap_top + raw_image.height + 6),
        "range=[{}, {}]".format(shorten_float(-raw_vmax), shorten_float(raw_vmax)),
        font=small_font,
        fill=(70, 70, 70),
    )
    draw.text(
        (abs_x, heatmap_top + abs_image.height + 6),
        "range=[0, {}]".format(shorten_float(abs_vmax)),
        font=small_font,
        fill=(70, 70, 70),
    )

    if label_lines:
        label_x = abs_x + abs_image.width + 20
        draw_multiline(draw, (label_x, heatmap_top), "\n".join(label_lines), small_font, (30, 30, 30))

    canvas.save(output_path)


def plot_overview(snapshots, output_path, dpi):
    del dpi
    if not snapshots:
        return

    cols = min(3, len(snapshots))
    rows = int(math.ceil(float(len(snapshots)) / float(cols)))
    tile_size = 150
    tile_gap = 18
    header_height = 26
    outer_margin = 20

    width = outer_margin * 2 + cols * tile_size + (cols - 1) * tile_gap
    height = outer_margin * 2 + rows * (tile_size + header_height) + (rows - 1) * tile_gap + 24
    canvas = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((outer_margin, 10), "Hessian Snapshot Overview", font=font, fill=(20, 20, 20))

    for index, snapshot in enumerate(snapshots):
        row = index // cols
        col = index % cols
        matrix = [[float(value) for value in line] for line in snapshot["hessian_matrix"]]
        vmax = max_abs_value(matrix)
        if vmax == 0.0:
            vmax = 1.0

        mini = render_heatmap_image(matrix, signed_heat_color, vmax, max(4, tile_size // max(len(matrix), 1)))
        if mini.size != (tile_size, tile_size):
            mini = mini.resize((tile_size, tile_size), Image.NEAREST)

        x = outer_margin + col * (tile_size + tile_gap)
        y = outer_margin + 24 + row * (tile_size + header_height + tile_gap)
        canvas.paste(mini, (x, y))
        draw.rectangle([x, y, x + tile_size, y + tile_size], outline=(180, 180, 180), width=1)

        metrics = snapshot.get("metrics", {})
        title = "ep{} r={}".format(
            snapshot.get("epoch", "?"),
            shorten_float(metrics.get("diag_to_offdiag_norm_ratio", 0.0)),
        )
        draw.text((x, y + tile_size + 4), title, font=font, fill=(40, 40, 40))

    canvas.save(output_path)


def write_index_file(snapshots, output_path):
    lines = ["# Hessian Heatmap Index", ""]
    for index, snapshot in enumerate(snapshots):
        metrics = snapshot.get("metrics", {})
        filename = build_snapshot_filename(snapshot, index)
        lines.append(
            "- `{}`: epoch={}, layer={}, diag/offdiag={}, diag_mass={}".format(
                filename,
                snapshot.get("epoch", "?"),
                snapshot.get("layer", "?"),
                shorten_float(metrics.get("diag_to_offdiag_norm_ratio", 0.0)),
                shorten_float(metrics.get("diag_mass_ratio", 0.0)),
            )
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def build_snapshot_filename(snapshot, index):
    return "snapshot_{:02d}_epoch{:03d}_{}.png".format(
        index,
        int(snapshot.get("epoch", 0)),
        str(snapshot.get("layer", "unknown")).replace("/", "_"),
    )


def main():
    args = parse_arguments()
    payload = load_payload(args.results_json)
    snapshots = collect_hessian_snapshots(payload)

    if args.only_epoch is not None:
        snapshots = [snapshot for snapshot in snapshots if int(snapshot.get("epoch", -1)) == args.only_epoch]

    if args.max_snapshots > 0:
        snapshots = snapshots[: args.max_snapshots]

    if not snapshots:
        raise ValueError("No Hessian snapshots found in {}".format(args.results_json))

    output_dir = infer_output_dir(args.results_json, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for index, snapshot in enumerate(snapshots):
        filename = build_snapshot_filename(snapshot, index)
        output_path = os.path.join(output_dir, filename)
        plot_single_snapshot(snapshot, output_path, args.tick_label_limit, args.dpi)
        print("Saved {}".format(output_path))

    overview_path = os.path.join(output_dir, "overview.png")
    plot_overview(snapshots, overview_path, args.dpi)
    print("Saved {}".format(overview_path))

    index_path = os.path.join(output_dir, "index.md")
    write_index_file(snapshots, index_path)
    print("Saved {}".format(index_path))


if __name__ == "__main__":
    main()

"""Reproduce Fig. 5 from existing HDF5 result files.

This script reads ``configs/reproduce_fig5.yaml`` by default. It does not train
models, alter checkpoints, rewrite source HDF5 files, or depend on notebook
execution order.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "outputs" / ".matplotlib"))
os.environ.setdefault("GMT_USERDIR", str(REPO_ROOT / "outputs" / ".gmt"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / "outputs" / ".cache"))

import h5py
import matplotlib
import numpy as np
import pygmt

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib import colors as mcolors
from matplotlib.ticker import AutoMinorLocator, FuncFormatter
from PIL import Image


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required to read the YAML config.") from exc

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config file {path} does not contain a mapping.")
    return config


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def depth_axis(config: dict[str, Any]) -> np.ndarray:
    sampling = config["depth_sampling"]
    start = float(sampling["start"])
    step = float(sampling["step"])
    count = int(sampling["count"])
    return start + step * np.arange(count, dtype=np.float64)


def nearest_depth_indices(axis: np.ndarray, depths_km: list[float]) -> list[int]:
    return [int(np.argmin(np.abs(axis - depth))) for depth in depths_km]


def profile_array(values: np.ndarray, field_name: str) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2:
        raise ValueError(f"Expected {field_name} shape [N, 301] or [N, 301, 1], got {values.shape}.")
    return values


def coordinate_vector(values: np.ndarray, sample_count: int, field_name: str) -> np.ndarray:
    values = np.asarray(values).squeeze()
    if values.ndim == 2 and values.shape[0] == sample_count:
        values = values[:, 0]
    values = values.reshape(-1)
    if values.shape[0] != sample_count:
        raise ValueError(f"Expected {field_name} length {sample_count}, got {values.shape[0]}.")
    return values


def read_result(config: dict[str, Any]) -> dict[str, np.ndarray]:
    result_path = resolve_path(config["paths"]["primary_result"])
    fields = config["hdf5_fields"]["result"]
    with h5py.File(result_path, "r") as handle:
        vs = profile_array(handle[fields["vs"]][:], fields["vs"])
        prediction = profile_array(handle[fields["prediction"]][:], fields["prediction"])
        longitude = coordinate_vector(handle[fields["longitude"]][:], vs.shape[0], fields["longitude"])
        latitude = coordinate_vector(handle[fields["latitude"]][:], vs.shape[0], fields["latitude"])

    if prediction.shape != vs.shape:
        raise ValueError(f"Prediction shape {prediction.shape} does not match reference shape {vs.shape}.")
    return {
        "vs": vs,
        "prediction": prediction,
        "error": prediction - vs,
        "longitude": longitude,
        "latitude": latitude,
    }


def read_gmt_segments(paths: list[str]) -> list[np.ndarray]:
    segments: list[np.ndarray] = []
    for path_value in paths:
        path = resolve_path(path_value)
        if not path.exists():
            continue
        current: list[tuple[float, float]] = []
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith(">"):
                    if len(current) > 1:
                        segments.append(np.asarray(current, dtype=float))
                    current = []
                    continue
                parts = stripped.split()
                if len(parts) >= 2:
                    current.append((float(parts[0]), float(parts[1])))
        if len(current) > 1:
            segments.append(np.asarray(current, dtype=float))
    return segments


def add_boundaries(ax: plt.Axes, segments: list[np.ndarray], *, linewidth: float, alpha: float) -> None:
    for segment in segments:
        ax.plot(segment[:, 0], segment[:, 1], color="0.0", linewidth=linewidth, alpha=alpha)


def make_triangulation(longitude: np.ndarray, latitude: np.ndarray) -> mtri.Triangulation:
    triangulation = mtri.Triangulation(longitude, latitude)
    triangles = triangulation.triangles
    x = longitude[triangles]
    y = latitude[triangles]
    edge_lengths = np.stack(
        [
            np.hypot(x[:, 0] - x[:, 1], y[:, 0] - y[:, 1]),
            np.hypot(x[:, 1] - x[:, 2], y[:, 1] - y[:, 2]),
            np.hypot(x[:, 2] - x[:, 0], y[:, 2] - y[:, 0]),
        ],
        axis=1,
    )
    triangulation.set_mask(edge_lengths.max(axis=1) > 4.0)
    return triangulation


def font_sizes(config: dict[str, Any]) -> dict[str, int]:
    configured = config["figure"].get("font_sizes", {})
    return {
        "panel": int(configured.get("panel_label", 14)),
        "depth": int(configured.get("depth_label", 14)),
        "label": int(configured.get("axis_label", 16)),
        "tick": int(configured.get("tick_label", 11)),
        "cbar": int(configured.get("colorbar_tick", 11)),
        "column": int(configured.get("column_title", 12)),
    }


def longitude_label(value: float, _position: int) -> str:
    return f"{value:.0f}°E"


def latitude_label(value: float, _position: int) -> str:
    return f"{value:.0f}°N"


def resolve_colormap(name: str) -> mcolors.Colormap:
    if name == "paper_velocity":
        return mcolors.LinearSegmentedColormap.from_list(
            "paper_velocity",
            [
                (0.00, "#0017b8"),
                (0.18, "#0074ff"),
                (0.40, "#00e5ff"),
                (0.55, "#ffff33"),
                (0.76, "#ff9b00"),
                (1.00, "#ff0000"),
            ],
            N=256,
        )
    if name == "paper_difference":
        return mcolors.LinearSegmentedColormap.from_list(
            "paper_difference",
            [
                (0.00, "#0017d9"),
                (0.50, "#ffffff"),
                (1.00, "#ff0000"),
            ],
            N=256,
        )
    return plt.get_cmap(name)


def style_geo_axes(
    ax: plt.Axes,
    region: list[float] | None,
    sizes: dict[str, int],
    *,
    show_xlabels: bool,
    show_ylabels: bool,
) -> None:
    if region:
        ax.set_xlim(float(region[0]), float(region[1]))
        ax.set_ylim(float(region[2]), float(region[3]))
    ax.set_xticks([80, 100, 120, 140])
    ax.set_yticks([20, 40])
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_major_formatter(FuncFormatter(longitude_label))
    ax.yaxis.set_major_formatter(FuncFormatter(latitude_label))
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        top=True,
        right=True,
        labelsize=sizes["tick"],
        width=0.8,
        length=4,
        labelbottom=show_xlabels,
        labelleft=show_ylabels,
    )
    ax.tick_params(axis="both", which="minor", direction="out", top=True, right=True, width=0.7, length=2.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def draw_surface(
    ax: plt.Axes,
    triangulation: mtri.Triangulation,
    values: np.ndarray,
    *,
    cmap: mcolors.Colormap,
    norm: mcolors.Normalize,
):
    return ax.tripcolor(
        triangulation,
        values,
        shading="gouraud",
        cmap=cmap,
        norm=norm,
        edgecolors="none",
        linewidth=0.0,
        antialiased=False,
        rasterized=True,
    )


def add_horizontal_colorbar(fig: plt.Figure, cax: plt.Axes, mappable: Any, sizes: dict[str, int]) -> None:
    colorbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    colorbar.ax.tick_params(labelsize=sizes["cbar"], width=0.8, length=3, pad=2)
    colorbar.outline.set_linewidth(0.8)


def fixed_velocity_limits(config: dict[str, Any], indices: list[int]) -> list[list[float]]:
    limits = config["figure"].get("velocity_color_limits", [])
    if len(limits) != len(indices):
        raise ValueError(
            f"Expected {len(indices)} fixed velocity color limits for Fig.5, got {len(limits)}. "
            "Automatic color scaling is disabled for manuscript-style reproduction outputs."
        )
    return limits


def save_figure(fig: plt.Figure, png_path: Path, pdf_path: Path, dpi: int) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")


def format_depth_token(depth_km: float) -> str:
    return str(int(round(depth_km))) if abs(depth_km - round(depth_km)) < 1e-6 else f"{depth_km:g}"


def pattern_path(pattern: str, depth_km: float) -> Path:
    return resolve_path(pattern.format(depth_km=format_depth_token(depth_km)))


def configured_depth_indices(config: dict[str, Any], axis: np.ndarray, depths_km: list[float]) -> list[int]:
    indices = config["figure"].get("layer_depth_indices")
    if indices is not None:
        return [int(index) for index in indices]
    return nearest_depth_indices(axis, depths_km)


def layer_triplets(data: dict[str, np.ndarray], indices: list[int]) -> np.ndarray:
    longitude = data["longitude"]
    latitude = data["latitude"]
    layers: list[np.ndarray] = []
    for depth_index in indices:
        reference = data["vs"][:, depth_index]
        prediction = data["prediction"][:, depth_index]
        error = prediction - reference
        layers.append(np.vstack((longitude, latitude, reference)).T)
        layers.append(np.vstack((longitude, latitude, prediction)).T)
        layers.append(np.vstack((longitude, latitude, error)).T)
    return np.asarray(layers)


def save_labeled_pygmt_png(
    temp_png: Path,
    output_png: Path,
    output_pdf: Path,
    *,
    font_size: int,
    font_weight: str,
) -> None:
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "STHeiti", "Heiti TC", "Heiti SC"]
    plt.rcParams["axes.unicode_minus"] = False
    image = Image.open(temp_png)
    fig, ax = plt.subplots(figsize=(image.width / 100, image.height / 100), dpi=100)
    ax.imshow(image)
    ax.axis("off")
    fig.text(
        0.1,
        0.5,
        "Latitude（°）",
        va="center",
        ha="center",
        rotation=90,
        fontsize=font_size,
        weight=font_weight,
    )
    fig.text(
        0.55,
        0.09,
        "Longitude（°）",
        va="center",
        ha="center",
        fontsize=font_size,
        weight=font_weight,
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight", pad_inches=0.3)
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def plot_figure_with_original_pygmt(
    output_path: Path,
    output_pdf_path: Path,
    config: dict[str, Any],
    axis: np.ndarray,
    indices: list[int],
    data: dict[str, np.ndarray],
) -> None:
    figure_config = config["figure"]
    region = figure_config.get("map_region", [70, 150, 15, 55])
    layer_depths = [float(value) for value in figure_config["layer_depths_km"]]
    cpt_list = [
        "no_green",
        "no_green",
        "polar",
        "no_green",
        "no_green",
        "polar",
        "no_green",
        "no_green",
        "polar",
        "no_green",
        "no_green",
        "polar",
    ]
    velocity_limits = fixed_velocity_limits(config, indices)
    error_limit = figure_config.get("error_color_limit", [-0.2, 0.2])
    velocity_step = float(figure_config.get("velocity_color_step", 0.01))
    error_step = float(figure_config.get("error_color_step", 0.001))
    color_ranges = []
    for velocity_limit in velocity_limits:
        color_ranges.append([float(velocity_limit[0]), float(velocity_limit[1]), velocity_step])
        color_ranges.append([float(velocity_limit[0]), float(velocity_limit[1]), velocity_step])
        color_ranges.append([float(error_limit[0]), float(error_limit[1]), error_step])
    label_list = ["(a)", "(e)", "(i)", "(b)", "(f)", "(j)", "(c)", "(g)", "(k)", "(d)", "(h)", "(l)"]
    layer_data = layer_triplets(data, indices)
    cn_block = config["paths"].get("gmt_boundaries", [])
    projection = str(figure_config.get("projection", "M?"))
    grid_spacing = str(figure_config.get("grid_spacing", "1d"))
    search_radius = str(figure_config.get("search_radius", "2d"))
    width_cm, height_cm = figure_config.get("subplot_figsize_cm", [20, 22])
    font_size = int(figure_config.get("final_axis_label_font_size", 28))
    font_weight = str(figure_config.get("final_axis_label_weight", "bold"))
    temp_png = output_path.with_name(f"{output_path.stem}_temp.png")

    temp_png.parent.mkdir(parents=True, exist_ok=True)
    pygmt.config(FONT_TITLE="10p,5", MAP_TITLE_OFFSET="1p", MAP_FRAME_TYPE="plain")
    fig = pygmt.Figure()
    with fig.subplot(nrows=4, ncols=3, figsize=(f"{width_cm}c", f"{height_cm}c"), sharex="b", sharey="l"):
        for i in range(4):
            for j in range(3):
                panel_index = i * 3 + j
                with fig.set_panel(panel=panel_index):
                    fig.basemap(region=region, projection=projection, frame=["a20f10"])
                    pygmt.makecpt(
                        cmap=cpt_list[panel_index],
                        series=color_ranges[panel_index],
                        background=True,
                        continuous=True,
                    )
                    grid = pygmt.nearneighbor(
                        data=layer_data[panel_index],
                        spacing=grid_spacing,
                        search_radius=search_radius,
                        region=region,
                    )
                    fig.grdimage(grid=grid, projection=projection, region=region, cmap=True)
                    fig.coast(shorelines="1/0.5p", region=region, projection=projection)
                    for block in cn_block:
                        fig.plot(data=resolve_path(block), projection=projection, region=region, pen="0.5")
                    pygmt.config(FONT_ANNOT_PRIMARY="20p")
                    fig.colorbar(position="JBC", frame="x")
                    pygmt.config(FONT_ANNOT_PRIMARY="12p")
                    fig.text(
                        text=label_list[panel_index],
                        position="TL",
                        justify="TL",
                        offset="0.1/-0.1",
                        font="14p",
                    )
                    fig.text(
                        text=f"depth={layer_depths[i]:.0f}",
                        position="BR",
                        justify="BR",
                        offset="-0.1/0.1",
                        font="14p",
                    )
    fig.savefig(temp_png, dpi=300)
    save_labeled_pygmt_png(
        temp_png,
        output_path,
        output_pdf_path,
        font_size=font_size,
        font_weight=font_weight,
    )
    temp_png.unlink(missing_ok=True)


def write_metrics(
    output_path: Path,
    axis: np.ndarray,
    indices: list[int],
    data: dict[str, np.ndarray],
    display_depths: list[float] | None = None,
) -> None:
    error = data["error"]
    depth_labels = display_depths if display_depths is not None else [float(axis[index]) for index in indices]
    rows: list[dict[str, float | int | str]] = []
    depth_entries = [(f"{depth_km:.1f}_km", index, depth_km) for depth_km, index in zip(depth_labels, indices)]
    for label, selected, display_depth in [("all_depths", None, None)] + depth_entries:
        values = error if selected is None else error[:, selected]
        reference = data["vs"] if selected is None else data["vs"][:, selected]
        prediction = data["prediction"] if selected is None else data["prediction"][:, selected]
        rows.append(
            {
                "depth_label": label,
                "depth_km": "" if selected is None else float(display_depth),
                "depth_index": "" if selected is None else int(selected),
                "sample_count": int(error.shape[0]),
                "mean_reference_vs": float(np.mean(reference)),
                "mean_prediction_vs": float(np.mean(prediction)),
                "mean_error": float(np.mean(values)),
                "median_error": float(np.median(values)),
                "mean_absolute_error": float(np.mean(np.abs(values))),
                "rmse": float(np.sqrt(np.mean(values**2))),
                "std_error": float(np.std(values)),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_figure(
    output_path: Path,
    output_pdf_path: Path,
    config: dict[str, Any],
    axis: np.ndarray,
    indices: list[int],
    data: dict[str, np.ndarray],
    triangulation: mtri.Triangulation,
) -> None:
    figure_config = config["figure"]
    region = figure_config.get("map_region")
    velocity_limits = fixed_velocity_limits(config, indices)
    error_limit = figure_config.get("error_color_limit", [-0.4, 0.4])
    error_vmin, error_vmax = map(float, error_limit)
    if not np.isclose(abs(error_vmin), abs(error_vmax)) or not error_vmin < 0.0 < error_vmax:
        raise ValueError("error_color_limit must be symmetric around zero, for example [-0.4, 0.4].")
    velocity_cmap = resolve_colormap(str(figure_config.get("velocity_cmap", "paper_velocity")))
    error_cmap = resolve_colormap(str(figure_config.get("error_cmap", "paper_difference")))
    error_norm = mcolors.TwoSlopeNorm(vmin=error_vmin, vcenter=0.0, vmax=error_vmax)
    boundaries = read_gmt_segments(config["paths"].get("gmt_boundaries", []))
    boundary_line_width = float(figure_config.get("boundary_line_width", 0.7))
    boundary_alpha = float(figure_config.get("boundary_alpha", 0.95))

    columns = [
        ("Reference Vs", "vs", velocity_cmap),
        ("Inverted Vs", "prediction", velocity_cmap),
        ("Error / Difference", "error", error_cmap),
    ]
    panel_labels = ["(a)", "(e)", "(i)", "(b)", "(f)", "(j)", "(c)", "(g)", "(k)", "(d)", "(h)", "(l)"]

    sizes = font_sizes(config)
    fig = plt.figure(figsize=(13.2, 15.2))
    grid = fig.add_gridspec(
        nrows=len(indices) * 2,
        ncols=3,
        height_ratios=[1.0, 0.055] * len(indices),
        hspace=0.32,
        wspace=0.10,
    )
    for row, depth_index in enumerate(indices):
        depth_km = float(axis[depth_index])
        for col, (title, key, cmap) in enumerate(columns):
            ax = fig.add_subplot(grid[row * 2, col])
            colorbar_ax = fig.add_subplot(grid[row * 2 + 1, col])
            if row == 0:
                ax.set_title(title, fontsize=sizes["column"], pad=5)
            values = data[key][:, depth_index]
            if key == "error":
                norm = error_norm
            else:
                vmin, vmax = map(float, velocity_limits[row])
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

            surface = draw_surface(ax, triangulation, values, cmap=cmap, norm=norm)
            add_boundaries(ax, boundaries, linewidth=boundary_line_width, alpha=boundary_alpha)
            style_geo_axes(
                ax,
                region,
                sizes,
                show_xlabels=row == len(indices) - 1,
                show_ylabels=col == 0,
            )
            ax.text(
                0.02,
                0.96,
                panel_labels[row * 3 + col],
                transform=ax.transAxes,
                va="top",
                fontsize=sizes["panel"],
            )
            ax.text(
                0.98,
                0.02,
                f"depth={depth_km:.0f}",
                transform=ax.transAxes,
                va="bottom",
                ha="right",
                fontsize=sizes["depth"],
            )
            add_horizontal_colorbar(fig, colorbar_ax, surface, sizes)

    fig.text(0.018, 0.5, "Latitude ( ° )", va="center", rotation="vertical", fontsize=sizes["label"])
    fig.text(0.5, 0.018, "Longitude ( ° )", ha="center", fontsize=sizes["label"])
    fig.subplots_adjust(left=0.085, right=0.985, top=0.985, bottom=0.065)
    save_figure(fig, output_path, output_pdf_path, int(figure_config.get("dpi", 600)))
    plt.close(fig)


def plot_depth_figures(
    config: dict[str, Any],
    axis: np.ndarray,
    indices: list[int],
    data: dict[str, np.ndarray],
    triangulation: mtri.Triangulation,
) -> list[Path]:
    figure_config = config["figure"]
    region = figure_config.get("map_region")
    velocity_limits = fixed_velocity_limits(config, indices)
    error_limit = figure_config.get("error_color_limit", [-0.4, 0.4])
    error_vmin, error_vmax = map(float, error_limit)
    velocity_cmap = resolve_colormap(str(figure_config.get("velocity_cmap", "paper_velocity")))
    error_cmap = resolve_colormap(str(figure_config.get("error_cmap", "paper_difference")))
    error_norm = mcolors.TwoSlopeNorm(vmin=error_vmin, vcenter=0.0, vmax=error_vmax)
    boundaries = read_gmt_segments(config["paths"].get("gmt_boundaries", []))
    boundary_line_width = float(figure_config.get("boundary_line_width", 0.7))
    boundary_alpha = float(figure_config.get("boundary_alpha", 0.95))
    png_pattern = config["paths"]["output_depth_figure_pattern"]
    pdf_pattern = config["paths"]["output_depth_figure_pdf_pattern"]
    columns = [
        ("Reference Vs", "vs", velocity_cmap),
        ("Inverted Vs", "prediction", velocity_cmap),
        ("Error / Difference", "error", error_cmap),
    ]
    display_depths = [float(value) for value in figure_config.get("layer_depths_km", [])]
    sizes = font_sizes(config)
    outputs: list[Path] = []
    for row, depth_index in enumerate(indices):
        depth_km = display_depths[row] if row < len(display_depths) else float(axis[depth_index])
        fig = plt.figure(figsize=(13.2, 4.6))
        grid = fig.add_gridspec(nrows=2, ncols=3, height_ratios=[1.0, 0.07], hspace=0.28, wspace=0.10)
        for col, (title, key, cmap) in enumerate(columns):
            ax = fig.add_subplot(grid[0, col])
            colorbar_ax = fig.add_subplot(grid[1, col])
            ax.set_title(title, fontsize=sizes["column"], pad=5)
            values = data[key][:, depth_index]
            if key == "error":
                norm = error_norm
            else:
                vmin, vmax = map(float, velocity_limits[row])
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            surface = draw_surface(ax, triangulation, values, cmap=cmap, norm=norm)
            add_boundaries(ax, boundaries, linewidth=boundary_line_width, alpha=boundary_alpha)
            style_geo_axes(ax, region, sizes, show_xlabels=True, show_ylabels=col == 0)
            ax.text(
                0.98,
                0.02,
                f"depth={depth_km:.0f}",
                transform=ax.transAxes,
                va="bottom",
                ha="right",
                fontsize=sizes["depth"],
            )
            add_horizontal_colorbar(fig, colorbar_ax, surface, sizes)

        png_path = pattern_path(png_pattern, depth_km)
        pdf_path = pattern_path(pdf_pattern, depth_km)
        fig.text(0.018, 0.5, "Latitude ( ° )", va="center", rotation="vertical", fontsize=sizes["label"])
        fig.text(0.5, 0.018, "Longitude ( ° )", ha="center", fontsize=sizes["label"])
        fig.subplots_adjust(left=0.085, right=0.985, top=0.92, bottom=0.16)
        save_figure(fig, png_path, pdf_path, int(figure_config.get("dpi", 600)))
        plt.close(fig)
        outputs.extend([png_path, pdf_path])
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Fig. 5 from existing HDF5 results.")
    parser.add_argument("--config", default="configs/reproduce_fig5.yaml", help="Path to reproduce_fig5 YAML config.")
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    config = load_config(config_path)
    axis = depth_axis(config)
    requested_depths = [float(value) for value in config["figure"]["layer_depths_km"]]
    indices = configured_depth_indices(config, axis, requested_depths)

    data = read_result(config)
    triangulation = make_triangulation(data["longitude"], data["latitude"])
    output_figure = resolve_path(config["paths"]["output_figure"])
    output_figure_pdf = resolve_path(config["paths"]["output_figure_pdf"])
    output_table = resolve_path(config["paths"]["output_table"])
    plot_figure_with_original_pygmt(output_figure, output_figure_pdf, config, axis, indices, data)
    depth_outputs = plot_depth_figures(config, axis, indices, data, triangulation)
    write_metrics(output_table, axis, indices, data, requested_depths)

    print(f"Fig.5 figure: {output_figure}")
    print(f"Fig.5 PDF: {output_figure_pdf}")
    print(f"Fig.5 metrics: {output_table}")
    for path in depth_outputs:
        print(f"Fig.5 depth figure: {path}")


if __name__ == "__main__":
    main()

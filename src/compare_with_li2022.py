"""Compare this study's real-data inversion with Li et al. (2022).

The script reads existing HDF5 outputs only:

* this study: ``data/duo_real_result.h5::vs_pred``
* Li et al. (2022): ``data/Li_2022_16.h5::vs``

It validates that longitude, latitude, and depth sampling are shared, computes
slice and depth-wise difference metrics, and writes comparison figures under
``outputs/``. It does not train models or modify original notebooks, data,
checkpoints, logs, or published figure files.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "outputs" / ".matplotlib"))

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib import colors as mcolors
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter, FuncFormatter


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


def nearest_depth_indices(axis: np.ndarray, depths_km: list[float], *, tolerance_km: float) -> list[int]:
    indices: list[int] = []
    for depth in depths_km:
        index = int(np.argmin(np.abs(axis - depth)))
        if abs(float(axis[index]) - float(depth)) > tolerance_km:
            raise ValueError(f"Requested depth {depth:g} km is not present in the shared depth axis.")
        indices.append(index)
    return indices


def profile_array(values: np.ndarray, field_name: str) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2:
        raise ValueError(f"Expected {field_name} shape [N, D] or [N, D, 1], got {values.shape}.")
    return values.astype(np.float64, copy=False)


def coordinate_vector(values: np.ndarray, sample_count: int, field_name: str) -> np.ndarray:
    values = np.asarray(values).squeeze()
    if values.ndim == 2 and values.shape[0] == sample_count:
        values = values[:, 0]
    values = values.reshape(-1)
    if values.shape[0] != sample_count:
        raise ValueError(f"Expected {field_name} length {sample_count}, got {values.shape[0]}.")
    return values.astype(np.float64, copy=False)


def read_inputs(config: dict[str, Any], axis: np.ndarray) -> dict[str, np.ndarray]:
    paths = config["paths"]
    fields = config["hdf5_fields"]
    study_path = resolve_path(paths["this_study_result"])
    li_path = resolve_path(paths["li2022_reference_model"])

    with h5py.File(study_path, "r") as handle:
        study_fields = fields["this_study"]
        prediction = profile_array(handle[study_fields["prediction"]][:], study_fields["prediction"])
        study_lon = coordinate_vector(handle[study_fields["longitude"]][:], prediction.shape[0], study_fields["longitude"])
        study_lat = coordinate_vector(handle[study_fields["latitude"]][:], prediction.shape[0], study_fields["latitude"])

    with h5py.File(li_path, "r") as handle:
        li_fields = fields["li2022"]
        reference = profile_array(handle[li_fields["vs"]][:], li_fields["vs"])
        li_lon = coordinate_vector(handle[li_fields["longitude"]][:], reference.shape[0], li_fields["longitude"])
        li_lat = coordinate_vector(handle[li_fields["latitude"]][:], reference.shape[0], li_fields["latitude"])
        li_depth = profile_array(handle[li_fields["depth"]][:], li_fields["depth"])

    validate_shared_grid(config, axis, prediction, reference, study_lon, study_lat, li_lon, li_lat, li_depth)
    return {
        "reference": reference,
        "prediction": prediction,
        "difference": prediction - reference,
        "longitude": study_lon,
        "latitude": study_lat,
        "depth": axis,
    }


def validate_shared_grid(
    config: dict[str, Any],
    axis: np.ndarray,
    prediction: np.ndarray,
    reference: np.ndarray,
    study_lon: np.ndarray,
    study_lat: np.ndarray,
    li_lon: np.ndarray,
    li_lat: np.ndarray,
    li_depth: np.ndarray,
) -> None:
    if prediction.shape != reference.shape:
        raise ValueError(f"Prediction shape {prediction.shape} does not match Li reference shape {reference.shape}.")
    if prediction.shape[1] != axis.shape[0]:
        raise ValueError(f"Prediction depth count {prediction.shape[1]} does not match configured axis {axis.shape[0]}.")

    coordinate_atol = float(config["comparison"].get("coordinate_atol", 1.0e-10))
    if not np.allclose(study_lon, li_lon, rtol=0.0, atol=coordinate_atol):
        raise ValueError("Longitude arrays differ between this study and Li et al. (2022).")
    if not np.allclose(study_lat, li_lat, rtol=0.0, atol=coordinate_atol):
        raise ValueError("Latitude arrays differ between this study and Li et al. (2022).")

    depth_atol = float(config["comparison"].get("depth_atol_km", 1.0e-10))
    if li_depth.shape != reference.shape:
        raise ValueError(f"Li depth shape {li_depth.shape} does not match Li Vs shape {reference.shape}.")
    if not np.allclose(li_depth, axis[None, :], rtol=0.0, atol=depth_atol):
        raise ValueError("Li et al. (2022) depth sampling differs from the configured shared depth axis.")


def paired_finite(reference: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(reference) & np.isfinite(prediction)
    return reference[mask], prediction[mask]


def pearson_r(reference: np.ndarray, prediction: np.ndarray) -> float:
    reference, prediction = paired_finite(reference, prediction)
    if reference.size < 2:
        return float("nan")
    ref_std = float(np.std(reference))
    pred_std = float(np.std(prediction))
    if ref_std == 0.0 or pred_std == 0.0:
        return float("nan")
    return float(np.corrcoef(reference, prediction)[0, 1])


def metric_row(scope: str, depth_km: float, depth_index: int, reference: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    reference, prediction = paired_finite(reference, prediction)
    if reference.size == 0:
        rmse = mae = bias = corr = float("nan")
    else:
        error = prediction - reference
        rmse = float(np.sqrt(np.mean(error**2)))
        mae = float(np.mean(np.abs(error)))
        bias = float(np.mean(error))
        corr = pearson_r(reference, prediction)

    return {
        "metric_scope": scope,
        "depth_km": float(depth_km),
        "depth_index": int(depth_index),
        "sample_count": int(reference.size),
        "rmse_km_s": rmse,
        "mae_km_s": mae,
        "bias_km_s": bias,
        "pearson_r": corr,
    }


def compute_metrics(axis: np.ndarray, slice_indices: list[int], data: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    reference = data["reference"]
    prediction = data["prediction"]
    rows: list[dict[str, Any]] = []

    for depth_index in slice_indices:
        rows.append(
            metric_row(
                "requested_slice",
                float(axis[depth_index]),
                depth_index,
                reference[:, depth_index],
                prediction[:, depth_index],
            )
        )

    for depth_index, depth_km in enumerate(axis):
        rows.append(
            metric_row(
                "depthwise",
                float(depth_km),
                depth_index,
                reference[:, depth_index],
                prediction[:, depth_index],
            )
        )

    return rows


def write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["metric_scope", "depth_km", "depth_index", "sample_count", "rmse_km_s", "mae_km_s", "bias_km_s", "pearson_r"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def add_boundaries(ax: plt.Axes, segments: list[np.ndarray]) -> None:
    for segment in segments:
        ax.plot(segment[:, 0], segment[:, 1], color="0.0", linewidth=0.7, alpha=0.95)


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
    *,
    show_xlabels: bool,
    show_ylabels: bool,
    tick_label_size: int,
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
        labelsize=tick_label_size,
        width=0.8,
        length=4,
        labelbottom=show_xlabels,
        labelleft=show_ylabels,
    )
    ax.tick_params(axis="both", which="minor", direction="out", top=True, right=True, width=0.7, length=2.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def slice_style(config: dict[str, Any]) -> dict[str, float]:
    configured = config["figure"].get("slice_style", {})
    return {
        "column_title": int(configured.get("column_title_font_size", 15)),
        "panel": int(configured.get("panel_label_font_size", 18)),
        "depth": int(configured.get("depth_label_font_size", 18)),
        "tick": int(configured.get("tick_label_font_size", 18)),
        "colorbar_tick": int(configured.get("colorbar_tick_label_font_size", 18)),
        "global_axis": int(configured.get("global_axis_label_font_size", 28)),
        "hspace": float(configured.get("subplot_hspace", 0.32)),
        "wspace": float(configured.get("subplot_wspace", 0.10)),
        "left": float(configured.get("subplot_left", 0.085)),
        "right": float(configured.get("subplot_right", 0.985)),
        "top": float(configured.get("subplot_top", 0.985)),
        "bottom": float(configured.get("subplot_bottom", 0.065)),
        "global_y_x": float(configured.get("global_y_label_x", 0.018)),
        "global_x_y": float(configured.get("global_x_label_y", 0.018)),
    }


def configured_colorbar_ticks(figure_config: dict[str, Any], key: str, row: int) -> list[float] | None:
    tick_config = figure_config.get("colorbar_ticks", {})
    if key == "difference":
        ticks = tick_config.get("difference")
    else:
        velocity_ticks = tick_config.get("velocity", [])
        ticks = velocity_ticks[row] if row < len(velocity_ticks) else None
    if ticks is None:
        return None
    return [float(value) for value in ticks]


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


def save_figure(fig: plt.Figure, png_path: Path, pdf_path: Path, dpi: int) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")


def plot_depthwise_error(png_path: Path, pdf_path: Path, axis: np.ndarray, rows: list[dict[str, Any]], dpi: int) -> None:
    depth_rows = [row for row in rows if row["metric_scope"] == "depthwise"]
    depth = np.asarray([row["depth_km"] for row in depth_rows], dtype=float)
    rmse = np.asarray([row["rmse_km_s"] for row in depth_rows], dtype=float)
    mae = np.asarray([row["mae_km_s"] for row in depth_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 8.5), constrained_layout=True)
    ax.plot(rmse, depth, color="#1f77b4", linewidth=1.6, label="RMSE")
    ax.plot(mae, depth, color="#ff7f0e", linewidth=1.6, label="MAE")
    ax.set_ylim(float(np.nanmax(depth)), float(np.nanmin(depth)))
    ax.set_xlabel("Velocity difference metric (km/s)", fontsize=12)
    ax.set_ylabel("Depth (km)", fontsize=12)
    ax.set_title("Depth-wise difference relative to the Li et al. (2022) reference model", fontsize=14, pad=10)
    ax.grid(True, color="0.85", linewidth=0.7)
    ax.legend(frameon=False, fontsize=11)
    ax.tick_params(axis="both", labelsize=11)
    save_figure(fig, png_path, pdf_path, dpi)
    plt.close(fig)


def plot_slice_comparison(
    png_path: Path,
    pdf_path: Path,
    config: dict[str, Any],
    axis: np.ndarray,
    slice_indices: list[int],
    data: dict[str, np.ndarray],
    triangulation: mtri.Triangulation,
) -> dict[str, Any]:
    figure_config = config["figure"]
    region = figure_config.get("map_region")
    velocity_limits = figure_config.get("velocity_color_limits", [])
    difference_limit = figure_config.get("difference_color_limit", [-0.4, 0.4])
    if len(velocity_limits) != len(slice_indices):
        raise ValueError(
            f"Expected {len(slice_indices)} fixed velocity color limits, got {len(velocity_limits)}. "
            "Automatic color scaling is disabled for manuscript-style comparison figures."
        )
    velocity_cmap_name = str(figure_config.get("velocity_cmap", "paper_velocity"))
    difference_cmap_name = str(figure_config.get("difference_cmap", "paper_difference"))
    velocity_cmap = resolve_colormap(velocity_cmap_name)
    difference_cmap = resolve_colormap(difference_cmap_name)
    difference_vmin, difference_vmax = map(float, difference_limit)
    if not np.isclose(abs(difference_vmin), abs(difference_vmax)) or not difference_vmin < 0.0 < difference_vmax:
        raise ValueError("difference_color_limit must be symmetric around zero, for example [-0.4, 0.4].")
    difference_norm = mcolors.TwoSlopeNorm(vmin=difference_vmin, vcenter=0.0, vmax=difference_vmax)
    boundaries = read_gmt_segments(config["paths"].get("gmt_boundaries", []))
    sizes = slice_style(config)
    columns = [
        ("Li et al. (2022) reference model", "reference", velocity_cmap),
        ("This study", "prediction", velocity_cmap),
        ("Difference", "difference", difference_cmap),
    ]
    panel_labels = ["(a)", "(e)", "(i)", "(b)", "(f)", "(j)", "(c)", "(g)", "(k)", "(d)", "(h)", "(l)"]

    fig = plt.figure(figsize=(13.2, 15.2))
    grid = fig.add_gridspec(
        nrows=len(slice_indices) * 2,
        ncols=3,
        height_ratios=[1.0, 0.055] * len(slice_indices),
        hspace=sizes["hspace"],
        wspace=sizes["wspace"],
    )
    for row, depth_index in enumerate(slice_indices):
        depth_km = float(axis[depth_index])
        for col, (title, key, cmap) in enumerate(columns):
            ax = fig.add_subplot(grid[row * 2, col])
            colorbar_ax = fig.add_subplot(grid[row * 2 + 1, col])
            if row == 0:
                ax.set_title(title, fontsize=sizes["column_title"], pad=6)
            values = data[key][:, depth_index]
            if key == "difference":
                norm = difference_norm
            else:
                vmin, vmax = map(float, velocity_limits[row])
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

            surface = draw_surface(ax, triangulation, values, cmap=cmap, norm=norm)
            add_boundaries(ax, boundaries)
            style_geo_axes(
                ax,
                region,
                show_xlabels=row == len(slice_indices) - 1,
                show_ylabels=col == 0,
                tick_label_size=sizes["tick"],
            )
            ax.text(
                0.02,
                0.96,
                panel_labels[row * 3 + col],
                transform=ax.transAxes,
                va="top",
                ha="left",
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
            colorbar = fig.colorbar(surface, cax=colorbar_ax, orientation="horizontal")
            ticks = configured_colorbar_ticks(figure_config, key, row)
            if ticks is not None:
                colorbar.set_ticks(ticks)
                colorbar.ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            colorbar.ax.tick_params(labelsize=sizes["colorbar_tick"], width=0.8, length=3, pad=2)
            colorbar.outline.set_linewidth(0.8)

    fig.text(sizes["global_y_x"], 0.5, "Latitude (°)", va="center", rotation="vertical", fontsize=sizes["global_axis"])
    fig.text(0.5, sizes["global_x_y"], "Longitude (°)", ha="center", fontsize=sizes["global_axis"])
    fig.subplots_adjust(
        left=sizes["left"],
        right=sizes["right"],
        top=sizes["top"],
        bottom=sizes["bottom"],
    )

    save_figure(fig, png_path, pdf_path, int(figure_config.get("dpi", 300)))
    plt.close(fig)
    return {
        "velocity_cmap": velocity_cmap.name,
        "difference_cmap": difference_cmap.name,
        "velocity_limits": [
            (float(axis[index]), float(limit[0]), float(limit[1]))
            for index, limit in zip(slice_indices, velocity_limits, strict=True)
        ],
        "difference_limit": (difference_vmin, difference_vmax),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare this study with the Li et al. (2022) reference model.")
    parser.add_argument("--config", default="configs/compare_li2022.yaml", help="Path to compare_li2022 YAML config.")
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    config = load_config(config_path)
    axis = depth_axis(config)
    requested_depths = [float(value) for value in config["comparison"]["slice_depths_km"]]
    slice_indices = nearest_depth_indices(
        axis,
        requested_depths,
        tolerance_km=float(config["comparison"].get("depth_atol_km", 1.0e-10)),
    )

    data = read_inputs(config, axis)
    rows = compute_metrics(axis, slice_indices, data)
    output_table = resolve_path(config["paths"]["output_table"])
    depthwise_png = resolve_path(config["paths"]["output_depthwise_figure"])
    depthwise_pdf = resolve_path(config["paths"]["output_depthwise_figure_pdf"])
    slice_png = resolve_path(config["paths"]["output_slice_comparison"])
    slice_pdf = resolve_path(config["paths"]["output_slice_comparison_pdf"])

    write_metrics(output_table, rows)
    plot_depthwise_error(depthwise_png, depthwise_pdf, axis, rows, int(config["figure"].get("dpi", 300)))
    triangulation = make_triangulation(data["longitude"], data["latitude"])
    style_report = plot_slice_comparison(slice_png, slice_pdf, config, axis, slice_indices, data, triangulation)

    print(f"Li2022 comparison metrics: {output_table}")
    print(f"Li2022 depth-wise difference PNG: {depthwise_png}")
    print(f"Li2022 depth-wise difference PDF: {depthwise_pdf}")
    print(f"Li2022 slice comparison PNG: {slice_png}")
    print(f"Li2022 slice comparison PDF: {slice_pdf}")
    print(f"Li2022 slice velocity cmap: {style_report['velocity_cmap']}")
    print(f"Li2022 slice difference cmap: {style_report['difference_cmap']}")
    for depth_km, vmin, vmax in style_report["velocity_limits"]:
        print(f"Li2022 slice velocity range at {depth_km:.0f} km: vmin={vmin:g}, vmax={vmax:g}")
    difference_vmin, difference_vmax = style_report["difference_limit"]
    print(f"Li2022 slice difference range: vmin={difference_vmin:g}, vmax={difference_vmax:g}")


if __name__ == "__main__":
    main()

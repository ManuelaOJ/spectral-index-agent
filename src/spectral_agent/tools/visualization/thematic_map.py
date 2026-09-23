"""
Static thematic-map generator for spectral index rasters.

Produces a publication-quality PNG map using matplotlib, with:

* Colour-mapped index raster
* Colourbar with range labels
* Title and subtitle
* Scale bar (approximate, based on pixel resolution)
* North arrow
* Coordinate grid (optional)

The module is intentionally kept free of external GIS viewers so it works
in headless / CI environments.
"""

from __future__ import annotations

import io
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless — must come before pyplot import

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import rasterio
from matplotlib.colors import Normalize
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Default colour-map per index category
# ─────────────────────────────────────────────────────────────────────────────

INDEX_CMAPS: dict[str, str] = {
    "NDVI": "RdYlGn",  # red (bare) → yellow → green (vegetated)
    "EVI": "YlGn",  # yellow (low) → dark green (high biomass)
    "SAVI": "BrBG",  # brown (bare soil) → white → blue-green (vegetated)
    "NDWI": "RdYlBu",  # red (dry) → yellow → blue (water)
    "NBR": "PiYG_r",  # green (unburned) → pink (burned)
    "NDBI": "YlOrRd",  # yellow (low) → orange → red (built-up)
}

INDEX_FULL_NAMES: dict[str, str] = {
    "NDVI": "Normalized Difference Vegetation Index",
    "EVI": "Enhanced Vegetation Index",
    "SAVI": "Soil Adjusted Vegetation Index",
    "NDWI": "Normalized Difference Water Index",
    "NBR": "Normalized Burn Ratio",
    "NDBI": "Normalized Difference Built-up Index",
}


# ─────────────────────────────────────────────────────────────────────────────
# Configuration dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MapConfig:
    """Configuration for static thematic map rendering."""

    title: str | None = None
    subtitle: str | None = None
    cmap: str | None = None  # None → auto from INDEX_CMAPS
    vmin: float | None = None  # None → auto from data
    vmax: float | None = None
    figsize: tuple[float, float] = (14, 10)
    dpi: int = 150
    add_colorbar: bool = True
    add_scale_bar: bool = True
    add_north_arrow: bool = True
    add_grid: bool = True
    nodata_color: str = "#d9d9d9"  # light grey for NaN pixels
    background_color: str = "#f0f0f0"
    grid_label_size: float = 7.5
    frame_linewidth: float = 1.2
    # ── Legend sidebar metadata ──────────────────────────────────────────
    scene_id: str | None = None
    acquisition_date: str | None = None
    satellite: str | None = None
    language: str = "en"  # "en" or "es"


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MapResult:
    """Metadata returned after generating a map."""

    output_path: Path
    index_name: str
    width_px: int
    height_px: int
    dpi: int
    crs: str
    bounds: list[float]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def generate_thematic_map(
    raster_path: Path,
    index_name: str,
    output_path: Path,
    *,
    config: MapConfig | None = None,
) -> MapResult:
    """
    Generate a static thematic map (PNG) from an index GeoTIFF.

    Parameters
    ----------
    raster_path : Path
        Single-band Float32 GeoTIFF (output of ``index_calculator``).
    index_name : str
        Spectral index name (``NDVI``, ``EVI``, …).
    output_path : Path
        Destination ``.png`` file.
    config : MapConfig | None
        Rendering options; uses sensible defaults when ``None``.

    Returns
    -------
    MapResult
        Metadata about the generated image.
    """
    raster_path = Path(raster_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = MapConfig()

    name = index_name.upper()

    # ── Read raster ─────────────────────────────────────────────────────
    with rasterio.open(raster_path) as src:
        data = src.read(1).astype(np.float32)
        transform = src.transform
        crs = str(src.crs)
        bounds = list(src.bounds)
        res_x, res_y = src.res
        epsg_code = src.crs.to_epsg() if src.crs else None
        is_projected = src.crs.is_projected if src.crs else False

    # ── Resolve display params ──────────────────────────────────────────
    cmap_name = config.cmap or INDEX_CMAPS.get(name, "viridis")
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color=config.nodata_color)

    valid = data[~np.isnan(data)]
    vmin = (
        config.vmin
        if config.vmin is not None
        else (float(valid.min()) if valid.size else -1.0)
    )
    vmax = (
        config.vmax
        if config.vmax is not None
        else (float(valid.max()) if valid.size else 1.0)
    )

    title = config.title or f"{name} — {INDEX_FULL_NAMES.get(name, index_name)}"
    subtitle = config.subtitle or ""

    # ── Build extent from transform ─────────────────────────────────────
    rows, cols = data.shape
    left = transform.c
    top = transform.f
    right = left + cols * transform.a
    bottom = top + rows * transform.e  # transform.e is negative
    extent = [left, right, bottom, top]

    # ── Determine coordinate type ───────────────────────────────────────
    coord_type = "Projected (m)" if is_projected else "Geographic (°)"

    # ── Plot: map + legend side panel ───────────────────────────────────
    fig = plt.figure(figsize=config.figsize)
    fig.patch.set_facecolor(config.background_color)

    # gridspec: 80% map, 20% legend panel
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    ax_legend = fig.add_subplot(gs[0, 1])

    ax.set_facecolor(config.nodata_color)

    masked = np.ma.masked_invalid(data)
    im = ax.imshow(
        masked,
        cmap=cmap,
        norm=Normalize(vmin=vmin, vmax=vmax),
        extent=extent,
        interpolation="nearest",
    )

    # Title on top of the map axes
    ax.set_title(title, fontsize=12, fontweight="bold", pad=14)
    if subtitle:
        ax.text(
            0.5,
            1.01,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#555555",
        )

    # ── Neatline frame ──────────────────────────────────────────────────
    for spine in ax.spines.values():
        spine.set_linewidth(config.frame_linewidth)
        spine.set_edgecolor("black")
        spine.set_visible(True)

    # ── Coordinate grid & tick labels ───────────────────────────────────
    if config.add_grid:
        _setup_coordinate_grid(ax, extent, config, is_projected=is_projected)
    else:
        lbl_e = "Easting (m)" if is_projected else "Longitude (°)"
        lbl_n = "Northing (m)" if is_projected else "Latitude (°)"
        ax.set_xlabel(lbl_e, fontsize=8)
        ax.set_ylabel(lbl_n, fontsize=8)
        ax.tick_params(labelsize=7)

    # ── Scale bar ───────────────────────────────────────────────────────
    if config.add_scale_bar:
        center_lat = (top + bottom) / 2.0
        _add_scale_bar(
            ax,
            res_x,
            cols,
            is_projected=is_projected,
            center_lat=center_lat,
        )

    # ── North arrow ─────────────────────────────────────────────────────
    if config.add_north_arrow:
        _add_north_arrow(ax)

    # ── Legend sidebar ──────────────────────────────────────────────────
    _build_legend_sidebar(
        fig=fig,
        ax_legend=ax_legend,
        im=im,
        index_name=name,
        title=title,
        vmin=vmin,
        vmax=vmax,
        scene_id=config.scene_id,
        acquisition_date=config.acquisition_date,
        satellite=config.satellite,
        coord_type=coord_type,
        epsg_code=epsg_code,
        language=config.language,
    )

    # ── Save ────────────────────────────────────────────────────────────
    fig.savefig(
        output_path,
        dpi=config.dpi,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)

    logger.info("Thematic map saved → %s (%d dpi)", output_path.name, config.dpi)

    return MapResult(
        output_path=output_path,
        index_name=name,
        width_px=cols,
        height_px=rows,
        dpi=config.dpi,
        crs=crs,
        bounds=bounds,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────


def _nice_grid_interval(span: float, target_ticks: int = 5) -> float:
    """Return a 'nice' grid interval for *span* aiming at ~*target_ticks*."""
    import math

    raw = span / target_ticks
    if raw <= 0:
        return 1.0
    exponent = math.floor(math.log10(raw))
    base = 10**exponent
    mantissa = raw / base
    if mantissa < 1.5:
        nice = base
    elif mantissa < 3.5:
        nice = 2 * base
    elif mantissa < 7.5:
        nice = 5 * base
    else:
        nice = 10 * base
    return nice


def _setup_coordinate_grid(
    ax,
    extent: list[float],
    config,
    *,
    is_projected: bool = True,
) -> None:
    """Configure a coordinate grid with labels on all four sides.

    * Bottom / Top   → horizontal labels (Easting / Longitude)
    * Left (ascending 90°) / Right (descending 270°) → Northing / Latitude
    """
    import matplotlib.ticker as mticker
    import math as _m

    left, right, bottom, top = extent
    dx = right - left
    dy = top - bottom

    ix = _nice_grid_interval(dx, target_ticks=4)
    iy = _nice_grid_interval(dy, target_ticks=4)

    x_start = _m.ceil(left / ix) * ix
    x_ticks = np.arange(x_start, right, ix)

    y_start = _m.ceil(bottom / iy) * iy
    y_ticks = np.arange(y_start, top, iy)

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    lbl_size = config.grid_label_size

    # Enable tick marks and labels on all four sides
    ax.tick_params(
        axis="x",
        which="both",
        bottom=True,
        top=True,
        labelbottom=True,
        labeltop=True,
        labelsize=lbl_size,
        direction="in",
        length=3,
        width=0.6,
        pad=3,
    )
    ax.tick_params(
        axis="y",
        which="both",
        left=True,
        right=True,
        labelleft=True,
        labelright=True,
        labelsize=lbl_size,
        direction="in",
        length=3,
        width=0.6,
        pad=3,
    )

    # CRS-aware formatter
    formatter = mticker.FuncFormatter(
        _projected_formatter if is_projected else _geographic_formatter
    )
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    # Rotate y-labels: left ascending (90°), right descending (270°)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_rotation(90)
        tick.label1.set_verticalalignment("center")
        tick.label2.set_rotation(270)
        tick.label2.set_verticalalignment("center")

    # Grid lines — subtle, semi-transparent
    ax.grid(
        True,
        which="major",
        color="#888888",
        linewidth=0.3,
        alpha=0.45,
        linestyle="--",
    )

    ax.set_xlabel("")
    ax.set_ylabel("")


def _projected_formatter(value: float, _pos) -> str:
    """Format projected coordinate values (metres) without scientific notation."""
    if abs(value) >= 1e6:
        return f"{value / 1e6:.2f}M"
    if abs(value) >= 1e3:
        return f"{value:,.0f}"
    return f"{value:.1f}"


def _geographic_formatter(value: float, _pos) -> str:
    """Format geographic coordinate values (degrees)."""
    return f"{value:.4f}°"


# ─────────────────────────────────────────────────────────────────────────────
# Legend sidebar (replaces the old framed-colorbar approach)
# ─────────────────────────────────────────────────────────────────────────────

_LEGEND_LABELS: dict[str, dict[str, str]] = {
    "en": {
        "header": "Legend",
        "scene_header": "Scene Information",
        "satellite": "Satellite",
        "scene": "Scene ID",
        "date": "Acq. Date",
        "spatial_header": "Spatial Reference",
        "coords": "Coordinates",
        "crs": "Reference",
    },
    "es": {
        "header": "Leyenda",
        "scene_header": "Info. de Escena",
        "satellite": "Satélite",
        "scene": "ID Escena",
        "date": "Fecha Acq.",
        "spatial_header": "Referencia Espacial",
        "coords": "Coordenadas",
        "crs": "Referencia",
    },
}


def _build_legend_sidebar(
    *,
    fig,
    ax_legend,
    im,
    index_name: str,
    title: str,
    vmin: float,
    vmax: float,
    scene_id: str | None,
    acquisition_date: str | None,
    satellite: str | None,
    coord_type: str,
    epsg_code: int | None,
    language: str = "en",
) -> None:
    """Populate the right-side legend panel with colorbar + metadata.

    Draws a card-style container with three visually separated sections:
    colorbar, scene information, and spatial reference.
    """
    from matplotlib.patches import FancyBboxPatch

    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    ax_legend.axis("off")

    labels = _LEGEND_LABELS.get(language, _LEGEND_LABELS["en"])

    # ── Card background ─────────────────────────────────────────────────
    card = FancyBboxPatch(
        (0.03, 0.03),
        0.94,
        0.94,
        boxstyle="round,pad=0.02",
        facecolor="#fafafa",
        edgecolor="#bbbbbb",
        linewidth=0.8,
        transform=ax_legend.transAxes,
        zorder=0,
    )
    ax_legend.add_patch(card)

    # ── Header ──────────────────────────────────────────────────────────
    y = 0.93
    ax_legend.text(
        0.50,
        y,
        labels["header"],
        transform=ax_legend.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#222222",
    )
    y -= 0.035
    ax_legend.plot(
        [0.10, 0.90],
        [y, y],
        transform=ax_legend.transAxes,
        color="#cccccc",
        linewidth=0.6,
        clip_on=False,
    )
    y -= 0.015

    # ── Colorbar ────────────────────────────────────────────────────────
    cb_height = 0.34
    cb_left = 0.18
    cb_width = 0.20
    cb_bottom = y - cb_height
    cax = ax_legend.inset_axes([cb_left, cb_bottom, cb_width, cb_height])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(index_name, fontsize=9, fontweight="bold", labelpad=10)
    cbar.ax.tick_params(labelsize=7, direction="in", length=2)
    cbar.outline.set_linewidth(0.8)
    cbar.outline.set_edgecolor("#555555")

    # Min / Max annotations beside the colorbar
    cbar.ax.text(
        1.35,
        0.0,
        f"Min {vmin:.3f}",
        transform=cbar.ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        color="#444444",
    )
    cbar.ax.text(
        1.35,
        1.0,
        f"Max {vmax:.3f}",
        transform=cbar.ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color="#444444",
    )

    y = cb_bottom - 0.02

    # ── Scene Information section ───────────────────────────────────────
    if scene_id or acquisition_date or satellite:
        ax_legend.plot(
            [0.10, 0.90],
            [y, y],
            transform=ax_legend.transAxes,
            color="#cccccc",
            linewidth=0.6,
            clip_on=False,
        )
        y -= 0.025
        ax_legend.text(
            0.10,
            y,
            labels["scene_header"],
            transform=ax_legend.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            fontweight="bold",
            color="#333333",
        )
        y -= 0.04

        if satellite:
            ax_legend.text(
                0.10,
                y,
                f"{labels['satellite']}:",
                transform=ax_legend.transAxes,
                ha="left",
                va="top",
                fontsize=6.5,
                fontweight="semibold",
                color="#555555",
            )
            y -= 0.025
            ax_legend.text(
                0.10,
                y,
                satellite,
                transform=ax_legend.transAxes,
                ha="left",
                va="top",
                fontsize=7.5,
                color="#333333",
            )
            y -= 0.03

        if scene_id:
            import textwrap

            ax_legend.text(
                0.10,
                y,
                f"{labels['scene']}:",
                transform=ax_legend.transAxes,
                ha="left",
                va="top",
                fontsize=6.5,
                fontweight="semibold",
                color="#555555",
            )
            y -= 0.025
            # Wrap long scene IDs so they stay inside the card
            wrapped = textwrap.fill(scene_id, width=22)
            n_lines = wrapped.count("\n") + 1
            ax_legend.text(
                0.10,
                y,
                wrapped,
                transform=ax_legend.transAxes,
                ha="left",
                va="top",
                fontsize=5.5,
                color="#333333",
                family="monospace",
                linespacing=1.2,
            )
            y -= 0.022 * n_lines + 0.01

        if acquisition_date:
            ax_legend.text(
                0.10,
                y,
                f"{labels['date']}:",
                transform=ax_legend.transAxes,
                ha="left",
                va="top",
                fontsize=6.5,
                fontweight="semibold",
                color="#555555",
            )
            y -= 0.025
            ax_legend.text(
                0.10,
                y,
                acquisition_date,
                transform=ax_legend.transAxes,
                ha="left",
                va="top",
                fontsize=7.5,
                color="#333333",
            )
            y -= 0.03

    # ── Spatial Reference section ───────────────────────────────────────
    ax_legend.plot(
        [0.10, 0.90],
        [y, y],
        transform=ax_legend.transAxes,
        color="#cccccc",
        linewidth=0.6,
        clip_on=False,
    )
    y -= 0.025
    ax_legend.text(
        0.10,
        y,
        labels["spatial_header"],
        transform=ax_legend.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
        color="#333333",
    )
    y -= 0.04

    ax_legend.text(
        0.10,
        y,
        f"{labels['coords']}:",
        transform=ax_legend.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        fontweight="semibold",
        color="#555555",
    )
    y -= 0.025
    ax_legend.text(
        0.10,
        y,
        coord_type,
        transform=ax_legend.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color="#333333",
    )
    y -= 0.035

    if epsg_code:
        ax_legend.text(
            0.10,
            y,
            f"{labels['crs']}:",
            transform=ax_legend.transAxes,
            ha="left",
            va="top",
            fontsize=6.5,
            fontweight="semibold",
            color="#555555",
        )
        y -= 0.03
        ax_legend.text(
            0.10,
            y,
            f"EPSG:{epsg_code}",
            transform=ax_legend.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            color="#333333",
            family="monospace",
        )


def _add_framed_colorbar(
    fig,
    ax,
    im,
    index_name: str,
    vmin: float = 0,
    vmax: float = 1,
) -> None:
    """Add a colour-bar legend on the right with a visible frame and min/max.

    .. deprecated:: Kept for backward compatibility; new code uses
       ``_build_legend_sidebar`` which embeds the colorbar in a side panel.
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.35)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(index_name, fontsize=8, labelpad=6)
    cbar.ax.tick_params(labelsize=6.5, direction="in", length=2)

    # Frame around the colour-bar
    cbar.outline.set_linewidth(0.8)
    cbar.outline.set_edgecolor("black")

    # Add explicit min / max annotations at the ends of the colour-bar
    cbar.ax.text(
        1.15,
        0.0,
        f"Min: {vmin:.3f}",
        transform=cbar.ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.5,
        fontstyle="italic",
    )
    cbar.ax.text(
        1.15,
        1.0,
        f"Max: {vmax:.3f}",
        transform=cbar.ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        fontstyle="italic",
    )


def _add_scale_bar(
    ax,
    pixel_res: float,
    ncols: int,
    n_divs: int = 2,
    *,
    is_projected: bool = True,
    center_lat: float = 0.0,
) -> None:
    """
    Draw a box-style scale bar with alternating black/white divisions.

    The bar is placed in the lower-left corner.  Each division is labelled
    with the distance it represents, and the unit (km or m) is shown to
    the right of the bar.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    pixel_res : float
        Pixel size in CRS units (metres for UTM, degrees for geographic).
    ncols : int
        Number of columns in the raster.
    n_divs : int
        Number of alternating divisions (default 2).
    is_projected : bool
        True when CRS units are metres; False for geographic (degrees).
    center_lat : float
        Approximate centre latitude — used only for geographic CRS to
        convert degrees to metres.
    """
    from matplotlib.patches import Rectangle
    import math as _m

    # ── Determine map width in metres ───────────────────────────────────
    if is_projected:
        map_width_m = pixel_res * ncols
    else:
        m_per_deg = 111_320 * _m.cos(_m.radians(abs(center_lat)))
        map_width_m = pixel_res * ncols * m_per_deg

    raw_len = map_width_m * 0.20
    nice_m = _nice_round(raw_len)  # nice distance in metres

    # ── Unit handling ───────────────────────────────────────────────────
    if nice_m >= 1000:
        unit = "km"
        divisor = 1000.0
    else:
        unit = "m"
        divisor = 1.0

    # ── Convert back to CRS units for drawing on axes ───────────────────
    if is_projected:
        bar_total_crs = nice_m
    else:
        m_per_deg = 111_320 * _m.cos(_m.radians(abs(center_lat)))
        bar_total_crs = nice_m / m_per_deg if m_per_deg > 0 else nice_m

    # ── Position (lower-left) ───────────────────────────────────────────
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    dx = xlim[1] - xlim[0]
    dy = ylim[1] - ylim[0]

    x0 = xlim[0] + dx * 0.05
    y0 = ylim[0] + dy * 0.045
    bar_height = dy * 0.012
    div_width = bar_total_crs / n_divs

    # ── Draw alternating black / white rectangles ───────────────────────
    colors = ["#000000", "#ffffff"]
    for i in range(n_divs):
        rect = Rectangle(
            (x0 + i * div_width, y0),
            div_width,
            bar_height,
            facecolor=colors[i % 2],
            edgecolor="black",
            linewidth=0.8,
            zorder=10,
        )
        ax.add_patch(rect)

    # ── Tick marks & labels at each division boundary ───────────────────
    tick_len = bar_height * 0.5
    text_offset = dy * 0.006
    div_dist_m = nice_m / n_divs  # distance per division in metres

    for i in range(n_divs + 1):
        xi = x0 + i * div_width
        ax.plot(
            [xi, xi],
            [y0 - tick_len, y0 + bar_height + tick_len],
            color="black",
            linewidth=0.7,
            zorder=11,
        )
        dist_value = (i * div_dist_m) / divisor
        lbl = _format_scale_label(dist_value)
        ax.text(
            xi,
            y0 - tick_len - text_offset,
            lbl,
            ha="center",
            va="top",
            fontsize=7,
            fontweight="bold",
            path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
            zorder=12,
        )

    # ── Unit label to the right of the bar ──────────────────────────────
    ax.text(
        x0 + bar_total_crs + dx * 0.012,
        y0 + bar_height / 2,
        unit,
        ha="left",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
        zorder=12,
    )


def _format_scale_label(value: float) -> str:
    """Format a scale bar distance label, adapting decimals to magnitude."""
    if value == 0:
        return "0"
    if value == int(value):
        return f"{int(value)}"
    if value >= 10:
        return f"{value:.0f}"
    if value >= 1:
        return f"{value:.1f}"
    if value >= 0.1:
        return f"{value:.2f}"
    return f"{value:.3f}"


# ─────────────────────────────────────────────────────────────────────────────
# SVG North-Arrow Rendering
# ─────────────────────────────────────────────────────────────────────────────

_SVG_NORTH_ARROW = Path(__file__).resolve().parents[4] / "data" / "north" / "5ZPyi.svg"
_north_arrow_cache: np.ndarray | None = None


def _tokenize_svg_d(d: str) -> list[str]:
    """Split an SVG path *d* attribute into command letters and numbers."""
    return re.findall(
        r"[MmCcLlHhVvSsQqTtAaZz]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?",
        d,
    )


def _parse_svg_d(d: str) -> MplPath:
    """Convert an SVG path *d* string to a :class:`matplotlib.path.Path`."""
    tokens = _tokenize_svg_d(d)
    verts: list[tuple[float, float]] = []
    codes: list[int] = []
    cx, cy = 0.0, 0.0
    sx, sy = 0.0, 0.0  # subpath start
    i = 0

    while i < len(tokens):
        cmd = tokens[i]
        i += 1

        if cmd in "Mm":
            rel = cmd == "m"
            x, y = float(tokens[i]), float(tokens[i + 1])
            i += 2
            if rel:
                cx, cy = cx + x, cy + y
            else:
                cx, cy = x, y
            sx, sy = cx, cy
            verts.append((cx, cy))
            codes.append(MplPath.MOVETO)
            # Subsequent coordinate pairs are implicit line-to
            while i < len(tokens) and tokens[i] not in "MmCcLlHhVvSsQqTtAaZz":
                x, y = float(tokens[i]), float(tokens[i + 1])
                i += 2
                if rel:
                    cx, cy = cx + x, cy + y
                else:
                    cx, cy = x, y
                verts.append((cx, cy))
                codes.append(MplPath.LINETO)

        elif cmd in "Ll":
            rel = cmd == "l"
            while i < len(tokens) and tokens[i] not in "MmCcLlHhVvSsQqTtAaZz":
                x, y = float(tokens[i]), float(tokens[i + 1])
                i += 2
                if rel:
                    cx, cy = cx + x, cy + y
                else:
                    cx, cy = x, y
                verts.append((cx, cy))
                codes.append(MplPath.LINETO)

        elif cmd in "Cc":
            rel = cmd == "c"
            while i < len(tokens) and tokens[i] not in "MmCcLlHhVvSsQqTtAaZz":
                x1, y1 = float(tokens[i]), float(tokens[i + 1])
                x2, y2 = float(tokens[i + 2]), float(tokens[i + 3])
                x3, y3 = float(tokens[i + 4]), float(tokens[i + 5])
                i += 6
                if rel:
                    cp1 = (cx + x1, cy + y1)
                    cp2 = (cx + x2, cy + y2)
                    end = (cx + x3, cy + y3)
                else:
                    cp1 = (x1, y1)
                    cp2 = (x2, y2)
                    end = (x3, y3)
                verts.extend([cp1, cp2, end])
                codes.extend([MplPath.CURVE4] * 3)
                cx, cy = end

        elif cmd in "Zz":
            verts.append((sx, sy))
            codes.append(MplPath.CLOSEPOLY)
            cx, cy = sx, sy

    return MplPath(verts, codes)


def _render_svg_to_image(svg_path: Path, height_px: int = 160) -> np.ndarray:
    """Parse SVG at *svg_path* and rasterise its paths to an RGBA array."""
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # Handle default SVG namespace
    tag = root.tag
    ns = tag[: tag.index("}") + 1] if "{" in tag else ""

    viewbox = root.get("viewBox", "0 0 189 267")
    vb = [float(v) for v in viewbox.split()]
    vb_w, vb_h = vb[2], vb[3]
    aspect = vb_h / vb_w

    g_elem = root.find(f".//{ns}g")

    # SVG transform: translate(0, 267) scale(0.1, -0.1)
    svg_tf = Affine2D().scale(0.1, -0.1).translate(0, vb_h)

    path_elems = g_elem.findall(f"{ns}path") if g_elem is not None else []

    # Render to a small temporary figure
    fig_w = 1.5
    fig_h = fig_w * aspect
    dpi = height_px / fig_h

    fig_tmp = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax_tmp = fig_tmp.add_axes([0, 0, 1, 1])
    ax_tmp.set_xlim(0, vb_w)
    ax_tmp.set_ylim(vb_h, 0)  # y-down like SVG so arrow points north
    ax_tmp.axis("off")
    fig_tmp.patch.set_alpha(0)

    fill_color = "#000000"
    if g_elem is not None:
        fill_color = g_elem.get("fill", fill_color)

    for p in path_elems:
        d = p.get("d", "")
        raw = _parse_svg_d(d)
        transformed = svg_tf.transform(raw.vertices)
        mpath = MplPath(transformed, raw.codes)
        patch = PathPatch(mpath, facecolor=fill_color, edgecolor="none", lw=0)
        ax_tmp.add_patch(patch)

    buf = io.BytesIO()
    fig_tmp.savefig(
        buf,
        format="png",
        transparent=True,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig_tmp)
    buf.seek(0)
    return plt.imread(buf)


def _get_north_arrow_image() -> np.ndarray | None:
    """Return the cached north-arrow RGBA array, rendering on first call."""
    global _north_arrow_cache  # noqa: PLW0603
    if _north_arrow_cache is not None:
        return _north_arrow_cache
    if not _SVG_NORTH_ARROW.exists():
        logger.warning("North arrow SVG not found: %s", _SVG_NORTH_ARROW)
        return None
    try:
        _north_arrow_cache = _render_svg_to_image(_SVG_NORTH_ARROW)
        return _north_arrow_cache
    except Exception:
        logger.warning("Failed to render SVG north arrow", exc_info=True)
        return None


def _add_north_arrow(ax) -> None:
    """Overlay the SVG north arrow in the upper-right corner of *ax*."""
    img = _get_north_arrow_image()
    if img is None:
        # Fallback: simple text arrow when SVG is unavailable
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        x = xlim[1] - (xlim[1] - xlim[0]) * 0.06
        y = ylim[1] - (ylim[1] - ylim[0]) * 0.06
        al = (ylim[1] - ylim[0]) * 0.06
        ax.annotate(
            "N",
            xy=(x, y),
            xytext=(x, y - al),
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        )
        return

    # Scale the north arrow to ~6 % of figure height
    fig_h = ax.figure.get_size_inches()[1]
    fig_dpi = ax.figure.dpi
    target_h = fig_h * 0.06 * fig_dpi  # desired height in pixels
    zoom = target_h / img.shape[0]

    imagebox = OffsetImage(img, zoom=zoom)
    imagebox.image.axes = ax
    ab = AnnotationBbox(
        imagebox,
        (0.95, 0.95),
        xycoords="axes fraction",
        box_alignment=(1.0, 1.0),
        frameon=False,
        pad=0,
    )
    ax.add_artist(ab)


def _nice_round(value: float) -> float:
    """
    Round *value* down to a 'nice' number for a scale bar.

    E.g. 3743 → 3000, 874 → 800, 47 → 40.
    """
    if value <= 0:
        return 1.0
    import math

    exponent = math.floor(math.log10(value))
    base = 10**exponent
    mantissa = value / base
    if mantissa >= 5:
        return 5 * base
    elif mantissa >= 2:
        return 2 * base
    else:
        return base

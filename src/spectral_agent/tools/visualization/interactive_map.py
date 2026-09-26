"""
Interactive map generator using Folium.

Overlays a spectral-index raster on an OpenStreetMap (or other) basemap
and exports the result as a self-contained HTML file that can be opened
in any browser or embedded in Streamlit.

Workflow
--------
1. Read the single-band index GeoTIFF.
2. Reproject bounds to WGS-84 (EPSG:4326) for Folium.
3. Render the raster as a colour-mapped image overlay.
4. Add a discrete legend, layer control, and optional draw plugin.
5. Save as ``.html``.

The HTML file is ~100–500 KB for typical AOIs and loads instantly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import folium
import numpy as np
import rasterio
from rasterio.warp import transform_bounds

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Default colourmap LUT — we bake a 256-entry RGBA LUT so folium
# receives a plain PNG overlay (no matplotlib runtime needed at view time).
# ─────────────────────────────────────────────────────────────────────────────


def _cmap_to_lut(cmap_name: str, n: int = 256) -> np.ndarray:
    """Return an (n, 4) uint8 RGBA lookup table from a matplotlib cmap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(cmap_name)
    lut = (cmap(np.linspace(0, 1, n)) * 255).astype(np.uint8)
    return lut


# Per-index defaults (mirrors thematic_map.py)
_INDEX_CMAPS: dict[str, str] = {
    "NDVI": "RdYlGn",  # red (bare) → yellow → green (vegetated)
    "EVI": "YlGn",  # yellow (low) → dark green (high biomass)
    "SAVI": "BrBG",  # brown (bare soil) → white → blue-green (vegetated)
    "NDWI": "RdYlBu",  # red (dry) → yellow → blue (water)
    "NBR": "PiYG_r",  # green (unburned) → pink (burned)
    "NDBI": "YlOrRd",  # yellow (low) → orange → red (built-up)
}


# ─────────────────────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class InteractiveMapResult:
    """Metadata about the generated HTML map."""

    output_path: Path
    index_name: str
    center_lat: float
    center_lon: float
    bounds_wgs84: list[float]  # [west, south, east, north]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def generate_interactive_map(
    raster_path: Path,
    index_name: str,
    output_path: Path,
    *,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    basemap: str = "OpenStreetMap",
    opacity: float = 0.75,
    zoom_start: int | None = None,
) -> InteractiveMapResult:
    """
    Create an interactive HTML map with the index raster overlaid.

    Parameters
    ----------
    raster_path : Path
        Single-band Float32 GeoTIFF (from ``index_calculator``).
    index_name : str
        Index name (``NDVI``, etc.).
    output_path : Path
        Destination ``.html`` file.
    cmap : str | None
        Matplotlib colourmap name; ``None`` → auto per index.
    vmin, vmax : float | None
        Value range for colour-mapping; ``None`` → auto from data.
    basemap : str
        Folium tile layer name (``"OpenStreetMap"``,
        ``"CartoDB positron"``, ``"Esri.WorldImagery"``, …).
    opacity : float
        Overlay opacity [0, 1].
    zoom_start : int | None
        Initial zoom; ``None`` → auto-fit.

    Returns
    -------
    InteractiveMapResult
    """
    raster_path = Path(raster_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    name = index_name.upper()

    # ── Read raster ─────────────────────────────────────────────────────
    with rasterio.open(raster_path) as src:
        data = src.read(1).astype(np.float32)
        raster_crs = src.crs
        raster_bounds = src.bounds  # in native CRS

    # ── Reproject bounds to WGS-84 ─────────────────────────────────────
    west, south, east, north = transform_bounds(
        raster_crs,
        "EPSG:4326",
        raster_bounds.left,
        raster_bounds.bottom,
        raster_bounds.right,
        raster_bounds.top,
    )
    center_lat = (south + north) / 2
    center_lon = (west + east) / 2

    # ── Normalise & colourise ───────────────────────────────────────────
    valid = data[~np.isnan(data)]
    v0 = vmin if vmin is not None else (float(valid.min()) if valid.size else -1.0)
    v1 = vmax if vmax is not None else (float(valid.max()) if valid.size else 1.0)

    cmap_name = cmap or _INDEX_CMAPS.get(name, "viridis")
    lut = _cmap_to_lut(cmap_name)

    # Map data → [0, 255] index
    normalised = np.clip((data - v0) / (v1 - v0 + 1e-10), 0, 1)
    indices_arr = (normalised * 255).astype(np.uint8)

    # Build RGBA image
    rgba = lut[indices_arr]  # shape (H, W, 4)

    # Make NaN pixels fully transparent
    nan_mask = np.isnan(data)
    rgba[nan_mask] = [0, 0, 0, 0]

    # ── Build Folium map ────────────────────────────────────────────────
    if zoom_start is None:
        # Rough heuristic: larger extent → lower zoom
        lat_span = north - south
        lon_span = east - west
        max_span = max(lat_span, lon_span)
        if max_span > 1:
            zoom_start = 9
        elif max_span > 0.1:
            zoom_start = 12
        else:
            zoom_start = 14

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles=basemap,
    )

    # ── Additional satellite basemap layer ──────────────────────────────
    # Esri World Imagery provides a high-resolution satellite basemap that
    # users can toggle via the layer control.
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics, USDA FSA, USGS, "
        "Aerogrid, IGN, IGP, and the GIS User Community",
        name="Satellite (Esri)",
        overlay=False,
        control=True,
    ).add_to(m)

    # Google Satellite as an alternative option
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google",
        name="Satellite (Google)",
        overlay=False,
        control=True,
    ).add_to(m)

    # Image overlay  (bounds format: [[south, west], [north, east]])
    # folium accepts a numpy RGBA array directly
    overlay_name = f"{name} Index"
    img_overlay = folium.raster_layers.ImageOverlay(
        image=rgba,
        bounds=[[south, west], [north, east]],
        opacity=opacity,
        name=overlay_name,
        interactive=True,
        cross_origin=False,
    )
    img_overlay.add_to(m)

    # Layer control
    folium.LayerControl().add_to(m)

    # ── Opacity slider ──────────────────────────────────────────────────
    opacity_html = _build_opacity_slider_html(opacity)
    root = cast(folium.Figure, m.get_root())
    root.html.add_child(folium.Element(opacity_html))

    # ── Legend as a simple HTML box ─────────────────────────────────────
    legend_html = _build_legend_html(name, cmap_name, v0, v1)
    root.html.add_child(folium.Element(legend_html))

    # ── Save ────────────────────────────────────────────────────────────
    m.save(str(output_path))

    logger.info("Interactive map saved → %s", output_path.name)

    return InteractiveMapResult(
        output_path=output_path,
        index_name=name,
        center_lat=center_lat,
        center_lon=center_lon,
        bounds_wgs84=[west, south, east, north],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _build_legend_html(
    index_name: str,
    cmap_name: str,
    vmin: float,
    vmax: float,
) -> str:
    """Build a small fixed-position HTML legend with a CSS gradient."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(cmap_name)
    stops = []
    for i in range(6):
        frac = i / 5
        r, g, b, _ = cmap(frac)
        stops.append(f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)}) {int(frac * 100)}%")
    gradient = ", ".join(stops)

    return f"""
    <div style="
        position: fixed;
        bottom: 30px; left: 30px;
        z-index: 1000;
        background: white;
        padding: 10px 14px;
        border-radius: 6px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        font-family: Arial, sans-serif;
        font-size: 12px;
        line-height: 1.4;
    ">
        <b>{index_name}</b><br>
        <div style="
            width: 160px; height: 14px;
            margin: 4px 0;
            border-radius: 3px;
            background: linear-gradient(to right, {gradient});
        "></div>
        <div style="display: flex; justify-content: space-between; font-size: 10px;">
            <span>{vmin:.3f}</span>
            <span>{vmax:.3f}</span>
        </div>
    </div>
    """


def _build_opacity_slider_html(initial_opacity: float = 0.75) -> str:
    """Return an HTML/JS snippet that adds an opacity slider for the raster overlay.

    The slider finds the *first* Leaflet ``ImageOverlay`` on the map and
    adjusts its opacity in real-time.
    """
    pct = int(initial_opacity * 100)
    return f"""
    <div id="opacity-ctrl" style="
        position: fixed;
        bottom: 30px; right: 30px;
        z-index: 1000;
        background: white;
        padding: 10px 14px;
        border-radius: 6px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        font-family: Arial, sans-serif;
        font-size: 12px;
        line-height: 1.6;
        min-width: 160px;
    ">
        <b>Opacity</b>
        <span id="opacity-val" style="float:right">{pct}%</span><br>
        <input id="opacity-slider" type="range" min="0" max="100"
               value="{pct}" style="width:100%; cursor:pointer">
    </div>
    <script>
    (function() {{
        // Wait for the map to be fully initialised
        var tries = 0;
        var timer = setInterval(function() {{
            tries++;
            if (tries > 50) {{ clearInterval(timer); return; }}
            // Find the leaflet map object
            var mapEl = document.querySelector('.folium-map');
            if (!mapEl || !mapEl._leaflet_id) return;
            var mapId = mapEl._leaflet_id;
            var mapObj = null;
            // Iterate window properties to find the L.Map instance
            for (var k in window) {{
                try {{
                    if (window[k] instanceof L.Map) {{ mapObj = window[k]; break; }}
                }} catch(e) {{}}
            }}
            if (!mapObj) return;
            clearInterval(timer);
            // Find the ImageOverlay layer
            var overlay = null;
            mapObj.eachLayer(function(layer) {{
                if (layer instanceof L.ImageOverlay) {{ overlay = layer; }}
            }});
            if (!overlay) return;
            var slider = document.getElementById('opacity-slider');
            var label = document.getElementById('opacity-val');
            slider.addEventListener('input', function() {{
                var val = parseInt(this.value, 10) / 100;
                overlay.setOpacity(val);
                label.textContent = this.value + '%';
            }});
        }}, 200);
    }})();
    </script>
    """

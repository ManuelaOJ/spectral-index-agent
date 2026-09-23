"""
LangChain tool wrapper for **thematic map** generation.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Annotated, Any, Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool
from pydantic import BaseModel, Field

from spectral_agent.tracking.metrics import track_step
from spectral_agent.utils.session_paths import get_session_id

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Input Schema
# ─────────────────────────────────────────────────────────────────────────────


class GenerateMapInput(BaseModel):
    """Input schema for map generation."""

    raster_path: str = Field(..., description="Path to the index GeoTIFF to visualise")
    index_name: str = Field(..., description="Spectral index name (e.g. 'NDVI')")
    map_type: Literal["static", "interactive", "both"] = Field(
        default="both",
        description="Type of map: 'static' (PNG), 'interactive' (HTML), or 'both'",
    )
    title: str | None = Field(
        default=None,
        description="Optional custom title for the map",
    )
    scene_id: str | None = Field(
        default=None,
        description="Satellite scene ID (e.g. 'LC90080552025023LGN00'). Auto-detected from path if omitted.",
    )
    acquisition_date: str | None = Field(
        default=None,
        description="Acquisition date (e.g. '2025-01-23'). Auto-detected from scene ID if omitted.",
    )
    satellite: str | None = Field(
        default=None,
        description="Satellite name (e.g. 'Landsat 9', 'Sentinel-2A'). Auto-detected from scene ID if omitted.",
    )
    language: str = Field(
        default="en",
        description="Legend language: 'en' (English) or 'es' (Spanish)",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tool
# ─────────────────────────────────────────────────────────────────────────────


@tool(args_schema=GenerateMapInput)
def generate_thematic_map_tool(
    raster_path: str,
    index_name: str,
    map_type: str = "both",
    title: str | None = None,
    scene_id: str | None = None,
    acquisition_date: str | None = None,
    satellite: str | None = None,
    language: str = "en",
    *,
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> dict[str, Any]:
    """
    Generate thematic maps from a spectral index GeoTIFF.

    Can produce:
    - A static PNG map with colorbar, scale bar, and north arrow.
    - An interactive HTML map with the index overlaid on OpenStreetMap.
    - Both at once.

    Use after compute_spectral_index_tool.
    """
    from spectral_agent.tools.visualization import (
        generate_thematic_map,
        generate_interactive_map,
        MapConfig,
    )

    rpath = Path(raster_path)
    output_dir = rpath.parent / "maps"

    # ── Auto-detect scene_id / acquisition_date from path ───────────────
    if scene_id is None:
        scene_id = _infer_scene_id(rpath)
    if acquisition_date is None and scene_id:
        acquisition_date = _infer_date_from_scene_id(scene_id)
    if satellite is None and scene_id:
        satellite = _infer_satellite_from_scene_id(scene_id)

    sid = get_session_id(config)
    logger.info(
        "generate_thematic_map_tool called: raster_path=%s, index=%s, "
        "map_type=%s, scene_id=%s, satellite=%s",
        raster_path, index_name, map_type, scene_id, satellite,
    )

    results: dict[str, Any] = {
        "success": True,
        "index_name": index_name.upper(),
        "outputs": {},
    }

    with track_step(
        "map_rendering", index_name=index_name.upper(), session_id=sid
    ) as step:
        try:
            if map_type in ("static", "both"):
                png_path = output_dir / f"{rpath.stem}_map.png"
                cfg = MapConfig(
                    title=title,
                    scene_id=scene_id,
                    acquisition_date=acquisition_date,
                    satellite=satellite,
                    language=language,
                )
                static_result = generate_thematic_map(
                    rpath, index_name, png_path, config=cfg
                )
                results["outputs"]["static_png"] = str(static_result.output_path)

            if map_type in ("interactive", "both"):
                html_path = output_dir / f"{rpath.stem}_map.html"
                interactive_result = generate_interactive_map(
                    rpath, index_name, html_path
                )
                results["outputs"]["interactive_html"] = str(
                    interactive_result.output_path
                )

            results["outputs"]["source_geotiff"] = str(rpath)
            results["message"] = f"Generated {map_type} map(s) for {index_name.upper()}"
            step.set_metadata(
                map_type=map_type, outputs=list(results["outputs"].keys())
            )

        except Exception as e:
            logger.error("Map generation failed: %s", e, exc_info=True)
            results["success"] = False
            results["error"] = str(e)
            results["message"] = f"Map generation failed: {e}"
            step.mark_failure(str(e))

    return results


def get_visualization_tools() -> list:
    """Return all visualisation LangChain tools."""
    return [generate_thematic_map_tool]


# ─────────────────────────────────────────────────────────────────────────────
# Auto-detection helpers
# ─────────────────────────────────────────────────────────────────────────────

# Landsat scene ID pattern: e.g. LC90080552025023LGN00
_LANDSAT_RE = re.compile(
    r"(L[CETOM]\d{2}_L[12][A-Z]{2}_\d{6}_(\d{8})_\d{8}_\d{2}_[A-Z]\d"
    r"|[LETC][CETOM]\d{2}\d{6}(\d{7})\w+)"
)

# Sentinel-2 scene ID pattern: e.g. S2A_MSIL2A_20200218T152631_...
_SENTINEL_RE = re.compile(r"(S2[AB]_\w+?_(\d{8})T\d{6}_\w+)")


def _infer_scene_id(rpath: Path) -> str | None:
    """Try to extract a satellite scene ID from the raster file path."""
    text = str(rpath)
    m = _LANDSAT_RE.search(text)
    if m:
        return m.group(1)
    m = _SENTINEL_RE.search(text)
    if m:
        return m.group(1)
    return None


def _infer_date_from_scene_id(scene_id: str) -> str | None:
    """Try to extract an acquisition date (YYYY-MM-DD) from a scene ID."""
    from datetime import datetime

    # Landsat Collection 2: ..._YYYYMMDD_...
    m = re.search(r"_(\d{8})_", scene_id)
    if m:
        try:
            dt = datetime.strptime(m.group(1), "%Y%m%d")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    # Landsat legacy: YYYYDDD (Julian day)
    m = re.search(r"\d{3}\d{6}(\d{7})", scene_id)
    if m:
        try:
            dt = datetime.strptime(m.group(1), "%Y%j")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    # Sentinel-2: ..._YYYYMMDDTHHMMSS_...
    m = re.search(r"_(\d{8})T\d{6}_", scene_id)
    if m:
        try:
            dt = datetime.strptime(m.group(1), "%Y%m%d")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    return None


# Satellite sensor prefix → human-readable name
_SATELLITE_NAMES: dict[str, str] = {
    "LC09": "Landsat 9",
    "LC08": "Landsat 8",
    "LE07": "Landsat 7",
    "LT05": "Landsat 5",
    "LT04": "Landsat 4",
    "S2A": "Sentinel-2A",
    "S2B": "Sentinel-2B",
}


def _infer_satellite_from_scene_id(scene_id: str) -> str | None:
    """Try to derive a satellite name from a scene ID prefix."""
    # Landsat Collection 2: first 4 chars (e.g. LC09, LE07)
    if len(scene_id) >= 4 and scene_id[:2] in ("LC", "LE", "LT", "LO", "LM"):
        prefix = scene_id[:4]
        return _SATELLITE_NAMES.get(prefix)
    # Sentinel-2: starts with S2A or S2B
    if scene_id.startswith("S2A"):
        return _SATELLITE_NAMES["S2A"]
    if scene_id.startswith("S2B"):
        return _SATELLITE_NAMES["S2B"]
    return None

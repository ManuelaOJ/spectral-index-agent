"""
LangChain tool wrappers for **raster processing** — crop, index, cache.

These tools wrap :mod:`spectral_agent.tools.raster` components to make
them callable by LangChain / LangGraph agents.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any, Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool
from pydantic import BaseModel, Field

from spectral_agent.config import get_settings
from spectral_agent.schemas.imagery import BoundingBox
from spectral_agent.schemas.spectral_request import LandsatSensor
from spectral_agent.tracking.debug_log import get_debug_logger
from spectral_agent.tracking.metrics import track_step
from spectral_agent.utils.session_paths import (
    get_session_id,
    session_raw_dir,
    session_processed_dir,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Input Schemas
# ─────────────────────────────────────────────────────────────────────────────


class CropLandsatBandsInput(BaseModel):
    """Input schema for Landsat band cropping."""

    scene_id: str = Field(
        ...,
        description="Landsat scene display ID (e.g. 'LC09_L2SP_008057_20240101_...')",
    )
    tar_path: str = Field(..., description="Path to the downloaded .tar archive")
    index_names: list[str] = Field(
        ...,
        description=(
            "Spectral indices to prepare bands for. "
            "Supported: NDVI, EVI, SAVI, NDWI, NBR, NDBI"
        ),
    )
    sensor: str = Field(
        ...,
        description=(
            "Landsat sensor name: 'Landsat 4', 'Landsat 5', 'Landsat 7', "
            "'Landsat 8', or 'Landsat 9'"
        ),
    )
    west: float = Field(
        ..., ge=-180, le=180, description="Western boundary longitude (WGS84)"
    )
    south: float = Field(
        ..., ge=-90, le=90, description="Southern boundary latitude (WGS84)"
    )
    east: float = Field(
        ..., ge=-180, le=180, description="Eastern boundary longitude (WGS84)"
    )
    north: float = Field(
        ..., ge=-90, le=90, description="Northern boundary latitude (WGS84)"
    )


class ComputeSpectralIndexInput(BaseModel):
    """Input schema for computing a spectral index."""

    scene_id: str = Field(..., description="Landsat scene display ID")
    index_names: list[str] = Field(
        ...,
        description=(
            "Spectral indices to compute. "
            "Supported: NDVI, EVI, SAVI, NDWI, NBR, NDBI"
        ),
    )
    band_paths: dict[str, str] | None = Field(
        default=None,
        description=(
            "Mapping of band names to file paths, e.g. "
            "{'SR_B4': '/path/to/SR_B4.tif', 'SR_B5': '/path/to/SR_B5.tif'}. "
            "If omitted, paths are auto-resolved from the session band cache."
        ),
    )
    sensor: str = Field(
        ...,
        description="Landsat sensor name ('Landsat 8', 'Landsat 9', etc.)",
    )


class ListCachedBandsInput(BaseModel):
    """Input for listing cached bands."""

    scene_id: str | None = Field(
        default=None,
        description="Filter by scene ID. If None, list all cached bands.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Crop Tool
# ─────────────────────────────────────────────────────────────────────────────


@tool(args_schema=CropLandsatBandsInput)
def crop_landsat_bands_tool(
    scene_id: str,
    tar_path: str,
    index_names: list[str],
    sensor: str,
    west: float,
    south: float,
    east: float,
    north: float,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> dict[str, Any]:
    """
    Extract and crop Landsat bands required for spectral indices.

    Extracts the .tar archive, identifies the bands needed for the
    requested indices, clips them to the bounding box, and caches
    them so shared bands are only cropped once.

    Use after download_landsat_tool to prepare bands for index computation.
    """
    from spectral_agent.tools.raster import LandsatProcessor

    sid = get_session_id(config)
    logger.info(
        "crop_landsat_bands_tool called: scene_id=%s, sensor=%s, "
        "indices=%s, bbox=(%s,%s,%s,%s), tar_path=%s",
        scene_id,
        sensor,
        index_names,
        west,
        south,
        east,
        north,
        tar_path,
    )
    with track_step("band_cropping", satellite="landsat", session_id=sid) as step:
        try:
            bbox = BoundingBox(west=west, south=south, east=east, north=north)
            sensor_enum = LandsatSensor(sensor)

            processor = LandsatProcessor(
                raw_dir=session_raw_dir(sid, "landsat"),
                processed_dir=session_processed_dir(sid, "landsat"),
            )

            scene_dir = processor.extract_tar(Path(tar_path))

            all_bands = processor.crop_multiple_indices(
                scene_id=scene_id,
                scene_dir=scene_dir,
                index_names=index_names,
                sensor=sensor_enum,
                bbox=bbox,
            )

            band_summary: dict[str, str] = {}
            for idx_name, band_dict in all_bands.items():
                for band_name, band_path in band_dict.items():
                    band_summary[band_name] = str(band_path)

            step.set_metadata(
                scene_id=scene_id,
                indices_prepared=list(all_bands.keys()),
                bands_cropped=len(band_summary),
            )
            return {
                "success": True,
                "scene_id": scene_id,
                "indices_prepared": list(all_bands.keys()),
                "bands_cropped": band_summary,
                "total_bands": len(band_summary),
                "message": (
                    f"Cropped {len(band_summary)} bands for "
                    f"{len(all_bands)} indices from {scene_id}"
                ),
            }

        except Exception as e:
            logger.error("Band cropping failed for %s: %s", scene_id, e, exc_info=True)
            step.mark_failure(str(e))
            return {
                "success": False,
                "scene_id": scene_id,
                "error": str(e),
                "message": f"Crop failed: {e}",
            }


# ─────────────────────────────────────────────────────────────────────────────
# Compute Index Tool
# ─────────────────────────────────────────────────────────────────────────────


@tool(args_schema=ComputeSpectralIndexInput)
def compute_spectral_index_tool(
    scene_id: str,
    index_names: list[str],
    sensor: str,
    band_paths: dict[str, str] | None = None,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> dict[str, Any]:
    """
    Compute spectral indices from cropped Landsat bands.

    Reads cropped GeoTIFF bands, applies Landsat Collection 2 Level-2
    reflectance scaling, computes the requested indices, and writes
    each result as a Float32 GeoTIFF.

    Use after crop_landsat_bands_tool.  The band_paths should come from
    that tool's output. If omitted, they are resolved from cached cropped
    bands in the current session.
    """
    from spectral_agent.tools.raster.index_calculator import (
        IndexResult,
        compute_indices,
    )

    sid = get_session_id(config)
    logger.info(
        "compute_spectral_index_tool called: scene_id=%s, sensor=%s, "
        "indices=%s, band_paths=%s",
        scene_id,
        sensor,
        index_names,
        band_paths if band_paths else "<auto>",
    )
    with track_step("index_computation", satellite="landsat", session_id=sid) as step:
        try:
            sensor_enum = LandsatSensor(sensor)
            output_dir = session_processed_dir(sid, "landsat") / scene_id / "indices"
            resolved_band_paths = band_paths or _resolve_band_paths_from_cache(
                scene_id=scene_id,
                index_names=index_names,
                sensor=sensor_enum,
                session_id=sid,
            )
            paths = {k: Path(v) for k, v in resolved_band_paths.items()}

            results: list[IndexResult] = compute_indices(
                index_names=index_names,
                band_paths=paths,
                sensor=sensor_enum,
                output_dir=output_dir,
                scene_id=scene_id,
            )

            # Resolve band mapping table for debug info
            from spectral_agent.schemas.spectral_request import (
                LANDSAT_89_INDEX_BANDS,
                LANDSAT_457_INDEX_BANDS,
            )
            from spectral_agent.tracking.debug_log import FORMULA_DESCRIPTIONS

            if sensor_enum in (LandsatSensor.LANDSAT_8, LandsatSensor.LANDSAT_9):
                band_table = LANDSAT_89_INDEX_BANDS
            else:
                band_table = LANDSAT_457_INDEX_BANDS

            dbg = get_debug_logger()
            index_summaries = []
            debug_entries = []
            for r in results:
                summary = {
                    "index": r.index_name,
                    "satellite": "landsat",
                    "sensor": sensor,
                    "output_path": str(r.output_path),
                    "value_min": round(r.value_min, 4),
                    "value_max": round(r.value_max, 4),
                    "value_mean": round(r.value_mean, 4),
                    "nodata_pct": r.nodata_pct,
                    "width": r.width,
                    "height": r.height,
                    "crs": r.crs,
                }
                index_summaries.append(summary)

                # Debug logging
                role_to_band = band_table.get(r.index_name, {})

                # Include index-specific parameters (e.g. L=0.5 for SAVI)
                extra_kwargs: dict = {"scene_id": scene_id}
                if r.index_name == "SAVI":
                    extra_kwargs["parameters"] = {"L": 0.5}

                rec = dbg.log_index_computation(
                    satellite="landsat",
                    sensor=sensor,
                    index_name=r.index_name,
                    bands_used=role_to_band,
                    output_path=str(r.output_path),
                    stats={
                        "min": round(r.value_min, 4),
                        "max": round(r.value_max, 4),
                        "mean": round(r.value_mean, 4),
                        "nodata_pct": r.nodata_pct,
                    },
                    session_id=sid,
                    **extra_kwargs,
                )
                debug_entry = {
                    "index": r.index_name,
                    "formula": rec.formula,
                    "bands": role_to_band,
                    "satellite": "landsat",
                    "sensor": sensor,
                }
                if r.index_name == "SAVI":
                    debug_entry["parameters"] = {"L": 0.5}
                debug_entries.append(debug_entry)

            step.set_metadata(
                scene_id=scene_id,
                indices_computed=[r["index"] for r in index_summaries],
            )
            return {
                "success": True,
                "scene_id": scene_id,
                "satellite": "landsat",
                "sensor": sensor,
                "indices_computed": [r["index"] for r in index_summaries],
                "results": index_summaries,
                "debug_info": debug_entries,
                "debug_log_path": str(dbg.log_path) if dbg.log_path else None,
                "band_paths_source": "input" if band_paths else "cache",
                "message": (
                    f"Computed {len(index_summaries)} indices for {scene_id} "
                    f"using {sensor} bands"
                ),
            }

        except Exception as e:
            logger.error(
                "Index computation failed for %s: %s", scene_id, e, exc_info=True
            )
            step.mark_failure(str(e))
            return {
                "success": False,
                "scene_id": scene_id,
                "error": str(e),
                "message": f"Index computation failed: {e}",
            }


def _required_bands_for_indices(
    index_names: list[str],
    sensor: LandsatSensor,
) -> set[str]:
    """Return the set of unique Landsat band names required for the indices."""
    from spectral_agent.schemas.spectral_request import (
        LANDSAT_89_INDEX_BANDS,
        LANDSAT_457_INDEX_BANDS,
    )

    band_table = (
        LANDSAT_89_INDEX_BANDS
        if sensor in (LandsatSensor.LANDSAT_8, LandsatSensor.LANDSAT_9)
        else LANDSAT_457_INDEX_BANDS
    )

    required: set[str] = set()
    for index_name in index_names:
        if index_name not in band_table:
            raise ValueError(
                f"Index '{index_name}' is not supported for {sensor.value}."
            )
        required.update(band_table[index_name].values())
    return required


def _resolve_band_paths_from_cache(
    scene_id: str,
    index_names: list[str],
    sensor: LandsatSensor,
    session_id: str,
) -> dict[str, str]:
    """Resolve required band paths from cached cropped bands for a scene."""
    from spectral_agent.tools.raster import BandCache

    required_bands = _required_bands_for_indices(index_names, sensor)
    cache = BandCache(root=session_processed_dir(session_id, "landsat"))
    entries = cache.list_all_entries(scene_id=scene_id)

    by_bbox: dict[str, dict[str, str]] = {}
    for entry in entries:
        band_name = entry["band_name"]
        if band_name not in required_bands:
            continue
        path_str = entry["path"]
        if not Path(path_str).exists():
            continue
        bbox_hash = entry["bbox_hash"]
        by_bbox.setdefault(bbox_hash, {})[band_name] = path_str

    candidates: list[dict[str, str]] = []
    for bands in by_bbox.values():
        if required_bands.issubset(set(bands)):
            candidates.append({bn: bands[bn] for bn in required_bands})

    if not candidates:
        raise ValueError(
            "Missing required band paths and no complete cached band set was found "
            f"for scene '{scene_id}'. Run crop_landsat_bands_tool first or pass "
            "band_paths explicitly."
        )

    def _candidate_score(candidate: dict[str, str]) -> float:
        # Prefer the most recently written cached set.
        return max(Path(p).stat().st_mtime for p in candidate.values())

    best = max(candidates, key=_candidate_score)
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Cache & Discovery Tools
# ─────────────────────────────────────────────────────────────────────────────


@tool(args_schema=ListCachedBandsInput)
def list_cached_bands_tool(
    scene_id: str | None = None,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> dict[str, Any]:
    """
    List bands that are already cropped and cached.

    Shows which bands have been processed, avoiding redundant work.
    Call this to check what's available before cropping or computing
    indices.
    """
    from spectral_agent.tools.raster import BandCache

    sid = get_session_id(config)
    with track_step("cache_listing", session_id=sid) as step:
        try:
            cache = BandCache(root=session_processed_dir(sid, "landsat"))
            all_bands = cache.list_all_entries(scene_id=scene_id)

            step.set_metadata(cached_bands=len(all_bands))
            return {
                "success": True,
                "total": len(all_bands),
                "bands": all_bands,
                "message": (
                    f"Found {len(all_bands)} cached band(s)"
                    + (f" for {scene_id}" if scene_id else "")
                ),
            }

        except Exception as e:
            logger.error("Cache listing failed: %s", e)
            step.mark_failure(str(e))
            return {
                "success": False,
                "error": str(e),
                "bands": [],
                "message": f"Cache listing failed: {e}",
            }


@tool
def list_available_indices_tool() -> dict[str, Any]:
    """
    List all spectral indices the system can compute, with their
    required Landsat bands and value ranges.

    Use this to discover what indices are available and what bands
    each one needs.
    """
    from spectral_agent.tools.raster.index_calculator import (
        INDEX_VALUE_RANGES,
        list_supported_indices,
    )
    from spectral_agent.schemas.spectral_request import (
        LANDSAT_89_INDEX_BANDS,
        LANDSAT_457_INDEX_BANDS,
    )
    from spectral_agent.tools.visualization.thematic_map import INDEX_FULL_NAMES

    indices_info = []
    for name in list_supported_indices():
        vmin, vmax = INDEX_VALUE_RANGES.get(name, (-1, 1))
        indices_info.append(
            {
                "name": name,
                "full_name": INDEX_FULL_NAMES.get(name, name),
                "value_range": [vmin, vmax],
                "landsat_89_bands": LANDSAT_89_INDEX_BANDS.get(name, {}),
                "landsat_457_bands": LANDSAT_457_INDEX_BANDS.get(name, {}),
            }
        )

    return {
        "success": True,
        "count": len(indices_info),
        "indices": indices_info,
        "message": f"{len(indices_info)} spectral indices available",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience registry
# ─────────────────────────────────────────────────────────────────────────────


def get_raster_tools() -> list:
    """Return all raster-processing LangChain tools."""
    return [
        crop_landsat_bands_tool,
        compute_spectral_index_tool,
        list_cached_bands_tool,
        list_available_indices_tool,
    ]

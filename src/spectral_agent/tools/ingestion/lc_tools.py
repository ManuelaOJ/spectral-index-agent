"""
LangChain tool wrappers for satellite imagery **ingestion**.

Provides search and download tools for Landsat and Sentinel-2 that are
callable by LangChain / LangGraph agents.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Annotated, Any, Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool
from pydantic import BaseModel, Field

from spectral_agent.schemas.imagery import BoundingBox
from spectral_agent.tools.ingestion.landsat import LandsatClient
from spectral_agent.tools.ingestion.sentinel import SentinelClient
from spectral_agent.tracking.debug_log import get_debug_logger
from spectral_agent.tracking.metrics import track_step
from spectral_agent.utils.session_paths import (
    get_session_id,
    session_processed_dir,
    session_raw_dir,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Input Schemas
# ─────────────────────────────────────────────────────────────────────────────


class SearchImageryInput(BaseModel):
    """Input schema for satellite imagery search."""

    west: float = Field(
        ...,
        ge=-180,
        le=180,
        description="Western boundary longitude in decimal degrees (WGS84)",
    )
    south: float = Field(
        ...,
        ge=-90,
        le=90,
        description="Southern boundary latitude in decimal degrees (WGS84)",
    )
    east: float = Field(
        ...,
        ge=-180,
        le=180,
        description="Eastern boundary longitude in decimal degrees (WGS84)",
    )
    north: float = Field(
        ...,
        ge=-90,
        le=90,
        description="Northern boundary latitude in decimal degrees (WGS84)",
    )
    start_date: str = Field(
        ...,
        description=(
            "Start date in YYYY-MM-DD format. "
            "If the user requests a specific day, set start_date = end_date = that day. "
            "If a month, use the 1st of that month. "
            "If a year, use YYYY-01-01."
        ),
    )
    end_date: str = Field(
        ...,
        description=(
            "End date in YYYY-MM-DD format. "
            "If the user requests a specific day, set end_date = start_date = that day. "
            "If a month, use the last day of that month. "
            "If a year, use YYYY-12-31."
        ),
    )
    max_cloud_cover: float = Field(
        default=100.0,
        ge=0,
        le=100,
        description=(
            "Maximum cloud cover percentage (0-100). "
            "Default is 100 (no filter) — the system returns all scenes "
            "sorted by lowest cloud cover. Only set a lower value if the "
            "user explicitly requests a cloud cover limit."
        ),
    )


class DownloadSceneInput(BaseModel):
    """Input schema for scene download."""

    scene_id: str = Field(..., description="Unique scene identifier from search results")
    satellite: Literal["landsat", "sentinel"] = Field(
        ..., description="Satellite type: 'landsat' or 'sentinel'"
    )
    entity_id: str | None = Field(
        default=None,
        description="USGS internal entity ID for Landsat scenes "
        "(returned by search as 'entity_id'). Required for download.",
    )
    collection: str | None = Field(
        default=None,
        description="Dataset collection name (e.g. 'landsat_ot_c2_l2').",
    )
    # Sentinel downloads require the AOI bbox
    west: float | None = Field(default=None, description="Western longitude")
    south: float | None = Field(default=None, description="Southern latitude")
    east: float | None = Field(default=None, description="Eastern longitude")
    north: float | None = Field(default=None, description="Northern latitude")


class MultiSatelliteSearchInput(SearchImageryInput):
    """Input for searching multiple satellites."""

    satellites: list[Literal["landsat", "sentinel"]] = Field(
        ...,
        description=(
            "List of satellites to search. MUST be explicitly provided — "
            "e.g. ['landsat'], ['sentinel'], or ['landsat', 'sentinel']. "
            "Only pass both when the user explicitly asks to compare."
        ),
    )


class SentinelIndexInput(BaseModel):
    """Input for computing spectral indices directly on Sentinel Hub."""

    scene_id: str = Field(..., description="Sentinel-2 scene ID from search results")
    index_names: list[str] = Field(
        ...,
        description=(
            "Spectral indices to compute server-side. Supported: NDVI, EVI, SAVI, NDWI, NBR, NDBI"
        ),
    )
    west: float = Field(..., ge=-180, le=180, description="Western boundary longitude (WGS84)")
    south: float = Field(..., ge=-90, le=90, description="Southern boundary latitude (WGS84)")
    east: float = Field(..., ge=-180, le=180, description="Eastern boundary longitude (WGS84)")
    north: float = Field(..., ge=-90, le=90, description="Northern boundary latitude (WGS84)")
    resolution: int = Field(default=10, description="Output resolution in meters (default 10)")


# ─────────────────────────────────────────────────────────────────────────────
# Landsat Tools
# ─────────────────────────────────────────────────────────────────────────────


@tool(args_schema=SearchImageryInput)
async def search_landsat_tool(
    west: float,
    south: float,
    east: float,
    north: float,
    start_date: str,
    end_date: str,
    max_cloud_cover: float = 100.0,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> dict[str, Any]:
    """
    Search for available Landsat satellite imagery.

    IMPORTANT: Only call this tool AFTER the user has confirmed they want
    Landsat.  If the user did not specify a satellite, ask first.

    Searches the USGS archive for Landsat 4-9 Collection 2 Level-2
    (surface reflectance) scenes matching the specified criteria.
    Returns ALL matching scenes sorted by lowest cloud cover.

    Leave max_cloud_cover at 100 (default) unless the user explicitly
    requests a cloud cover limit.
    """
    sid = get_session_id(config)
    with track_step("scene_search", satellite="landsat", session_id=sid) as step:
        try:
            bbox = BoundingBox(west=west, south=south, east=east, north=north)
            start = date.fromisoformat(start_date)
            end = date.fromisoformat(end_date)

            logger.info(
                "Landsat search: bbox=%s, dates=%s..%s, cloud<=%s",
                bbox,
                start,
                end,
                max_cloud_cover,
            )

            async with LandsatClient() as client:
                result = await client.search(
                    bbox=bbox,
                    start_date=start,
                    end_date=end,
                    max_cloud_cover=max_cloud_cover,
                )
                logger.info("Landsat search returned %d scenes", result.total_count)

            scenes_summary = [
                {
                    "scene_id": scene.scene_id,
                    "entity_id": scene.raw_metadata.get("entityId", ""),
                    "satellite": "landsat",
                    "acquisition_date": scene.acquisition_date.isoformat(),
                    "cloud_cover": scene.cloud_cover,
                    "collection": scene.collection,
                }
                for scene in result.scenes
            ]

            step.set_metadata(scenes_found=result.total_count)
            return {
                "success": True,
                "total_found": result.total_count,
                "scenes": scenes_summary,
            }

        except Exception as e:
            logger.error("Landsat search failed: %s", e, exc_info=True)
            step.mark_failure(str(e))
            return {
                "success": False,
                "error": str(e),
                "scenes": [],
                "message": f"Search failed: {e}",
            }


@tool(args_schema=DownloadSceneInput)
async def download_landsat_tool(
    scene_id: str,
    satellite: str = "landsat",
    entity_id: str | None = None,
    collection: str | None = None,
    west: float | None = None,
    south: float | None = None,
    east: float | None = None,
    north: float | None = None,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> dict[str, Any]:
    """
    Download a Landsat scene to local storage.

    Downloads the full Landsat scene (all bands) as a tar archive.
    The scene_id AND entity_id must come from a previous search_landsat_tool result.

    IMPORTANT: Always pass entity_id and collection from the search results.
    The entity_id is the USGS internal identifier required for download.

    Returns the path to the downloaded file and download statistics.

    Use this tool after search_landsat_tool to actually download the imagery.
    """
    sid = get_session_id(config)
    output_dir = session_raw_dir(sid, "landsat")
    logger.info(
        "download_landsat_tool called: scene_id=%s, entity_id=%s, "
        "collection=%s, bbox=(%s,%s,%s,%s)",
        scene_id,
        entity_id,
        collection,
        west,
        south,
        east,
        north,
    )

    # Use entity_id for the actual download; fall back to scene_id
    download_id = entity_id or scene_id

    with track_step("scene_download", satellite="landsat", session_id=sid) as step:
        try:
            async with LandsatClient() as client:
                file_path = await client.download(
                    scene_id=download_id,
                    output_dir=output_dir,
                    collection=collection,
                )

            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            step.set_metadata(scene_id=scene_id, file_size_mb=round(file_size_mb, 2))

            return {
                "success": True,
                "scene_id": scene_id,
                "file_path": str(file_path),
                "file_size_mb": round(file_size_mb, 2),
                "message": f"Downloaded {scene_id} ({file_size_mb:.2f} MB)",
            }

        except Exception as e:
            logger.error("Landsat download failed: %s", e, exc_info=True)
            step.mark_failure(str(e))
            return {
                "success": False,
                "scene_id": scene_id,
                "error": str(e),
                "message": f"Download failed: {e}",
            }


# ─────────────────────────────────────────────────────────────────────────────
# Sentinel Tools
# ─────────────────────────────────────────────────────────────────────────────


@tool(args_schema=SearchImageryInput)
async def search_sentinel_tool(
    west: float,
    south: float,
    east: float,
    north: float,
    start_date: str,
    end_date: str,
    max_cloud_cover: float = 100.0,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> dict[str, Any]:
    """
    Search for available Sentinel-2 satellite imagery.

    IMPORTANT: Only call this tool AFTER the user has confirmed they want
    Sentinel-2.  If the user did not specify a satellite, ask first.

    Searches the Copernicus Data Space for Sentinel-2 MSI Level-2A
    (surface reflectance) scenes matching the specified criteria.
    Returns ALL matching scenes sorted by lowest cloud cover.

    Leave max_cloud_cover at 100 (default) unless the user explicitly
    requests a cloud cover limit.
    """
    sid = get_session_id(config)
    with track_step("scene_search", satellite="sentinel", session_id=sid) as step:
        try:
            bbox = BoundingBox(west=west, south=south, east=east, north=north)
            start = date.fromisoformat(start_date)
            end = date.fromisoformat(end_date)

            async with SentinelClient() as client:
                result = await client.search(
                    bbox=bbox,
                    start_date=start,
                    end_date=end,
                    max_cloud_cover=max_cloud_cover,
                )

            scenes_summary = [
                {
                    "scene_id": scene.scene_id,
                    "satellite": "sentinel",
                    "acquisition_date": scene.acquisition_date.isoformat(),
                    "cloud_cover": scene.cloud_cover,
                    "collection": scene.collection,
                }
                for scene in result.scenes
            ]

            step.set_metadata(
                scenes_found=result.total_count,
                search_start_date=start_date,
                search_end_date=end_date,
            )
            return {
                "success": True,
                "total_found": result.total_count,
                "search_dates": {"start": start_date, "end": end_date},
                "scenes": scenes_summary,
            }

        except Exception as e:
            logger.error("Sentinel search failed: %s", e)
            step.mark_failure(str(e))
            return {
                "success": False,
                "error": str(e),
                "scenes": [],
                "message": f"Search failed: {e}",
            }


@tool(args_schema=DownloadSceneInput)
async def download_sentinel_tool(
    scene_id: str,
    satellite: str = "sentinel",
    entity_id: str | None = None,
    collection: str | None = None,
    west: float | None = None,
    south: float | None = None,
    east: float | None = None,
    north: float | None = None,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> dict[str, Any]:
    """
    Download a Sentinel-2 scene to local storage.

    Downloads all 12 Sentinel-2 spectral bands for the specified area.
    The scene_id must be from a previous search_sentinel_tool result.

    IMPORTANT: You MUST provide the bounding box (west, south, east, north)
    for Sentinel downloads. Use the same bbox from the search.

    Returns the path to the downloaded file and download statistics.

    Use this tool after search_sentinel_tool to actually download the imagery.
    """
    sid = get_session_id(config)
    output_dir = session_raw_dir(sid, "sentinel")

    # Sentinel downloads require a bbox
    bbox = None
    if west is not None and south is not None and east is not None and north is not None:
        bbox = BoundingBox(west=west, south=south, east=east, north=north)

    with track_step("scene_download", satellite="sentinel", session_id=sid) as step:
        try:
            async with SentinelClient() as client:
                file_path = await client.download(
                    scene_id=scene_id,
                    output_dir=output_dir,
                    bbox=bbox,
                )

            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            step.set_metadata(scene_id=scene_id, file_size_mb=round(file_size_mb, 2))

            return {
                "success": True,
                "scene_id": scene_id,
                "file_path": str(file_path),
                "file_size_mb": round(file_size_mb, 2),
                "message": f"Downloaded {scene_id} ({file_size_mb:.2f} MB)",
            }

        except Exception as e:
            logger.error("Sentinel download failed: %s", e)
            step.mark_failure(str(e))
            return {
                "success": False,
                "scene_id": scene_id,
                "error": str(e),
                "message": f"Download failed: {e}",
            }


# ─────────────────────────────────────────────────────────────────────────────
# Sentinel Index Download Tool (server-side computation)
# ─────────────────────────────────────────────────────────────────────────────


@tool(args_schema=SentinelIndexInput)
async def download_sentinel_index_tool(
    scene_id: str,
    index_names: list[str],
    west: float,
    south: float,
    east: float,
    north: float,
    resolution: int = 10,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> dict[str, Any]:
    """
    Compute spectral indices for a Sentinel-2 scene and download the result.

    This tool calculates indices (NDVI, EVI, SAVI, NDWI, NBR, NDBI, etc.)
    directly on Sentinel Hub servers and downloads the computed GeoTIFF.
    This is much faster than downloading all raw bands.

    Use this tool for Sentinel-2 INSTEAD of download_sentinel_tool +
    crop + compute.  Pass the scene_id from search_sentinel_tool and
    the bounding box.

    The output GeoTIFF contains one band per requested index as FLOAT32
    values (e.g. NDVI ranges from -1 to 1).
    """
    sid = get_session_id(config)
    output_dir = session_processed_dir(sid, "sentinel")

    bbox = BoundingBox(west=west, south=south, east=east, north=north)

    with track_step(
        "sentinel_index_download",
        satellite="sentinel",
        index_name=",".join(index_names),
        session_id=sid,
    ) as step:
        try:
            async with SentinelClient() as client:
                file_path = await client.download_indices(
                    scene_id=scene_id,
                    output_dir=output_dir,
                    bbox=bbox,
                    indices=index_names,
                    resolution=resolution,
                )

            file_size_mb = file_path.stat().st_size / (1024 * 1024)

            step.set_metadata(
                scene_id=scene_id,
                indices=index_names,
                file_size_mb=round(file_size_mb, 2),
            )

            # ── Debug logging for Sentinel indices ──────────────────
            from spectral_agent.tools.indices.spectral_indices import (
                SPECTRAL_INDICES,
            )

            dbg = get_debug_logger()
            debug_entries = []
            for idx_name in index_names:
                idx_upper = idx_name.upper()
                idx_def = SPECTRAL_INDICES.get(idx_upper)
                sentinel_bands = {b: b for b in idx_def.bands} if idx_def else {}
                extra: dict[str, Any] = {"parameters": {"L": 0.5}} if idx_upper == "SAVI" else {}
                rec = dbg.log_index_computation(
                    satellite="sentinel",
                    sensor="Sentinel-2",
                    index_name=idx_upper,
                    bands_used=sentinel_bands,
                    evalscript=(idx_def.formula if idx_def else ""),
                    output_path=str(file_path),
                    stats={"file_size_mb": round(file_size_mb, 2)},
                    session_id=sid,
                    scene_id=scene_id,
                    resolution=resolution,
                    **extra,
                )
                debug_entry: dict[str, Any] = {
                    "index": idx_upper,
                    "formula": rec.formula,
                    "bands": list(sentinel_bands.values()),
                    "satellite": "sentinel",
                    "sensor": "Sentinel-2",
                    "evalscript_formula": rec.evalscript,
                }
                if idx_upper == "SAVI":
                    debug_entry["parameters"] = {"L": 0.5}
                debug_entries.append(debug_entry)

            return {
                "success": True,
                "scene_id": scene_id,
                "satellite": "sentinel",
                "sensor": "Sentinel-2",
                "indices_computed": index_names,
                "output_path": str(file_path),
                "file_size_mb": round(file_size_mb, 2),
                "debug_info": debug_entries,
                "debug_log_path": str(dbg.log_path) if dbg.log_path else None,
                "message": (
                    f"Computed {', '.join(index_names)} for {scene_id} "
                    f"using Sentinel-2 bands ({file_size_mb:.2f} MB)"
                ),
            }

        except Exception as e:
            logger.error("Sentinel index download failed: %s", e)
            step.mark_failure(str(e))
            return {
                "success": False,
                "scene_id": scene_id,
                "error": str(e),
                "message": f"Sentinel index download failed: {e}",
            }


# ─────────────────────────────────────────────────────────────────────────────
# Unified Multi-Satellite Tool
# ─────────────────────────────────────────────────────────────────────────────


@tool(args_schema=MultiSatelliteSearchInput)
async def search_satellite_imagery_tool(
    west: float,
    south: float,
    east: float,
    north: float,
    start_date: str,
    end_date: str,
    max_cloud_cover: float = 100.0,
    satellites: list[str] | None = None,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> dict[str, Any]:
    """
    Search for satellite imagery from one or more satellites.

    The 'satellites' parameter MUST be provided explicitly — there is no
    default. Pass ['landsat'], ['sentinel'], or ['landsat', 'sentinel']
    only when the user requests a comparison.

    Prefer using search_landsat_tool or search_sentinel_tool directly
    when the satellite is already known.
    """
    if not satellites:
        return {
            "success": False,
            "total_found": 0,
            "scenes": [],
            "errors": [
                "'satellites' parameter is required. Ask the user: 'Landsat or Sentinel-2?'"
            ],
        }
    sid = get_session_id(config)
    with track_step("multi_satellite_search", session_id=sid) as step:
        bbox = BoundingBox(west=west, south=south, east=east, north=north)
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)

        all_scenes: list[dict[str, Any]] = []
        errors: list[str] = []

        if "landsat" in satellites:
            try:
                async with LandsatClient() as client:
                    result = await client.search(
                        bbox=bbox,
                        start_date=start,
                        end_date=end,
                        max_cloud_cover=max_cloud_cover,
                    )
                    for scene in result.scenes:
                        all_scenes.append(
                            {
                                "scene_id": scene.scene_id,
                                "entity_id": scene.raw_metadata.get("entityId", ""),
                                "satellite": "landsat",
                                "acquisition_date": scene.acquisition_date.isoformat(),
                                "cloud_cover": scene.cloud_cover,
                                "collection": scene.collection,
                            }
                        )
            except Exception as e:
                errors.append(f"Landsat search error: {e}")

        if "sentinel" in satellites:
            try:
                async with SentinelClient() as client:
                    result = await client.search(
                        bbox=bbox,
                        start_date=start,
                        end_date=end,
                        max_cloud_cover=max_cloud_cover,
                    )
                    for scene in result.scenes:
                        all_scenes.append(
                            {
                                "scene_id": scene.scene_id,
                                "satellite": "sentinel",
                                "acquisition_date": scene.acquisition_date.isoformat(),
                                "cloud_cover": scene.cloud_cover,
                                "collection": scene.collection,
                            }
                        )
            except Exception as e:
                errors.append(f"Sentinel search error: {e}")

        all_scenes.sort(key=lambda s: s["cloud_cover"])

        step.set_metadata(scenes_found=len(all_scenes), errors=errors or None)
        return {
            "success": len(errors) == 0,
            "total_found": len(all_scenes),
            "scenes": all_scenes,
            "errors": errors if errors else None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience registry
# ─────────────────────────────────────────────────────────────────────────────


def get_ingestion_tools() -> list:
    """Return all ingestion-related LangChain tools."""
    return [
        search_landsat_tool,
        search_sentinel_tool,
        download_landsat_tool,
        download_sentinel_tool,
        download_sentinel_index_tool,
        search_satellite_imagery_tool,
    ]

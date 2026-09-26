"""
LangGraph workflow for satellite imagery ingestion.

This module defines a stateful graph that orchestrates the complete
ingestion process: validation → search → download → finalization.

The graph can be executed standalone or integrated into larger agent workflows.
"""

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from spectral_agent.config import get_settings
from spectral_agent.schemas.imagery import (
    BoundingBox,
    DownloadResult,
    IngestionResponse,
)
from spectral_agent.tools.ingestion.landsat import LandsatClient
from spectral_agent.tools.ingestion.sentinel import SentinelClient

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# State Definition
# ─────────────────────────────────────────────────────────────────────────────


class IngestionState(TypedDict, total=False):
    """
    State schema for the ingestion workflow graph.

    This state is passed through each node and accumulates results.
    All fields are optional to allow incremental updates.
    """

    # Input parameters
    bbox: dict  # BoundingBox as dict
    start_date: str
    end_date: str
    satellites: list[str]
    max_cloud_cover: float
    max_scenes: int | None

    # Spectral indices to compute (if None, downloads all bands)
    indices: list[str] | None  # e.g., ["NDVI", "NDWI", "NBR"]
    include_rgb: bool  # Include RGB bands with indices

    # Validation
    is_valid: bool
    validation_errors: list[str]

    # Search results
    landsat_scenes: list[dict]
    sentinel_scenes: list[dict]
    total_scenes_found: int

    # Download results
    landsat_downloads: list[dict]
    sentinel_downloads: list[dict]
    total_downloaded: int
    total_failed: int

    # Final output
    success: bool
    message: str
    errors: list[str]

    # Execution metadata
    started_at: str
    completed_at: str
    duration_seconds: float


# ─────────────────────────────────────────────────────────────────────────────
# Node Functions
# ─────────────────────────────────────────────────────────────────────────────


async def validate_input(state: IngestionState) -> IngestionState:
    """
    Validate the input parameters for the ingestion workflow.

    Checks:
    - Valid bounding box coordinates
    - Valid date range
    - At least one satellite selected
    - Credentials available for selected satellites
    """
    errors = []
    settings = get_settings()

    # Validate bounding box
    try:
        bbox = state.get("bbox", {})
        if not bbox:
            errors.append("Bounding box is required")
        else:
            BoundingBox(**bbox)  # Validate with pydantic
    except Exception as e:
        errors.append(f"Invalid bounding box: {str(e)}")

    # Validate dates
    try:
        start = date.fromisoformat(state.get("start_date", ""))
        end = date.fromisoformat(state.get("end_date", ""))
        if start > end:
            errors.append("start_date must be before end_date")
        if end > date.today():
            errors.append("end_date cannot be in the future")
    except ValueError as e:
        errors.append(f"Invalid date format: {str(e)}")

    # Validate satellites
    satellites = state.get("satellites", [])
    if not satellites:
        errors.append("At least one satellite must be selected")

    # Check credentials
    if "landsat" in satellites and not settings.has_usgs_credentials():
        errors.append(
            "USGS credentials required for Landsat (SPECTRAL_USGS_USERNAME, SPECTRAL_USGS_TOKEN)"
        )

    if "sentinel" in satellites and not settings.has_copernicus_credentials():
        errors.append(
            "Copernicus credentials required for Sentinel (SPECTRAL_COPERNICUS_CLIENT_ID, SPECTRAL_COPERNICUS_CLIENT_SECRET)"
        )

    return {
        **state,
        "is_valid": len(errors) == 0,
        "validation_errors": errors,
        "errors": state.get("errors", []) + errors,
    }


async def search_landsat(state: IngestionState) -> IngestionState:
    """Search for Landsat scenes matching the criteria."""

    if "landsat" not in state.get("satellites", []):
        return {**state, "landsat_scenes": []}

    try:
        bbox = BoundingBox(**state["bbox"])
        start = date.fromisoformat(state["start_date"])
        end = date.fromisoformat(state["end_date"])
        max_cloud = state.get("max_cloud_cover", 100.0)

        async with LandsatClient() as client:
            result = await client.search(
                bbox=bbox,
                start_date=start,
                end_date=end,
                max_cloud_cover=max_cloud,
            )

        scenes = [
            {
                "scene_id": s.scene_id,  # displayId (e.g. LC09_L2SP_...)
                "entity_id": s.raw_metadata.get("entityId", s.scene_id),  # For API calls
                "acquisition_date": s.acquisition_date.isoformat(),
                "cloud_cover": s.cloud_cover,
                "collection": s.collection,
            }
            for s in result.scenes
        ]

        logger.info("Found %d Landsat scenes", len(scenes))

        return {
            **state,
            "landsat_scenes": scenes,
        }

    except Exception as e:
        logger.error("Landsat search failed: %s", e)
        return {
            **state,
            "landsat_scenes": [],
            "errors": state.get("errors", []) + [f"Landsat search failed: {str(e)}"],
        }


async def search_sentinel(state: IngestionState) -> IngestionState:
    """Search for Sentinel-2 scenes matching the criteria."""

    if "sentinel" not in state.get("satellites", []):
        return {**state, "sentinel_scenes": []}

    try:
        bbox = BoundingBox(**state["bbox"])
        start = date.fromisoformat(state["start_date"])
        end = date.fromisoformat(state["end_date"])
        max_cloud = state.get("max_cloud_cover", 100.0)

        async with SentinelClient() as client:
            result = await client.search(
                bbox=bbox,
                start_date=start,
                end_date=end,
                max_cloud_cover=max_cloud,
            )

        scenes = [
            {
                "scene_id": s.scene_id,
                "acquisition_date": s.acquisition_date.isoformat(),
                "cloud_cover": s.cloud_cover,
                "collection": s.collection,
                "product_name": s.raw_metadata.get("Name", ""),
            }
            for s in result.scenes
        ]

        logger.info("Found %d Sentinel-2 scenes", len(scenes))

        return {
            **state,
            "sentinel_scenes": scenes,
        }

    except Exception as e:
        logger.error("Sentinel search failed: %s", e)
        return {
            **state,
            "sentinel_scenes": [],
            "errors": state.get("errors", []) + [f"Sentinel search failed: {str(e)}"],
        }


async def aggregate_search_results(state: IngestionState) -> IngestionState:
    """Aggregate search results from all satellites."""

    landsat_count = len(state.get("landsat_scenes", []))
    sentinel_count = len(state.get("sentinel_scenes", []))
    total = landsat_count + sentinel_count

    logger.info(
        "Total scenes found: %d (Landsat: %d, Sentinel: %d)", total, landsat_count, sentinel_count
    )

    return {
        **state,
        "total_scenes_found": total,
    }


async def download_landsat_scenes(state: IngestionState) -> IngestionState:
    """Download Landsat scenes."""

    scenes = state.get("landsat_scenes", [])
    if not scenes:
        return {**state, "landsat_downloads": []}

    settings = get_settings()
    output_dir = settings.landsat_raw_dir
    downloads = []

    async with LandsatClient() as client:
        for scene in scenes:
            scene_id = scene["scene_id"]  # Display name
            entity_id = scene.get("entity_id", scene_id)  # API entity ID
            collection = scene.get("collection")
            try:
                logger.info("Downloading Landsat scene: %s", scene_id)
                file_path = await client.download(
                    scene_id=entity_id,  # API needs entityId
                    output_dir=output_dir,
                    collection=collection,
                )

                downloads.append(
                    {
                        "scene_id": scene_id,
                        "success": True,
                        "file_path": str(file_path),
                        "file_size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
                    }
                )

            except Exception as e:
                logger.error("Failed to download %s: %s", scene_id, e)
                downloads.append(
                    {
                        "scene_id": scene_id,
                        "success": False,
                        "error": str(e),
                    }
                )

    return {
        **state,
        "landsat_downloads": downloads,
    }


async def download_sentinel_scenes(state: IngestionState) -> IngestionState:
    """
    Download Sentinel-2 scenes.

    If 'indices' is specified in state, downloads computed indices.
    Otherwise downloads all 12 spectral bands.
    """
    scenes = state.get("sentinel_scenes", [])
    if not scenes:
        return {**state, "sentinel_downloads": []}

    settings = get_settings()
    output_dir = settings.sentinel_raw_dir
    downloads = []

    # Get the search bbox for downloading (Sentinel Hub limits to 2500x2500 pixels)
    bbox = BoundingBox(**state["bbox"])

    # Check if we should download indices or all bands
    indices = state.get("indices")  # e.g., ["NDVI", "NDWI"] or None

    async with SentinelClient() as client:
        for scene in scenes:
            scene_id = scene["scene_id"]
            try:
                if indices:
                    # Download computed indices
                    logger.info("Downloading Sentinel scene: %s", scene_id)
                    logger.info("Computing indices: %s", indices)
                    file_path = await client.download_indices(
                        scene_id=scene_id,
                        output_dir=output_dir,
                        bbox=bbox,
                        indices=indices,
                        resolution=10,
                        include_rgb=state.get("include_rgb", False),
                    )
                else:
                    # Download all spectral bands
                    logger.info("Downloading Sentinel scene: %s", scene_id)
                    logger.info("Downloading all 12 spectral bands for AOI")
                    file_path = await client.download(
                        scene_id=scene_id,
                        output_dir=output_dir,
                        bbox=bbox,
                        resolution=10,
                    )

                downloads.append(
                    {
                        "scene_id": scene_id,
                        "success": True,
                        "file_path": str(file_path),
                        "file_size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
                        "indices": indices,
                    }
                )

            except Exception as e:
                logger.error("Failed to download %s: %s", scene_id, e)
                downloads.append(
                    {
                        "scene_id": scene_id,
                        "success": False,
                        "error": str(e),
                    }
                )

    return {
        **state,
        "sentinel_downloads": downloads,
    }


async def finalize(state: IngestionState) -> IngestionState:
    """Finalize the workflow and compute summary statistics."""

    landsat_downloads = state.get("landsat_downloads", [])
    sentinel_downloads = state.get("sentinel_downloads", [])

    all_downloads = landsat_downloads + sentinel_downloads
    successful = [d for d in all_downloads if d.get("success", False)]
    failed = [d for d in all_downloads if not d.get("success", True)]

    total_size_mb = sum(d.get("file_size_mb", 0) for d in successful)

    # Determine overall success
    has_downloads = len(successful) > 0
    has_critical_errors = len(state.get("validation_errors", [])) > 0
    success = has_downloads and not has_critical_errors

    if success:
        message = f"Successfully downloaded {len(successful)} scenes ({total_size_mb:.2f} MB total)"
    elif has_critical_errors:
        message = "Workflow failed due to validation errors"
    else:
        message = f"No scenes were downloaded. {len(failed)} downloads failed."

    completed_at = datetime.now()
    started_at = datetime.fromisoformat(state.get("started_at", completed_at.isoformat()))
    duration = (completed_at - started_at).total_seconds()

    return {
        **state,
        "success": success,
        "message": message,
        "total_downloaded": len(successful),
        "total_failed": len(failed),
        "completed_at": completed_at.isoformat(),
        "duration_seconds": duration,
    }


def handle_validation_error(state: IngestionState) -> IngestionState:
    """Handle validation errors - this is a terminal state."""
    return {
        **state,
        "success": False,
        "message": f"Validation failed: {'; '.join(state.get('validation_errors', []))}",
        "completed_at": datetime.now().isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Conditional Routing
# ─────────────────────────────────────────────────────────────────────────────


def should_continue_after_validation(
    state: IngestionState,
) -> Literal["search", "error"]:
    """Decide next step after validation."""
    if state.get("is_valid", False):
        return "search"
    return "error"


def should_download(state: IngestionState) -> Literal["download", "finalize"]:
    """Decide whether to proceed with downloads."""
    total_found = state.get("total_scenes_found", 0)
    if total_found > 0:
        return "download"
    return "finalize"


# ─────────────────────────────────────────────────────────────────────────────
# Graph Construction
# ─────────────────────────────────────────────────────────────────────────────


def create_ingestion_graph() -> CompiledStateGraph:
    """
    Create the ingestion workflow graph.

    Graph structure:

        START
          │
          ▼
        validate_input
          │
          ├─[valid]──► search_landsat ──┐
          │                             │
          │            search_sentinel ◄┘
          │                  │
          │                  ▼
          │         aggregate_results
          │                  │
          │        ┌────────┴────────┐
          │        │                  │
          │  [has scenes]      [no scenes]
          │        │                  │
          │        ▼                  │
          │   download_landsat        │
          │        │                  │
          │   download_sentinel       │
          │        │                  │
          │        ▼                  │
          │     finalize ◄────────────┘
          │        │
          └─[error]─► handle_error
                           │
                           ▼
                          END

    Returns:
        StateGraph: Compiled workflow graph
    """

    # Create the graph
    graph = StateGraph(IngestionState)

    # Add nodes
    graph.add_node("validate", validate_input)
    graph.add_node("search_landsat", search_landsat)
    graph.add_node("search_sentinel", search_sentinel)
    graph.add_node("aggregate", aggregate_search_results)
    graph.add_node("download_landsat", download_landsat_scenes)
    graph.add_node("download_sentinel", download_sentinel_scenes)
    graph.add_node("finalize", finalize)
    graph.add_node("handle_error", handle_validation_error)

    # Set entry point
    graph.set_entry_point("validate")

    # Add conditional edge after validation
    graph.add_conditional_edges(
        "validate",
        should_continue_after_validation,
        {
            "search": "search_landsat",
            "error": "handle_error",
        },
    )

    # Search flow (sequential for now, could be parallel)
    graph.add_edge("search_landsat", "search_sentinel")
    graph.add_edge("search_sentinel", "aggregate")

    # Conditional download
    graph.add_conditional_edges(
        "aggregate",
        should_download,
        {
            "download": "download_landsat",
            "finalize": "finalize",
        },
    )

    # Download flow
    graph.add_edge("download_landsat", "download_sentinel")
    graph.add_edge("download_sentinel", "finalize")

    # Terminal nodes
    graph.add_edge("finalize", END)
    graph.add_edge("handle_error", END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Workflow Execution
# ─────────────────────────────────────────────────────────────────────────────


async def run_ingestion_workflow(
    bbox: BoundingBox | dict,
    start_date: date | str,
    end_date: date | str,
    satellites: list[str] | None = None,
    max_cloud_cover: float = 100.0,
    max_scenes: int | None = None,
) -> IngestionResponse:
    """
    Execute the ingestion workflow.

    This is the main entry point for running the ingestion graph.

    Args:
        bbox: Area of interest (BoundingBox or dict)
        start_date: Start of date range
        end_date: End of date range
        satellites: List of satellites ['landsat', 'sentinel']
        max_cloud_cover: Maximum cloud cover percentage
        max_scenes: Maximum scenes to download per satellite

    Returns:
        IngestionResponse: Workflow results

    Example:
        >>> response = await run_ingestion_workflow(
        ...     bbox={"west": -122.5, "south": 37.5, "east": -122.0, "north": 38.0},
        ...     start_date="2024-01-01",
        ...     end_date="2024-06-01",
        ...     satellites=["landsat", "sentinel"],
        ...     max_cloud_cover=15.0,
        ... )
        >>> print(response.message)
    """
    # Normalize inputs
    if isinstance(bbox, BoundingBox):
        bbox_dict = bbox.model_dump()
    else:
        bbox_dict = bbox

    if isinstance(start_date, date):
        start_str = start_date.isoformat()
    else:
        start_str = start_date

    if isinstance(end_date, date):
        end_str = end_date.isoformat()
    else:
        end_str = end_date

    satellites = satellites or ["landsat", "sentinel"]

    # Initial state
    initial_state: IngestionState = {
        "bbox": bbox_dict,
        "start_date": start_str,
        "end_date": end_str,
        "satellites": satellites,
        "max_cloud_cover": max_cloud_cover,
        "max_scenes": max_scenes,
        "started_at": datetime.now().isoformat(),
        "errors": [],
    }

    # Create and run the graph
    graph = create_ingestion_graph()

    logger.info(
        "Starting ingestion workflow for bbox=%s, dates=%s to %s", bbox_dict, start_str, end_str
    )

    # Execute the workflow
    final_state = await graph.ainvoke(initial_state)

    logger.info("Workflow completed: %s", final_state.get("message"))

    # Convert to response model
    return IngestionResponse(
        success=final_state.get("success", False),
        message=final_state.get("message", "Unknown error"),
        landsat_results=[
            DownloadResult(
                scene_id=d["scene_id"],
                success=d.get("success", False),
                file_path=Path(d["file_path"]) if d.get("file_path") else None,
                file_size_bytes=(
                    int(d.get("file_size_mb", 0) * 1024 * 1024) if d.get("file_size_mb") else None
                ),
                error_message=d.get("error"),
            )
            for d in final_state.get("landsat_downloads", [])
        ],
        sentinel_results=[
            DownloadResult(
                scene_id=d["scene_id"],
                success=d.get("success", False),
                file_path=Path(d["file_path"]) if d.get("file_path") else None,
                file_size_bytes=(
                    int(d.get("file_size_mb", 0) * 1024 * 1024) if d.get("file_size_mb") else None
                ),
                error_message=d.get("error"),
            )
            for d in final_state.get("sentinel_downloads", [])
        ],
        total_scenes_found=final_state.get("total_scenes_found", 0),
        total_scenes_downloaded=final_state.get("total_downloaded", 0),
        total_download_time_seconds=final_state.get("duration_seconds", 0.0),
        errors=final_state.get("errors", []),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Graph Visualization
# ─────────────────────────────────────────────────────────────────────────────


def get_graph_mermaid() -> str:
    """
    Get a Mermaid diagram representation of the ingestion graph.

    Returns:
        str: Mermaid diagram code
    """
    return """
```mermaid
graph TD
    START([Start]) --> validate[Validate Input]
    validate -->|valid| search_landsat[Search Landsat]
    validate -->|invalid| handle_error[Handle Error]
    search_landsat --> search_sentinel[Search Sentinel]
    search_sentinel --> aggregate[Aggregate Results]
    aggregate -->|scenes found| download_landsat[Download Landsat]
    aggregate -->|no scenes| finalize[Finalize]
    download_landsat --> download_sentinel[Download Sentinel]
    download_sentinel --> finalize
    finalize --> END([End])
    handle_error --> END
```
"""

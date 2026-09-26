"""
Central tool registry — single place to collect **all** LangChain tools.

Other modules should use :func:`get_all_tools` instead of importing
individual tool objects directly.
"""

from __future__ import annotations

from spectral_agent.tools.ingestion.lc_tools import (
    DownloadSceneInput,
    MultiSatelliteSearchInput,
    # Re-export schemas
    SearchImageryInput,
    SentinelIndexInput,
    download_landsat_tool,
    download_sentinel_index_tool,
    download_sentinel_tool,
    get_ingestion_tools,
    # Re-export individual tools for backward compatibility
    search_landsat_tool,
    search_satellite_imagery_tool,
    search_sentinel_tool,
)
from spectral_agent.tools.raster.lc_tools import (
    ComputeSpectralIndexInput,
    CropLandsatBandsInput,
    ListCachedBandsInput,
    compute_spectral_index_tool,
    crop_landsat_bands_tool,
    get_raster_tools,
    list_available_indices_tool,
    list_cached_bands_tool,
)
from spectral_agent.tools.visualization.lc_tools import (
    GenerateMapInput,
    generate_thematic_map_tool,
    get_visualization_tools,
)


def get_all_tools() -> list:
    """Return every tool available for the spectral agent."""
    return get_ingestion_tools() + get_raster_tools() + get_visualization_tools()


__all__ = [
    # Registries
    "get_all_tools",
    "get_ingestion_tools",
    "get_raster_tools",
    "get_visualization_tools",
    # Ingestion tools
    "search_landsat_tool",
    "search_sentinel_tool",
    "download_landsat_tool",
    "download_sentinel_tool",
    "download_sentinel_index_tool",
    "search_satellite_imagery_tool",
    # Raster tools
    "crop_landsat_bands_tool",
    "compute_spectral_index_tool",
    "list_cached_bands_tool",
    "list_available_indices_tool",
    # Visualisation tools
    "generate_thematic_map_tool",
    # Schemas
    "SearchImageryInput",
    "DownloadSceneInput",
    "MultiSatelliteSearchInput",
    "SentinelIndexInput",
    "CropLandsatBandsInput",
    "ComputeSpectralIndexInput",
    "ListCachedBandsInput",
    "GenerateMapInput",
]

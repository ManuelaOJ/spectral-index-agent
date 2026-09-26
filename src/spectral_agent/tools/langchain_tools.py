"""
LangChain tool wrappers — backward-compatibility re-exports.

.. deprecated::
    Import from :mod:`spectral_agent.tools._registry` or from the
    sub-module ``lc_tools`` files directly.
"""

from spectral_agent.tools._registry import (  # noqa: F401
    ComputeSpectralIndexInput,
    CropLandsatBandsInput,
    DownloadSceneInput,
    GenerateMapInput,
    ListCachedBandsInput,
    MultiSatelliteSearchInput,
    SearchImageryInput,
    compute_spectral_index_tool,
    # Raster
    crop_landsat_bands_tool,
    download_landsat_tool,
    download_sentinel_tool,
    # Visualisation
    generate_thematic_map_tool,
    get_all_tools,
    # Registries
    get_ingestion_tools,
    get_raster_tools,
    get_visualization_tools,
    list_available_indices_tool,
    list_cached_bands_tool,
    # Ingestion
    search_landsat_tool,
    search_satellite_imagery_tool,
    search_sentinel_tool,
)

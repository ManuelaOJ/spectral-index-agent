"""
Raster processing tools for Landsat band extraction, cropping, index
computation, and caching.

This package provides:
  - ``LandsatProcessor``  – extract .tar archives, crop bands to an AOI
  - ``BandCache``         – track previously-cropped bands to avoid duplication
  - ``IndexCalculator``   – compute spectral indices from cropped bands
  - ``IndexResult``       – metadata dataclass for computed index outputs
"""

from .landsat_processor import LandsatProcessor
from .band_cache import BandCache
from .index_calculator import (
    IndexResult,
    compute_index,
    compute_and_save,
    compute_indices,
    list_supported_indices,
    INDEX_FORMULAS,
    INDEX_VALUE_RANGES,
    LANDSAT_C2_L2_SCALE,
    LANDSAT_C2_L2_OFFSET,
)

__all__ = [
    "LandsatProcessor",
    "BandCache",
    "IndexResult",
    "compute_index",
    "compute_and_save",
    "compute_indices",
    "list_supported_indices",
    "INDEX_FORMULAS",
    "INDEX_VALUE_RANGES",
    "LANDSAT_C2_L2_SCALE",
    "LANDSAT_C2_L2_OFFSET",
]

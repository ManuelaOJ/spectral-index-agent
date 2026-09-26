"""
Visualization tools for spectral index rasters.

This package provides:
  - ``generate_thematic_map``     — static PNG map (matplotlib)
  - ``generate_interactive_map``  — interactive HTML map (folium)
  - ``MapConfig``                 — configuration for static maps
  - ``MapResult``                 — metadata from static map generation
  - ``InteractiveMapResult``      — metadata from interactive map generation
"""

from .interactive_map import (
    InteractiveMapResult,
    generate_interactive_map,
)
from .thematic_map import (
    INDEX_CMAPS,
    INDEX_FULL_NAMES,
    MapConfig,
    MapResult,
    generate_thematic_map,
)

__all__ = [
    "generate_thematic_map",
    "generate_interactive_map",
    "MapConfig",
    "MapResult",
    "InteractiveMapResult",
    "INDEX_CMAPS",
    "INDEX_FULL_NAMES",
]

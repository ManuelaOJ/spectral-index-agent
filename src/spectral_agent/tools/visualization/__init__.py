"""
Visualization tools for spectral index rasters.

This package provides:
  - ``generate_thematic_map``     — static PNG map (matplotlib)
  - ``generate_interactive_map``  — interactive HTML map (folium)
  - ``MapConfig``                 — configuration for static maps
  - ``MapResult``                 — metadata from static map generation
  - ``InteractiveMapResult``      — metadata from interactive map generation
"""

from .thematic_map import (
    generate_thematic_map,
    MapConfig,
    MapResult,
    INDEX_CMAPS,
    INDEX_FULL_NAMES,
)
from .interactive_map import (
    generate_interactive_map,
    InteractiveMapResult,
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

"""
Phase 1 processing pipeline.

Layers:
  1. Extraction  – LLM parses natural language into ExtractionResult
  2. Normalization – resolves dates, geometry, validates consistency
  3. Rules Engine – applies sensor priority, cloud cover strategy
  4. Request Builder – produces the final SpectralIndexRequest

Utilities:
  - geo_input – reads KML, Shapefile, GeoJSON files into ExtractedGeometry
"""

from .extraction import extract_request
from .normalization import normalize_request
from .rules_engine import apply_rules
from .request_builder import build_request
from .geo_input import read_geometry_file, geometry_to_bbox, geometry_to_geojson

__all__ = [
    "extract_request",
    "normalize_request",
    "apply_rules",
    "build_request",
    "read_geometry_file",
    "geometry_to_bbox",
    "geometry_to_geojson",
]

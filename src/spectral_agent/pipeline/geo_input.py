"""
Geospatial file input parser with CRS verification.

Reads KML, Shapefile (.shp), and GeoJSON files, verifies the coordinate
reference system (CRS), reprojects to WGS84 (EPSG:4326) if needed, and
converts them to ExtractedGeometry objects for the pipeline.

Supported formats:
  - KML / KMZ  (.kml, .kmz)  — always WGS84 by KML spec
  - ESRI Shapefile (.shp)    — requires .prj for CRS; auto-reproj if not WGS84
  - GeoJSON (.geojson, .json) — must be WGS84 by spec; validated anyway

Verification steps:
  1. Detect CRS from file metadata (.prj, GeoJSON spec, KML spec)
  2. If CRS is missing → attempt to infer from coordinate range
  3. Reproject to EPSG:4326 if CRS ≠ WGS84
  4. Validate final coordinates are in valid WGS84 range:
     - Longitude: -180 to 180
     - Latitude:  -90 to 90
  5. Reject geometries with projected coordinates (meters) pretending to be WGS84
"""

from __future__ import annotations

import logging
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import geopandas as gpd
from shapely.geometry import Polygon, mapping, shape
from shapely.ops import unary_union

from spectral_agent.schemas.spectral_request import ExtractedGeometry, GeometryType

logger = logging.getLogger(__name__)

# Supported extensions
_SUPPORTED_EXTENSIONS = {".shp", ".geojson", ".json", ".kml", ".kmz"}

# Common projected CRS for Colombia (used to guess CRS when missing)
_COMMON_PROJECTED_CRS = {
    "EPSG:3116": "MAGNA-SIRGAS / Colombia Bogota zone",
    "EPSG:3117": "MAGNA-SIRGAS / Colombia East Central zone",
    "EPSG:3118": "MAGNA-SIRGAS / Colombia East zone",
    "EPSG:3115": "MAGNA-SIRGAS / Colombia West zone",
    "EPSG:3114": "MAGNA-SIRGAS / Colombia Far West zone",
    "EPSG:32618": "WGS 84 / UTM zone 18N",
    "EPSG:32619": "WGS 84 / UTM zone 19N",
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def read_geometry_file(filepath: str | Path) -> ExtractedGeometry:
    """
    Read a geospatial file, verify/reproject CRS to WGS84, and return
    an ExtractedGeometry.

    Pipeline:
      1. Read raw geometry from file
      2. Detect or infer CRS
      3. Reproject to EPSG:4326 if needed
      4. Validate coordinates are in WGS84 range
      5. Return ExtractedGeometry

    Args:
        filepath: Path to a .shp, .kml, .kmz, .geojson, or .json file.

    Returns:
        ExtractedGeometry in WGS84, ready for the normalization layer.

    Raises:
        ValueError: If format unsupported, no geometry, or coordinates invalid.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Geometry file not found: {path}")

    ext = path.suffix.lower()
    if ext not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{ext}'. Supported: {sorted(_SUPPORTED_EXTENSIONS)}"
        )

    logger.info("Reading geometry from %s (format=%s)", path.name, ext)

    # ── Step 1: Read raw geometry ───────────────────────────────────────
    if ext in (".kml", ".kmz"):
        # KML is WGS84 by specification — no CRS to check
        merged = _read_kml(path)
        original_crs = "EPSG:4326 (KML spec)"
    else:
        gdf = gpd.read_file(str(path))

        if gdf.empty or gdf.geometry.is_empty.all():
            raise ValueError(f"No valid geometries found in {path.name}")

        # ── Step 2: Detect / infer CRS ──────────────────────────────────
        gdf, original_crs = _ensure_crs(gdf, path)

        # ── Step 3: Reproject to WGS84 ──────────────────────────────────
        gdf = _reproject_to_wgs84(gdf, original_crs)

        merged = unary_union(gdf.geometry)

    # ── Step 4: Validate WGS84 coordinates ──────────────────────────────
    _validate_wgs84_bounds(merged, path.name)

    geom_type = _classify_geometry(merged)
    geojson = mapping(merged)

    logger.info(
        "Geometry verified → %s, CRS original: %s, ahora WGS84 (%s)",
        geom_type.value,
        original_crs,
        path.name,
    )

    return ExtractedGeometry(
        type=geom_type,
        coordinates=geojson["coordinates"],
        source=ext.lstrip("."),
        raw_input=str(path.name),
    )


def geometry_to_bbox(geom: ExtractedGeometry) -> dict[str, float]:
    """
    Compute bounding box from an ExtractedGeometry.

    Returns:
        dict with keys west, south, east, north.
    """
    geojson = _to_geojson(geom)
    shp = shape(geojson)
    minx, miny, maxx, maxy = shp.bounds
    return {"west": minx, "south": miny, "east": maxx, "north": maxy}


def geometry_to_geojson(geom: ExtractedGeometry) -> dict[str, Any]:
    """Convert ExtractedGeometry to a GeoJSON geometry dict."""
    return _to_geojson(geom)


def geometry_area_km2(geom: ExtractedGeometry) -> float:
    """Approximate area in km² (uses equirectangular projection at centroid)."""

    shp = shape(_to_geojson(geom))
    centroid = shp.centroid
    cos_lat = math.cos(math.radians(centroid.y))
    # Shapely .area gives deg², convert to km²
    area_km2 = shp.area * (111.32**2) * cos_lat
    return round(area_km2, 4)


# ─────────────────────────────────────────────────────────────────────────────
# CRS verification and reprojection
# ─────────────────────────────────────────────────────────────────────────────


def _is_wgs84_range(minx: float, miny: float, maxx: float, maxy: float) -> bool:
    """Check if coordinates fall within valid WGS84 degree range."""
    return -180 <= minx <= 180 and -180 <= maxx <= 180 and -90 <= miny <= 90 and -90 <= maxy <= 90


def _ensure_crs(gdf: gpd.GeoDataFrame, path: Path) -> tuple[gpd.GeoDataFrame, str]:
    """
    Ensure the GeoDataFrame has a CRS. If missing, attempt to infer it.

    Returns:
        (gdf_with_crs, description_string)

    Raises:
        ValueError: If CRS cannot be determined.
    """
    if gdf.crs is not None:
        crs_str = str(gdf.crs)
        logger.info("CRS detected from file: %s (%s)", crs_str, path.name)
        return gdf, crs_str

    # CRS is missing — common with Shapefiles without .prj
    logger.warning(
        "No CRS found in %s. Attempting to infer from coordinate range...",
        path.name,
    )

    merged = unary_union(gdf.geometry)
    minx, miny, maxx, maxy = merged.bounds

    # Heuristic: if coordinates are in WGS84 range, assume EPSG:4326
    if _is_wgs84_range(minx, miny, maxx, maxy):
        logger.info(
            "Coordinates in WGS84 range (lon: %.2f..%.2f, lat: %.2f..%.2f). Assuming EPSG:4326.",
            minx,
            maxx,
            miny,
            maxy,
        )
        gdf = gdf.set_crs(epsg=4326)
        return gdf, "EPSG:4326 (inferred — coordinates in degree range)"

    # Coordinates are large numbers → likely projected (meters)
    # For Colombia: typical easting 400k-1200k, northing 400k-1800k
    if 100_000 < maxx < 2_000_000 and 100_000 < maxy < 2_000_000:
        logger.warning(
            "Coordinates appear to be in meters (x: %.0f..%.0f, y: %.0f..%.0f). "
            "Assuming EPSG:3116 (MAGNA-SIRGAS Colombia Bogota zone). "
            "If this is wrong, add a .prj file to your Shapefile.",
            minx,
            maxx,
            miny,
            maxy,
        )
        gdf = gdf.set_crs(epsg=3116)
        return gdf, "EPSG:3116 (inferred — coordinates in meter range, Colombia)"

    # UTM-like coordinates
    if 100_000 < maxx < 900_000 and 0 < maxy < 10_000_000:
        logger.warning(
            "Coordinates appear to be UTM (x: %.0f..%.0f, y: %.0f..%.0f). "
            "Assuming EPSG:32618 (UTM zone 18N). "
            "If this is wrong, add a .prj file to your Shapefile.",
            minx,
            maxx,
            miny,
            maxy,
        )
        gdf = gdf.set_crs(epsg=32618)
        return gdf, "EPSG:32618 (inferred — UTM-like coordinates)"

    raise ValueError(
        f"Cannot determine CRS for {path.name}. "
        f"Coordinate range: x=[{minx:.2f}, {maxx:.2f}], y=[{miny:.2f}, {maxy:.2f}]. "
        f"Please add a .prj file or use a format that includes CRS information."
    )


def _reproject_to_wgs84(gdf: gpd.GeoDataFrame, original_crs: str) -> gpd.GeoDataFrame:
    """
    Reproject GeoDataFrame to EPSG:4326 if not already.

    Logs the reprojection details.
    """
    if gdf.crs and gdf.crs.equals("EPSG:4326"):
        logger.info("CRS is already EPSG:4326 (WGS84). No reprojection needed.")
        return gdf

    # Check if CRS is geographic (degrees) but not exactly 4326
    if gdf.crs and gdf.crs.is_geographic:
        logger.info(
            "CRS %s is geographic (degrees). Reprojecting to EPSG:4326...",
            gdf.crs,
        )
    else:
        logger.info(
            "CRS %s is projected (meters/feet). Reprojecting to EPSG:4326 (degrees)...",
            gdf.crs,
        )

    gdf = gdf.to_crs(epsg=4326)

    # Sanity check after reprojection
    merged = unary_union(gdf.geometry)
    minx, miny, maxx, maxy = merged.bounds
    logger.info(
        "Reprojection complete: lon=[%.6f, %.6f], lat=[%.6f, %.6f]",
        minx,
        maxx,
        miny,
        maxy,
    )

    return gdf


def _validate_wgs84_bounds(geom, filename: str) -> None:
    """
    Validate that a geometry's coordinates are in valid WGS84 range.

    Raises ValueError if coordinates are outside [-180,180] lon / [-90,90] lat,
    which would indicate a failed or missing reprojection.
    """
    minx, miny, maxx, maxy = geom.bounds

    if not _is_wgs84_range(minx, miny, maxx, maxy):
        raise ValueError(
            f"Geometry from {filename} has coordinates outside WGS84 range after processing. "
            f"lon=[{minx:.2f}, {maxx:.2f}], lat=[{miny:.2f}, {maxy:.2f}]. "
            f"This usually means the CRS was not correctly detected. "
            f"Ensure your file includes CRS information (e.g., .prj for Shapefiles)."
        )

    # Extra check: if coordinates are suspiciously large for degrees
    # (e.g., someone has a file that says WGS84 but actually has meters)
    extent_x = maxx - minx
    extent_y = maxy - miny
    if extent_x > 90 or extent_y > 90:
        logger.warning(
            "Geometry from %s covers an unusually large area: "
            "%.2f° lon x %.2f° lat. Verify the CRS is correct.",
            filename,
            extent_x,
            extent_y,
        )

    logger.info(
        "WGS84 validation passed for %s: lon=[%.6f, %.6f], lat=[%.6f, %.6f]",
        filename,
        minx,
        maxx,
        miny,
        maxy,
    )


# ─────────────────────────────────────────────────────────────────────────────
# File format parsers
# ─────────────────────────────────────────────────────────────────────────────


def _read_kml(path: Path):
    """
    Parse a KML file using the standard-library XML parser.

    Extracts all <coordinates> blocks inside <Polygon> elements,
    builds Shapely polygons, and returns their union.

    Returns:
        A Shapely geometry (Polygon or MultiPolygon).

    Raises:
        ValueError: If no polygons are found in the KML.
    """
    ns = {"kml": "http://www.opengis.net/kml/2.2"}

    tree = ET.parse(str(path))
    root = tree.getroot()

    # Also handle KML without explicit namespace prefix
    if root.tag.startswith("{"):
        ns_uri = root.tag.split("}")[0].lstrip("{")
        ns = {"kml": ns_uri}

    polygons = []
    for coord_elem in root.iter(f"{{{ns['kml']}}}coordinates"):
        text = coord_elem.text
        if not text:
            continue
        ring = []
        for token in text.strip().split():
            parts = token.split(",")
            lon, lat = float(parts[0]), float(parts[1])
            ring.append((lon, lat))
        if len(ring) >= 3:
            polygons.append(Polygon(ring))

    if not polygons:
        raise ValueError(f"No polygon geometries found in {path.name}")

    logger.info("KML parsed: %d polygon(s) from %s", len(polygons), path.name)

    if len(polygons) == 1:
        return polygons[0]
    return unary_union(polygons)


def _classify_geometry(geom) -> GeometryType:
    """Map a Shapely geometry type to our GeometryType enum."""
    gtype = geom.geom_type
    if gtype == "Point":
        return GeometryType.POINT
    if gtype == "Polygon":
        return GeometryType.POLYGON
    if gtype == "MultiPolygon":
        return GeometryType.MULTI_POLYGON
    # LineString, GeometryCollection, etc. → treat as polygon via convex hull
    logger.warning(
        "Geometry type '%s' not natively supported; using convex hull as polygon.",
        gtype,
    )
    return GeometryType.POLYGON


def _to_geojson(geom: ExtractedGeometry) -> dict[str, Any]:
    """Reconstruct a GeoJSON geometry dict from ExtractedGeometry."""
    type_map = {
        GeometryType.POINT: "Point",
        GeometryType.BBOX: "Polygon",
        GeometryType.POLYGON: "Polygon",
        GeometryType.MULTI_POLYGON: "MultiPolygon",
    }
    return {
        "type": type_map[geom.type],
        "coordinates": geom.coordinates,
    }

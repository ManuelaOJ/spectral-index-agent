"""
Geospatial utility functions.

Common geospatial operations used throughout the platform.
"""

import math
from typing import Any

from spectral_agent.schemas.imagery import BoundingBox


def validate_bbox(
    west: float,
    south: float,
    east: float,
    north: float,
) -> BoundingBox:
    """
    Validate and create a BoundingBox.

    Args:
        west: Western longitude (-180 to 180)
        south: Southern latitude (-90 to 90)
        east: Eastern longitude (-180 to 180)
        north: Northern latitude (-90 to 90)

    Returns:
        BoundingBox: Validated bounding box

    Raises:
        ValueError: If coordinates are invalid
    """
    return BoundingBox(west=west, south=south, east=east, north=north)


def bbox_to_geojson(bbox: BoundingBox) -> dict[str, Any]:
    """
    Convert a BoundingBox to GeoJSON Polygon.

    Args:
        bbox: Bounding box to convert

    Returns:
        dict: GeoJSON Polygon geometry
    """
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [bbox.west, bbox.south],
                [bbox.east, bbox.south],
                [bbox.east, bbox.north],
                [bbox.west, bbox.north],
                [bbox.west, bbox.south],
            ]
        ],
    }


def calculate_area_km2(bbox: BoundingBox) -> float:
    """
    Calculate the approximate area of a bounding box in km².

    Uses the Haversine formula approximation for small areas.

    Args:
        bbox: Bounding box

    Returns:
        float: Approximate area in square kilometers
    """
    # Earth's radius in km
    R = 6371.0

    # Convert to radians
    lat1 = math.radians(bbox.south)
    lat2 = math.radians(bbox.north)
    lon1 = math.radians(bbox.west)
    lon2 = math.radians(bbox.east)

    # Width at the center latitude
    center_lat = (lat1 + lat2) / 2
    width_km = R * math.cos(center_lat) * (lon2 - lon1)

    # Height
    height_km = R * (lat2 - lat1)

    return abs(width_km * height_km)


def meters_to_degrees(meters: float, latitude: float) -> tuple[float, float]:
    """
    Convert meters to approximate degrees at a given latitude.

    Args:
        meters: Distance in meters
        latitude: Reference latitude

    Returns:
        tuple: (lon_degrees, lat_degrees)
    """
    # Approximate meters per degree
    lat_meters_per_deg = 111320.0
    lon_meters_per_deg = 111320.0 * math.cos(math.radians(latitude))

    lat_deg = meters / lat_meters_per_deg
    lon_deg = meters / lon_meters_per_deg if lon_meters_per_deg > 0 else 0

    return (lon_deg, lat_deg)


def buffer_bbox(bbox: BoundingBox, buffer_km: float) -> BoundingBox:
    """
    Add a buffer around a bounding box.

    Args:
        bbox: Original bounding box
        buffer_km: Buffer distance in kilometers

    Returns:
        BoundingBox: Expanded bounding box
    """
    center_lat = (bbox.south + bbox.north) / 2
    buffer_m = buffer_km * 1000

    lon_buffer, lat_buffer = meters_to_degrees(buffer_m, center_lat)

    return BoundingBox(
        west=max(-180, bbox.west - lon_buffer),
        south=max(-90, bbox.south - lat_buffer),
        east=min(180, bbox.east + lon_buffer),
        north=min(90, bbox.north + lat_buffer),
    )

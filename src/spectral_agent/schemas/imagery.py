"""
Pydantic models for imagery data structures.

Defines schemas for search requests, scene metadata, and download results.
"""

from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class SatelliteType(str, Enum):
    """Supported satellite types."""

    LANDSAT = "landsat"
    SENTINEL = "sentinel"


class BoundingBox(BaseModel):
    """Geographic bounding box in WGS84 coordinates."""

    west: float = Field(..., ge=-180, le=180, description="Western longitude")
    south: float = Field(..., ge=-90, le=90, description="Southern latitude")
    east: float = Field(..., ge=-180, le=180, description="Eastern longitude")
    north: float = Field(..., ge=-90, le=90, description="Northern latitude")

    @model_validator(mode="after")
    def validate_bounds(self) -> "BoundingBox":
        """Ensure west < east and south < north."""
        if self.west >= self.east:
            raise ValueError("West longitude must be less than east longitude")
        if self.south >= self.north:
            raise ValueError("South latitude must be less than north latitude")
        return self

    def to_wkt(self) -> str:
        """Convert to Well-Known Text format."""
        return (
            f"POLYGON(({self.west} {self.south}, {self.east} {self.south}, "
            f"{self.east} {self.north}, {self.west} {self.north}, {self.west} {self.south}))"
        )

    def to_geojson(self) -> dict:
        """Convert to GeoJSON Polygon."""
        return {
            "type": "Polygon",
            "coordinates": [
                [
                    [self.west, self.south],
                    [self.east, self.south],
                    [self.east, self.north],
                    [self.west, self.north],
                    [self.west, self.south],
                ]
            ],
        }

    @classmethod
    def from_geojson(cls, geojson: dict) -> "BoundingBox":
        """Create BoundingBox from GeoJSON geometry."""
        coords = geojson.get("coordinates", [[]])[0]
        if not coords:
            raise ValueError("Invalid GeoJSON: no coordinates found")

        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]

        return cls(
            west=min(lons),
            east=max(lons),
            south=min(lats),
            north=max(lats),
        )


class GeoJSON(BaseModel):
    """GeoJSON geometry model."""

    type: Literal["Point", "Polygon", "MultiPolygon"]
    coordinates: list[Any]

    def to_bbox(self) -> BoundingBox:
        """Extract bounding box from geometry."""
        return BoundingBox.from_geojson(self.model_dump())


class SceneMetadata(BaseModel):
    """Metadata for a single satellite scene."""

    scene_id: str = Field(..., description="Unique scene identifier")
    satellite: SatelliteType = Field(..., description="Satellite type")
    collection: str = Field(..., description="Data collection name")
    acquisition_date: datetime = Field(..., description="Scene acquisition timestamp")
    cloud_cover: float = Field(..., ge=0, le=100, description="Cloud cover percentage")

    # Spatial information
    footprint: dict | None = Field(
        default=None, description="Scene footprint as GeoJSON"
    )
    bbox: BoundingBox | None = Field(default=None, description="Scene bounding box")

    # Sensor details
    sensor: str | None = Field(default=None, description="Sensor name")
    processing_level: str | None = Field(default=None, description="Processing level")

    # Data access
    download_url: str | None = Field(default=None, description="Direct download URL")
    browse_url: str | None = Field(default=None, description="Browse image URL")

    # Additional metadata
    sun_azimuth: float | None = Field(default=None, description="Sun azimuth angle")
    sun_elevation: float | None = Field(default=None, description="Sun elevation angle")

    # Raw provider metadata
    raw_metadata: dict = Field(
        default_factory=dict, description="Original provider metadata"
    )


class SearchResult(BaseModel):
    """Results from a scene search operation."""

    total_count: int = Field(..., ge=0, description="Total matching scenes")
    returned_count: int = Field(..., ge=0, description="Number of scenes returned")
    scenes: list[SceneMetadata] = Field(
        default_factory=list, description="Scene metadata list"
    )

    # Query information
    query_bbox: BoundingBox | None = Field(
        default=None, description="Query bounding box"
    )
    query_start_date: date | None = Field(default=None, description="Query start date")
    query_end_date: date | None = Field(default=None, description="Query end date")

    # Pagination
    next_page_token: str | None = Field(default=None, description="Token for next page")


class DownloadResult(BaseModel):
    """Result of a download operation."""

    scene_id: str = Field(..., description="Scene identifier")
    success: bool = Field(..., description="Whether download succeeded")
    file_path: Path | None = Field(default=None, description="Path to downloaded file")
    file_size_bytes: int | None = Field(default=None, description="File size in bytes")
    download_time_seconds: float | None = Field(
        default=None, description="Download duration"
    )
    error_message: str | None = Field(
        default=None, description="Error message if failed"
    )


class IngestionRequest(BaseModel):
    """Request schema for the ingestion tool."""

    # Required parameters
    bbox: BoundingBox = Field(..., description="Area of interest bounding box")
    start_date: date = Field(..., description="Start date for imagery search")
    end_date: date = Field(..., description="End date for imagery search")

    # Optional parameters
    satellites: list[SatelliteType] = Field(
        default=[SatelliteType.LANDSAT, SatelliteType.SENTINEL],
        description="Satellites to query",
    )
    max_cloud_cover: float = Field(
        default=100.0, ge=0, le=100, description="Maximum cloud cover percentage. Default 100 (no filter)."
    )
    max_scenes_per_satellite: int | None = Field(
        default=None, ge=1, description="Maximum scenes per satellite. None = no limit."
    )
    bands: list[str] | None = Field(
        default=None, description="Specific bands to download (None = all)"
    )

    @model_validator(mode="after")
    def validate_dates(self) -> "IngestionRequest":
        """Ensure start_date <= end_date."""
        if self.start_date > self.end_date:
            raise ValueError("start_date must be before or equal to end_date")
        return self


class IngestionResponse(BaseModel):
    """Response from the ingestion workflow."""

    success: bool = Field(..., description="Overall success status")
    message: str = Field(..., description="Status message")

    # Results per satellite
    landsat_results: list[DownloadResult] = Field(default_factory=list)
    sentinel_results: list[DownloadResult] = Field(default_factory=list)

    # Summary statistics
    total_scenes_found: int = Field(
        default=0, description="Total scenes found across all satellites"
    )
    total_scenes_downloaded: int = Field(
        default=0, description="Total scenes successfully downloaded"
    )
    total_download_size_mb: float = Field(
        default=0.0, description="Total download size in MB"
    )
    total_download_time_seconds: float = Field(
        default=0.0, description="Total download time"
    )

    # Errors
    errors: list[str] = Field(
        default_factory=list, description="List of error messages"
    )

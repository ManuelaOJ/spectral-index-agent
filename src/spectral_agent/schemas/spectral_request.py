"""
Phase 1 schemas for spectral index request extraction and processing.

Defines the structured output that the NLP extraction layer must produce,
the business rules engine input/output, and the final API request payload.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────


class SpectralIndexName(str, Enum):
    """Supported spectral indices."""

    NDVI = "NDVI"
    EVI = "EVI"
    SAVI = "SAVI"
    NDWI = "NDWI"
    NBR = "NBR"
    NDBI = "NDBI"


class LandsatSensor(str, Enum):
    """Landsat sensor identifiers with operation periods."""

    LANDSAT_4 = "Landsat 4"
    LANDSAT_5 = "Landsat 5"
    LANDSAT_7 = "Landsat 7"
    LANDSAT_8 = "Landsat 8"
    LANDSAT_9 = "Landsat 9"


class CloudCoverStrategy(str, Enum):
    """Strategy for selecting scenes based on cloud cover."""

    MINIMUM_AVAILABLE = "minimum_available"
    USER_SPECIFIED = "user_specified"


class GeometryType(str, Enum):
    """Supported geometry input types."""

    POINT = "point"
    BBOX = "bbox"
    POLYGON = "polygon"
    MULTI_POLYGON = "multi_polygon"


class SelectionRule(str, Enum):
    """Rules that can be applied during scene selection."""

    DEFAULT_SENSOR_PRIORITY = "default_sensor_priority"
    USER_SPECIFIED_SENSOR = "user_specified_sensor"
    MINIMUM_CLOUD_COVER = "minimum_cloud_cover"
    USER_CLOUD_COVER_THRESHOLD = "user_cloud_cover_threshold"
    SAME_DATE_FOR_MULTIPLE_SCENES = "same_date_for_multiple_scenes"
    TEMPORAL_WINDOW_VALIDATION = "temporal_window_validation"


# ─────────────────────────────────────────────────────────────────────────────
# Sensor operation windows
# ─────────────────────────────────────────────────────────────────────────────

SENSOR_OPERATION_WINDOWS: dict[LandsatSensor, tuple[date, date]] = {
    LandsatSensor.LANDSAT_4: (date(1982, 7, 1), date(1993, 12, 31)),
    LandsatSensor.LANDSAT_5: (date(1984, 3, 1), date(2013, 1, 31)),
    LandsatSensor.LANDSAT_7: (date(1999, 4, 1), date(2025, 6, 30)),
    LandsatSensor.LANDSAT_8: (date(2013, 2, 1), date(2026, 12, 31)),
    LandsatSensor.LANDSAT_9: (date(2021, 9, 1), date(2026, 12, 31)),
}

# Default priority order (highest to lowest)
SENSOR_PRIORITY: list[LandsatSensor] = [
    LandsatSensor.LANDSAT_9,
    LandsatSensor.LANDSAT_8,
    LandsatSensor.LANDSAT_7,
    LandsatSensor.LANDSAT_5,
    LandsatSensor.LANDSAT_4,
]

# Map sensor to M2M collection name
SENSOR_TO_COLLECTION: dict[LandsatSensor, str] = {
    LandsatSensor.LANDSAT_9: "landsat_ot_c2_l2",
    LandsatSensor.LANDSAT_8: "landsat_ot_c2_l2",
    LandsatSensor.LANDSAT_7: "landsat_etm_c2_l2",
    LandsatSensor.LANDSAT_5: "landsat_tm_c2_l2",
    LandsatSensor.LANDSAT_4: "landsat_tm_c2_l2",
}


# ─────────────────────────────────────────────────────────────────────────────
# Landsat band mapping per index
# ─────────────────────────────────────────────────────────────────────────────

# Landsat 8-9 OLI band mapping for spectral indices
LANDSAT_89_INDEX_BANDS: dict[str, dict[str, str]] = {
    "NDVI": {"RED": "SR_B4", "NIR": "SR_B5"},
    "EVI": {"BLUE": "SR_B2", "RED": "SR_B4", "NIR": "SR_B5"},
    "SAVI": {"RED": "SR_B4", "NIR": "SR_B5"},
    "NDWI": {"GREEN": "SR_B3", "NIR": "SR_B5"},
    "NBR": {"NIR": "SR_B5", "SWIR2": "SR_B7"},
    "NDBI": {"NIR": "SR_B5", "SWIR1": "SR_B6"},
}

# Landsat 4-5 TM / 7 ETM+ band mapping
LANDSAT_457_INDEX_BANDS: dict[str, dict[str, str]] = {
    "NDVI": {"RED": "SR_B3", "NIR": "SR_B4"},
    "EVI": {"BLUE": "SR_B1", "RED": "SR_B3", "NIR": "SR_B4"},
    "SAVI": {"RED": "SR_B3", "NIR": "SR_B4"},
    "NDWI": {"GREEN": "SR_B2", "NIR": "SR_B4"},
    "NBR": {"NIR": "SR_B4", "SWIR2": "SR_B7"},
    "NDBI": {"NIR": "SR_B4", "SWIR1": "SR_B5"},
}


# ─────────────────────────────────────────────────────────────────────────────
# Extraction output (Layer 1 → Layer 2 input)
# ─────────────────────────────────────────────────────────────────────────────


class ExtractedGeometry(BaseModel):
    """Geometry extracted from user input."""

    type: GeometryType
    coordinates: list[Any] = Field(
        description="Coordinates: [lon, lat] for point; [[w,s],[e,n]] for bbox; polygon rings"
    )
    source: str = Field(
        description="Where the geometry came from: 'text', 'kml', 'shapefile', 'geojson'"
    )
    raw_input: str | None = Field(
        default=None, description="Original text that was parsed"
    )


class ExtractionResult(BaseModel):
    """
    Output of Layer 1 (NLP Extraction).

    Contains everything the LLM extracted from the user's natural language request.
    """

    indices: list[str] = Field(
        description="Spectral indices requested (e.g. ['NDVI', 'NBR'])"
    )
    location_description: str | None = Field(
        default=None, description="Location as described by user (e.g. 'Medellin')"
    )
    geometry: ExtractedGeometry | None = Field(
        default=None, description="Parsed geometry if coordinates were provided"
    )
    geometry_file: str | None = Field(
        default=None,
        description=(
            "Path to a geometry file (.kml, .kmz, .shp, .geojson) "
            "provided by the user. Processed during normalization."
        ),
    )
    start_date: str | None = Field(
        default=None, description="Start date (ISO format) or year"
    )
    end_date: str | None = Field(
        default=None, description="End date (ISO format) or year"
    )
    year: int | None = Field(
        default=None, description="Single year if only a year was specified"
    )
    years: list[int] | None = Field(
        default=None, description="Multiple years if comparison requested"
    )
    month: int | None = Field(default=None, description="Specific month if mentioned")
    sensor_requested: str | None = Field(
        default=None, description="Sensor explicitly requested (e.g. 'Landsat 8')"
    )
    cloud_cover_max: float | None = Field(
        default=None, description="Max cloud cover if user specified"
    )
    comparison_mode: bool = Field(
        default=False,
        description="True if user wants to compare the same index across different dates/years",
    )
    raw_query: str = Field(description="Original user query")


# ─────────────────────────────────────────────────────────────────────────────
# Normalized request (Layer 2 output)
# ─────────────────────────────────────────────────────────────────────────────


class DateRange(BaseModel):
    """Resolved date range."""

    start: date
    end: date

    @model_validator(mode="after")
    def validate_order(self) -> "DateRange":
        if self.start > self.end:
            raise ValueError("start must be <= end")
        return self


class NormalizedRequest(BaseModel):
    """
    Output of Layer 2 (Normalization).

    Fully resolved, validated request ready for the rules engine.
    """

    indices: list[SpectralIndexName]
    date_ranges: list[DateRange] = Field(
        description="One or more date ranges to process"
    )
    geometry: ExtractedGeometry
    sensor_requested: LandsatSensor | None = Field(
        default=None, description="Sensor if explicitly requested"
    )
    cloud_cover_max: float | None = Field(
        default=None, description="User-specified cloud cover threshold"
    )
    comparison_mode: bool = False
    location_label: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Rules engine output (Layer 3)
# ─────────────────────────────────────────────────────────────────────────────


class SensorSelection(BaseModel):
    """Sensor selection result from the rules engine."""

    sensor: LandsatSensor
    collection: str = Field(description="M2M API collection name")
    reason: str


class SceneSelectionPlan(BaseModel):
    """
    Output of Layer 3 (Rules Engine).

    Describes HOW scenes should be selected for each date range.
    """

    indices: list[SpectralIndexName]
    date_range: DateRange
    sensors: list[SensorSelection] = Field(
        description="Sensors to query, in priority order"
    )
    cloud_cover_strategy: CloudCoverStrategy
    cloud_cover_threshold: float | None = Field(
        default=None, description="Threshold if user-specified"
    )
    geometry_type: GeometryType
    needs_multiple_scenes: bool = Field(
        default=False,
        description="True when geometry spans multiple Landsat tiles",
    )
    selection_rules_applied: list[SelectionRule]


# ─────────────────────────────────────────────────────────────────────────────
# Final structured output (Layer 4 – API request)
# ─────────────────────────────────────────────────────────────────────────────


class SpectralIndexRequest(BaseModel):
    """
    Complete structured request ready for the API / download phase.

    This is the Phase 1 deliverable: proves the agent can correctly
    interpret a natural language request and produce an actionable
    search/download plan.
    """

    indices: list[str]
    date_range: dict[str, str] = Field(
        description="{'start': 'YYYY-MM-DD', 'end': 'YYYY-MM-DD'}"
    )
    sensor_selected: str
    collection: str
    cloud_cover_strategy: str
    cloud_cover_threshold: float | None = None
    geometry_type: str
    geometry: dict[str, Any]
    needs_multiple_scenes: bool
    selection_rules_applied: list[str]
    bands_required: dict[str, dict[str, str]] = Field(
        default_factory=dict,
        description="Per-index band mapping for the selected sensor",
    )

    # Metadata
    original_query: str = ""
    location_label: str | None = None

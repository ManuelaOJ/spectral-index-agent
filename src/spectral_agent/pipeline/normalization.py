"""
Layer 2 – Normalization.

Converts the raw ExtractionResult into a fully validated NormalizedRequest:
  - Resolves year/years/month into concrete DateRange objects.
  - Validates index names against the supported catalogue.
  - Standardises geometry to a common representation.
  - Reads geometry from KML, Shapefile or GeoJSON files when provided.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

from spectral_agent.schemas.spectral_request import (
    DateRange,
    ExtractionResult,
    ExtractedGeometry,
    GeometryType,
    LandsatSensor,
    NormalizedRequest,
    SpectralIndexName,
)

logger = logging.getLogger(__name__)


def normalize_request(extraction: ExtractionResult) -> NormalizedRequest:
    """
    Transform raw extraction into a validated, normalized request.

    Raises ValueError if critical information is missing.
    """

    # ── Validate indices ────────────────────────────────────────────────
    valid_indices: list[SpectralIndexName] = []
    for idx_name in extraction.indices:
        try:
            valid_indices.append(SpectralIndexName(idx_name.upper()))
        except ValueError:
            raise ValueError(
                f"Unknown spectral index '{idx_name}'. "
                f"Supported: {[e.value for e in SpectralIndexName]}"
            )

    if not valid_indices:
        raise ValueError("No valid spectral indices found in the request.")

    # ── Resolve date ranges ─────────────────────────────────────────────
    date_ranges = _resolve_dates(extraction)
    if not date_ranges:
        raise ValueError(
            "Could not resolve any date range from the request. "
            "Provide a year, date range, or explicit start/end dates."
        )

    # ── Resolve geometry ────────────────────────────────────────────────
    geometry = extraction.geometry

    # Priority: geometry_file > inline geometry > location name
    if extraction.geometry_file:
        from spectral_agent.pipeline.geo_input import read_geometry_file

        try:
            geometry = read_geometry_file(extraction.geometry_file)
            logger.info(
                "Geometry loaded from file: %s (%s)",
                extraction.geometry_file,
                geometry.type.value,
            )
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(f"Error reading geometry file: {e}") from e

    if geometry is None:
        if extraction.location_description:
            # Placeholder: in production this would geocode the location.
            # For Phase 1 we accept name-only and flag geometry as pending.
            geometry = ExtractedGeometry(
                type=GeometryType.POLYGON,
                coordinates=[],
                source="geocode_pending",
                raw_input=extraction.location_description,
            )
        else:
            raise ValueError(
                "No geometry or location provided. "
                "Specify coordinates, a bounding box, or a location name."
            )

    # ── Resolve sensor ──────────────────────────────────────────────────
    sensor_req: LandsatSensor | None = None
    if extraction.sensor_requested:
        sensor_text = extraction.sensor_requested.strip()
        for member in LandsatSensor:
            if member.value.lower() == sensor_text.lower():
                sensor_req = member
                break
        if sensor_req is None:
            logger.warning(
                "Could not match sensor '%s' to a known Landsat sensor. "
                "Falling back to automatic selection.",
                sensor_text,
            )

    return NormalizedRequest(
        indices=valid_indices,
        date_ranges=date_ranges,
        geometry=geometry,
        sensor_requested=sensor_req,
        cloud_cover_max=extraction.cloud_cover_max,
        comparison_mode=extraction.comparison_mode,
        location_label=extraction.location_description,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Date resolution helpers
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_dates(ext: ExtractionResult) -> list[DateRange]:
    """Build date ranges from all possible extraction fields."""

    ranges: list[DateRange] = []

    # Case 1: explicit start_date / end_date
    if ext.start_date and ext.end_date:
        ranges.append(
            DateRange(
                start=date.fromisoformat(ext.start_date),
                end=date.fromisoformat(ext.end_date),
            )
        )

    # Case 2: comparison mode (years list)
    elif ext.years:
        for y in ext.years:
            if ext.month:
                s = date(y, ext.month, 1)
                e = _end_of_month(y, ext.month)
            else:
                s = date(y, 1, 1)
                e = date(y, 12, 31)
            ranges.append(DateRange(start=s, end=e))

    # Case 3: single year
    elif ext.year:
        if ext.month:
            s = date(ext.year, ext.month, 1)
            e = _end_of_month(ext.year, ext.month)
        else:
            s = date(ext.year, 1, 1)
            e = date(ext.year, 12, 31)
        ranges.append(DateRange(start=s, end=e))

    # Case 4: only start_date given
    elif ext.start_date:
        s = date.fromisoformat(ext.start_date)
        ranges.append(DateRange(start=s, end=s))

    return ranges


def _end_of_month(year: int, month: int) -> date:
    """Return the last day of the given month."""
    import calendar

    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, last_day)

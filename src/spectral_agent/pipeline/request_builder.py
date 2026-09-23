"""
Layer 4 – Request Builder.

Converts the SceneSelectionPlan(s) into the final SpectralIndexRequest
objects that are ready for the download / API phase.
"""

from __future__ import annotations

import logging
from typing import Any

from spectral_agent.schemas.spectral_request import (
    LANDSAT_89_INDEX_BANDS,
    LANDSAT_457_INDEX_BANDS,
    LandsatSensor,
    NormalizedRequest,
    SceneSelectionPlan,
    SpectralIndexRequest,
)

logger = logging.getLogger(__name__)


def build_request(
    plan: SceneSelectionPlan,
    normalized: NormalizedRequest,
) -> SpectralIndexRequest:
    """
    Build the final structured request from a plan + normalized input.

    Args:
        plan: Output from the rules engine (one per date range).
        normalized: The normalized request (carries geometry, location, etc.).

    Returns:
        SpectralIndexRequest ready for the API / download phase.
    """
    # Pick primary sensor (first in priority list)
    primary = plan.sensors[0] if plan.sensors else None
    sensor_label = primary.sensor.value if primary else "Unknown"
    collection = primary.collection if primary else "unknown"

    # Resolve band mapping based on sensor
    bands_required = _resolve_bands(plan, primary.sensor if primary else None)

    # Build geometry dict
    geo_dict: dict[str, Any] = {
        "type": normalized.geometry.type.value,
        "coordinates": normalized.geometry.coordinates,
        "source": normalized.geometry.source,
    }

    return SpectralIndexRequest(
        indices=[idx.value for idx in plan.indices],
        date_range={
            "start": plan.date_range.start.isoformat(),
            "end": plan.date_range.end.isoformat(),
        },
        sensor_selected=sensor_label,
        collection=collection,
        cloud_cover_strategy=plan.cloud_cover_strategy.value,
        cloud_cover_threshold=plan.cloud_cover_threshold,
        geometry_type=plan.geometry_type.value,
        geometry=geo_dict,
        needs_multiple_scenes=plan.needs_multiple_scenes,
        selection_rules_applied=[r.value for r in plan.selection_rules_applied],
        bands_required=bands_required,
        original_query=normalized.location_label or "",
        location_label=normalized.location_label,
    )


def build_requests(
    plans: list[SceneSelectionPlan],
    normalized: NormalizedRequest,
) -> list[SpectralIndexRequest]:
    """Build one SpectralIndexRequest per plan (one per date range)."""
    return [build_request(p, normalized) for p in plans]


# ─────────────────────────────────────────────────────────────────────────────
# Band resolution
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_bands(
    plan: SceneSelectionPlan,
    sensor: LandsatSensor | None,
) -> dict[str, dict[str, str]]:
    """Map each index to the concrete Landsat bands for the chosen sensor."""
    if sensor is None:
        return {}

    if sensor in (LandsatSensor.LANDSAT_8, LandsatSensor.LANDSAT_9):
        table = LANDSAT_89_INDEX_BANDS
    else:
        table = LANDSAT_457_INDEX_BANDS

    result: dict[str, dict[str, str]] = {}
    for idx in plan.indices:
        mapping = table.get(idx.value)
        if mapping:
            result[idx.value] = mapping
        else:
            logger.warning("No band mapping for %s on %s", idx.value, sensor.value)

    return result

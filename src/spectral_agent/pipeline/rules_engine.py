"""
Layer 3 – Rules Engine.

Applies the mandatory business rules:
  1. Sensor priority (L9 > L8 > L7 > L5 > L4) filtered by operation window.
  2. Cloud cover strategy (minimum available vs. user threshold).
  3. Multiple-scene consistency (same date across tiles).
"""

from __future__ import annotations

import logging
from datetime import date

from spectral_agent.schemas.spectral_request import (
    SENSOR_OPERATION_WINDOWS,
    SENSOR_PRIORITY,
    SENSOR_TO_COLLECTION,
    CloudCoverStrategy,
    DateRange,
    LandsatSensor,
    NormalizedRequest,
    SceneSelectionPlan,
    SelectionRule,
    SensorSelection,
)

logger = logging.getLogger(__name__)


def apply_rules(request: NormalizedRequest) -> list[SceneSelectionPlan]:
    """
    Apply business rules and produce a SceneSelectionPlan per date range.

    Returns one plan per date range in the request.
    """
    plans: list[SceneSelectionPlan] = []

    for dr in request.date_ranges:
        plan = _build_plan(request, dr)
        plans.append(plan)

    return plans


# ─────────────────────────────────────────────────────────────────────────────
# Internal
# ─────────────────────────────────────────────────────────────────────────────


def _build_plan(req: NormalizedRequest, dr: DateRange) -> SceneSelectionPlan:
    """Build a single SceneSelectionPlan for one date range."""

    rules_applied: list[SelectionRule] = []

    # ── Sensor selection ────────────────────────────────────────────────
    if req.sensor_requested is not None:
        # User explicitly chose a sensor → validate it covers the range
        sensors = _validate_user_sensor(req.sensor_requested, dr)
        rules_applied.append(SelectionRule.USER_SPECIFIED_SENSOR)
    else:
        sensors = _select_sensors_by_priority(dr)
        rules_applied.append(SelectionRule.DEFAULT_SENSOR_PRIORITY)

    rules_applied.append(SelectionRule.TEMPORAL_WINDOW_VALIDATION)

    # ── Cloud cover strategy ────────────────────────────────────────────
    if req.cloud_cover_max is not None:
        cc_strategy = CloudCoverStrategy.USER_SPECIFIED
        cc_threshold = req.cloud_cover_max
        rules_applied.append(SelectionRule.USER_CLOUD_COVER_THRESHOLD)
    else:
        cc_strategy = CloudCoverStrategy.MINIMUM_AVAILABLE
        cc_threshold = None
        rules_applied.append(SelectionRule.MINIMUM_CLOUD_COVER)

    # ── Multiple scenes ─────────────────────────────────────────────────
    needs_multiple = _check_multiple_scenes(req)
    if needs_multiple:
        rules_applied.append(SelectionRule.SAME_DATE_FOR_MULTIPLE_SCENES)

    return SceneSelectionPlan(
        indices=req.indices,
        date_range=dr,
        sensors=sensors,
        cloud_cover_strategy=cc_strategy,
        cloud_cover_threshold=cc_threshold,
        geometry_type=req.geometry.type,
        needs_multiple_scenes=needs_multiple,
        selection_rules_applied=rules_applied,
    )


def _select_sensors_by_priority(dr: DateRange) -> list[SensorSelection]:
    """
    Return sensors available for the date range, in priority order.

    A sensor is available if its operation window overlaps with the
    requested date range.
    """
    result: list[SensorSelection] = []

    for sensor in SENSOR_PRIORITY:
        win_start, win_end = SENSOR_OPERATION_WINDOWS[sensor]
        if _ranges_overlap(dr.start, dr.end, win_start, win_end):
            result.append(
                SensorSelection(
                    sensor=sensor,
                    collection=SENSOR_TO_COLLECTION[sensor],
                    reason=f"{sensor.value} available: {win_start} to {win_end}",
                )
            )

    if not result:
        logger.warning("No Landsat sensor covers the range %s – %s", dr.start, dr.end)

    return result


def _validate_user_sensor(sensor: LandsatSensor, dr: DateRange) -> list[SensorSelection]:
    """Validate that the user-requested sensor covers the date range."""
    win_start, win_end = SENSOR_OPERATION_WINDOWS[sensor]

    if _ranges_overlap(dr.start, dr.end, win_start, win_end):
        return [
            SensorSelection(
                sensor=sensor,
                collection=SENSOR_TO_COLLECTION[sensor],
                reason=f"User requested {sensor.value} (active {win_start} – {win_end})",
            )
        ]

    # Sensor doesn't cover the range → fall back to priority
    logger.warning(
        "%s not available for %s – %s. Falling back to auto-priority.",
        sensor.value,
        dr.start,
        dr.end,
    )
    return _select_sensors_by_priority(dr)


def _ranges_overlap(a_start: date, a_end: date, b_start: date, b_end: date) -> bool:
    """Check whether two date ranges overlap."""
    return a_start <= b_end and b_start <= a_end


def _check_multiple_scenes(req: NormalizedRequest) -> bool:
    """
    Heuristic: if geometry spans a large area it likely needs multiple tiles.

    For Phase 1 we flag MultiPolygon or large bounding boxes.
    """
    from spectral_agent.schemas.spectral_request import GeometryType

    if req.geometry.type == GeometryType.MULTI_POLYGON:
        return True

    if req.geometry.type == GeometryType.BBOX and len(req.geometry.coordinates) >= 2:
        try:
            sw = req.geometry.coordinates[0]
            ne = req.geometry.coordinates[1]
            lon_span = abs(ne[0] - sw[0])
            lat_span = abs(ne[1] - sw[1])
            # Landsat tiles are ~185 km wide ≈ 1.65 degrees
            if lon_span > 1.65 or lat_span > 1.65:
                return True
        except (IndexError, TypeError):
            pass

    return False

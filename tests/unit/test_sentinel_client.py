"""Unit tests for Sentinel client search behavior."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest

from spectral_agent.schemas.imagery import BoundingBox
from spectral_agent.tools.ingestion.sentinel import SentinelClient


@pytest.mark.asyncio
async def test_search_clamps_limit_to_100_when_default() -> None:
    """Copernicus Catalog enforces limit <= 100; default path must respect it."""
    client = SentinelClient()

    feature = {
        "id": "S2A_MSIL2A_20220301T152631_N0500_R025_T18NXN_20220301T190000.SAFE",
        "collection": "sentinel-2-l2a",
        "geometry": {"type": "Polygon", "coordinates": []},
        "properties": {
            "datetime": "2022-03-01T15:26:31Z",
            "eo:cloud_cover": 7.5,
            "platform": "sentinel-2a",
        },
    }

    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"numberMatched": 1, "features": [feature]}

    client._ensure_authenticated = AsyncMock(return_value=None)
    client._oauth_session = MagicMock()
    client._oauth_session.post.return_value = response

    bbox = BoundingBox(west=-73.9, south=6.8, east=-73.8, north=6.9)
    result = await client.search(
        bbox=bbox,
        start_date=date(2022, 1, 1),
        end_date=date(2022, 3, 31),
    )

    assert result.returned_count == 1
    assert result.scenes[0].scene_id.startswith("S2A_MSIL2A_20220301")

    _, kwargs = client._oauth_session.post.call_args
    assert kwargs["json"]["limit"] == 100


@pytest.mark.asyncio
async def test_search_honors_lower_max_results() -> None:
    """A caller-provided max_results below 100 should be used directly."""
    client = SentinelClient()

    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"numberMatched": 0, "features": []}

    client._ensure_authenticated = AsyncMock(return_value=None)
    client._oauth_session = MagicMock()
    client._oauth_session.post.return_value = response

    bbox = BoundingBox(west=-73.9, south=6.8, east=-73.8, north=6.9)
    await client.search(
        bbox=bbox,
        start_date=date(2022, 1, 1),
        end_date=date(2022, 3, 31),
        max_results=5,
    )

    _, kwargs = client._oauth_session.post.call_args
    assert kwargs["json"]["limit"] == 5

"""
Pytest configuration and shared fixtures.
"""

import os
import sys

# ── Fix TLS CA bundle (PostgreSQL 18 sets SSL_CERT_FILE to bad path) ─────────
import certifi as _certifi
os.environ.setdefault("SSL_CERT_FILE", _certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", _certifi.where())

# ── Fix PROJ_DATA before any geo-library import ──────────────────────────────
# A system-wide PROJ_LIB (e.g. from PostgreSQL/PostGIS) or pyproj's bundled
# proj.db may be incompatible with the version that rasterio's GDAL expects.
# We locate rasterio's proj_data directory via filesystem inspection
# (without importing rasterio, which would trigger GDAL init with the
# wrong PROJ_LIB) and force-set the env vars.
import importlib.util as _ilu
_rasterio_spec = _ilu.find_spec("rasterio")
if _rasterio_spec and _rasterio_spec.submodule_search_locations:
    from pathlib import Path as _Path
    _proj_dir = _Path(_rasterio_spec.submodule_search_locations[0]) / "proj_data"
    if (_proj_dir / "proj.db").exists():
        os.environ["PROJ_DATA"] = str(_proj_dir)
        os.environ["PROJ_LIB"] = str(_proj_dir)
del _ilu

import pytest
from datetime import date, datetime
from pathlib import Path

from spectral_agent.schemas.imagery import BoundingBox


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_bbox() -> BoundingBox:
    """Sample bounding box for testing (San Francisco Bay Area)."""
    return BoundingBox(
        west=-122.5,
        south=37.5,
        east=-122.0,
        north=38.0,
    )


@pytest.fixture
def sample_date_range() -> tuple[date, date]:
    """Sample date range for testing."""
    return (date(2024, 1, 1), date(2024, 6, 1))


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Temporary directory for test data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    return data_dir


# ─────────────────────────────────────────────────────────────────────────────
# Markers
# ─────────────────────────────────────────────────────────────────────────────


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests requiring credentials"
    )

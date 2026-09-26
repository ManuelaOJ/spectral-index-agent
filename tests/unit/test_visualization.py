"""
Unit tests for the visualization package — thematic_map and interactive_map.

Uses the same synthetic-raster approach as test_index_calculator:
tiny GeoTIFFs with known values, so tests run in < 2 s and need no real
imagery data.
"""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from spectral_agent.tools.visualization.interactive_map import (
    InteractiveMapResult,
    generate_interactive_map,
)
from spectral_agent.tools.visualization.thematic_map import (
    INDEX_CMAPS,
    MapConfig,
    MapResult,
    _nice_round,
    generate_thematic_map,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers / fixtures
# ─────────────────────────────────────────────────────────────────────────────

RASTER_CRS = CRS.from_epsg(32618)
WIDTH = 20
HEIGHT = 20
TRANSFORM = from_bounds(500000, 700000, 500600, 700600, WIDTH, HEIGHT)


def _write_index_tif(path: Path, *, fill: float = 0.6, add_nan: bool = False) -> Path:
    """Write a synthetic single-band Float32 GeoTIFF mimicking an index raster."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((1, HEIGHT, WIDTH), fill, dtype="float32")
    if add_nan:
        data[0, 0, :] = np.nan  # first row NoData
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": WIDTH,
        "height": HEIGHT,
        "count": 1,
        "crs": RASTER_CRS,
        "transform": TRANSFORM,
        "nodata": float("nan"),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return path


@pytest.fixture
def ndvi_raster(tmp_path: Path) -> Path:
    return _write_index_tif(tmp_path / "NDVI_test.tif", fill=0.65)


@pytest.fixture
def ndvi_raster_with_nan(tmp_path: Path) -> Path:
    return _write_index_tif(tmp_path / "NDVI_nan.tif", fill=0.65, add_nan=True)


@pytest.fixture
def gradient_raster(tmp_path: Path) -> Path:
    """Raster with a gradient from -1 to 1 (row-wise)."""
    path = tmp_path / "gradient.tif"
    path.parent.mkdir(parents=True, exist_ok=True)
    grad = np.linspace(-1, 1, HEIGHT).reshape(-1, 1)
    data = np.broadcast_to(grad, (HEIGHT, WIDTH)).astype("float32")
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": WIDTH,
        "height": HEIGHT,
        "count": 1,
        "crs": RASTER_CRS,
        "transform": TRANSFORM,
        "nodata": float("nan"),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data[np.newaxis, :, :])
    return path


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    d = tmp_path / "maps"
    d.mkdir()
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Tests — static thematic map
# ─────────────────────────────────────────────────────────────────────────────


class TestThematicMap:
    """Tests for generate_thematic_map."""

    def test_basic_png(self, ndvi_raster, output_dir):
        out = output_dir / "NDVI.png"
        result = generate_thematic_map(ndvi_raster, "NDVI", out)
        assert isinstance(result, MapResult)
        assert result.output_path == out
        assert out.exists()
        assert out.stat().st_size > 1000  # non-trivial PNG
        assert result.index_name == "NDVI"
        assert result.width_px == WIDTH
        assert result.height_px == HEIGHT

    def test_custom_config(self, ndvi_raster, output_dir):
        cfg = MapConfig(
            title="My Custom Title",
            subtitle="Test subtitle",
            cmap="coolwarm",
            vmin=-0.5,
            vmax=0.5,
            dpi=72,
            figsize=(6, 6),
            add_grid=True,
        )
        out = output_dir / "custom.png"
        result = generate_thematic_map(ndvi_raster, "NDVI", out, config=cfg)
        assert result.dpi == 72
        assert out.exists()

    def test_no_colorbar_no_scale(self, ndvi_raster, output_dir):
        cfg = MapConfig(add_colorbar=False, add_scale_bar=False, add_north_arrow=False)
        out = output_dir / "bare.png"
        result = generate_thematic_map(ndvi_raster, "NDVI", out, config=cfg)
        assert out.exists()

    def test_nan_pixels(self, ndvi_raster_with_nan, output_dir):
        """NoData pixels should render without crashing."""
        out = output_dir / "nan.png"
        result = generate_thematic_map(ndvi_raster_with_nan, "NDVI", out)
        assert out.exists()

    def test_gradient(self, gradient_raster, output_dir):
        out = output_dir / "gradient.png"
        result = generate_thematic_map(gradient_raster, "NDVI", out)
        assert out.exists()
        assert result.width_px == WIDTH

    def test_all_supported_indices(self, ndvi_raster, output_dir):
        """Every index name in INDEX_CMAPS should render without error."""
        for idx_name in INDEX_CMAPS:
            out = output_dir / f"{idx_name}.png"
            result = generate_thematic_map(ndvi_raster, idx_name, out)
            assert out.exists(), f"Failed to generate map for {idx_name}"

    def test_unknown_index_uses_viridis(self, ndvi_raster, output_dir):
        """An unrecognised index should fall back to viridis and still work."""
        out = output_dir / "UNKNOWN.png"
        result = generate_thematic_map(ndvi_raster, "UNKNOWN_INDEX", out)
        assert out.exists()

    def test_output_dir_created(self, ndvi_raster, tmp_path):
        """Non-existent output directory should be created automatically."""
        out = tmp_path / "new" / "sub" / "dir" / "map.png"
        result = generate_thematic_map(ndvi_raster, "NDVI", out)
        assert out.exists()

    def test_crs_in_result(self, ndvi_raster, output_dir):
        out = output_dir / "crs.png"
        result = generate_thematic_map(ndvi_raster, "NDVI", out)
        assert "32618" in result.crs


# Tests for _nice_round helper
class TestNiceRound:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (3743, 2000),
            (874, 500),
            (47, 20),
            (9, 5),
            (1.5, 1),
            (0.3, 0.2),
            (15000, 10000),
            (5500, 5000),
        ],
    )
    def test_nice_values(self, value, expected):
        assert _nice_round(value) == pytest.approx(expected)

    def test_zero_returns_one(self):
        assert _nice_round(0) == 1.0

    def test_negative_returns_one(self):
        assert _nice_round(-5) == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Tests — interactive map
# ─────────────────────────────────────────────────────────────────────────────


class TestInteractiveMap:
    """Tests for generate_interactive_map."""

    def test_basic_html(self, ndvi_raster, output_dir):
        out = output_dir / "NDVI.html"
        result = generate_interactive_map(ndvi_raster, "NDVI", out)
        assert isinstance(result, InteractiveMapResult)
        assert result.output_path == out
        assert out.exists()
        assert out.stat().st_size > 500

        # HTML should contain folium / leaflet markers
        html = out.read_text(encoding="utf-8")
        assert "leaflet" in html.lower()
        assert result.index_name == "NDVI"

    def test_center_coordinates(self, ndvi_raster, output_dir):
        out = output_dir / "center.html"
        result = generate_interactive_map(ndvi_raster, "NDVI", out)
        # UTM 18N north of equator → lat in range ~(6, 7) roughly
        assert -90 <= result.center_lat <= 90
        assert -180 <= result.center_lon <= 180

    def test_custom_cmap_and_range(self, ndvi_raster, output_dir):
        out = output_dir / "custom.html"
        result = generate_interactive_map(
            ndvi_raster,
            "NDWI",
            out,
            cmap="coolwarm",
            vmin=-0.5,
            vmax=0.5,
            opacity=0.5,
        )
        assert out.exists()
        assert result.index_name == "NDWI"

    def test_nan_pixels(self, ndvi_raster_with_nan, output_dir):
        out = output_dir / "nan.html"
        result = generate_interactive_map(ndvi_raster_with_nan, "NDVI", out)
        assert out.exists()

    def test_gradient(self, gradient_raster, output_dir):
        out = output_dir / "gradient.html"
        result = generate_interactive_map(gradient_raster, "NDVI", out)
        assert out.exists()

    def test_legend_in_html(self, ndvi_raster, output_dir):
        out = output_dir / "legend.html"
        generate_interactive_map(ndvi_raster, "NDVI", out)
        html = out.read_text(encoding="utf-8")
        assert "NDVI" in html  # legend should mention index name

    def test_output_dir_created(self, ndvi_raster, tmp_path):
        out = tmp_path / "new" / "deep" / "map.html"
        result = generate_interactive_map(ndvi_raster, "NDVI", out)
        assert out.exists()

    def test_bounds_wgs84(self, ndvi_raster, output_dir):
        out = output_dir / "bounds.html"
        result = generate_interactive_map(ndvi_raster, "NDVI", out)
        w, s, e, n = result.bounds_wgs84
        assert w < e
        assert s < n
        assert -180 <= w <= 180
        assert -90 <= s <= 90

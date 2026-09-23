"""
Unit tests for the new LangChain tool wrappers (Step 4).

Tests the 5 new raster-processing tools using synthetic data.
No real API keys, no real satellite imagery — everything is mocked
or uses tiny GeoTIFFs in tmp directories.
"""

import json
import numpy as np
import pytest
import rasterio
import tarfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from spectral_agent.tools import (
    crop_landsat_bands_tool,
    compute_spectral_index_tool,
    generate_thematic_map_tool,
    list_cached_bands_tool,
    list_available_indices_tool,
    get_ingestion_tools,
    get_raster_tools,
    get_all_tools,
)
from spectral_agent.tools.raster.index_calculator import (
    LANDSAT_C2_L2_SCALE,
    LANDSAT_C2_L2_OFFSET,
)
from spectral_agent.tools.raster.band_cache import BandCache


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

RASTER_CRS = CRS.from_epsg(32618)
WIDTH = 20
HEIGHT = 20
# Raster covers 500000-500600 E, 700000-700600 N in UTM 18N
RASTER_TRANSFORM = from_bounds(500000, 700000, 500600, 700600, WIDTH, HEIGHT)

# Corresponding WGS-84 bbox that covers the synthetic raster
# UTM 18N (500000,700000)→(500600,700600) ≈ lon -75.0..-74.995, lat 6.333..6.338
WGS84_WEST = -75.001
WGS84_SOUTH = 6.332
WGS84_EAST = -74.993
WGS84_NORTH = 6.339

TRANSFORM = RASTER_TRANSFORM


def reflectance_to_dn(refl: float) -> int:
    return int(round((refl - LANDSAT_C2_L2_OFFSET) / LANDSAT_C2_L2_SCALE))


def write_band_tif(path: Path, fill_dn: int) -> Path:
    """Write a synthetic Landsat band GeoTIFF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((1, HEIGHT, WIDTH), fill_dn, dtype="uint16")
    profile = {
        "driver": "GTiff",
        "dtype": "uint16",
        "width": WIDTH,
        "height": HEIGHT,
        "count": 1,
        "crs": RASTER_CRS,
        "transform": TRANSFORM,
        "nodata": 0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return path


def write_index_tif(path: Path, fill: float = 0.6) -> Path:
    """Write a synthetic Float32 index GeoTIFF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((1, HEIGHT, WIDTH), fill, dtype="float32")
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


def make_scene_tar(tmp_path: Path, scene_id: str = "LC09_TEST") -> Path:
    """
    Build a .tar with synthetic Landsat 8/9 band files
    (SR_B2..SR_B7) so crop_landsat_bands_tool can process them.
    """
    scene_dir = tmp_path / "tar_staging" / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    band_spec = {
        "SR_B2": 0.05,
        "SR_B3": 0.08,
        "SR_B4": 0.10,
        "SR_B5": 0.50,
        "SR_B6": 0.25,
        "SR_B7": 0.15,
    }
    for band, refl in band_spec.items():
        write_band_tif(scene_dir / f"{scene_id}_{band}.TIF", reflectance_to_dn(refl))

    tar_path = tmp_path / f"{scene_id}.tar"
    with tarfile.open(tar_path, "w") as tf:
        for f in scene_dir.iterdir():
            tf.add(f, arcname=f.name)

    return tar_path


class FakeSettings:
    """Minimal mock of Settings for tools that call get_settings()."""

    def __init__(self, tmp_path: Path):
        self.landsat_raw_dir = tmp_path / "raw" / "landsat"
        self.processed_data_dir = tmp_path / "processed"
        self.landsat_raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def fake_settings(tmp_path):
    return FakeSettings(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# Tests — list_available_indices_tool (no mocks needed)
# ─────────────────────────────────────────────────────────────────────────────


class TestListAvailableIndicesTool:

    def test_returns_all_six(self):
        result = list_available_indices_tool.invoke({})
        assert result["success"] is True
        assert result["count"] == 6
        names = {i["name"] for i in result["indices"]}
        assert names == {"NDVI", "EVI", "SAVI", "NDWI", "NBR", "NDBI"}

    def test_includes_band_mappings(self):
        result = list_available_indices_tool.invoke({})
        ndvi = next(i for i in result["indices"] if i["name"] == "NDVI")
        assert "NIR" in ndvi["landsat_89_bands"]
        assert "RED" in ndvi["landsat_89_bands"]
        assert ndvi["value_range"] == [-1.0, 1.0]

    def test_includes_full_names(self):
        result = list_available_indices_tool.invoke({})
        ndvi = next(i for i in result["indices"] if i["name"] == "NDVI")
        assert "Vegetation" in ndvi["full_name"]


# ─────────────────────────────────────────────────────────────────────────────
# Tests — crop_landsat_bands_tool
# ─────────────────────────────────────────────────────────────────────────────


class TestCropLandsatBandsTool:

    def test_crop_ndvi(self, tmp_path, fake_settings):
        tar_path = make_scene_tar(tmp_path, "LC09_CROP_TEST")

        with patch(
            "spectral_agent.tools.raster.lc_tools.get_settings",
            return_value=fake_settings,
        ):
            result = crop_landsat_bands_tool.invoke(
                {
                    "scene_id": "LC09_CROP_TEST",
                    "tar_path": str(tar_path),
                    "index_names": ["NDVI"],
                    "sensor": "Landsat 9",
                    "west": WGS84_WEST,
                    "south": WGS84_SOUTH,
                    "east": WGS84_EAST,
                    "north": WGS84_NORTH,
                }
            )

        assert result["success"] is True
        assert "NDVI" in result["indices_prepared"]
        assert result["total_bands"] >= 2  # RED + NIR
        for path_str in result["bands_cropped"].values():
            assert Path(path_str).exists()

    def test_crop_multiple_indices(self, tmp_path, fake_settings):
        tar_path = make_scene_tar(tmp_path, "LC09_MULTI")

        with patch(
            "spectral_agent.tools.raster.lc_tools.get_settings",
            return_value=fake_settings,
        ):
            result = crop_landsat_bands_tool.invoke(
                {
                    "scene_id": "LC09_MULTI",
                    "tar_path": str(tar_path),
                    "index_names": ["NDVI", "NDWI", "NBR"],
                    "sensor": "Landsat 9",
                    "west": WGS84_WEST,
                    "south": WGS84_SOUTH,
                    "east": WGS84_EAST,
                    "north": WGS84_NORTH,
                }
            )

        assert result["success"] is True
        assert len(result["indices_prepared"]) == 3

    def test_bad_sensor_returns_error(self, tmp_path, fake_settings):
        tar_path = make_scene_tar(tmp_path, "LC09_BAD")

        with patch(
            "spectral_agent.tools.raster.lc_tools.get_settings",
            return_value=fake_settings,
        ):
            result = crop_landsat_bands_tool.invoke(
                {
                    "scene_id": "LC09_BAD",
                    "tar_path": str(tar_path),
                    "index_names": ["NDVI"],
                    "sensor": "InvalidSensor",
                    "west": WGS84_WEST,
                    "south": WGS84_SOUTH,
                    "east": WGS84_EAST,
                    "north": WGS84_NORTH,
                }
            )

        assert result["success"] is False
        assert "error" in result


# ─────────────────────────────────────────────────────────────────────────────
# Tests — compute_spectral_index_tool
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeSpectralIndexTool:

    def test_compute_ndvi(self, tmp_path, fake_settings):
        # Create cropped band files
        bands_dir = tmp_path / "bands"
        band_paths = {
            "SR_B4": str(
                write_band_tif(bands_dir / "SR_B4.tif", reflectance_to_dn(0.10))
            ),
            "SR_B5": str(
                write_band_tif(bands_dir / "SR_B5.tif", reflectance_to_dn(0.50))
            ),
        }

        with patch(
            "spectral_agent.tools.raster.lc_tools.get_settings",
            return_value=fake_settings,
        ):
            result = compute_spectral_index_tool.invoke(
                {
                    "scene_id": "TEST_SCENE",
                    "index_names": ["NDVI"],
                    "band_paths": band_paths,
                    "sensor": "Landsat 9",
                }
            )

        assert result["success"] is True
        assert result["indices_computed"] == ["NDVI"]
        ndvi_info = result["results"][0]
        assert ndvi_info["value_mean"] > 0.5  # healthy vegetation
        assert Path(ndvi_info["output_path"]).exists()

    def test_compute_multiple(self, tmp_path, fake_settings):
        bands_dir = tmp_path / "bands"
        band_paths = {
            "SR_B4": str(
                write_band_tif(bands_dir / "SR_B4.tif", reflectance_to_dn(0.10))
            ),
            "SR_B5": str(
                write_band_tif(bands_dir / "SR_B5.tif", reflectance_to_dn(0.50))
            ),
        }

        with patch(
            "spectral_agent.tools.raster.lc_tools.get_settings",
            return_value=fake_settings,
        ):
            result = compute_spectral_index_tool.invoke(
                {
                    "scene_id": "TEST_SCENE",
                    "index_names": ["NDVI", "SAVI"],
                    "band_paths": band_paths,
                    "sensor": "Landsat 9",
                }
            )

        assert result["success"] is True
        assert len(result["results"]) == 2

    def test_missing_band_returns_error(self, tmp_path, fake_settings):
        bands_dir = tmp_path / "bands"
        band_paths = {
            "SR_B5": str(
                write_band_tif(bands_dir / "SR_B5.tif", reflectance_to_dn(0.50))
            ),
        }

        with patch(
            "spectral_agent.tools.raster.lc_tools.get_settings",
            return_value=fake_settings,
        ):
            result = compute_spectral_index_tool.invoke(
                {
                    "scene_id": "TEST_SCENE",
                    "index_names": ["NDVI"],
                    "band_paths": band_paths,
                    "sensor": "Landsat 9",
                }
            )

        assert result["success"] is False

    def test_compute_ndvi_without_band_paths_uses_cache(self, tmp_path):
        processed_root = tmp_path / "processed" / "default" / "landsat"
        scene_id = "TEST_SCENE_CACHE"
        bbox_hash = "abc123def456"
        scene_dir = processed_root / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)

        b4_path = write_band_tif(
            scene_dir / f"{scene_id}_SR_B4_crop_{bbox_hash}.tif",
            reflectance_to_dn(0.10),
        )
        b5_path = write_band_tif(
            scene_dir / f"{scene_id}_SR_B5_crop_{bbox_hash}.tif",
            reflectance_to_dn(0.50),
        )

        cache = BandCache(root=processed_root)
        cache.register_band(scene_id, "SR_B4", bbox_hash, b4_path)
        cache.register_band(scene_id, "SR_B5", bbox_hash, b5_path)

        with patch(
            "spectral_agent.tools.raster.lc_tools.session_processed_dir",
            return_value=processed_root,
        ):
            result = compute_spectral_index_tool.invoke(
                {
                    "scene_id": scene_id,
                    "index_names": ["NDVI"],
                    "sensor": "Landsat 9",
                }
            )

        assert result["success"] is True
        assert result["indices_computed"] == ["NDVI"]
        assert result["band_paths_source"] == "cache"


# ─────────────────────────────────────────────────────────────────────────────
# Tests — generate_thematic_map_tool
# ─────────────────────────────────────────────────────────────────────────────


class TestGenerateThematicMapTool:

    def test_both_maps(self, tmp_path):
        idx_path = write_index_tif(tmp_path / "scene" / "NDVI.tif")

        result = generate_thematic_map_tool.invoke(
            {
                "raster_path": str(idx_path),
                "index_name": "NDVI",
                "map_type": "both",
            }
        )

        assert result["success"] is True
        assert "static_png" in result["outputs"]
        assert "interactive_html" in result["outputs"]
        assert Path(result["outputs"]["static_png"]).exists()
        assert Path(result["outputs"]["interactive_html"]).exists()

    def test_static_only(self, tmp_path):
        idx_path = write_index_tif(tmp_path / "scene" / "NDVI.tif")

        result = generate_thematic_map_tool.invoke(
            {
                "raster_path": str(idx_path),
                "index_name": "NDVI",
                "map_type": "static",
            }
        )

        assert result["success"] is True
        assert "static_png" in result["outputs"]
        assert "interactive_html" not in result["outputs"]

    def test_interactive_only(self, tmp_path):
        idx_path = write_index_tif(tmp_path / "scene" / "NDVI.tif")

        result = generate_thematic_map_tool.invoke(
            {
                "raster_path": str(idx_path),
                "index_name": "NDVI",
                "map_type": "interactive",
            }
        )

        assert result["success"] is True
        assert "interactive_html" in result["outputs"]
        assert "static_png" not in result["outputs"]

    def test_custom_title(self, tmp_path):
        idx_path = write_index_tif(tmp_path / "scene" / "NDVI.tif")

        result = generate_thematic_map_tool.invoke(
            {
                "raster_path": str(idx_path),
                "index_name": "NDVI",
                "map_type": "static",
                "title": "Custom Title Test",
            }
        )

        assert result["success"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Tests — list_cached_bands_tool
# ─────────────────────────────────────────────────────────────────────────────


class TestListCachedBandsTool:

    def test_empty_cache(self, tmp_path, fake_settings):
        empty_dir = tmp_path / "processed" / "default" / "landsat"
        empty_dir.mkdir(parents=True, exist_ok=True)
        with patch(
            "spectral_agent.tools.raster.lc_tools.session_processed_dir",
            return_value=empty_dir,
        ):
            result = list_cached_bands_tool.invoke({})

        assert result["success"] is True
        assert result["total"] == 0
        assert result["bands"] == []

    def test_with_scene_filter(self, tmp_path, fake_settings):
        empty_dir = tmp_path / "processed" / "default" / "landsat"
        empty_dir.mkdir(parents=True, exist_ok=True)
        with patch(
            "spectral_agent.tools.raster.lc_tools.session_processed_dir",
            return_value=empty_dir,
        ):
            result = list_cached_bands_tool.invoke({"scene_id": "NONEXISTENT"})

        assert result["success"] is True
        assert result["total"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests — tool registries
# ─────────────────────────────────────────────────────────────────────────────


class TestToolRegistries:

    def test_ingestion_tools_count(self):
        tools = get_ingestion_tools()
        assert (
            len(tools) == 6
        )  # search/download landsat/sentinel + sentinel_index + multi_search

    def test_raster_tools_count(self):
        tools = get_raster_tools()
        assert (
            len(tools) == 4
        )  # crop, compute, list_cached, list_indices (map moved to visualization)

    def test_all_tools_count(self):
        all_tools = get_all_tools()
        assert len(all_tools) == 11  # 6 ingestion + 4 raster + 1 visualization

    def test_all_tools_have_names(self):
        for t in get_all_tools():
            assert hasattr(t, "name")
            assert t.name  # non-empty

    def test_no_duplicate_tool_names(self):
        names = [t.name for t in get_all_tools()]
        assert len(names) == len(set(names))

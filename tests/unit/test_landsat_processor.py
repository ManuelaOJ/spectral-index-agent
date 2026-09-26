"""
Unit tests for LandsatProcessor and BandCache.

These tests use synthetic GeoTIFF fixtures (tiny rasters) so they run
fast and require no real Landsat data.
"""

import tarfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from spectral_agent.schemas.imagery import BoundingBox
from spectral_agent.schemas.spectral_request import LandsatSensor
from spectral_agent.tools.raster.band_cache import BandCache
from spectral_agent.tools.raster.landsat_processor import LandsatProcessor

# ─────────────────────────────────────────────────────────────────────────────
# Helpers – synthetic raster / tar creation
# ─────────────────────────────────────────────────────────────────────────────

SCENE_ID = "LC09_L2SP_008057_20240101_20240105_02_T1"

# Small raster covering part of Colombia (UTM 18N)
RASTER_CRS = CRS.from_epsg(32618)
# Approx bounds in UTM 18N for lon ~-74, lat ~7
RASTER_WEST = 580_000.0
RASTER_SOUTH = 760_000.0
RASTER_WIDTH_M = 20_000.0  # 20 km
RASTER_HEIGHT_M = 20_000.0
RASTER_RES = 30  # 30 m pixel

COLS = int(RASTER_WIDTH_M / RASTER_RES)
ROWS = int(RASTER_HEIGHT_M / RASTER_RES)

RASTER_TRANSFORM = from_bounds(
    RASTER_WEST,
    RASTER_SOUTH,
    RASTER_WEST + RASTER_WIDTH_M,
    RASTER_SOUTH + RASTER_HEIGHT_M,
    COLS,
    ROWS,
)


def _write_synthetic_tif(path: Path, value: int = 5000) -> None:
    """Write a tiny single-band int16 GeoTIFF."""
    data = np.full((1, ROWS, COLS), value, dtype=np.int16)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=ROWS,
        width=COLS,
        count=1,
        dtype="int16",
        crs=RASTER_CRS,
        transform=RASTER_TRANSFORM,
        nodata=0,
    ) as dst:
        dst.write(data)


def _create_fake_scene(scene_dir: Path, bands: list[str]) -> None:
    """Create a directory with synthetic band TIFs."""
    scene_dir.mkdir(parents=True, exist_ok=True)
    for i, band in enumerate(bands, start=1):
        fname = f"{SCENE_ID}_{band}.TIF"
        _write_synthetic_tif(scene_dir / fname, value=1000 * i)


def _create_fake_tar(tar_path: Path, bands: list[str]) -> None:
    """Create a .tar containing synthetic band TIFs."""
    tmp_dir = tar_path.parent / "_tar_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for i, band in enumerate(bands, start=1):
        fname = f"{SCENE_ID}_{band}.TIF"
        _write_synthetic_tif(tmp_dir / fname, value=1000 * i)

    with tarfile.open(tar_path, "w") as tf:
        for f in tmp_dir.iterdir():
            tf.add(f, arcname=f.name)

    # clean up temp
    for f in tmp_dir.iterdir():
        f.unlink()
    tmp_dir.rmdir()


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

# BBox that falls INSIDE the synthetic raster (in WGS84 coords)
# UTM 18N (580000, 760000) ≈ (-74.15, 6.87) in WGS84
BBOX_COLOMBIA = BoundingBox(west=-74.10, south=6.90, east=-74.00, north=7.00)


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    d = tmp_path / "raw" / "landsat"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def processed_dir(tmp_path: Path) -> Path:
    d = tmp_path / "processed" / "landsat"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def fake_scene_dir(raw_dir: Path) -> Path:
    """Extracted scene with SR_B2..B7 and QA_PIXEL."""
    scene_dir = raw_dir / SCENE_ID
    _create_fake_scene(
        scene_dir, ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "QA_PIXEL"]
    )
    return scene_dir


@pytest.fixture
def fake_tar(raw_dir: Path) -> Path:
    """A .tar archive with two bands."""
    tar_path = raw_dir / f"{SCENE_ID}.tar"
    _create_fake_tar(tar_path, ["SR_B4", "SR_B5"])
    return tar_path


@pytest.fixture
def processor(raw_dir: Path, processed_dir: Path) -> LandsatProcessor:
    return LandsatProcessor(raw_dir=raw_dir, processed_dir=processed_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: BandCache
# ─────────────────────────────────────────────────────────────────────────────


class TestBandCache:
    """Tests for BandCache persistence and query logic."""

    def test_empty_cache(self, processed_dir: Path):
        cache = BandCache(root=processed_dir)
        assert len(cache) == 0
        assert cache.has_band("scene1", "SR_B4", "abc123") is False
        assert cache.get_band_path("scene1", "SR_B4", "abc123") is None

    def test_register_and_query(self, processed_dir: Path):
        cache = BandCache(root=processed_dir)
        # Create a dummy file
        dummy = processed_dir / "test.tif"
        dummy.write_text("fake")

        cache.register_band("scene1", "SR_B4", "abc123", dummy)

        assert cache.has_band("scene1", "SR_B4", "abc123")
        assert cache.get_band_path("scene1", "SR_B4", "abc123") == dummy
        assert cache.list_bands("scene1", "abc123") == ["SR_B4"]

    def test_persistence(self, processed_dir: Path):
        """Cache survives reload from disk."""
        dummy = processed_dir / "test.tif"
        dummy.write_text("fake")

        cache1 = BandCache(root=processed_dir)
        cache1.register_band("scene1", "SR_B4", "aaa", dummy)

        # New instance — should load from manifest
        cache2 = BandCache(root=processed_dir)
        assert cache2.has_band("scene1", "SR_B4", "aaa")

    def test_missing_file_evicts(self, processed_dir: Path):
        """If the file vanishes, has_band returns False and entry is removed."""
        dummy = processed_dir / "gone.tif"
        dummy.write_text("fake")

        cache = BandCache(root=processed_dir)
        cache.register_band("scene1", "SR_B5", "bbb", dummy)
        dummy.unlink()

        assert cache.has_band("scene1", "SR_B5", "bbb") is False

    def test_bbox_hash_deterministic(self):
        bbox = {"west": -74.1, "south": 6.9, "east": -74.0, "north": 7.0}
        h1 = BandCache.make_bbox_hash(bbox)
        h2 = BandCache.make_bbox_hash(bbox)
        assert h1 == h2
        assert len(h1) == 12

    def test_get_bands_for_index(self, processed_dir: Path):
        cache = BandCache(root=processed_dir)
        d1 = processed_dir / "b4.tif"
        d1.write_text("fake")
        cache.register_band("s1", "SR_B4", "h", d1)

        result = cache.get_bands_for_index("s1", ["SR_B4", "SR_B5"], "h")
        assert result["SR_B4"] is not None
        assert result["SR_B5"] is None

    def test_clear(self, processed_dir: Path):
        cache = BandCache(root=processed_dir)
        d = processed_dir / "x.tif"
        d.write_text("f")
        cache.register_band("s1", "SR_B4", "h", d)
        cache.register_band("s2", "SR_B5", "h", d)
        assert len(cache) == 2

        removed = cache.clear("s1")
        assert removed == 1
        assert len(cache) == 1

    def test_summary(self, processed_dir: Path):
        cache = BandCache(root=processed_dir)
        s = cache.summary()
        assert s["total_entries"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests: LandsatProcessor
# ─────────────────────────────────────────────────────────────────────────────


class TestLandsatProcessor:
    """Tests for tar extraction, band finding, and cropping."""

    def test_extract_tar(self, processor: LandsatProcessor, fake_tar: Path):
        scene_dir = processor.extract_tar(fake_tar)
        assert scene_dir.is_dir()
        tifs = list(scene_dir.glob("*.TIF"))
        assert len(tifs) == 2

    def test_extract_tar_idempotent(self, processor: LandsatProcessor, fake_tar: Path):
        """Second extraction reuses existing directory."""
        d1 = processor.extract_tar(fake_tar)
        d2 = processor.extract_tar(fake_tar)
        assert d1 == d2

    def test_extract_tar_not_found(self, processor: LandsatProcessor, raw_dir: Path):
        with pytest.raises(FileNotFoundError):
            processor.extract_tar(raw_dir / "nonexistent.tar")

    def test_find_band_file(self, fake_scene_dir: Path):
        path = LandsatProcessor.find_band_file(fake_scene_dir, "SR_B4")
        assert path is not None
        assert "SR_B4" in path.name

    def test_find_band_file_missing(self, fake_scene_dir: Path):
        assert LandsatProcessor.find_band_file(fake_scene_dir, "SR_B99") is None

    def test_crop_band(
        self, processor: LandsatProcessor, fake_scene_dir: Path, processed_dir: Path
    ):
        src = LandsatProcessor.find_band_file(fake_scene_dir, "SR_B4")
        out = processed_dir / "cropped.tif"
        meta = processor.crop_band(src, BBOX_COLOMBIA, out)

        assert out.exists()
        assert meta["width"] > 0
        assert meta["height"] > 0
        # Cropped should be smaller than original
        assert meta["width"] < COLS
        assert meta["height"] < ROWS

    def test_crop_band_preserves_crs(
        self, processor: LandsatProcessor, fake_scene_dir: Path, processed_dir: Path
    ):
        src = LandsatProcessor.find_band_file(fake_scene_dir, "SR_B5")
        out = processed_dir / "cropped_b5.tif"
        meta = processor.crop_band(src, BBOX_COLOMBIA, out)

        with rasterio.open(out) as ds:
            assert ds.crs == RASTER_CRS

    def test_crop_bands_for_index_ndvi(self, processor: LandsatProcessor, fake_scene_dir: Path):
        """NDVI needs SR_B4 (RED) + SR_B5 (NIR) for L9."""
        paths = processor.crop_bands_for_index(
            scene_id=SCENE_ID,
            scene_dir=fake_scene_dir,
            index_name="NDVI",
            sensor=LandsatSensor.LANDSAT_9,
            bbox=BBOX_COLOMBIA,
        )
        assert "SR_B4" in paths
        assert "SR_B5" in paths
        assert all(p.exists() for p in paths.values())

    def test_cache_reuses_bands(self, processor: LandsatProcessor, fake_scene_dir: Path):
        """Second call to crop_bands_for_index should hit cache."""
        processor.crop_bands_for_index(
            SCENE_ID,
            fake_scene_dir,
            "NDVI",
            LandsatSensor.LANDSAT_9,
            BBOX_COLOMBIA,
        )
        # SAVI also needs SR_B4 + SR_B5 — should be cached
        paths = processor.crop_bands_for_index(
            SCENE_ID,
            fake_scene_dir,
            "SAVI",
            LandsatSensor.LANDSAT_9,
            BBOX_COLOMBIA,
        )
        assert "SR_B4" in paths
        assert "SR_B5" in paths

    def test_crop_multiple_indices(self, processor: LandsatProcessor, fake_scene_dir: Path):
        result = processor.crop_multiple_indices(
            SCENE_ID,
            fake_scene_dir,
            ["NDVI", "NDWI"],
            LandsatSensor.LANDSAT_9,
            BBOX_COLOMBIA,
        )
        assert "NDVI" in result
        assert "NDWI" in result
        # NDWI needs SR_B3 + SR_B5; SR_B5 should be reused from NDVI crop
        assert "SR_B3" in result["NDWI"]
        assert "SR_B5" in result["NDWI"]

    def test_unsupported_index_raises(self, processor: LandsatProcessor, fake_scene_dir: Path):
        with pytest.raises(ValueError, match="not in band table"):
            processor.crop_bands_for_index(
                SCENE_ID,
                fake_scene_dir,
                "FAKE_IDX",
                LandsatSensor.LANDSAT_9,
                BBOX_COLOMBIA,
            )

    def test_list_scene_bands(self, fake_scene_dir: Path):
        bands = LandsatProcessor.list_scene_bands(LandsatProcessor, fake_scene_dir)
        assert "SR_B4" in bands
        assert "QA_PIXEL" in bands

    def test_get_raster_info(self, processor: LandsatProcessor, fake_scene_dir: Path):
        src = LandsatProcessor.find_band_file(fake_scene_dir, "SR_B4")
        info = processor.get_raster_info(src)
        assert info["width"] == COLS
        assert info["height"] == ROWS
        assert info["dtype"] == "int16"

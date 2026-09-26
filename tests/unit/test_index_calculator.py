"""
Unit tests for index_calculator — spectral index computation on Landsat
rasters.

All tests use synthetic GeoTIFFs with known DN values so we can verify
the formulas produce the correct results analytically.

Landsat Collection 2 Level 2 scaling:
    reflectance = DN * 0.0000275 − 0.2

To get a desired reflectance *r*, we reverse:
    DN = (r + 0.2) / 0.0000275
    https://www.usgs.gov/faqs/how-do-i-use-a-scale-factor-landsat-level-2-science-products
"""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from spectral_agent.schemas.spectral_request import LandsatSensor
from spectral_agent.tools.raster.index_calculator import (
    _EPS,
    LANDSAT_C2_L2_OFFSET,
    LANDSAT_C2_L2_SCALE,
    IndexResult,
    compute_and_save,
    compute_index,
    compute_indices,
    list_supported_indices,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

# Raster parameters for synthetic data
RASTER_CRS = CRS.from_epsg(32618)
RASTER_WIDTH = 20
RASTER_HEIGHT = 20
RASTER_TRANSFORM = from_bounds(500000, 700000, 500600, 700600, RASTER_WIDTH, RASTER_HEIGHT)
NODATA = 0


def reflectance_to_dn(reflectance: float) -> int:
    """Convert a reflectance value to Landsat C2 L2 DN."""
    return int(round((reflectance - LANDSAT_C2_L2_OFFSET) / LANDSAT_C2_L2_SCALE))


def write_synthetic_band(
    path: Path,
    fill_dn: int,
    *,
    nodata: int = NODATA,
    dtype: str = "uint16",
) -> Path:
    """Write a tiny single-band GeoTIFF with a constant DN value."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((1, RASTER_HEIGHT, RASTER_WIDTH), fill_dn, dtype=dtype)
    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "width": RASTER_WIDTH,
        "height": RASTER_HEIGHT,
        "count": 1,
        "crs": RASTER_CRS,
        "transform": RASTER_TRANSFORM,
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return path


def write_band_with_nodata(
    path: Path,
    fill_dn: int,
    nodata_dn: int = NODATA,
) -> Path:
    """Write a band where the first row is NoData and the rest is fill_dn."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((1, RASTER_HEIGHT, RASTER_WIDTH), fill_dn, dtype="uint16")
    data[0, 0, :] = nodata_dn  # first row = nodata
    profile = {
        "driver": "GTiff",
        "dtype": "uint16",
        "width": RASTER_WIDTH,
        "height": RASTER_HEIGHT,
        "count": 1,
        "crs": RASTER_CRS,
        "transform": RASTER_TRANSFORM,
        "nodata": nodata_dn,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def bands_dir(tmp_path: Path) -> Path:
    """Directory for synthetic band GeoTIFFs."""
    d = tmp_path / "bands"
    d.mkdir()
    return d


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    d = tmp_path / "output"
    d.mkdir()
    return d


@pytest.fixture
def ndvi_band_paths(bands_dir: Path) -> dict[str, Path]:
    """
    Create SR_B4 (RED) and SR_B5 (NIR) with known reflectance.

    RED = 0.10, NIR = 0.50
    Expected NDVI ≈ (0.50 − 0.10) / (0.50 + 0.10) ≈ 0.6667
    """
    dn_red = reflectance_to_dn(0.10)
    dn_nir = reflectance_to_dn(0.50)
    return {
        "SR_B4": write_synthetic_band(bands_dir / "SR_B4.tif", dn_red),
        "SR_B5": write_synthetic_band(bands_dir / "SR_B5.tif", dn_nir),
    }


@pytest.fixture
def evi_band_paths(bands_dir: Path) -> dict[str, Path]:
    """
    SR_B2 (BLUE=0.05), SR_B4 (RED=0.10), SR_B5 (NIR=0.50)

    EVI = 2.5*(NIR-RED)/(NIR + 6*RED - 7.5*BLUE + 1)
        = 2.5*(0.4)/(0.5 + 0.6 - 0.375 + 1)
        = 1.0 / 1.725
        ≈ 0.5797
    """
    return {
        "SR_B2": write_synthetic_band(bands_dir / "SR_B2.tif", reflectance_to_dn(0.05)),
        "SR_B4": write_synthetic_band(bands_dir / "SR_B4.tif", reflectance_to_dn(0.10)),
        "SR_B5": write_synthetic_band(bands_dir / "SR_B5.tif", reflectance_to_dn(0.50)),
    }


@pytest.fixture
def all_l89_bands(bands_dir: Path) -> dict[str, Path]:
    """
    All Landsat 8/9 SR bands needed for every supported index.
                  reflectance
    SR_B2 BLUE    0.05
    SR_B3 GREEN   0.08
    SR_B4 RED     0.10
    SR_B5 NIR     0.50
    SR_B6 SWIR1   0.25
    SR_B7 SWIR2   0.15
    """
    spec = {
        "SR_B2": 0.05,
        "SR_B3": 0.08,
        "SR_B4": 0.10,
        "SR_B5": 0.50,
        "SR_B6": 0.25,
        "SR_B7": 0.15,
    }
    paths = {}
    for band, refl in spec.items():
        paths[band] = write_synthetic_band(bands_dir / f"{band}.tif", reflectance_to_dn(refl))
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Tests — pure compute_index (arrays)
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeIndex:
    """Test the pure NumPy compute_index function."""

    def test_ndvi_basic(self):
        nir = np.full((5, 5), 0.50)
        red = np.full((5, 5), 0.10)
        result = compute_index("NDVI", {"NIR": nir, "RED": red})
        expected = (0.50 - 0.10) / (0.50 + 0.10 + _EPS)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_ndvi_negative(self):
        """When RED > NIR, NDVI should be negative (bare soil / water)."""
        nir = np.full((5, 5), 0.05)
        red = np.full((5, 5), 0.30)
        result = compute_index("NDVI", {"NIR": nir, "RED": red})
        assert result[0, 0] < 0

    def test_evi_basic(self):
        nir = np.full((5, 5), 0.50)
        red = np.full((5, 5), 0.10)
        blue = np.full((5, 5), 0.05)
        result = compute_index("EVI", {"NIR": nir, "RED": red, "BLUE": blue})
        expected = 2.5 * (0.40) / (0.50 + 0.60 - 0.375 + 1.0)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_savi_basic(self):
        nir = np.full((5, 5), 0.50)
        red = np.full((5, 5), 0.10)
        result = compute_index("SAVI", {"NIR": nir, "RED": red})
        expected = 1.5 * (0.40) / (0.50 + 0.10 + 0.5)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_ndwi_basic(self):
        green = np.full((5, 5), 0.08)
        nir = np.full((5, 5), 0.50)
        result = compute_index("NDWI", {"GREEN": green, "NIR": nir})
        expected = (0.08 - 0.50) / (0.08 + 0.50 + _EPS)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_nbr_basic(self):
        nir = np.full((5, 5), 0.50)
        swir2 = np.full((5, 5), 0.15)
        result = compute_index("NBR", {"NIR": nir, "SWIR2": swir2})
        expected = (0.50 - 0.15) / (0.50 + 0.15 + _EPS)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_ndbi_basic(self):
        swir1 = np.full((5, 5), 0.25)
        nir = np.full((5, 5), 0.50)
        result = compute_index("NDBI", {"SWIR1": swir1, "NIR": nir})
        expected = (0.25 - 0.50) / (0.25 + 0.50 + _EPS)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_nodata_mask(self):
        """Pixels flagged as nodata should become NaN."""
        nir = np.full((5, 5), 0.50)
        red = np.full((5, 5), 0.10)
        mask = np.zeros((5, 5), dtype=bool)
        mask[0, :] = True  # first row is nodata
        result = compute_index("NDVI", {"NIR": nir, "RED": red}, nodata_mask=mask)
        assert np.all(np.isnan(result[0, :]))
        assert not np.any(np.isnan(result[1:, :]))

    def test_unknown_index_raises(self):
        with pytest.raises(ValueError, match="Unknown index"):
            compute_index("FAKE_INDEX", {"NIR": np.zeros(5)})

    def test_missing_role_raises(self):
        """Missing a required role should produce a clear error."""
        with pytest.raises(ValueError, match="Missing band role"):
            compute_index("NDVI", {"NIR": np.ones(5)})  # missing RED

    def test_case_insensitive(self):
        nir = np.full((5, 5), 0.50)
        red = np.full((5, 5), 0.10)
        result = compute_index("ndvi", {"NIR": nir, "RED": red})
        assert result.shape == (5, 5)


# ─────────────────────────────────────────────────────────────────────────────
# Tests — compute_and_save (file I/O)
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeAndSave:
    """Test end-to-end compute + GeoTIFF write."""

    def test_ndvi_file(self, ndvi_band_paths, output_dir):
        out = output_dir / "NDVI.tif"
        result = compute_and_save(
            "NDVI",
            ndvi_band_paths,
            LandsatSensor.LANDSAT_9,
            out,
        )
        assert isinstance(result, IndexResult)
        assert result.output_path == out
        assert out.exists()
        assert result.width == RASTER_WIDTH
        assert result.height == RASTER_HEIGHT
        assert result.crs == str(RASTER_CRS)

        # Value check: NDVI ≈ 0.6667
        expected = (0.50 - 0.10) / (0.50 + 0.10 + _EPS)
        assert abs(result.value_mean - expected) < 0.02  # small tolerance for scale rounding

    def test_evi_file(self, evi_band_paths, output_dir):
        out = output_dir / "EVI.tif"
        result = compute_and_save("EVI", evi_band_paths, LandsatSensor.LANDSAT_8, out)
        assert result.output_path.exists()
        expected = 2.5 * 0.40 / (0.50 + 0.60 - 0.375 + 1.0)
        assert abs(result.value_mean - expected) < 0.02

    def test_output_is_float32(self, ndvi_band_paths, output_dir):
        out = output_dir / "NDVI.tif"
        compute_and_save("NDVI", ndvi_band_paths, LandsatSensor.LANDSAT_9, out)
        with rasterio.open(out) as src:
            assert src.dtypes[0] == "float32"
            assert src.count == 1
            assert src.nodata is not None  # NaN
            tags = src.tags()
            assert tags["index"] == "NDVI"
            assert tags["sensor"] == "Landsat 9"

    def test_nodata_pixels_in_output(self, bands_dir, output_dir):
        """Verify that nodata in input bands propagates to NaN in the index."""
        # Write NIR and RED where first row = nodata
        dn_red = reflectance_to_dn(0.10)
        dn_nir = reflectance_to_dn(0.50)
        paths = {
            "SR_B4": write_band_with_nodata(bands_dir / "SR_B4_nd.tif", dn_red),
            "SR_B5": write_band_with_nodata(bands_dir / "SR_B5_nd.tif", dn_nir),
        }
        out = output_dir / "NDVI_nd.tif"
        result = compute_and_save("NDVI", paths, LandsatSensor.LANDSAT_9, out)
        assert result.nodata_pct > 0

        with rasterio.open(out) as src:
            data = src.read(1)
            # First row should be NaN
            assert np.all(np.isnan(data[0, :]))
            # Other rows should have valid NDVI
            assert not np.any(np.isnan(data[1:, :]))

    def test_without_scale(self, bands_dir, output_dir):
        """apply_scale=False should treat raw values as reflectance."""
        # Write bands with raw reflectance values (float)
        nir_path = bands_dir / "SR_B5_raw.tif"
        red_path = bands_dir / "SR_B4_raw.tif"
        nir_path.parent.mkdir(exist_ok=True)

        for path, val in [(nir_path, 0.50), (red_path, 0.10)]:
            data = np.full((1, RASTER_HEIGHT, RASTER_WIDTH), val, dtype="float32")
            profile = {
                "driver": "GTiff",
                "dtype": "float32",
                "width": RASTER_WIDTH,
                "height": RASTER_HEIGHT,
                "count": 1,
                "crs": RASTER_CRS,
                "transform": RASTER_TRANSFORM,
                "nodata": None,
            }
            with rasterio.open(path, "w", **profile) as dst:
                dst.write(data)

        out = output_dir / "NDVI_raw.tif"
        result = compute_and_save(
            "NDVI",
            {"SR_B4": red_path, "SR_B5": nir_path},
            LandsatSensor.LANDSAT_9,
            out,
            apply_scale=False,
        )
        expected = (0.50 - 0.10) / (0.50 + 0.10 + _EPS)
        assert abs(result.value_mean - expected) < 1e-4

    def test_missing_band_raises(self, ndvi_band_paths, output_dir):
        """Providing incomplete band_paths should raise ValueError."""
        del ndvi_band_paths["SR_B4"]
        with pytest.raises(ValueError, match="requires band"):
            compute_and_save(
                "NDVI",
                ndvi_band_paths,
                LandsatSensor.LANDSAT_9,
                output_dir / "fail.tif",
            )

    def test_missing_file_raises(self, output_dir):
        """band_paths that point to nonexistent files should raise."""
        paths = {
            "SR_B4": Path("/nonexistent/SR_B4.tif"),
            "SR_B5": Path("/nonexistent/SR_B5.tif"),
        }
        with pytest.raises(FileNotFoundError):
            compute_and_save(
                "NDVI",
                paths,
                LandsatSensor.LANDSAT_9,
                output_dir / "fail.tif",
            )

    def test_landsat_457_sensor(self, bands_dir, output_dir):
        """Landsat 5 uses different band names (SR_B3=RED, SR_B4=NIR)."""
        dn_red = reflectance_to_dn(0.10)
        dn_nir = reflectance_to_dn(0.50)
        paths = {
            "SR_B3": write_synthetic_band(bands_dir / "SR_B3.tif", dn_red),
            "SR_B4": write_synthetic_band(bands_dir / "SR_B4.tif", dn_nir),
        }
        out = output_dir / "NDVI_L5.tif"
        result = compute_and_save("NDVI", paths, LandsatSensor.LANDSAT_5, out)
        expected = (0.50 - 0.10) / (0.50 + 0.10 + _EPS)
        assert abs(result.value_mean - expected) < 0.02
        assert result.output_path.exists()


# ─────────────────────────────────────────────────────────────────────────────
# Tests — compute_indices (batch)
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeIndices:
    """Test batch computation of multiple indices."""

    def test_batch_two_indices(self, ndvi_band_paths, output_dir):
        """NDVI and SAVI share the same bands (RED, NIR)."""
        results = compute_indices(
            ["NDVI", "SAVI"],
            ndvi_band_paths,
            LandsatSensor.LANDSAT_9,
            output_dir,
            scene_id="TEST_SCENE",
        )
        assert len(results) == 2
        assert results[0].index_name == "NDVI"
        assert results[1].index_name == "SAVI"
        for r in results:
            assert r.output_path.exists()
            assert "TEST_SCENE" in r.output_path.name

    def test_batch_all_indices(self, all_l89_bands, output_dir):
        """Compute all 6 supported indices for Landsat 8/9."""
        results = compute_indices(
            list_supported_indices(),
            all_l89_bands,
            LandsatSensor.LANDSAT_9,
            output_dir,
            scene_id="FULL_TEST",
        )
        assert len(results) == 6
        names = {r.index_name for r in results}
        assert names == {"NDVI", "EVI", "SAVI", "NDWI", "NBR", "NDBI"}
        for r in results:
            assert r.output_path.exists()
            assert not np.isnan(r.value_mean)

    def test_batch_empty_list(self, ndvi_band_paths, output_dir):
        results = compute_indices([], ndvi_band_paths, LandsatSensor.LANDSAT_9, output_dir)
        assert results == []


# ─────────────────────────────────────────────────────────────────────────────
# Tests — utilities
# ─────────────────────────────────────────────────────────────────────────────


class TestUtilities:
    def test_list_supported_indices(self):
        indices = list_supported_indices()
        assert isinstance(indices, list)
        assert len(indices) == 6
        assert "NDVI" in indices
        assert indices == sorted(indices)  # should be sorted

    def test_reflectance_round_trip(self):
        """Verify our test helper reflectance→DN→reflectance round-trips."""
        for refl in [0.05, 0.10, 0.25, 0.50, 0.80]:
            dn = reflectance_to_dn(refl)
            recovered = dn * LANDSAT_C2_L2_SCALE + LANDSAT_C2_L2_OFFSET
            assert abs(recovered - refl) < 0.001

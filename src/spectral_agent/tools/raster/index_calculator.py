"""
Spectral index calculator for Landsat rasters.

Reads cropped GeoTIFF bands, applies Landsat Collection 2 Level-2 scale
factors, computes the requested spectral index, and writes the result as
a single-band Float32 GeoTIFF.

Workflow
--------
1. ``compute_index()``       — pure NumPy: arrays in → index array out.
2. ``compute_and_save()``    — end-to-end for a *single* index: read crop
                               GeoTIFFs, scale, compute, write GeoTIFF.
3. ``compute_indices()``     — batch: compute several indices for one scene.

Landsat Collection 2 Level 2 reflectance scaling
--------------------------------------------------
  ``reflectance = DN * 0.0000275 − 0.2``

The module clamps reflectance to [0, 1] after scaling. Negative values
are fill artefacts and are marked NoData; exactly 0 is physically valid.

Value ranges
------------
  * Normalised-difference indices (NDVI, NDWI, NBR, NDBI): [−1, 1]
  * EVI: approximately [−1, 1]
  * SAVI: approximately [−1.5, 1.5]

NoData pixels (where *any* input band is NoData or reflectance ≤ 0) are
written as ``NaN`` in the output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import rasterio

from spectral_agent.schemas.spectral_request import (
    LANDSAT_89_INDEX_BANDS,
    LANDSAT_457_INDEX_BANDS,
    LandsatSensor,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

#: Landsat Collection 2 Level-2 surface-reflectance multiplicative factor.
LANDSAT_C2_L2_SCALE: float = 0.0000275
#: Landsat Collection 2 Level-2 surface-reflectance additive offset.
LANDSAT_C2_L2_OFFSET: float = -0.2

#: Small epsilon to prevent division by zero.
_EPS: float = 1e-10

#: SAVI soil brightness correction factor (L).
#: L = 0.5 accommodates most land cover types.
_SAVI_L: float = 0.5

#: Index value ranges for documentation / clamping (min, max).
INDEX_VALUE_RANGES: dict[str, tuple[float, float]] = {
    "NDVI": (-1.0, 1.0),
    "EVI": (-1.0, 1.0),
    "SAVI": (-1.5, 1.5),
    "NDWI": (-1.0, 1.0),
    "NBR": (-1.0, 1.0),
    "NDBI": (-1.0, 1.0),
}


# ─────────────────────────────────────────────────────────────────────────────
# Formula registry
# ─────────────────────────────────────────────────────────────────────────────
#
# Each formula is a callable that receives a ``dict[str, np.ndarray]`` where
# keys are spectral *roles* (RED, NIR, BLUE, …) and arrays are Float64
# reflectance values in [0, 1].  Returns a single ``np.ndarray``.


def _ndvi(b: dict[str, np.ndarray]) -> np.ndarray:
    """Normalized Difference Vegetation Index."""
    return (b["NIR"] - b["RED"]) / (b["NIR"] + b["RED"] + _EPS)


def _evi(b: dict[str, np.ndarray]) -> np.ndarray:
    """Enhanced Vegetation Index."""
    return (
        2.5
        * (b["NIR"] - b["RED"])
        / (b["NIR"] + 6.0 * b["RED"] - 7.5 * b["BLUE"] + 1.0)
    )


def _savi(b: dict[str, np.ndarray]) -> np.ndarray:
    """Soil Adjusted Vegetation Index.

    Formula: ((NIR - RED) / (NIR + RED + L)) * (1 + L)
    where L = 0.5 (soil brightness correction factor).

    Landsat 4-7: NIR=Band 4, RED=Band 3
    Landsat 8-9: NIR=Band 5, RED=Band 4
    """
    return (1 + _SAVI_L) * (b["NIR"] - b["RED"]) / (b["NIR"] + b["RED"] + _SAVI_L)


def _ndwi(b: dict[str, np.ndarray]) -> np.ndarray:
    """Normalized Difference Water Index."""
    return (b["GREEN"] - b["NIR"]) / (b["GREEN"] + b["NIR"] + _EPS)


def _nbr(b: dict[str, np.ndarray]) -> np.ndarray:
    """Normalized Burn Ratio."""
    return (b["NIR"] - b["SWIR2"]) / (b["NIR"] + b["SWIR2"] + _EPS)


def _ndbi(b: dict[str, np.ndarray]) -> np.ndarray:
    """Normalized Difference Built-up Index."""
    return (b["SWIR1"] - b["NIR"]) / (b["SWIR1"] + b["NIR"] + _EPS)


INDEX_FORMULAS: dict[str, Callable[[dict[str, np.ndarray]], np.ndarray]] = {
    "NDVI": _ndvi,
    "EVI": _evi,
    "SAVI": _savi,
    "NDWI": _ndwi,
    "NBR": _nbr,
    "NDBI": _ndbi,
}


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class IndexResult:
    """Metadata returned after computing and saving an index."""

    index_name: str
    output_path: Path
    width: int
    height: int
    crs: str
    bounds: list[float]
    value_min: float
    value_max: float
    value_mean: float
    nodata_pct: float  # percentage of NaN pixels


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def list_supported_indices() -> list[str]:
    """Return sorted list of index names this module can compute."""
    return sorted(INDEX_FORMULAS)


def compute_index(
    index_name: str,
    role_arrays: dict[str, np.ndarray],
    *,
    nodata_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute a spectral index from role-keyed reflectance arrays.

    Parameters
    ----------
    index_name : str
        One of ``NDVI``, ``EVI``, ``SAVI``, ``NDWI``, ``NBR``, ``NDBI``.
    role_arrays : dict[str, ndarray]
        ``{"RED": array, "NIR": array, ...}`` — Float64 reflectance in
        [0, 1].  All arrays must have the same shape.
    nodata_mask : ndarray | None
        Boolean mask (``True`` = nodata).  If supplied, those pixels are
        set to ``NaN`` in the output.

    Returns
    -------
    ndarray
        Float64 array with the computed index values.  NoData → ``NaN``.

    Raises
    ------
    ValueError
        Unknown index name or missing role.
    """
    name = index_name.upper()
    if name not in INDEX_FORMULAS:
        raise ValueError(
            f"Unknown index '{index_name}'. Supported: {list_supported_indices()}"
        )

    formula = INDEX_FORMULAS[name]

    # Validate roles
    # Peek at what roles the formula needs by checking the band-table
    # (We accept any dict that provides all required keys.)
    try:
        result = formula(role_arrays)
    except KeyError as exc:
        raise ValueError(
            f"Missing band role {exc} for index '{name}'. "
            f"Provided roles: {sorted(role_arrays)}"
        ) from exc

    # Apply nodata mask
    if nodata_mask is not None:
        result = np.where(nodata_mask, np.nan, result)

    return result


def compute_and_save(
    index_name: str,
    band_paths: dict[str, Path],
    sensor: LandsatSensor,
    output_path: Path,
    *,
    apply_scale: bool = True,
) -> IndexResult:
    """
    End-to-end: read cropped band TIFFs → compute index → write GeoTIFF.

    Parameters
    ----------
    index_name : str
        Spectral index to compute (e.g. ``"NDVI"``).
    band_paths : dict[str, Path]
        ``{band_name: path}`` — the cropped GeoTIFFs produced by
        :class:`LandsatProcessor`.  Keys are Landsat band names like
        ``"SR_B4"``, ``"SR_B5"``, etc.
    sensor : LandsatSensor
        Determines which band table to use for resolving roles.
    output_path : Path
        Where to write the resulting index GeoTIFF.
    apply_scale : bool
        If ``True`` (default), apply the Landsat Collection 2 Level-2
        scale/offset before computing the index.  Set to ``False`` if
        the input bands are already in reflectance units.

    Returns
    -------
    IndexResult
        Metadata about the written GeoTIFF.

    Raises
    ------
    ValueError
        If index or sensor is unsupported, or required bands are missing.
    FileNotFoundError
        If a band file doesn't exist.
    """
    name = index_name.upper()
    if name not in INDEX_FORMULAS:
        raise ValueError(
            f"Unknown index '{index_name}'. Supported: {list_supported_indices()}"
        )

    # Resolve role → band_name mapping
    band_table = _get_band_table(sensor)
    if name not in band_table:
        raise ValueError(f"Index '{name}' not in band table for {sensor.value}.")

    role_to_band: dict[str, str] = band_table[name]

    # Check that all required band files are provided
    for role, band_name in role_to_band.items():
        if band_name not in band_paths:
            raise ValueError(
                f"Index '{name}' requires band '{band_name}' (role={role}) "
                f"but it was not provided. Available: {sorted(band_paths)}"
            )
        if not Path(band_paths[band_name]).exists():
            raise FileNotFoundError(f"Band file not found: {band_paths[band_name]}")

    # Read one reference band for shape / CRS / transform
    ref_band_name = list(role_to_band.values())[0]
    with rasterio.open(band_paths[ref_band_name]) as ref:
        height = ref.height
        width = ref.width
        crs = ref.crs
        transform = ref.transform
        bounds = list(ref.bounds)
        ref_nodata = ref.nodata

    # Read all bands and build role→array mapping
    role_arrays: dict[str, np.ndarray] = {}
    nodata_mask = np.zeros((height, width), dtype=bool)

    for role, band_name in role_to_band.items():
        with rasterio.open(band_paths[band_name]) as src:
            raw = src.read(1).astype(np.float64)
            nd = src.nodata

        # Build nodata mask (union across all bands)
        if nd is not None:
            nodata_mask |= raw == nd

        if apply_scale:
            scaled = raw * LANDSAT_C2_L2_SCALE + LANDSAT_C2_L2_OFFSET
            # Values < 0 are fill artefacts; exactly 0 is physically valid
            nodata_mask |= scaled < 0
            scaled = np.clip(scaled, 0.0, 1.0)
        else:
            scaled = raw

        role_arrays[role] = scaled

    # Compute the index
    index_arr = compute_index(name, role_arrays, nodata_mask=nodata_mask)
    index_arr = index_arr.astype(np.float32)

    # Write output GeoTIFF
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": width,
        "height": height,
        "count": 1,
        "crs": crs,
        "transform": transform,
        "nodata": float("nan"),
        "compress": "deflate",
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(index_arr, 1)
        dst.update_tags(index=name, sensor=sensor.value)

    # Compute stats (ignoring NaN)
    valid = index_arr[~np.isnan(index_arr)]
    nodata_count = int(np.isnan(index_arr).sum())
    total = index_arr.size

    result = IndexResult(
        index_name=name,
        output_path=output_path,
        width=width,
        height=height,
        crs=str(crs),
        bounds=bounds,
        value_min=float(valid.min()) if valid.size > 0 else float("nan"),
        value_max=float(valid.max()) if valid.size > 0 else float("nan"),
        value_mean=float(valid.mean()) if valid.size > 0 else float("nan"),
        nodata_pct=round(100 * nodata_count / total, 2) if total > 0 else 0.0,
    )

    logger.info(
        "Computed %s → %s  (min=%.3f, max=%.3f, mean=%.3f, nodata=%.1f%%)",
        name,
        output_path.name,
        result.value_min,
        result.value_max,
        result.value_mean,
        result.nodata_pct,
    )

    return result


def compute_indices(
    index_names: list[str],
    band_paths: dict[str, Path],
    sensor: LandsatSensor,
    output_dir: Path,
    scene_id: str = "scene",
    *,
    apply_scale: bool = True,
) -> list[IndexResult]:
    """
    Batch-compute several indices for the same scene.

    Parameters
    ----------
    index_names : list[str]
        Indices to compute.
    band_paths : dict[str, Path]
        All cropped band paths available for the scene.
    sensor : LandsatSensor
        Sensor for band-role resolution.
    output_dir : Path
        Directory for output GeoTIFFs.
    scene_id : str
        Used in output filenames.
    apply_scale : bool
        Apply Landsat C2 L2 scaling.

    Returns
    -------
    list[IndexResult]
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[IndexResult] = []
    for idx_name in index_names:
        out_path = output_dir / f"{scene_id}_{idx_name.upper()}.tif"
        result = compute_and_save(
            index_name=idx_name,
            band_paths=band_paths,
            sensor=sensor,
            output_path=out_path,
            apply_scale=apply_scale,
        )
        results.append(result)

    logger.info(
        "Computed %d indices for %s: %s",
        len(results),
        scene_id,
        [r.index_name for r in results],
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────


def _get_band_table(sensor: LandsatSensor) -> dict[str, dict[str, str]]:
    """Return the index → {role: band_name} mapping for the given sensor."""
    if sensor in (LandsatSensor.LANDSAT_8, LandsatSensor.LANDSAT_9):
        return LANDSAT_89_INDEX_BANDS
    return LANDSAT_457_INDEX_BANDS

"""
Landsat raster processor — .tar extraction and band cropping.

This module handles the gap between the raw `.tar` archive downloaded
from USGS M2M and the cropped single-band GeoTIFFs the index calculator
needs.

Workflow
--------
1. ``extract_tar()``   – unpack the archive into a scene directory.
2. ``find_band_file()``– locate a specific band TIF inside the scene dir.
3. ``crop_band()``     – read one band TIF, clip it to a bbox, write out a
                         cropped GeoTIFF.
4. ``crop_bands_for_index()`` – high-level: given an index name + sensor,
                                resolve required bands, check cache, crop
                                only the missing ones, return paths.

All output GeoTIFFs keep the original CRS and spatial resolution.
"""

from __future__ import annotations

import logging
import tarfile
from pathlib import Path
from typing import Any

import rasterio
from rasterio.mask import mask as rasterio_mask
from shapely.geometry import box, mapping

from spectral_agent.schemas.imagery import BoundingBox
from spectral_agent.schemas.spectral_request import (
    LANDSAT_89_INDEX_BANDS,
    LANDSAT_457_INDEX_BANDS,
    LandsatSensor,
)

from .band_cache import BandCache

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


class LandsatProcessor:
    """
    Extracts and crops Landsat bands from downloaded .tar archives.

    Parameters
    ----------
    raw_dir : Path
        Where the raw ``.tar`` files live (e.g. ``data/raw/landsat``).
    processed_dir : Path
        Where cropped bands will be stored (e.g. ``data/processed/landsat``).
    cache : BandCache | None
        Optional band cache; if ``None`` a new one is created under
        *processed_dir*.

    Example
    -------
    >>> proc = LandsatProcessor(
    ...     raw_dir=Path("data/raw/landsat"),
    ...     processed_dir=Path("data/processed/landsat"),
    ... )
    >>> scene_dir = proc.extract_tar(Path("data/raw/landsat/LC09_L2SP_xxx.tar"))
    >>> paths = proc.crop_bands_for_index(
    ...     scene_id="LC09_L2SP_xxx",
    ...     scene_dir=scene_dir,
    ...     index_name="NDVI",
    ...     sensor=LandsatSensor.LANDSAT_9,
    ...     bbox=BoundingBox(west=-74.2, south=6.9, east=-73.8, north=7.2),
    ... )
    """

    def __init__(
        self,
        raw_dir: Path,
        processed_dir: Path,
        cache: BandCache | None = None,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.cache = cache or BandCache(root=self.processed_dir)

    # ── 1. Tar extraction ───────────────────────────────────────────────

    def extract_tar(self, tar_path: Path, *, force: bool = False) -> Path:
        """
        Extract a Landsat ``.tar`` archive into a scene directory.

        Parameters
        ----------
        tar_path : Path
            Path to the ``.tar`` file.
        force : bool
            Re-extract even if the directory already exists.

        Returns
        -------
        Path
            The directory containing the extracted files.

        Raises
        ------
        FileNotFoundError
            If *tar_path* does not exist.
        tarfile.TarError
            If the archive is corrupt.
        """
        tar_path = Path(tar_path)
        if not tar_path.exists():
            raise FileNotFoundError(f"Tar file not found: {tar_path}")

        scene_id = tar_path.stem  # e.g. LC09_L2SP_008057_20240101_...
        scene_dir = self.raw_dir / scene_id

        if scene_dir.exists() and not force:
            tifs = list(scene_dir.glob("*.TIF")) + list(scene_dir.glob("*.tif"))
            if tifs:
                logger.info("Scene already extracted (%d TIFs): %s", len(tifs), scene_dir)
                return scene_dir

        scene_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Extracting %s → %s ...", tar_path.name, scene_dir)

        with tarfile.open(tar_path, "r") as tf:
            # Security: prevent path-traversal attacks
            for member in tf.getmembers():
                member_path = Path(scene_dir / member.name).resolve()
                if not str(member_path).startswith(str(scene_dir.resolve())):
                    raise tarfile.TarError(f"Path traversal detected in tar member: {member.name}")
            tf.extractall(path=scene_dir, filter="data")

        tifs = list(scene_dir.glob("*.TIF")) + list(scene_dir.glob("*.tif"))
        logger.info("Extracted %d files (%d TIFs)", len(list(scene_dir.iterdir())), len(tifs))
        return scene_dir

    # ── 2. Find a band file ─────────────────────────────────────────────

    @staticmethod
    def find_band_file(scene_dir: Path, band_name: str) -> Path | None:
        """
        Locate the GeoTIFF for a given band inside an extracted scene.

        Landsat naming convention:
          ``<scene_id>_<band_name>.TIF``
          e.g. ``LC09_L2SP_008057_20240101_..._SR_B4.TIF``

        Parameters
        ----------
        scene_dir : Path
            Directory with extracted scene files.
        band_name : str
            Band identifier, e.g. ``"SR_B4"``.

        Returns
        -------
        Path | None
            Full path to the TIF, or ``None`` if not found.
        """
        # Try both cases (.TIF and .tif)
        for pattern in (f"*_{band_name}.TIF", f"*_{band_name}.tif"):
            matches = list(scene_dir.glob(pattern))
            if matches:
                return matches[0]
        return None

    # ── 3. Crop a single band ──────────────────────────────────────────

    def crop_band(
        self,
        band_path: Path,
        bbox: BoundingBox,
        output_path: Path,
    ) -> dict[str, Any]:
        """
        Read a Landsat band GeoTIFF and clip it to the given bounding box.

        The bbox (WGS84) is reprojected on-the-fly to the raster's native
        CRS before clipping, so this works for any UTM zone.

        Parameters
        ----------
        band_path : Path
            Path to the input full-scene band TIF.
        bbox : BoundingBox
            Area of interest in WGS84.
        output_path : Path
            Where to write the cropped GeoTIFF.

        Returns
        -------
        dict
            Metadata: ``{width, height, crs, transform, bounds, nodata}``.

        Raises
        ------
        FileNotFoundError
            If *band_path* does not exist.
        ValueError
            If the bbox does not overlap the raster.
        """
        band_path = Path(band_path)
        if not band_path.exists():
            raise FileNotFoundError(f"Band file not found: {band_path}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(band_path) as src:
            # Build a shapely bbox in WGS84, then reproject to raster CRS
            bbox_geom_wgs84 = box(bbox.west, bbox.south, bbox.east, bbox.north)

            if str(src.crs) != "EPSG:4326":
                from pyproj import Transformer

                transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                bbox_geom_native = _transform_geom(bbox_geom_wgs84, transformer)
            else:
                bbox_geom_native = bbox_geom_wgs84

            # Clip using rasterio.mask
            try:
                out_image, out_transform = rasterio_mask(
                    src,
                    [mapping(bbox_geom_native)],
                    crop=True,
                    all_touched=True,
                    nodata=src.nodata if src.nodata is not None else 0,
                )
            except ValueError as exc:
                raise ValueError(
                    f"Bounding box does not overlap raster {band_path.name}: {exc}"
                ) from exc

            out_meta = src.meta.copy()
            out_meta.update(
                {
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "compress": "deflate",
                }
            )

            with rasterio.open(output_path, "w", **out_meta) as dst:
                dst.write(out_image)

        logger.info(
            "Cropped %s → %s (%dx%d px)",
            band_path.name,
            output_path.name,
            out_meta["width"],
            out_meta["height"],
        )

        return {
            "width": out_meta["width"],
            "height": out_meta["height"],
            "crs": str(src.crs),
            "transform": list(out_transform)[:6],
            "bounds": list(rasterio.open(output_path).bounds),
            "nodata": out_meta.get("nodata"),
        }

    # ── 4. High-level: crop all bands needed for an index ──────────────

    def crop_bands_for_index(
        self,
        scene_id: str,
        scene_dir: Path,
        index_name: str,
        sensor: LandsatSensor,
        bbox: BoundingBox,
    ) -> dict[str, Path]:
        """
        Ensure all bands required for *index_name* are cropped and cached.

        Skips bands already in the :pyclass:`BandCache`.

        Parameters
        ----------
        scene_id : str
            Display id of the scene (used as cache key).
        scene_dir : Path
            Extracted scene directory.
        index_name : str
            Spectral index name (e.g. ``"NDVI"``).
        sensor : LandsatSensor
            Which Landsat sensor (determines band mapping table).
        bbox : BoundingBox
            AOI in WGS84.

        Returns
        -------
        dict[str, Path]
            ``{band_name: path_to_cropped_tif}`` for every band the index
            needs.

        Raises
        ------
        ValueError
            If the index is unsupported or a required band file is missing.
        """
        # Resolve band table
        band_table = _get_band_table(sensor)
        if index_name not in band_table:
            raise ValueError(
                f"Index '{index_name}' not in band table for {sensor.value}. "
                f"Available: {sorted(band_table)}"
            )

        role_to_band: dict[str, str] = band_table[index_name]
        required_bands: list[str] = list(role_to_band.values())

        bbox_hash = BandCache.make_bbox_hash(bbox.model_dump())

        # Check cache
        cached = self.cache.get_bands_for_index(scene_id, required_bands, bbox_hash)
        already = {bn: p for bn, p in cached.items() if p is not None}
        missing = [bn for bn in required_bands if bn not in already]

        if not missing:
            logger.info(
                "All bands for %s/%s already cached: %s",
                scene_id,
                index_name,
                list(already),
            )
            return already

        logger.info(
            "Index %s requires %s — cached: %s, to crop: %s",
            index_name,
            required_bands,
            list(already),
            missing,
        )

        # Crop missing bands
        scene_out_dir = self.processed_dir / scene_id
        scene_out_dir.mkdir(parents=True, exist_ok=True)

        for band_name in missing:
            src_path = self.find_band_file(scene_dir, band_name)
            if src_path is None:
                raise ValueError(
                    f"Band file for '{band_name}' not found in {scene_dir}. "
                    f"Files present: {[f.name for f in scene_dir.iterdir()]}"
                )

            out_name = f"{scene_id}_{band_name}_crop_{bbox_hash}.tif"
            out_path = scene_out_dir / out_name

            meta = self.crop_band(src_path, bbox, out_path)

            self.cache.register_band(
                scene_id=scene_id,
                band_name=band_name,
                bbox_hash=bbox_hash,
                path=out_path,
                width=meta["width"],
                height=meta["height"],
            )

            already[band_name] = out_path

        return already

    def crop_multiple_indices(
        self,
        scene_id: str,
        scene_dir: Path,
        index_names: list[str],
        sensor: LandsatSensor,
        bbox: BoundingBox,
    ) -> dict[str, dict[str, Path]]:
        """
        Crop bands for several indices at once (shared bands are only cropped once).

        Returns
        -------
        dict[str, dict[str, Path]]
            ``{index_name: {band_name: path}}``.
        """
        results: dict[str, dict[str, Path]] = {}
        for idx in index_names:
            results[idx] = self.crop_bands_for_index(scene_id, scene_dir, idx, sensor, bbox)
        return results

    # ── Convenience ────────────────────────────────────────────────────

    def list_scene_bands(self, scene_dir: Path) -> list[str]:
        """List all band TIF files present in an extracted scene directory."""
        bands = []
        for f in sorted(scene_dir.iterdir()):
            if f.suffix.upper() == ".TIF":
                # Extract band name from filename
                # e.g. "LC09_..._SR_B4.TIF" → "SR_B4"
                parts = f.stem.rsplit("_", 2)
                if len(parts) >= 2:
                    band = "_".join(parts[-2:])
                    bands.append(band)
        return bands

    def get_raster_info(self, raster_path: Path) -> dict[str, Any]:
        """Read quick metadata from a GeoTIFF."""
        with rasterio.open(raster_path) as src:
            return {
                "path": str(raster_path),
                "width": src.width,
                "height": src.height,
                "crs": str(src.crs),
                "bounds": list(src.bounds),
                "res": src.res,
                "dtype": str(src.dtypes[0]),
                "nodata": src.nodata,
                "count": src.count,
            }


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────


def _get_band_table(sensor: LandsatSensor) -> dict[str, dict[str, str]]:
    """Return the index → band mapping for the given sensor."""
    if sensor in (LandsatSensor.LANDSAT_8, LandsatSensor.LANDSAT_9):
        return LANDSAT_89_INDEX_BANDS
    return LANDSAT_457_INDEX_BANDS


def _transform_geom(geom, transformer):
    """Reproject a shapely geometry using a pyproj Transformer."""
    from shapely.ops import transform as shapely_transform

    return shapely_transform(transformer.transform, geom)

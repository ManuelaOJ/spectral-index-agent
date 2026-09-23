"""
Sentinel-2 data ingestion client.

Implements access to Copernicus Data Space Ecosystem for Sentinel-2
Level-2A (Surface Reflectance) data using Sentinel Hub APIs.

API Documentation: https://documentation.dataspace.copernicus.eu/
"""

import logging
import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from oauthlib.oauth2 import BackendApplicationClient
from pyproj import Transformer
from requests_oauthlib import OAuth2Session
import requests
import rasterio
from rasterio.crs import CRS as RasterioCRS
from rasterio.transform import from_bounds

from sentinelhub import (
    SHConfig,
    CRS,
    BBox as SHBBox,
    DataCollection,
    MimeType,
    MosaickingOrder,
    SentinelHubRequest,
    bbox_to_dimensions,
)

from spectral_agent.config import get_settings
from spectral_agent.core.base_tool import BaseIngestionTool
from spectral_agent.core.exceptions import (
    AuthenticationError,
    DownloadError,
    IngestionError,
    RateLimitError,
)
from spectral_agent.schemas.imagery import (
    BoundingBox,
    DownloadResult,
    SceneMetadata,
    SearchResult,
    SatelliteType,
)

logger = logging.getLogger(__name__)


def _sentinelhub_compliance_hook(response: requests.Response) -> requests.Response:
    """Compliance hook to handle server errors correctly."""
    response.raise_for_status()
    return response


# Regex to extract the acquisition timestamp from a Sentinel-2 scene ID.
# Example: S2B_MSIL2A_20200801T152639_N0500_R025_T18NXN
#                     ^^^^^^^^^^^^^^^^
_S2_DATE_RE = re.compile(r"_(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})_")


def _parse_date_from_scene_id(scene_id: str) -> datetime | None:
    """Extract acquisition datetime from a Sentinel-2 scene ID string."""
    m = _S2_DATE_RE.search(scene_id)
    if not m:
        return None
    return datetime(
        int(m.group(1)),
        int(m.group(2)),
        int(m.group(3)),
        int(m.group(4)),
        int(m.group(5)),
        int(m.group(6)),
    )


def _utm_crs_from_bbox(bbox: "BoundingBox") -> tuple[int, CRS]:
    """Derive the UTM EPSG code and sentinelhub CRS from a WGS84 bounding box.

    Uses the centre of the bbox to pick the UTM zone.

    Returns
    -------
    tuple[int, CRS]
        (epsg_code, sentinelhub CRS)
    """
    lon = (bbox.west + bbox.east) / 2
    lat = (bbox.south + bbox.north) / 2
    zone_number = int((lon + 180) / 6) + 1
    if lat >= 0:
        epsg = 32600 + zone_number  # UTM North
    else:
        epsg = 32700 + zone_number  # UTM South
    return epsg, CRS(f"EPSG:{epsg}")


def _reproject_bbox_to_utm(
    bbox: "BoundingBox", utm_epsg: int
) -> tuple[float, float, float, float]:
    """Reproject a WGS84 bounding box to a UTM CRS.

    Returns
    -------
    tuple
        (west, south, east, north) in UTM metres.
    """
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
    west, south = transformer.transform(bbox.west, bbox.south)
    east, north = transformer.transform(bbox.east, bbox.north)
    return (west, south, east, north)


def _snap_bbox_to_grid(
    bbox_utm: tuple[float, float, float, float], resolution: int = 10
) -> tuple[float, float, float, float]:
    """Snap a UTM bounding box to an exact pixel grid.

    Expands the bbox outward so that each edge is a multiple of *resolution*,
    guaranteeing pixel sizes of exactly ``resolution × resolution`` metres
    (equivalent to GDAL's ``-tap`` flag).

    Returns
    -------
    tuple
        (west, south, east, north) snapped to the grid.
    """
    import math

    west, south, east, north = bbox_utm
    west = math.floor(west / resolution) * resolution
    south = math.floor(south / resolution) * resolution
    east = math.ceil(east / resolution) * resolution
    north = math.ceil(north / resolution) * resolution
    return (west, south, east, north)


class SentinelClient(BaseIngestionTool):
    """
    Client for downloading Sentinel-2 imagery via Copernicus Data Space.

    Supports Sentinel-2 Level-2A (MSI) Surface Reflectance data.
    Uses Sentinel Hub Catalog API for searching.

    Example:
        >>> async with SentinelClient() as client:
        ...     results = await client.search(
        ...         bbox=BoundingBox(west=-122.5, south=37.5, east=-122.0, north=38.0),
        ...         start_date=date(2024, 1, 1),
        ...         end_date=date(2024, 6, 1),
        ...     )
    """

    # Sentinel-2 band information (wavelengths in nm, resolution in meters)
    # Wavelengths shown are for S2A; S2B has slightly different values
    BAND_INFO = {
        # Spectral bands
        "B01": {"name": "Coastal Aerosol", "wavelength": 442.7, "resolution": 60},
        "B02": {"name": "Blue", "wavelength": 492.4, "resolution": 10},
        "B03": {"name": "Green", "wavelength": 559.8, "resolution": 10},
        "B04": {"name": "Red", "wavelength": 664.6, "resolution": 10},
        "B05": {"name": "Vegetation Red Edge 1", "wavelength": 704.1, "resolution": 20},
        "B06": {"name": "Vegetation Red Edge 2", "wavelength": 740.5, "resolution": 20},
        "B07": {"name": "Vegetation Red Edge 3", "wavelength": 782.8, "resolution": 20},
        "B08": {"name": "NIR", "wavelength": 832.8, "resolution": 10},
        "B8A": {"name": "Narrow NIR", "wavelength": 864.7, "resolution": 20},
        "B09": {"name": "Water Vapour", "wavelength": 945.1, "resolution": 60},
        "B11": {"name": "SWIR 1", "wavelength": 1613.7, "resolution": 20},
        "B12": {"name": "SWIR 2", "wavelength": 2202.4, "resolution": 20},
        # Sen2Cor derived products
        "AOT": {"name": "Aerosol Optical Thickness", "resolution": 10},
        "SCL": {"name": "Scene Classification", "resolution": 20},
        "SNW": {"name": "Snow Probability", "resolution": 20},
        "CLD": {"name": "Cloud Probability", "resolution": 20},
        # Geometry bands
        "sunAzimuthAngles": {"name": "Sun Azimuth Angle", "resolution": 5000},
        "sunZenithAngles": {"name": "Sun Zenith Angle", "resolution": 5000},
        "viewAzimuthMean": {"name": "Viewing Azimuth Angle", "resolution": 5000},
        "viewZenithMean": {"name": "Viewing Zenith Angle", "resolution": 5000},
        # Mask
        "dataMask": {"name": "Data/No-Data Mask", "resolution": None},
    }

    def __init__(self):
        """Initialize the Sentinel client."""
        self.settings = get_settings()
        self._oauth_session: OAuth2Session | None = None
        self._token: dict | None = None
        self._token_expiry: datetime | None = None

    @property
    def satellite_name(self) -> str:
        return "Sentinel-2"

    @property
    def supported_collections(self) -> list[str]:
        return ["sentinel-2-l2a"]

    def _is_token_expired(self) -> bool:
        """Check if the current token is expired."""
        if self._token_expiry is None:
            return True
        # Add 60 second buffer before expiry
        return datetime.now() >= (self._token_expiry - timedelta(seconds=60))

    async def authenticate(self) -> bool:
        """
        Authenticate with Copernicus Data Space using OAuth2.

        Uses requests-oauthlib for proper OAuth2 handling.

        Returns:
            bool: True if authentication successful

        Raises:
            AuthenticationError: If credentials are missing or invalid
        """
        if not self.settings.has_copernicus_credentials():
            raise AuthenticationError(
                "Copernicus credentials not configured",
                details={
                    "required": [
                        "SPECTRAL_COPERNICUS_CLIENT_ID",
                        "SPECTRAL_COPERNICUS_CLIENT_SECRET",
                    ]
                },
            )

        try:
            # Create OAuth2 session using BackendApplicationClient (client credentials flow)
            client = BackendApplicationClient(
                client_id=self.settings.copernicus_client_id
            )
            self._oauth_session = OAuth2Session(client=client)

            # Register compliance hook to handle server errors correctly
            self._oauth_session.register_compliance_hook(
                "access_token_response", _sentinelhub_compliance_hook
            )

            # Fetch access token
            self._token = self._oauth_session.fetch_token(
                token_url=self.settings.copernicus_token_url,
                client_secret=self.settings.copernicus_client_secret,
                include_client_id=True,
            )

            # Calculate token expiry
            expires_in = self._token.get("expires_in", 3600)
            self._token_expiry = datetime.now() + timedelta(seconds=expires_in)

            logger.info("Successfully authenticated with Copernicus Data Space")
            return True

        except Exception as e:
            raise AuthenticationError(f"Copernicus authentication failed: {str(e)}")

    async def _ensure_authenticated(self) -> None:
        """Ensure we have a valid access token."""
        if self._oauth_session is None or self._is_token_expired():
            await self.authenticate()

    async def search(
        self,
        bbox: BoundingBox,
        start_date: date,
        end_date: date,
        max_cloud_cover: float = 100.0,
        collections: list[str] | None = None,
        max_results: int | None = None,
    ) -> SearchResult:
        """
        Search for Sentinel-2 L2A scenes using Sentinel Hub Catalog API.

        Args:
            bbox: Area of interest bounding box
            start_date: Start of date range
            end_date: End of date range
            max_cloud_cover: Maximum cloud cover (0-100)
            collections: Collection IDs (default: sentinel-2-l2a)
            max_results: Maximum results to return.
                Copernicus Catalog enforces ``limit <= 100`` per request.
                If ``None``, returns up to 100 scenes.

        Returns:
            SearchResult with matching scenes
        """
        await self._ensure_authenticated()

        collections = collections or self.settings.sentinel_collections

        # Build Catalog API search request (STAC format)
        # Use CQL2-JSON filter format for cloud cover
        # Copernicus Catalog requires limit <= 100.
        requested_limit = max_results if max_results is not None else 100
        request_limit = max(1, min(int(requested_limit), 100))

        search_request = {
            "bbox": [bbox.west, bbox.south, bbox.east, bbox.north],
            "datetime": f"{start_date.isoformat()}T00:00:00Z/{end_date.isoformat()}T23:59:59Z",
            "collections": collections,
            "limit": request_limit,
            "filter-lang": "cql2-json",
            "filter": {
                "op": "<",
                "args": [{"property": "eo:cloud_cover"}, max_cloud_cover],
            },
        }

        url = f"{self.settings.sentinel_hub_catalog_url}/search"

        try:
            response = self._oauth_session.post(
                url,
                json=search_request,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

            data = response.json()
            features = data.get("features", [])

            scenes = []
            for feature in features:
                try:
                    scene = self._parse_stac_feature(feature)
                    scenes.append(scene)
                except Exception as e:
                    logger.warning("Error parsing feature %s: %s", feature.get("id"), e)

            return SearchResult(
                total_count=data.get("numberMatched", len(scenes)),
                returned_count=len(scenes),
                scenes=scenes,
                query_bbox=bbox,
                query_start_date=start_date,
                query_end_date=end_date,
            )

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                raise RateLimitError("Copernicus API rate limit exceeded")
            raise IngestionError(
                f"Search failed: {e.response.status_code}",
                details={"error": e.response.text},
            )

    def _parse_stac_feature(self, feature: dict) -> SceneMetadata:
        """Parse STAC feature from Catalog API into SceneMetadata."""
        properties = feature.get("properties", {})

        # Extract cloud cover
        cloud_cover = properties.get("eo:cloud_cover", 0.0)

        # Parse acquisition date
        datetime_str = properties.get("datetime", "")
        if datetime_str:
            acq_date = datetime.fromisoformat(datetime_str.replace("Z", "+00:00"))
        else:
            acq_date = datetime.now()

        # Get feature ID and collection
        feature_id = feature.get("id", "")
        collection = feature.get("collection", "sentinel-2-l2a")

        # Extract geometry
        geometry = feature.get("geometry", {})

        # Build scene metadata
        return SceneMetadata(
            scene_id=feature_id,
            satellite=SatelliteType.SENTINEL,
            collection=collection,
            acquisition_date=acq_date,
            cloud_cover=cloud_cover,
            footprint=geometry if geometry else None,
            processing_level="L2A",
            sensor=properties.get("platform", "sentinel-2"),
            raw_metadata={
                "platform": properties.get("platform", "sentinel-2"),
                "instrument": properties.get("instruments", ["msi"]),
                "gsd": properties.get("gsd"),
                "processing_level": "L2A",
                "tile_id": properties.get("grid:code", ""),
                "stac_id": feature_id,
            },
        )

    # Evalscript for all Sentinel-2 L2A spectral bands (12 bands).
    # Requests REFLECTANCE units so Sentinel Hub returns surface-reflectance
    # values directly as FLOAT32 (typical range 0–0.4, max ~1.0).
    # No additional scaling (÷10000) is needed downstream.
    EVALSCRIPT_ALL_BANDS = """//VERSION=3
function setup() {
    return {
        input: [{
            bands: ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12"],
            units: "REFLECTANCE"
        }],
        output: {
            bands: 12,
            sampleType: "FLOAT32"
        }
    };
}

function evaluatePixel(sample) {
    return [sample.B01,
            sample.B02,
            sample.B03,
            sample.B04,
            sample.B05,
            sample.B06,
            sample.B07,
            sample.B08,
            sample.B8A,
            sample.B09,
            sample.B11,
            sample.B12];
}
"""

    # Band order in the output array (for reference)
    ALL_BANDS = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B11",
        "B12",
    ]

    def _get_sh_config(self) -> SHConfig:
        """Create SentinelHub configuration for Copernicus Data Space."""
        config = SHConfig()
        config.sh_client_id = self.settings.copernicus_client_id
        config.sh_client_secret = self.settings.copernicus_client_secret
        config.sh_token_url = self.settings.copernicus_token_url
        config.sh_base_url = "https://sh.dataspace.copernicus.eu"
        return config

    async def download(
        self,
        scene_id: str,
        output_dir: Path,
        bbox: BoundingBox | None = None,
        resolution: int = 10,
    ) -> Path:
        """
        Download all Sentinel-2 L2A spectral bands using Sentinel Hub Process API.

        Downloads all 12 spectral bands (B01-B12, B8A, excluding B10 which is L1C only)
        as raw digital numbers (DN) in INT16 format.

        Args:
            scene_id: Scene/feature ID (STAC item ID)
            output_dir: Directory to save the file
            bbox: Bounding box for the area to download (required, max ~25km x 25km at 10m)
            resolution: Output resolution in meters (default: 10)

        Returns:
            Path to downloaded GeoTIFF file with 12 bands

        Note:
            - Values are surface reflectance (FLOAT32, typical range 0–0.4)
            - No additional scaling needed — ready for index computation
            - Band order: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12
            - Sentinel Hub limits output to 2500x2500 pixels per request
        """
        await self._ensure_authenticated()

        if bbox is None:
            raise DownloadError(
                "bbox is required for download (Sentinel Hub limits to 2500x2500 pixels)",
                details={"scene_id": scene_id},
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Get acquisition date — try STAC catalog first, fall back to
            # parsing the date embedded in the scene_id string.
            acq_date: datetime | None = None
            try:
                scene_metadata = await self.get_scene_metadata(scene_id)
                acq_date = scene_metadata.acquisition_date
            except Exception as meta_exc:
                logger.warning(
                    "STAC metadata lookup failed for %s (%s), "
                    "falling back to scene_id date parsing.",
                    scene_id,
                    meta_exc,
                )
                acq_date = _parse_date_from_scene_id(scene_id)
                if acq_date is None:
                    raise DownloadError(
                        f"Cannot determine acquisition date for {scene_id}: "
                        f"STAC lookup failed ({meta_exc}) and scene_id "
                        "could not be parsed.",
                        details={"scene_id": scene_id},
                    )

            # Reproject bbox to the native UTM zone so Sentinel Hub returns
            # data in the raster's native CRS (avoids WGS84 resampling artefacts).
            utm_epsg, utm_crs = _utm_crs_from_bbox(bbox)
            bbox_coords = _reproject_bbox_to_utm(bbox, utm_epsg)
            bbox_coords = _snap_bbox_to_grid(bbox_coords, resolution)

            # Build time interval from acquisition date
            time_interval = (
                acq_date.strftime("%Y-%m-%d"),
                (acq_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            )

            # Create Sentinel Hub config and request
            sh_config = self._get_sh_config()
            sh_bbox = SHBBox(bbox=bbox_coords, crs=utm_crs)
            sh_size = bbox_to_dimensions(sh_bbox, resolution=resolution)

            # Check size limits (Sentinel Hub max is 2500x2500)
            if sh_size[0] > 2500 or sh_size[1] > 2500:
                raise DownloadError(
                    f"Requested image size {sh_size[0]}x{sh_size[1]} exceeds Sentinel Hub limit of 2500x2500. "
                    f"Reduce bbox size or increase resolution.",
                    details={
                        "scene_id": scene_id,
                        "size": sh_size,
                        "resolution": resolution,
                    },
                )

            logger.info(
                "Downloading %s - all 12 bands at %sm resolution", scene_id, resolution
            )
            logger.info("Bands: %s", self.ALL_BANDS)
            logger.info("Image size: %s x %s pixels", sh_size[0], sh_size[1])

            # Create output folder for this scene
            scene_folder = output_dir / scene_id.replace(".SAFE", "")
            scene_folder.mkdir(parents=True, exist_ok=True)

            # Use SentinelHubRequest with data_folder for native caching and saving
            request = SentinelHubRequest(
                data_folder=str(scene_folder),
                evalscript=self.EVALSCRIPT_ALL_BANDS,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A.define_from(
                            "s2l2a", service_url=sh_config.sh_base_url
                        ),
                        time_interval=time_interval,
                        mosaicking_order=MosaickingOrder.LEAST_CC,
                    )
                ],
                responses=[
                    SentinelHubRequest.output_response("default", MimeType.TIFF)
                ],
                bbox=sh_bbox,
                size=sh_size,
                config=sh_config,
            )

            # Download and save directly to disk
            # This uses sentinelhub's native caching - if data exists, it won't redownload
            logger.info("Requesting data from Sentinel Hub Process API...")
            data = request.get_data(save_data=True)

            if not data or len(data) == 0:
                raise DownloadError(
                    f"No data returned for scene {scene_id}",
                    details={"scene_id": scene_id},
                )

            # Find the downloaded TIFF file
            output_path = None
            for folder, _, filenames in os.walk(scene_folder):
                for filename in filenames:
                    if filename.endswith(".tif") or filename.endswith(".tiff"):
                        output_path = Path(folder) / filename
                        break
                if output_path:
                    break

            if not output_path or not output_path.exists():
                # Fallback: save manually if sentinelhub didn't save
                output_filename = f"{scene_id.replace('.SAFE', '')}_{resolution}m.tif"
                output_path = scene_folder / output_filename

                image_data = data[0]
                transform = from_bounds(
                    bbox_coords[0],
                    bbox_coords[1],
                    bbox_coords[2],
                    bbox_coords[3],
                    sh_size[0],
                    sh_size[1],
                )

                with rasterio.open(
                    output_path,
                    "w",
                    driver="GTiff",
                    height=image_data.shape[0],
                    width=image_data.shape[1],
                    count=len(self.ALL_BANDS),
                    dtype=image_data.dtype,
                    crs=RasterioCRS.from_epsg(utm_epsg),
                    transform=transform,
                ) as dst:
                    for i, band_name in enumerate(self.ALL_BANDS):
                        dst.write(image_data[:, :, i], i + 1)
                        dst.set_band_description(i + 1, band_name)

            file_size_mb = output_path.stat().st_size / 1024 / 1024
            logger.info("Successfully saved %s (%.2f MB)", output_path, file_size_mb)
            logger.info(
                "Values are surface reflectance (FLOAT32). No scaling needed."
            )
            return output_path

        except DownloadError:
            raise
        except Exception as e:
            logger.error("Download failed for %s: %s", scene_id, e)
            raise DownloadError(
                f"Failed to download scene {scene_id}: {str(e)}",
                details={"scene_id": scene_id},
            )

    async def download_indices(
        self,
        scene_id: str,
        output_dir: Path,
        bbox: BoundingBox,
        indices: list[str] | str,
        resolution: int = 10,
        include_rgb: bool = False,
    ) -> Path:
        """
        Download computed spectral indices using Sentinel Hub Process API.

        Calculates indices directly on Sentinel Hub servers and returns
        the computed values, which is more efficient than downloading raw bands.

        Args:
            scene_id: Scene/feature ID (STAC item ID)
            output_dir: Directory to save the file
            bbox: Bounding box for the area to download (max ~25km x 25km at 10m)
            indices: Index name(s) to calculate (e.g., "NDVI" or ["NDVI", "NDWI"])
            resolution: Output resolution in meters (default: 10)
            include_rgb: If True, include RGB bands (B04, B03, B02) as first 3 bands

        Returns:
            Path to downloaded GeoTIFF file with computed indices

        Available indices:
            Vegetation: NDVI, EVI, SAVI, MSAVI2, GNDVI, NDRE, LAI, ARVI, SIPI
            Water: NDWI, MNDWI, NDMI
            Soil: BSI, NDTI
            Burn: NBR, BAI
            Snow: NDSI
            Built-up: NDBI, UI
            Other: ARI, CRI1

        Example:
            >>> path = await client.download_indices(
            ...     scene_id="S2A_...",
            ...     output_dir=Path("data"),
            ...     bbox=bbox,
            ...     indices=["NDVI", "NDWI", "NBR"],
            ... )
        """
        from spectral_agent.tools.indices import generate_evalscript, get_index
        from spectral_agent.tools.indices.spectral_indices import (
            generate_evalscript_with_rgb,
        )

        await self._ensure_authenticated()

        if isinstance(indices, str):
            indices = [indices]

        # Validate indices exist
        index_defs = [get_index(name) for name in indices]

        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Get acquisition date — try STAC catalog first, fall back to
            # parsing the date embedded in the scene_id string.
            acq_date: datetime | None = None
            try:
                scene_metadata = await self.get_scene_metadata(scene_id)
                acq_date = scene_metadata.acquisition_date
            except Exception as meta_exc:
                logger.warning(
                    "STAC metadata lookup failed for %s (%s), "
                    "falling back to scene_id date parsing.",
                    scene_id,
                    meta_exc,
                )
                acq_date = _parse_date_from_scene_id(scene_id)
                if acq_date is None:
                    raise DownloadError(
                        f"Cannot determine acquisition date for {scene_id}: "
                        f"STAC lookup failed ({meta_exc}) and scene_id "
                        "could not be parsed.",
                        details={"scene_id": scene_id},
                    )

            # Reproject bbox to native UTM zone (same as download())
            utm_epsg, utm_crs = _utm_crs_from_bbox(bbox)
            bbox_coords = _reproject_bbox_to_utm(bbox, utm_epsg)
            bbox_coords = _snap_bbox_to_grid(bbox_coords, resolution)

            # Build time interval from acquisition date
            time_interval = (
                acq_date.strftime("%Y-%m-%d"),
                (acq_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            )

            # Create Sentinel Hub config and request
            sh_config = self._get_sh_config()
            sh_bbox = SHBBox(bbox=bbox_coords, crs=utm_crs)
            sh_size = bbox_to_dimensions(sh_bbox, resolution=resolution)

            # Check size limits
            if sh_size[0] > 2500 or sh_size[1] > 2500:
                raise DownloadError(
                    f"Requested image size {sh_size[0]}x{sh_size[1]} exceeds limit of 2500x2500. "
                    f"Reduce bbox size or increase resolution.",
                    details={"scene_id": scene_id, "size": sh_size},
                )

            # Generate evalscript for indices
            if include_rgb:
                evalscript = generate_evalscript_with_rgb(indices)
                output_bands = 3 + len(indices)
                band_names = ["Red", "Green", "Blue"] + indices
            else:
                evalscript = generate_evalscript(indices)
                output_bands = len(indices)
                band_names = indices

            logger.info("Downloading indices for %s: %s", scene_id, indices)
            logger.info("Image size: %s x %s pixels", sh_size[0], sh_size[1])

            # Create output folder
            scene_folder = output_dir / scene_id.replace(".SAFE", "")
            scene_folder.mkdir(parents=True, exist_ok=True)

            request = SentinelHubRequest(
                data_folder=str(scene_folder),
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A.define_from(
                            "s2l2a", service_url=sh_config.sh_base_url
                        ),
                        time_interval=time_interval,
                        mosaicking_order=MosaickingOrder.LEAST_CC,
                    )
                ],
                responses=[
                    SentinelHubRequest.output_response("default", MimeType.TIFF)
                ],
                bbox=sh_bbox,
                size=sh_size,
                config=sh_config,
            )

            logger.info("Requesting indices from Sentinel Hub Process API...")
            data = request.get_data(save_data=True)

            if not data or len(data) == 0:
                raise DownloadError(
                    f"No data returned for scene {scene_id}",
                    details={"scene_id": scene_id, "indices": indices},
                )

            # Find the downloaded TIFF file
            output_path = None
            for folder, _, filenames in os.walk(scene_folder):
                for filename in filenames:
                    if filename.endswith(".tif") or filename.endswith(".tiff"):
                        output_path = Path(folder) / filename
                        break
                if output_path:
                    break

            if not output_path or not output_path.exists():
                # Fallback: save manually
                indices_str = "_".join(indices)
                output_filename = (
                    f"{scene_id.replace('.SAFE', '')}_{indices_str}_{resolution}m.tif"
                )
                output_path = scene_folder / output_filename

                image_data = data[0]
                transform = from_bounds(
                    bbox_coords[0],
                    bbox_coords[1],
                    bbox_coords[2],
                    bbox_coords[3],
                    sh_size[0],
                    sh_size[1],
                )

                with rasterio.open(
                    output_path,
                    "w",
                    driver="GTiff",
                    height=image_data.shape[0],
                    width=image_data.shape[1],
                    count=output_bands,
                    dtype=image_data.dtype,
                    crs=RasterioCRS.from_epsg(utm_epsg),
                    transform=transform,
                ) as dst:
                    for i, band_name in enumerate(band_names):
                        if len(image_data.shape) == 3:
                            dst.write(image_data[:, :, i], i + 1)
                        else:
                            dst.write(image_data, 1)
                        dst.set_band_description(i + 1, band_name)

            file_size_mb = output_path.stat().st_size / 1024 / 1024
            logger.info("Successfully saved %s (%.2f MB)", output_path, file_size_mb)
            logger.info("Indices: %s", band_names)
            return output_path

        except DownloadError:
            raise
        except Exception as e:
            logger.error("Index download failed for %s: %s", scene_id, e)
            raise DownloadError(
                f"Failed to download indices for {scene_id}: {str(e)}",
                details={"scene_id": scene_id, "indices": indices},
            )

    async def get_scene_metadata(self, scene_id: str) -> SceneMetadata:
        """
        Get detailed metadata for a specific Sentinel-2 scene.

        Args:
            scene_id: The scene/feature ID (STAC item ID)

        Returns:
            SceneMetadata for the scene
        """
        await self._ensure_authenticated()

        # Query Catalog API for specific feature
        url = f"{self.settings.sentinel_hub_catalog_url}/collections/sentinel-2-l2a/items/{scene_id}"
        response = self._oauth_session.get(url)
        response.raise_for_status()

        feature = response.json()
        return self._parse_stac_feature(feature)

    async def close(self) -> None:
        """Close the OAuth session."""
        if self._oauth_session:
            self._oauth_session.close()
            self._oauth_session = None
        self._token = None
        self._token_expiry = None

    async def __aenter__(self) -> "SentinelClient":
        await self.authenticate()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

"""
Landsat data ingestion client.

Implements the USGS M2M (Machine-to-Machine) API for accessing
Landsat Collection 2 Level-2 data.

API Documentation: https://m2m.cr.usgs.gov/
"""

import asyncio
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, cast

import httpx

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
    SatelliteType,
    SceneMetadata,
    SearchResult,
)

logger = logging.getLogger(__name__)


class LandsatClient(BaseIngestionTool):
    """
    Client for downloading Landsat imagery via USGS M2M API.

    Supports Landsat 8/9 Collection 2 Level-2 Surface Reflectance data.

    Example:
        >>> client = LandsatClient()
        >>> await client.authenticate()
        >>> results = await client.search(
        ...     bbox=BoundingBox(west=-122.5, south=37.5, east=-122.0, north=38.0),
        ...     start_date=date(2024, 1, 1),
        ...     end_date=date(2024, 6, 1),
        ... )
    """

    # Collection mappings
    COLLECTION_MAPPING = {
        "landsat_ot_c2_l2": "Landsat 8-9 OLI/TIRS C2 L2",
        "landsat_etm_c2_l2": "Landsat 7 ETM+ C2 L2",
        "landsat_tm_c2_l2": "Landsat 4-5 TM C2 L2",
    }

    # Landsat 8-9 OLI/TIRS Collection 2 Level 2 bands
    LANDSAT_89_BANDS = {
        "SR_B1": {
            "name": "Coastal Aerosol",
            "wavelength_range": (0.43, 0.45),
            "resolution": 30,
        },
        "SR_B2": {"name": "Blue", "wavelength_range": (0.45, 0.51), "resolution": 30},
        "SR_B3": {"name": "Green", "wavelength_range": (0.53, 0.59), "resolution": 30},
        "SR_B4": {"name": "Red", "wavelength_range": (0.64, 0.67), "resolution": 30},
        "SR_B5": {"name": "NIR", "wavelength_range": (0.85, 0.88), "resolution": 30},
        "SR_B6": {"name": "SWIR 1", "wavelength_range": (1.57, 1.65), "resolution": 30},
        "SR_B7": {"name": "SWIR 2", "wavelength_range": (2.11, 2.29), "resolution": 30},
        "SR_B8": {
            "name": "Panchromatic",
            "wavelength_range": (0.50, 0.68),
            "resolution": 15,
        },
        "SR_B9": {"name": "Cirrus", "wavelength_range": (1.36, 1.38), "resolution": 30},
        "ST_B10": {
            "name": "Thermal Infrared 1",
            "wavelength_range": (10.60, 11.19),
            "resolution": 100,
        },
        "QA_PIXEL": {"name": "Quality Assessment", "resolution": 30},
        "QA_RADSAT": {"name": "Radiometric Saturation QA", "resolution": 30},
        "SR_QA_AEROSOL": {"name": "Aerosol QA", "resolution": 30},
    }

    # Landsat 7 ETM+ Collection 2 Level 2 bands
    LANDSAT_7_BANDS = {
        "SR_B1": {"name": "Blue", "wavelength_range": (0.45, 0.52), "resolution": 30},
        "SR_B2": {"name": "Green", "wavelength_range": (0.52, 0.60), "resolution": 30},
        "SR_B3": {"name": "Red", "wavelength_range": (0.63, 0.69), "resolution": 30},
        "SR_B4": {"name": "NIR", "wavelength_range": (0.77, 0.90), "resolution": 30},
        "SR_B5": {"name": "SWIR 1", "wavelength_range": (1.55, 1.75), "resolution": 30},
        "ST_B6": {
            "name": "Thermal",
            "wavelength_range": (10.40, 12.50),
            "resolution": 60,
        },
        "SR_B7": {"name": "SWIR 2", "wavelength_range": (2.08, 2.35), "resolution": 30},
        "SR_B8": {
            "name": "Panchromatic",
            "wavelength_range": (0.52, 0.90),
            "resolution": 15,
        },
        "QA_PIXEL": {"name": "Quality Assessment", "resolution": 30},
        "QA_RADSAT": {"name": "Radiometric Saturation QA", "resolution": 30},
        "SR_ATMOS_OPACITY": {"name": "Atmospheric Opacity", "resolution": 30},
        "SR_CLOUD_QA": {"name": "Cloud QA", "resolution": 30},
    }

    # Landsat 4-5 TM Collection 2 Level 2 bands
    LANDSAT_45_BANDS = {
        "SR_B1": {"name": "Blue", "wavelength_range": (0.45, 0.52), "resolution": 30},
        "SR_B2": {"name": "Green", "wavelength_range": (0.52, 0.60), "resolution": 30},
        "SR_B3": {"name": "Red", "wavelength_range": (0.63, 0.69), "resolution": 30},
        "SR_B4": {"name": "NIR", "wavelength_range": (0.76, 0.90), "resolution": 30},
        "SR_B5": {"name": "SWIR 1", "wavelength_range": (1.55, 1.75), "resolution": 30},
        "ST_B6": {
            "name": "Thermal",
            "wavelength_range": (10.40, 12.50),
            "resolution": 120,
        },
        "SR_B7": {"name": "SWIR 2", "wavelength_range": (2.08, 2.35), "resolution": 30},
        "QA_PIXEL": {"name": "Quality Assessment", "resolution": 30},
        "QA_RADSAT": {"name": "Radiometric Saturation QA", "resolution": 30},
        "SR_ATMOS_OPACITY": {"name": "Atmospheric Opacity", "resolution": 30},
        "SR_CLOUD_QA": {"name": "Cloud QA", "resolution": 30},
    }

    # Band mapping by collection
    BAND_INFO_BY_COLLECTION = {
        "landsat_ot_c2_l2": LANDSAT_89_BANDS,
        "landsat_etm_c2_l2": LANDSAT_7_BANDS,
        "landsat_tm_c2_l2": LANDSAT_45_BANDS,
    }

    def __init__(self):
        """Initialize the Landsat client."""
        self.settings = get_settings()
        self._api_key: str | None = None
        self._client: httpx.AsyncClient | None = None

    @property
    def satellite_name(self) -> str:
        return "Landsat"

    @property
    def supported_collections(self) -> list[str]:
        return list(self.COLLECTION_MAPPING.keys())

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.settings.request_timeout),
                follow_redirects=True,
            )
        return self._client

    async def _make_request(
        self,
        endpoint: str,
        data: dict | None = None,
        require_auth: bool = True,
    ) -> Any:
        """
        Make a request to the M2M API.

        Args:
            endpoint: API endpoint name
            data: Request payload
            require_auth: Whether to include API key

        Returns:
            Response data

        Raises:
            IngestionError: If request fails
        """
        client = await self._get_client()
        url = f"{self.settings.usgs_m2m_endpoint}{endpoint}"

        payload = data or {}
        if require_auth and self._api_key:
            headers = {"X-Auth-Token": self._api_key}
        else:
            headers = {}

        max_attempts = max(1, int(self.settings.max_retries))
        last_error: str | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()

                if result.get("errorCode"):
                    error_msg = result.get("errorMessage", "Unknown error")
                    raise IngestionError(
                        f"M2M API error: {error_msg}",
                        details={
                            "endpoint": endpoint,
                            "error_code": result.get("errorCode"),
                        },
                    )

                return result.get("data", {})

            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                last_error = f"HTTP {status}"

                # Retry transient server/rate-limit failures.
                should_retry = status == 429 or status >= 500
                if should_retry and attempt < max_attempts:
                    delay_s = float(self.settings.retry_delay) * attempt
                    logger.warning(
                        "USGS %s failed with HTTP %s (attempt %s/%s). Retrying in %.1fs...",
                        endpoint,
                        status,
                        attempt,
                        max_attempts,
                        delay_s,
                    )
                    await asyncio.sleep(delay_s)
                    continue

                if status == 429:
                    raise RateLimitError(
                        "USGS API rate limit exceeded",
                        details={"endpoint": endpoint, "attempts": attempt},
                    )

                raise IngestionError(
                    f"HTTP error: {status}",
                    details={
                        "endpoint": endpoint,
                        "status": status,
                        "attempts": attempt,
                    },
                )

            except httpx.RequestError as e:
                last_error = str(e)
                if attempt < max_attempts:
                    delay_s = float(self.settings.retry_delay) * attempt
                    logger.warning(
                        "USGS %s request failed (attempt %s/%s): %s. Retrying in %.1fs...",
                        endpoint,
                        attempt,
                        max_attempts,
                        e,
                        delay_s,
                    )
                    await asyncio.sleep(delay_s)
                    continue

                raise IngestionError(
                    f"Request failed: {str(e)}",
                    details={"endpoint": endpoint, "attempts": attempt},
                )

        raise IngestionError(
            "Request failed after retries",
            details={
                "endpoint": endpoint,
                "attempts": max_attempts,
                "error": last_error,
            },
        )

    async def authenticate(self) -> bool:
        """
        Authenticate with the USGS M2M API.

        Returns:
            bool: True if authentication successful

        Raises:
            AuthenticationError: If credentials are missing or invalid
        """
        if not self.settings.has_usgs_credentials():
            raise AuthenticationError(
                "USGS credentials not configured",
                details={"required": ["SPECTRAL_USGS_USERNAME", "SPECTRAL_USGS_TOKEN"]},
            )

        try:
            result = await self._make_request(
                "login-token",
                data={
                    "username": self.settings.usgs_username,
                    "token": self.settings.usgs_token,
                },
                require_auth=False,
            )

            self._api_key = result
            logger.info("Successfully authenticated with USGS M2M API")
            return True

        except IngestionError as e:
            raise AuthenticationError("USGS authentication failed", details={"error": str(e)})

    async def logout(self) -> None:
        """Logout and invalidate the API key."""
        if self._api_key:
            try:
                await self._make_request("logout")
            except Exception:
                pass  # Best effort logout
            finally:
                self._api_key = None

    def get_band_info(self, collection: str = "landsat_ot_c2_l2") -> dict:
        """
        Get band information for a specific Landsat collection.

        Args:
            collection: Collection ID (landsat_ot_c2_l2, landsat_etm_c2_l2, landsat_tm_c2_l2)

        Returns:
            Dictionary of band information
        """
        if collection not in self.BAND_INFO_BY_COLLECTION:
            raise ValueError(
                f"Unknown collection '{collection}'. "
                f"Available: {list(self.BAND_INFO_BY_COLLECTION.keys())}"
            )
        return self.BAND_INFO_BY_COLLECTION[collection]

    def _validate_bands(self, bands: list[str], collection: str = "landsat_ot_c2_l2") -> list[str]:
        """
        Validate band names for a specific collection.

        Args:
            bands: List of band names
            collection: Collection ID

        Returns:
            Validated list of band names

        Raises:
            ValueError: If invalid band names are provided
        """
        collection_bands = self.get_band_info(collection)
        invalid_bands = [b for b in bands if b not in collection_bands]

        if invalid_bands:
            raise ValueError(
                f"Invalid bands for {collection}: {invalid_bands}. "
                f"Valid bands: {list(collection_bands.keys())}"
            )
        return bands

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
        Search for Landsat scenes matching the criteria.

        Args:
            bbox: Area of interest bounding box
            start_date: Start of date range
            end_date: End of date range
            max_cloud_cover: Maximum cloud cover (0-100)
            collections: Collections to search (default: all)
            max_results: Maximum results to return.
                ``None`` returns all matching scenes (up to USGS API max).

        Returns:
            SearchResult with matching scenes
        """
        if not self._api_key:
            await self.authenticate()

        collections = collections or self.settings.landsat_collections
        all_scenes = []

        for collection in collections:
            try:
                scenes = await self._search_collection(
                    collection=collection,
                    bbox=bbox,
                    start_date=start_date,
                    end_date=end_date,
                    max_cloud_cover=max_cloud_cover,
                )
                all_scenes.extend(scenes)
            except Exception as e:
                logger.warning("Error searching collection %s: %s", collection, e)

        # Sort by acquisition date (most recent first), then cloud cover.
        # This ensures the scene closest to the user's requested date wins
        # when tied on cloud cover, and that newer imagery is preferred.
        target_mid = start_date + (end_date - start_date) / 2
        all_scenes.sort(
            key=lambda s: (
                abs((s.acquisition_date.date() - target_mid).days),
                s.cloud_cover,
            )
        )

        # Limit results only when explicitly requested
        if max_results is not None:
            all_scenes = all_scenes[:max_results]

        return SearchResult(
            total_count=len(all_scenes),
            returned_count=len(all_scenes),
            scenes=all_scenes,
            query_bbox=bbox,
            query_start_date=start_date,
            query_end_date=end_date,
        )

    async def _search_collection(
        self,
        collection: str,
        bbox: BoundingBox,
        start_date: date,
        end_date: date,
        max_cloud_cover: float,
        max_results: int | None = None,
    ) -> list[SceneMetadata]:
        """Search a specific Landsat collection."""

        # Build spatial filter
        spatial_filter = {
            "filterType": "mbr",
            "lowerLeft": {"longitude": bbox.west, "latitude": bbox.south},
            "upperRight": {"longitude": bbox.east, "latitude": bbox.north},
        }

        # Build scene filter following USGS M2M API spec:
        # - acquisitionFilter (not temporalFilter) with start/end keys
        # - cloudCoverFilter only when user requests a limit
        scene_filter: dict[str, Any] = {
            "spatialFilter": spatial_filter,
            "acquisitionFilter": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
        }

        # Add cloud cover filter only if user explicitly requested a limit
        if max_cloud_cover < 100:
            scene_filter["cloudCoverFilter"] = {
                "min": 0,
                "max": int(max_cloud_cover),
            }

        search_payload: dict[str, Any] = {
            "datasetName": collection,
            "sceneFilter": scene_filter,
        }

        logger.info(
            "Searching %s: dates=%s to %s, bbox=%s",
            collection,
            start_date,
            end_date,
            bbox,
        )
        result = await self._make_request("scene-search", data=search_payload)

        logger.info(
            "Collection %s: API returned %s records, total=%s",
            collection,
            result.get("recordsReturned", 0),
            result.get("totalHits", 0),
        )

        scenes = []
        for scene_data in result.get("results", []):
            try:
                scene = self._parse_scene_metadata(scene_data, collection)
                scenes.append(scene)
            except Exception as e:
                logger.warning("Error parsing scene %s: %s", scene_data.get("entityId"), e)

        return scenes

    def _parse_scene_metadata(self, data: dict, collection: str) -> SceneMetadata:
        """Parse raw scene data into SceneMetadata."""

        # Extract cloud cover from metadata
        cloud_cover = 0.0
        for meta in data.get("metadata", []):
            if "cloud" in meta.get("fieldName", "").lower():
                try:
                    cloud_cover = float(meta.get("value", 0))
                    break
                except ValueError:
                    pass

        # Parse acquisition date
        acq_date_str = data.get("temporalCoverage", {}).get("startDate")
        if acq_date_str:
            acq_date = datetime.fromisoformat(acq_date_str.replace("Z", "+00:00"))
        else:
            acq_date = datetime.now()

        # Extract browse URL
        browse_url = None
        for browse in data.get("browse", []):
            browse_url = browse.get("browsePath")
            break

        # Use displayId (e.g. LC09_L2SP_008055_20260211_...) as the user-facing ID
        # Store entityId (e.g. LC90080552026042LGN00) in raw_metadata for API calls
        display_id = data.get("displayId", data.get("entityId", ""))
        entity_id = data.get("entityId", "")

        return SceneMetadata(
            scene_id=display_id,
            satellite=SatelliteType.LANDSAT,
            collection=collection,
            acquisition_date=acq_date,
            cloud_cover=cloud_cover,
            footprint=data.get("spatialCoverage"),
            sensor=self.COLLECTION_MAPPING.get(collection, collection),
            processing_level="L2",
            browse_url=browse_url,
            raw_metadata={**data, "entityId": entity_id},
        )

    async def get_download_options(self, scene_id: str, collection: str) -> list[dict]:
        """Get available download options for a scene."""

        result = await self._make_request(
            "download-options",
            data={
                "datasetName": collection,
                "entityIds": [scene_id],
            },
        )

        return result

    async def request_download(
        self,
        scene_id: str,
        collection: str,
        max_retries: int = 30,
        retry_delay: float = 30.0,
    ) -> str | None:
        """
        Request download URL for a scene following the M2M workflow:

        1. download-options  → filter available products (available=True, not folder)
        2. download-request  → submit {entityId, productId} list
        3. download-retrieve → poll for preparingDownloads until ready

        Args:
            scene_id: Scene entity ID
            collection: Collection name
            max_retries: Maximum poll attempts for preparing downloads
            retry_delay: Seconds between poll attempts

        Returns:
            Download URL or None if unavailable
        """
        # ── Step 1: download-options → build available products list ──
        options = await self.get_download_options(scene_id, collection)

        available_products: list[dict] = []
        for product in options:
            if product.get("available") is True and product.get("downloadSystem") != "folder":
                available_products.append(
                    {
                        "entityId": product["entityId"],
                        "productId": product["id"],
                    }
                )

        if not available_products:
            logger.warning(
                "No available (non-folder) products for %s. download-options returned %d items.",
                scene_id,
                len(options),
            )
            logger.debug("download-options response: %s", options)
            return None

        logger.info("Found %d downloadable product(s) for %s", len(available_products), scene_id)

        # ── Step 2: download-request ──
        label = f"spectral-agent-{scene_id[:20]}"
        result = await self._make_request(
            "download-request",
            data={
                "downloads": available_products,
                "label": label,
            },
        )

        logger.info(
            "download-request response keys: %s",
            list(result.keys()) if isinstance(result, dict) else type(result),
        )

        # Helper to extract first URL from a download list
        def _extract_url(download_list: list) -> str | None:
            for item in download_list:
                url = item.get("url")
                if url:
                    return url
            return None

        # Check availableDownloads (ready immediately)
        url = _extract_url(result.get("availableDownloads", []))
        if url:
            logger.info("Download for %s is immediately available", scene_id)
            return url

        # Check duplicateProducts (already requested earlier, URL still valid)
        url = _extract_url(result.get("duplicateProducts", []))
        if url:
            logger.info("Download for %s found in duplicateProducts", scene_id)
            return url

        # ── Step 3: download-retrieve for preparingDownloads ──
        preparing = result.get("preparingDownloads", [])
        if not preparing:
            logger.warning(
                "Download for %s not in any response category. Full response: %s",
                scene_id,
                result,
            )
            return None

        # Collect downloadIds that are still being prepared
        preparing_ids = {item["downloadId"] for item in preparing}
        logger.info(
            "%d download(s) being prepared for %s, "
            "polling download-retrieve every %ss "
            "(max %d attempts \u2248 %.0f min)...",
            len(preparing_ids),
            scene_id,
            retry_delay,
            max_retries,
            max_retries * retry_delay / 60,
        )

        retrieve_payload = {"label": label}

        for attempt in range(1, max_retries + 1):
            await asyncio.sleep(retry_delay)
            logger.info("download-retrieve attempt %d/%d...", attempt, max_retries)

            retrieve_result = await self._make_request("download-retrieve", data=retrieve_payload)

            # Check 'available' list
            for item in retrieve_result.get("available", []):
                if item.get("downloadId") in preparing_ids and item.get("url"):
                    logger.info(
                        "Download for %s ready via download-retrieve (attempt %d)",
                        scene_id,
                        attempt,
                    )
                    return item["url"]

            # Check 'requested' list (may also contain ready URLs)
            for item in retrieve_result.get("requested", []):
                if item.get("downloadId") in preparing_ids and item.get("url"):
                    logger.info(
                        "Download for %s ready via download-retrieve/requested (attempt %d)",
                        scene_id,
                        attempt,
                    )
                    return item["url"]

        logger.warning(
            "Download for %s still preparing after %d attempts (%.0f min)",
            scene_id,
            max_retries,
            max_retries * retry_delay / 60,
        )
        return None

    async def download(
        self,
        scene_id: str,
        output_dir: Path,
        bands: list[str] | None = None,
        collection: str | None = None,
    ) -> Path:
        """
        Download a Landsat scene.

        Args:
            scene_id: Scene entity ID
            output_dir: Directory to save the file
            bands: Not used for Landsat (full product download)
            collection: Collection name (if known, avoids probing all collections)

        Returns:
            Path to downloaded file
        """
        if not self._api_key:
            await self.authenticate()

        output_dir.mkdir(parents=True, exist_ok=True)

        # Use provided collection or probe all collections to find the scene
        if not collection:
            for coll in self.settings.landsat_collections:
                try:
                    options = await self.get_download_options(scene_id, coll)
                    if options:
                        collection = coll
                        break
                except Exception:
                    continue

        if not collection:
            raise DownloadError("Could not find scene in any collection", scene_id=scene_id)

        # Get download URL
        download_url = await self.request_download(scene_id, collection)

        if not download_url:
            raise DownloadError(
                "Download URL not available (may still be preparing)", scene_id=scene_id
            )

        # Download the file
        output_path = output_dir / f"{scene_id}.tar"

        client = await self._get_client()

        try:
            async with client.stream(
                "GET",
                download_url,
                headers={"X-Auth-Token": cast(str, self._api_key)},
            ) as response:
                response.raise_for_status()

                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0

                with open(output_path, "wb") as f:
                    async for chunk in response.aiter_bytes(self.settings.download_chunk_size):
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            if downloaded % (10 * 1024 * 1024) == 0:  # Log every 10MB
                                logger.info("Download progress: %.1f%%", progress)

            logger.info("Downloaded %s to %s", scene_id, output_path)
            return output_path

        except Exception as e:
            if output_path.exists():
                output_path.unlink()
            raise DownloadError(f"Download failed: {str(e)}", scene_id=scene_id)

    async def get_scene_metadata(self, scene_id: str) -> SceneMetadata:
        """Get detailed metadata for a scene."""

        if not self._api_key:
            await self.authenticate()

        for collection in self.settings.landsat_collections:
            try:
                result = await self._make_request(
                    "scene-metadata",
                    data={
                        "datasetName": collection,
                        "entityId": scene_id,
                    },
                )
                return self._parse_scene_metadata(result, collection)
            except Exception:
                continue

        raise IngestionError(f"Scene {scene_id} not found in any collection")

    async def close(self) -> None:
        """Close the client and logout."""
        await self.logout()
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "LandsatClient":
        await self.authenticate()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

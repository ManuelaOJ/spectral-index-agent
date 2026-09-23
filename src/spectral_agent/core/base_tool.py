"""
Base classes for ingestion tools.

Provides abstract base classes that define the interface for satellite
data ingestion tools.
"""

from abc import ABC, abstractmethod
from datetime import date
import logging
from pathlib import Path
from typing import Any

from spectral_agent.schemas.imagery import BoundingBox, SceneMetadata, SearchResult

logger = logging.getLogger(__name__)


class BaseIngestionTool(ABC):
    """
    Abstract base class for satellite data ingestion tools.

    All satellite-specific ingestion tools should inherit from this class
    and implement the required methods.
    """

    @property
    @abstractmethod
    def satellite_name(self) -> str:
        """Name of the satellite system (e.g., 'Landsat', 'Sentinel-2')."""
        pass

    @property
    @abstractmethod
    def supported_collections(self) -> list[str]:
        """List of supported data collections."""
        pass

    @abstractmethod
    async def authenticate(self) -> bool:
        """
        Authenticate with the satellite data provider.

        Returns:
            bool: True if authentication successful

        Raises:
            AuthenticationError: If authentication fails
        """
        pass

    @abstractmethod
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
        Search for available scenes matching the criteria.

        Args:
            bbox: Bounding box for the area of interest
            start_date: Start of the date range
            end_date: End of the date range
            max_cloud_cover: Maximum cloud cover percentage (0-100)
            collections: Specific collections to search (None = all)
            max_results: Maximum number of results to return.
                ``None`` means return all matches (subject to API limits).

        Returns:
            SearchResult: Search results with scene metadata

        Raises:
            IngestionError: If search fails
        """
        pass

    @abstractmethod
    async def download(
        self,
        scene_id: str,
        output_dir: Path,
        bands: list[str] | None = None,
    ) -> Path:
        """
        Download a scene to the local filesystem.

        Args:
            scene_id: Unique identifier of the scene
            output_dir: Directory to save the downloaded data
            bands: Specific bands to download (None = all)

        Returns:
            Path: Path to the downloaded file/directory

        Raises:
            DownloadError: If download fails
        """
        pass

    @abstractmethod
    async def get_scene_metadata(self, scene_id: str) -> SceneMetadata:
        """
        Get detailed metadata for a specific scene.

        Args:
            scene_id: Unique identifier of the scene

        Returns:
            SceneMetadata: Detailed scene metadata
        """
        pass

    async def search_and_download(
        self,
        bbox: BoundingBox,
        start_date: date,
        end_date: date,
        output_dir: Path,
        max_cloud_cover: float = 100.0,
        max_scenes: int | None = None,
        bands: list[str] | None = None,
    ) -> list[Path]:
        """
        Convenience method to search and download scenes in one operation.

        Args:
            bbox: Bounding box for the area of interest
            start_date: Start of the date range
            end_date: End of the date range
            output_dir: Directory to save downloaded data
            max_cloud_cover: Maximum cloud cover percentage
            max_scenes: Maximum number of scenes to download.
                ``None`` means download all matching scenes.
            bands: Specific bands to download

        Returns:
            list[Path]: Paths to downloaded files
        """
        search_result = await self.search(
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            max_cloud_cover=max_cloud_cover,
            max_results=max_scenes,
        )

        scenes_to_download = (
            search_result.scenes[:max_scenes]
            if max_scenes is not None
            else search_result.scenes
        )
        downloaded_paths = []
        for scene in scenes_to_download:
            try:
                path = await self.download(
                    scene_id=scene.scene_id,
                    output_dir=output_dir,
                    bands=bands,
                )
                downloaded_paths.append(path)
                logger.info("Downloaded scene %s to %s", scene.scene_id, path)
            except Exception:
                logger.error(
                    "Failed to download scene %s",
                    scene.scene_id,
                    exc_info=True,
                )

        return downloaded_paths

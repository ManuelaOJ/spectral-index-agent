"""
Unit tests for imagery schemas.
"""

from datetime import date, datetime

import pytest

from spectral_agent.schemas.imagery import (
    BoundingBox,
    IngestionRequest,
    SatelliteType,
    SceneMetadata,
)


class TestBoundingBox:
    """Tests for BoundingBox schema."""

    def test_valid_bbox(self):
        """Test creating a valid bounding box."""
        bbox = BoundingBox(west=-122.5, south=37.5, east=-122.0, north=38.0)
        assert bbox.west == -122.5
        assert bbox.south == 37.5
        assert bbox.east == -122.0
        assert bbox.north == 38.0

    def test_invalid_bbox_west_greater_than_east(self):
        """Test that west > east raises error."""
        with pytest.raises(ValueError, match="West longitude must be less than east"):
            BoundingBox(west=-120.0, south=37.5, east=-122.0, north=38.0)

    def test_invalid_bbox_south_greater_than_north(self):
        """Test that south > north raises error."""
        with pytest.raises(ValueError, match="South latitude must be less than north"):
            BoundingBox(west=-122.5, south=39.0, east=-122.0, north=38.0)

    def test_to_wkt(self, sample_bbox):
        """Test WKT conversion."""
        wkt = sample_bbox.to_wkt()
        assert "POLYGON" in wkt
        assert "-122.5" in wkt
        assert "37.5" in wkt

    def test_to_geojson(self, sample_bbox):
        """Test GeoJSON conversion."""
        geojson = sample_bbox.to_geojson()
        assert geojson["type"] == "Polygon"
        assert len(geojson["coordinates"]) == 1
        assert len(geojson["coordinates"][0]) == 5  # Closed ring

    def test_from_geojson(self):
        """Test creating BoundingBox from GeoJSON."""
        geojson = {
            "type": "Polygon",
            "coordinates": [
                [
                    [-122.5, 37.5],
                    [-122.0, 37.5],
                    [-122.0, 38.0],
                    [-122.5, 38.0],
                    [-122.5, 37.5],
                ]
            ],
        }
        bbox = BoundingBox.from_geojson(geojson)
        assert bbox.west == -122.5
        assert bbox.east == -122.0


class TestIngestionRequest:
    """Tests for IngestionRequest schema."""

    def test_valid_request(self, sample_bbox):
        """Test creating a valid ingestion request."""
        request = IngestionRequest(
            bbox=sample_bbox,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 1),
        )
        assert request.satellites == [SatelliteType.LANDSAT, SatelliteType.SENTINEL]
        assert request.max_cloud_cover == 100.0

    def test_invalid_date_range(self, sample_bbox):
        """Test that start_date > end_date raises error."""
        with pytest.raises(ValueError, match="start_date must be before"):
            IngestionRequest(
                bbox=sample_bbox,
                start_date=date(2024, 6, 1),
                end_date=date(2024, 1, 1),
            )

    def test_custom_satellites(self, sample_bbox):
        """Test specifying custom satellites."""
        request = IngestionRequest(
            bbox=sample_bbox,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 1),
            satellites=[SatelliteType.SENTINEL],
        )
        assert request.satellites == [SatelliteType.SENTINEL]


class TestSceneMetadata:
    """Tests for SceneMetadata schema."""

    def test_valid_metadata(self):
        """Test creating valid scene metadata."""
        metadata = SceneMetadata(
            scene_id="LC08_L2SP_044034_20240101_20240105_02_T1",
            satellite=SatelliteType.LANDSAT,
            collection="landsat_ot_c2_l2",
            acquisition_date=datetime(2024, 1, 1, 10, 30, 0),
            cloud_cover=5.5,
        )
        assert metadata.scene_id == "LC08_L2SP_044034_20240101_20240105_02_T1"
        assert metadata.satellite == SatelliteType.LANDSAT
        assert metadata.cloud_cover == 5.5

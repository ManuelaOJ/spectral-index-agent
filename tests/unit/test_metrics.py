"""
Unit tests for the PipelineMetrics observability module.
"""

import pytest

from spectral_agent.tracking.metrics import (
    PipelineMetrics,
    StepRecord,
    StepTimer,
    get_metrics,
    track_step,
)


class TestStepRecord:
    """Tests for StepRecord creation and serialisation."""

    def test_create_success(self):
        rec = StepRecord(
            timestamp="2025-01-01T00:00:00Z",
            step_name="scene_search",
            duration_s=1.234,
            status="success",
            satellite="landsat",
        )
        assert rec.status == "success"
        assert rec.satellite == "landsat"
        assert rec.error is None

    def test_create_failure(self):
        rec = StepRecord(
            timestamp="2025-01-01T00:00:00Z",
            step_name="scene_download",
            duration_s=5.0,
            status="failure",
            error="ConnectionError: timeout",
        )
        assert rec.status == "failure"
        assert "ConnectionError" in rec.error

    def test_to_json_line(self):
        rec = StepRecord(
            timestamp="2025-01-01T00:00:00Z",
            step_name="index_computation",
            duration_s=2.5,
            status="success",
            index_name="NDVI",
            metadata={"scene_id": "LC09_test"},
        )
        line = rec.to_json_line()
        assert '"NDVI"' in line
        assert '"LC09_test"' in line


class TestPipelineMetrics:
    """Tests for the PipelineMetrics store."""

    def test_empty_summary(self):
        pm = PipelineMetrics()
        s = pm.summary()
        assert s["total_steps"] == 0
        assert s["successes"] == 0
        assert s["failures"] == 0

    def test_record_and_summary(self):
        pm = PipelineMetrics()
        pm.record(StepRecord(
            timestamp="2025-01-01T00:00:00Z",
            step_name="scene_search",
            duration_s=1.0,
            status="success",
            satellite="landsat",
        ))
        pm.record(StepRecord(
            timestamp="2025-01-01T00:00:01Z",
            step_name="scene_search",
            duration_s=2.0,
            status="failure",
            satellite="sentinel",
            error="API error",
        ))

        s = pm.summary()
        assert s["total_steps"] == 2
        assert s["successes"] == 1
        assert s["failures"] == 1
        assert s["total_duration_s"] == 3.0

        step_data = s["by_step"]["scene_search"]
        assert step_data["calls"] == 2
        assert step_data["avg_s"] == 1.5
        assert "landsat" in step_data["satellites"]
        assert "sentinel" in step_data["satellites"]

    def test_last_n(self):
        pm = PipelineMetrics()
        for i in range(5):
            pm.record(StepRecord(
                timestamp=f"2025-01-01T00:00:0{i}Z",
                step_name=f"step_{i}",
                duration_s=float(i),
                status="success",
            ))

        last2 = pm.last_n(2)
        assert len(last2) == 2
        assert last2[0].step_name == "step_3"
        assert last2[1].step_name == "step_4"

    def test_reset(self):
        pm = PipelineMetrics()
        pm.record(StepRecord(
            timestamp="2025-01-01T00:00:00Z",
            step_name="t",
            duration_s=0.1,
            status="success",
        ))
        assert len(pm.records) == 1
        pm.reset()
        assert len(pm.records) == 0

    def test_persist_to_file(self, tmp_path):
        log_file = tmp_path / "metrics.jsonl"
        pm = PipelineMetrics(log_path=log_file)
        pm.record(StepRecord(
            timestamp="2025-01-01T00:00:00Z",
            step_name="map_rendering",
            duration_s=3.14,
            status="success",
            index_name="NDVI",
        ))
        assert log_file.exists()
        content = log_file.read_text()
        assert '"NDVI"' in content
        assert '"map_rendering"' in content


class TestTrackStepContextManager:
    """Tests for the track_step context manager."""

    def test_success_path(self):
        pm = PipelineMetrics()
        with track_step(
            "test_step", satellite="landsat", metrics=pm,
        ) as step:
            step.set_metadata(foo="bar")

        assert len(pm.records) == 1
        rec = pm.records[0]
        assert rec.status == "success"
        assert rec.satellite == "landsat"
        assert rec.metadata["foo"] == "bar"
        assert rec.duration_s >= 0

    def test_failure_path(self):
        pm = PipelineMetrics()
        with pytest.raises(ValueError, match="boom"):
            with track_step("failing_step", metrics=pm):
                raise ValueError("boom")

        assert len(pm.records) == 1
        rec = pm.records[0]
        assert rec.status == "failure"
        assert "boom" in rec.error

    def test_index_name_tracking(self):
        pm = PipelineMetrics()
        with track_step(
            "index_computation", index_name="NDWI", metrics=pm,
        ):
            pass

        rec = pm.records[0]
        assert rec.index_name == "NDWI"


class TestGetMetricsSingleton:
    """Test the module-level singleton."""

    def test_returns_same_instance(self):
        # Reset the module-level singleton for clean test
        import spectral_agent.tracking.metrics as mod
        mod._metrics = None

        m1 = get_metrics(log_path=None)
        m2 = get_metrics(log_path=None)
        assert m1 is m2

        # Clean up
        mod._metrics = None

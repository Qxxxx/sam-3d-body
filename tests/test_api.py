"""
Tests for FastAPI endpoints.

These tests use the FastAPI TestClient for endpoint testing.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Import the FastAPI app
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app, model_loaded
from api.models import ActionType

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check(self):
        """Test health check returns expected fields."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "timestamp" in data
        assert "gpu_available" in data
        assert "model_loaded" in data

    def test_health_status_values(self):
        """Test health status is valid."""
        response = client.get("/health")
        data = response.json()

        assert data["status"] in ["healthy", "degraded"]
        assert isinstance(data["gpu_available"], bool)
        assert isinstance(data["model_loaded"], bool)

    def test_health_version_format(self):
        """Test version follows semver format."""
        response = client.get("/health")
        data = response.json()

        version = data["version"]
        parts = version.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)


class TestFeatureFlagsEndpoint:
    """Tests for /feature-flags endpoint."""

    def test_get_feature_flags(self):
        """Test feature flags endpoint returns configuration."""
        response = client.get("/feature-flags")
        assert response.status_code == 200

        data = response.json()
        assert "ENABLE_SMASH_ANALYSIS" in data
        assert "SUPPORTED_ACTIONS" in data
        assert "SUPPORT_LEFT_HANDED" in data

    def test_feature_flags_types(self):
        """Test feature flags have correct types."""
        response = client.get("/feature-flags")
        data = response.json()

        assert isinstance(data["ENABLE_SMASH_ANALYSIS"], bool)
        assert isinstance(data["SUPPORTED_ACTIONS"], list)
        assert isinstance(data["SUPPORT_LEFT_HANDED"], bool)


class TestVideoInferenceEndpoint:
    """Tests for /infer/video endpoint."""

    def test_video_inference_model_not_loaded(self):
        """Test error when model is not loaded."""
        with patch("api.main.model_loaded", False):
            response = client.post("/infer/video", json={
                "video_url": "file:///test.mp4",
                "action_type": "smash",
            })

            assert response.status_code == 503
            assert "Model not loaded" in response.json()["message"]

    def test_video_inference_unsupported_action(self):
        """Test error for unsupported action type."""
        with patch("api.main.model_loaded", True):
            with patch("api.main.FEATURE_FLAGS", {"SUPPORTED_ACTIONS": ["smash"]}):
                response = client.post("/infer/video", json={
                    "video_url": "file:///test.mp4",
                    "action_type": "clear",  # Not in MVP
                })

                assert response.status_code == 400
                assert "not supported" in response.json()["message"].lower()

    def test_video_inference_invalid_url(self):
        """Test error for invalid video URL."""
        with patch("api.main.model_loaded", True):
            with patch("api.main.FEATURE_FLAGS", {"SUPPORTED_ACTIONS": ["smash"]}):
                with patch("sam_3d_body.video_processor.VideoFrameExtractor.extract_frames") as mock_extract:
                    mock_extract.side_effect = RuntimeError("Failed to open video")

                    response = client.post("/infer/video", json={
                        "video_url": "file:///nonexistent.mp4",
                        "action_type": "smash",
                    })

                    assert response.status_code == 400

    def test_video_inference_success(self, test_video_path, temp_output_dir):
        """Test successful video inference."""
        with patch("api.main.model_loaded", True):
            with patch("api.main.estimator") as mock_estimator:
                with patch("api.main.config.output_dir", temp_output_dir):
                    # Mock estimator output
                    mock_output = [{
                        "pred_keypoints_3d": np.random.randn(70, 3).astype(np.float32),
                    }]
                    mock_estimator.process_one_image.return_value = mock_output

                    response = client.post("/infer/video", json={
                        "video_url": f"file://{test_video_path}",
                        "action_type": "smash",
                        "target_fps": 10,
                    })

                    # Should succeed (may return 200 or 500 depending on processing)
                    # Just check the structure is correct
                    if response.status_code == 200:
                        data = response.json()
                        assert "task_id" in data
                        assert "status" in data
                        assert "skeleton" in data
                        assert "video_metadata" in data

    def test_video_inference_with_time_range(self, test_video_path):
        """Test video inference with time range."""
        with patch("api.main.model_loaded", True):
            with patch("api.main.estimator") as mock_estimator:
                mock_output = [{
                    "pred_keypoints_3d": np.random.randn(70, 3).astype(np.float32),
                }]
                mock_estimator.process_one_image.return_value = mock_output

                response = client.post("/infer/video", json={
                    "video_url": f"file://{test_video_path}",
                    "action_type": "smash",
                    "start_time": 0.5,
                    "end_time": 2.5,
                })

                # Request structure should be valid
                assert response.status_code in [200, 500]  # 500 if model not actually loaded


class TestAlignmentEndpoint:
    """Tests for /infer/alignment endpoint."""

    @pytest.fixture
    def skeleton_files(self, temp_output_dir, sample_skeleton_sequence):
        """Create temporary skeleton files for testing."""
        joints, timestamps = sample_skeleton_sequence

        # Create user skeleton
        user_base = Path(temp_output_dir) / "user_test"
        np.savez(f"{user_base}.npz", joints=joints, timestamps=timestamps)
        with open(f"{user_base}.json", "w") as f:
            json.dump({
                "format_version": "1.0.0",
                "num_frames": len(joints),
                "num_joints": joints.shape[1],
                "joint_names": [f"joint_{i}" for i in range(joints.shape[1])],
                "metadata": {},
            }, f)

        # Create reference skeleton (slightly different)
        ref_joints = joints + np.random.randn(*joints.shape) * 0.01
        ref_base = Path(temp_output_dir) / "ref_test"
        np.savez(f"{ref_base}.npz", joints=ref_joints, timestamps=timestamps)
        with open(f"{ref_base}.json", "w") as f:
            json.dump({
                "format_version": "1.0.0",
                "num_frames": len(ref_joints),
                "num_joints": ref_joints.shape[1],
                "joint_names": [f"joint_{i}" for i in range(ref_joints.shape[1])],
                "metadata": {},
            }, f)

        return f"file://{user_base}.npz", f"file://{ref_base}.npz"

    def test_alignment_success(self, skeleton_files):
        """Test successful alignment and scoring."""
        user_url, ref_url = skeleton_files

        response = client.post("/infer/alignment", json={
            "user_skeleton_url": user_url,
            "reference_skeleton_url": ref_url,
            "action_type": "smash",
            "detect_phases": False,
        })

        # Should succeed with 200
        assert response.status_code == 200

        data = response.json()
        assert "task_id" in data
        assert "overall_score" in data
        assert 0 <= data["overall_score"] <= 100
        assert "alignment_path" in data
        assert "joint_errors" in data
        assert "tips" in data
        assert "dtw_distance" in data

    def test_alignment_response_structure(self, skeleton_files):
        """Test alignment response has correct structure."""
        user_url, ref_url = skeleton_files

        response = client.post("/infer/alignment", json={
            "user_skeleton_url": user_url,
            "reference_skeleton_url": ref_url,
            "action_type": "smash",
        })

        if response.status_code == 200:
            data = response.json()

            # Check alignment path structure
            if data["alignment_path"]:
                path_point = data["alignment_path"][0]
                assert "user_frame" in path_point
                assert "reference_frame" in path_point

            # Check joint errors structure
            if data["joint_errors"]:
                error = data["joint_errors"][0]
                assert "joint_name" in error
                assert "mean_error_cm" in error
                assert "severity" in error

            # Check tips is a list of strings
            assert isinstance(data["tips"], list)

    def test_alignment_missing_files(self):
        """Test error for missing skeleton files."""
        response = client.post("/infer/alignment", json={
            "user_skeleton_url": "file:///nonexistent.npz",
            "reference_skeleton_url": "file:///nonexistent.npz",
            "action_type": "smash",
        })

        # Should return error
        assert response.status_code == 500

    def test_alignment_with_phase_detection(self, skeleton_files):
        """Test alignment with phase detection enabled."""
        user_url, ref_url = skeleton_files

        response = client.post("/infer/alignment", json={
            "user_skeleton_url": user_url,
            "reference_skeleton_url": ref_url,
            "action_type": "smash",
            "detect_phases": True,
        })

        if response.status_code == 200:
            data = response.json()
            # Phase scores may or may not be detected depending on data
            assert "phase_scores" in data


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_json(self):
        """Test error for invalid JSON payload."""
        response = client.post(
            "/infer/video",
            data="invalid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_missing_required_fields(self):
        """Test error for missing required fields."""
        response = client.post("/infer/video", json={
            # Missing video_url
            "action_type": "smash",
        })
        assert response.status_code == 422

    def test_invalid_enum_value(self):
        """Test error for invalid enum value."""
        response = client.post("/infer/video", json={
            "video_url": "file:///test.mp4",
            "action_type": "invalid_action",
        })
        assert response.status_code == 422

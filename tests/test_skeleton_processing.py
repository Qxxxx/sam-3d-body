"""
Tests for skeleton_processing module.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from sam_3d_body.skeleton_processing import (
    SkeletonSequence,
    SkeletonNormalizer,
    TemporalSmoother,
    SkeletonSerializer,
    process_skeleton_sequence,
    NormalizationParams,
    MHR70_JOINT_NAMES,
)


class TestSkeletonSequence:
    """Tests for SkeletonSequence dataclass."""

    def test_initialization(self, sample_skeleton_sequence):
        """Test skeleton sequence initialization."""
        joints, timestamps = sample_skeleton_sequence

        seq = SkeletonSequence(
            joints=joints,
            timestamps=timestamps,
            joint_names=MHR70_JOINT_NAMES,
        )

        assert seq.num_frames == len(joints)
        assert seq.num_joints == 70
        assert len(seq.joint_names) == 70

    def test_get_joint(self, sample_skeleton_sequence):
        """Test getting joint trajectory by name."""
        joints, timestamps = sample_skeleton_sequence

        seq = SkeletonSequence(
            joints=joints,
            timestamps=timestamps,
            joint_names=MHR70_JOINT_NAMES,
        )

        nose_trajectory = seq.get_joint("nose")
        assert nose_trajectory is not None
        assert nose_trajectory.shape == (seq.num_frames, 3)

    def test_get_joint_invalid(self, sample_skeleton_sequence):
        """Test getting invalid joint name."""
        joints, timestamps = sample_skeleton_sequence

        seq = SkeletonSequence(
            joints=joints,
            timestamps=timestamps,
            joint_names=MHR70_JOINT_NAMES,
        )

        assert seq.get_joint("invalid_joint") is None

    def test_copy(self, sample_skeleton_sequence):
        """Test copying a skeleton sequence."""
        joints, timestamps = sample_skeleton_sequence

        seq = SkeletonSequence(joints=joints, timestamps=timestamps)
        seq_copy = seq.copy()

        assert seq_copy.num_frames == seq.num_frames
        assert not np.shares_memory(seq_copy.joints, seq.joints)


class TestSkeletonNormalizer:
    """Tests for SkeletonNormalizer class."""

    def test_normalization_translation(self, sample_skeleton_sequence):
        """Test that normalization centers at pelvis."""
        joints, timestamps = sample_skeleton_sequence

        seq = SkeletonSequence(
            joints=joints,
            timestamps=timestamps,
            joint_names=MHR70_JOINT_NAMES,
        )

        normalizer = SkeletonNormalizer()
        normalized, params = normalizer.normalize(seq)

        # After normalization, pelvis should be near origin
        left_hip_idx = MHR70_JOINT_NAMES.index("left_hip")
        right_hip_idx = MHR70_JOINT_NAMES.index("right_hip")

        pelvis = (normalized.joints[:, left_hip_idx, :] +
                  normalized.joints[:, right_hip_idx, :]) / 2.0

        assert np.allclose(pelvis, 0, atol=1e-5)

    def test_normalization_scale(self, sample_skeleton_sequence):
        """Test that normalization scales to unit height."""
        joints, timestamps = sample_skeleton_sequence

        seq = SkeletonSequence(
            joints=joints,
            timestamps=timestamps,
            joint_names=MHR70_JOINT_NAMES,
        )

        normalizer = SkeletonNormalizer()
        normalized, params = normalizer.normalize(seq)

        # Height should be approximately 1.0
        neck_idx = MHR70_JOINT_NAMES.index("neck")
        left_hip_idx = MHR70_JOINT_NAMES.index("left_hip")
        right_hip_idx = MHR70_JOINT_NAMES.index("right_hip")

        neck = normalized.joints[:, neck_idx, :]
        pelvis = (normalized.joints[:, left_hip_idx, :] +
                  normalized.joints[:, right_hip_idx, :]) / 2.0

        height = np.mean(np.linalg.norm(neck - pelvis, axis=1))
        assert height == pytest.approx(1.0, abs=0.1)

    def test_normalization_params(self, sample_skeleton_sequence):
        """Test normalization parameters are tracked."""
        joints, timestamps = sample_skeleton_sequence

        seq = SkeletonSequence(joints=joints, timestamps=timestamps)
        normalizer = SkeletonNormalizer(version="1.0.0")
        normalized, params = normalizer.normalize(seq)

        assert params.version == "1.0.0"
        assert params.scale > 0
        assert params.translation.shape == (3,)
        assert params.rotation.shape == (3, 3)

    def test_denormalize(self, sample_skeleton_sequence):
        """Test denormalization reverses normalization."""
        joints, timestamps = sample_skeleton_sequence

        seq = SkeletonSequence(joints=joints, timestamps=timestamps)
        normalizer = SkeletonNormalizer()

        normalized, params = normalizer.normalize(seq)
        denormalized = normalizer.denormalize(normalized, params)

        # Should approximately recover original
        assert np.allclose(denormalized.joints, seq.joints, atol=1e-4)

    def test_normalization_params_serialization(self):
        """Test serialization of normalization parameters."""
        params = NormalizationParams(
            version="1.0.0",
            translation=np.array([1.0, 2.0, 3.0]),
            scale=2.0,
            rotation=np.eye(3),
        )

        data = params.to_dict()
        restored = NormalizationParams.from_dict(data)

        assert restored.version == params.version
        assert restored.scale == params.scale
        assert np.array_equal(restored.translation, params.translation)
        assert np.array_equal(restored.rotation, params.rotation)


class TestTemporalSmoother:
    """Tests for TemporalSmoother class."""

    def test_gaussian_smoothing(self, sample_skeleton_sequence):
        """Test Gaussian smoothing reduces jitter."""
        joints, timestamps = sample_skeleton_sequence

        # Add noise
        noisy_joints = joints + np.random.randn(*joints.shape) * 0.01

        seq = SkeletonSequence(joints=noisy_joints, timestamps=timestamps)
        smoother = TemporalSmoother(method="gaussian", gaussian_sigma=2.0)

        smoothed = smoother.smooth(seq)

        # Variance should be reduced
        original_var = np.var(noisy_joints)
        smoothed_var = np.var(smoothed.joints)

        assert smoothed_var < original_var

    def test_savgol_smoothing(self, sample_skeleton_sequence):
        """Test Savitzky-Golay smoothing."""
        joints, timestamps = sample_skeleton_sequence

        seq = SkeletonSequence(joints=joints, timestamps=timestamps)
        smoother = TemporalSmoother(
            method="savgol",
            savgol_window=7,
            savgol_polyorder=3,
        )

        smoothed = smoother.smooth(seq)

        assert smoothed.num_frames == seq.num_frames
        assert "smoothed" in smoothed.metadata

    def test_smoothing_short_sequence(self):
        """Test handling of very short sequences."""
        joints = np.random.randn(2, 70, 3).astype(np.float32)
        timestamps = np.array([0.0, 0.033], dtype=np.float32)

        seq = SkeletonSequence(joints=joints, timestamps=timestamps)
        smoother = TemporalSmoother()

        # Should handle short sequences gracefully
        smoothed = smoother.smooth(seq)
        assert smoothed.num_frames == 2


class TestSkeletonSerializer:
    """Tests for SkeletonSerializer class."""

    def test_save_and_load(self, sample_skeleton_sequence, temp_output_dir):
        """Test saving and loading skeleton sequence."""
        joints, timestamps = sample_skeleton_sequence

        seq = SkeletonSequence(
            joints=joints,
            timestamps=timestamps,
            joint_names=MHR70_JOINT_NAMES,
        )

        serializer = SkeletonSerializer()
        output_base = Path(temp_output_dir) / "test_skeleton"

        npz_path, json_path = serializer.save(seq, output_base)

        # Check files exist
        assert Path(npz_path).exists()
        assert Path(json_path).exists()

        # Load and verify
        loaded = serializer.load(npz_path, json_path)

        assert loaded.num_frames == seq.num_frames
        assert loaded.num_joints == seq.num_joints
        assert np.allclose(loaded.joints, seq.joints)
        assert np.allclose(loaded.timestamps, seq.timestamps)

    def test_save_with_normalization(self, sample_skeleton_sequence, temp_output_dir):
        """Test saving sequence with normalization params."""
        joints, timestamps = sample_skeleton_sequence

        normalizer = SkeletonNormalizer()
        seq = SkeletonSequence(joints=joints, timestamps=timestamps)
        normalized, params = normalizer.normalize(seq)

        serializer = SkeletonSerializer()
        output_base = Path(temp_output_dir) / "test_norm"

        npz_path, json_path = serializer.save(normalized, output_base)

        # Check metadata contains normalization
        with open(json_path) as f:
            metadata = json.load(f)

        assert "normalization" in metadata
        assert metadata["normalization"]["version"] == "1.0.0"

    def test_load_with_normalization(self, sample_skeleton_sequence, temp_output_dir):
        """Test loading sequence with normalization params."""
        joints, timestamps = sample_skeleton_sequence

        normalizer = SkeletonNormalizer()
        seq = SkeletonSequence(joints=joints, timestamps=timestamps)
        normalized, params = normalizer.normalize(seq)

        serializer = SkeletonSerializer()
        output_base = Path(temp_output_dir) / "test_load_norm"
        npz_path, json_path = serializer.save(normalized, output_base)

        loaded = serializer.load(npz_path, json_path)

        assert loaded.normalization_params is not None
        assert loaded.normalization_params.version == "1.0.0"


class TestProcessSkeletonSequence:
    """Tests for process_skeleton_sequence convenience function."""

    def test_full_pipeline(self, sample_skeleton_sequence):
        """Test full processing pipeline."""
        joints, timestamps = sample_skeleton_sequence

        result = process_skeleton_sequence(
            joints=joints,
            timestamps=timestamps,
            normalize=True,
            smooth=True,
            smoothing_method="gaussian",
            smoothing_sigma=1.0,
        )

        assert isinstance(result, SkeletonSequence)
        assert result.metadata.get("normalized") is True
        assert result.metadata.get("smoothed") is True

    def test_pipeline_without_normalization(self, sample_skeleton_sequence):
        """Test pipeline without normalization."""
        joints, timestamps = sample_skeleton_sequence

        result = process_skeleton_sequence(
            joints=joints,
            timestamps=timestamps,
            normalize=False,
            smooth=True,
        )

        assert result.normalization_params is None

    def test_pipeline_without_smoothing(self, sample_skeleton_sequence):
        """Test pipeline without smoothing."""
        joints, timestamps = sample_skeleton_sequence

        result = process_skeleton_sequence(
            joints=joints,
            timestamps=timestamps,
            normalize=True,
            smooth=False,
        )

        assert result.metadata.get("smoothed") is None

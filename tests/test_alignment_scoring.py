"""
Tests for alignment_scoring module.
"""

import numpy as np
import pytest

from sam_3d_body.alignment_scoring import (
    FastDTWAligner,
    PoseScorer,
    align_and_score,
    AlignmentResult,
    ScoringResult,
    JointError,
)
from sam_3d_body.skeleton_processing import SkeletonSequence, MHR70_JOINT_NAMES


class TestFastDTWAligner:
    """Tests for FastDTWAligner class."""

    def test_align_identical_sequences(self, sample_skeleton_sequence):
        """Test alignment of identical sequences."""
        joints, timestamps = sample_skeleton_sequence

        seq1 = SkeletonSequence(joints=joints, timestamps=timestamps)
        seq2 = SkeletonSequence(joints=joints, timestamps=timestamps)

        aligner = FastDTWAligner()
        result = aligner.align(seq1, seq2)

        assert isinstance(result, AlignmentResult)
        assert len(result.path) > 0
        assert result.dtw_distance >= 0

    def test_align_different_length_sequences(self):
        """Test alignment of sequences with different lengths."""
        # Create sequences of different lengths
        joints1 = np.random.randn(50, 70, 3).astype(np.float32)
        joints2 = np.random.randn(80, 70, 3).astype(np.float32)

        timestamps1 = np.arange(50, dtype=np.float32) / 30.0
        timestamps2 = np.arange(80, dtype=np.float32) / 30.0

        seq1 = SkeletonSequence(joints=joints1, timestamps=timestamps1)
        seq2 = SkeletonSequence(joints=joints2, timestamps=timestamps2)

        aligner = FastDTWAligner()
        result = aligner.align(seq1, seq2)

        # Warped sequences should have same length
        assert result.query_warped.shape[0] == result.reference_warped.shape[0]

    def test_alignment_path_valid(self, sample_skeleton_sequence):
        """Test that alignment path is valid."""
        joints, timestamps = sample_skeleton_sequence

        # Create slightly different sequences
        seq1 = SkeletonSequence(joints=joints, timestamps=timestamps)
        seq2 = SkeletonSequence(
            joints=joints + np.random.randn(*joints.shape) * 0.01,
            timestamps=timestamps,
        )

        aligner = FastDTWAligner()
        result = aligner.align(seq1, seq2)

        # Path should be monotonic
        for i in range(1, len(result.path)):
            assert result.path[i][0] >= result.path[i-1][0]
            assert result.path[i][1] >= result.path[i-1][1]

    def test_frame_distances_shape(self, sample_skeleton_sequence):
        """Test frame distances have correct shape."""
        joints, timestamps = sample_skeleton_sequence

        seq1 = SkeletonSequence(joints=joints, timestamps=timestamps)
        seq2 = SkeletonSequence(joints=joints, timestamps=timestamps)

        aligner = FastDTWAligner()
        result = aligner.align(seq1, seq2)

        assert len(result.frame_distances) == len(result.path)
        assert result.joint_distances.shape[0] == len(result.path)
        assert result.joint_distances.shape[1] == 70  # num_joints


class TestPoseScorer:
    """Tests for PoseScorer class."""

    def test_score_identical_sequences(self, sample_skeleton_sequence):
        """Test scoring of identical sequences."""
        joints, timestamps = sample_skeleton_sequence

        seq1 = SkeletonSequence(joints=joints, timestamps=timestamps)
        seq2 = SkeletonSequence(joints=joints, timestamps=timestamps)

        scorer = PoseScorer()
        result = scorer.score(seq1, seq2)

        assert isinstance(result, ScoringResult)
        assert 0 <= result.overall_score <= 100

        # Identical sequences should have high score
        assert result.overall_score > 90

    def test_score_different_sequences(self):
        """Test scoring of significantly different sequences."""
        joints1 = np.random.randn(50, 70, 3).astype(np.float32)
        joints2 = np.random.randn(50, 70, 3).astype(np.float32) + 1.0

        timestamps = np.arange(50, dtype=np.float32) / 30.0

        seq1 = SkeletonSequence(joints=joints1, timestamps=timestamps)
        seq2 = SkeletonSequence(joints=joints2, timestamps=timestamps)

        scorer = PoseScorer()
        result = scorer.score(seq1, seq2)

        assert isinstance(result, ScoringResult)
        assert 0 <= result.overall_score <= 100

    def test_joint_errors_sorted(self, sample_skeleton_sequence):
        """Test that joint errors are sorted by severity."""
        joints, timestamps = sample_skeleton_sequence

        # Add error to specific joints
        joints_error = joints.copy()
        joints_error[:, 0, :] += 0.2  # Add error to first joint

        seq1 = SkeletonSequence(joints=joints_error, timestamps=timestamps)
        seq2 = SkeletonSequence(joints=joints, timestamps=timestamps)

        scorer = PoseScorer()
        result = scorer.score(seq1, seq2)

        # Check errors are sorted by mean_error (descending)
        for i in range(len(result.joint_errors) - 1):
            assert result.joint_errors[i].mean_error >= result.joint_errors[i+1].mean_error

    def test_joint_error_severity_levels(self, sample_skeleton_sequence):
        """Test joint error severity classification."""
        joints, timestamps = sample_skeleton_sequence

        scorer = PoseScorer()

        # Test severity thresholds
        assert scorer.CRITICAL_THRESHOLD > scorer.MODERATE_THRESHOLD
        assert scorer.MODERATE_THRESHOLD > scorer.MINOR_THRESHOLD

    def test_tips_generation(self, sample_skeleton_sequence):
        """Test that tips are generated."""
        joints, timestamps = sample_skeleton_sequence

        seq1 = SkeletonSequence(joints=joints, timestamps=timestamps)
        seq2 = SkeletonSequence(joints=joints, timestamps=timestamps)

        scorer = PoseScorer()
        result = scorer.score(seq1, seq2)

        assert isinstance(result.tips, list)
        assert len(result.tips) > 0
        assert all(isinstance(tip, str) for tip in result.tips)

    def test_top_errors_format(self, sample_skeleton_sequence):
        """Test top errors have correct format."""
        joints, timestamps = sample_skeleton_sequence

        # Create sequences with errors
        joints_error = joints.copy()
        joints_error[:, 5, :] += 0.15  # Error in shoulders

        seq1 = SkeletonSequence(joints=joints_error, timestamps=timestamps)
        seq2 = SkeletonSequence(joints=joints, timestamps=timestamps)

        scorer = PoseScorer()
        result = scorer.score(seq1, seq2)

        for error in result.top_errors:
            assert "joint" in error
            assert "severity" in error
            assert "mean_error_cm" in error
            assert "description" in error

    def test_phase_detection(self, sample_skeleton_sequence):
        """Test phase detection and scoring."""
        joints, timestamps = sample_skeleton_sequence

        seq1 = SkeletonSequence(joints=joints, timestamps=timestamps)
        seq2 = SkeletonSequence(joints=joints, timestamps=timestamps)

        scorer = PoseScorer()
        result = scorer.score(seq1, seq2, detect_phases=True)

        # May or may not detect phases depending on data
        if result.phase_scores:
            for phase_name, score in result.phase_scores.items():
                assert 0 <= score <= 100

    def test_detailed_scores(self, sample_skeleton_sequence):
        """Test detailed scores are generated."""
        joints, timestamps = sample_skeleton_sequence

        seq1 = SkeletonSequence(joints=joints, timestamps=timestamps)
        seq2 = SkeletonSequence(joints=joints, timestamps=timestamps)

        scorer = PoseScorer()
        result = scorer.score(seq1, seq2)

        assert len(result.detailed_scores) > 0

        for score in result.detailed_scores:
            assert hasattr(score, 'name')
            assert hasattr(score, 'score')
            assert 0 <= score.score <= 100
            assert hasattr(score, 'weight')
            assert hasattr(score, 'description')


class TestAlignAndScore:
    """Tests for align_and_score convenience function."""

    def test_full_pipeline(self, sample_skeleton_sequence):
        """Test full alignment and scoring pipeline."""
        joints, timestamps = sample_skeleton_sequence

        # Create user and reference sequences
        user_joints = joints + np.random.randn(*joints.shape) * 0.05

        user_seq = SkeletonSequence(joints=user_joints, timestamps=timestamps)
        ref_seq = SkeletonSequence(joints=joints, timestamps=timestamps)

        alignment, scoring = align_and_score(user_seq, ref_seq)

        assert isinstance(alignment, AlignmentResult)
        assert isinstance(scoring, ScoringResult)
        assert 0 <= scoring.overall_score <= 100

    def test_pipeline_with_phase_detection(self, sample_skeleton_sequence):
        """Test pipeline with phase detection enabled."""
        joints, timestamps = sample_skeleton_sequence

        user_seq = SkeletonSequence(joints=joints, timestamps=timestamps)
        ref_seq = SkeletonSequence(joints=joints, timestamps=timestamps)

        alignment, scoring = align_and_score(user_seq, ref_seq, detect_phases=True)

        assert isinstance(alignment, AlignmentResult)
        assert isinstance(scoring, ScoringResult)

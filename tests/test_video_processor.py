"""
Tests for video_processor module.
"""

import os
from unittest.mock import Mock, patch

import numpy as np
import pytest

from sam_3d_body.video_processor import (
    FrameData,
    SubjectTracker,
    VideoFrameExtractor,
    create_video_processor,
)


class TestVideoFrameExtractor:
    """Tests for VideoFrameExtractor class."""

    def test_initialization(self):
        """Test extractor initialization with default parameters."""
        extractor = VideoFrameExtractor()
        assert extractor.target_fps == 30.0
        assert extractor.max_retries == 3
        assert extractor.retry_delay == 1.0

    def test_initialization_custom_params(self):
        """Test extractor initialization with custom parameters."""
        extractor = VideoFrameExtractor(
            target_fps=60.0,
            max_retries=5,
            retry_delay=2.0,
        )
        assert extractor.target_fps == 60.0
        assert extractor.max_retries == 5
        assert extractor.retry_delay == 2.0

    def test_extract_frames_from_file(self, test_video_path):
        """Test frame extraction from a local video file."""
        extractor = VideoFrameExtractor(target_fps=30.0)
        frames, metadata = extractor.extract_frames(test_video_path)

        assert len(frames) > 0
        assert metadata.fps == 30.0
        assert metadata.width == 640
        assert metadata.height == 480
        assert metadata.duration == 3.0
        assert metadata.extracted_frames == len(frames)

        # Check frame data
        for i, frame in enumerate(frames):
            assert isinstance(frame, FrameData)
            assert frame.frame_idx == i
            assert frame.timestamp >= 0
            assert frame.image.shape == (480, 640, 3)
            assert frame.image.dtype == np.uint8

    def test_extract_frames_at_different_fps(self, test_video_path):
        """Test frame extraction at different target FPS."""
        # Extract at 10 FPS (should get fewer frames)
        extractor_10 = VideoFrameExtractor(target_fps=10.0)
        frames_10, _ = extractor_10.extract_frames(test_video_path)

        # Extract at 30 FPS
        extractor_30 = VideoFrameExtractor(target_fps=30.0)
        frames_30, _ = extractor_30.extract_frames(test_video_path)

        # 10 FPS should get approximately 1/3 the frames
        assert len(frames_10) < len(frames_30)
        assert len(frames_10) == pytest.approx(len(frames_30) / 3, abs=2)

    def test_extract_frames_with_time_range(self, test_video_path):
        """Test frame extraction with start and end time."""
        extractor = VideoFrameExtractor(target_fps=30.0)

        # Extract only middle second (1s to 2s)
        frames, metadata = extractor.extract_frames(
            test_video_path,
            start_time=1.0,
            end_time=2.0,
        )

        # Should get approximately 30 frames (1 second at 30fps)
        assert len(frames) == pytest.approx(30, abs=2)

        # Check timestamps are within range
        for frame in frames:
            assert 1.0 <= frame.timestamp <= 2.0

    def test_extract_frames_invalid_video(self):
        """Test handling of invalid video file."""
        extractor = VideoFrameExtractor()

        with pytest.raises(RuntimeError):
            extractor.extract_frames("/nonexistent/path/video.mp4")

    def test_download_video_file_url(self, test_video_path):
        """Test downloading video from file:// URL."""
        extractor = VideoFrameExtractor()
        url = f"file://{test_video_path}"

        downloaded_path = extractor._download_video(url)
        assert os.path.exists(downloaded_path)
        assert downloaded_path == test_video_path

    def test_download_video_invalid_file(self):
        """Test downloading from invalid file:// URL."""
        extractor = VideoFrameExtractor()

        with pytest.raises(FileNotFoundError):
            extractor._download_video("file:///nonexistent/video.mp4")

    @patch("sam_3d_body.video_processor.httpx.Client")
    def test_download_video_http(self, mock_client_class, temp_output_dir):
        """Test downloading video from HTTP URL."""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.content = b"fake video content"
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.get = Mock(return_value=mock_response)
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        extractor = VideoFrameExtractor(temp_dir=temp_output_dir)
        url = "https://example.com/video.mp4"

        downloaded_path = extractor._download_video(url)

        assert os.path.exists(downloaded_path)
        mock_client.get.assert_called_once_with(url)

        # Cleanup
        if os.path.exists(downloaded_path):
            os.remove(downloaded_path)

    def test_temp_file_cleanup(self, test_video_path):
        """Test that temporary files are cleaned up after processing."""
        extractor = VideoFrameExtractor()
        url = f"file://{test_video_path}"

        # Extract frames
        frames, _ = extractor.extract_frames(url)

        # Check no temp files remain
        temp_files = [f for f in os.listdir(extractor.temp_dir) if f.startswith("video_")]
        assert len(temp_files) == 0


class TestSubjectTracker:
    """Tests for SubjectTracker class."""

    def test_initialization_with_bbox(self, sample_bbox):
        """Test tracker initialization with bounding box."""
        tracker = SubjectTracker(initial_bbox=sample_bbox)
        assert np.array_equal(tracker.initial_bbox, sample_bbox)
        assert tracker.previous_bbox is None

    def test_iou_calculation_same_box(self, sample_bbox):
        """Test IoU calculation with identical boxes."""
        tracker = SubjectTracker()
        iou = tracker._compute_iou(sample_bbox, sample_bbox)
        assert iou == pytest.approx(1.0, abs=1e-6)

    def test_iou_calculation_no_overlap(self):
        """Test IoU calculation with non-overlapping boxes."""
        tracker = SubjectTracker()
        bbox1 = np.array([0, 0, 10, 10], dtype=np.float32)
        bbox2 = np.array([20, 20, 30, 30], dtype=np.float32)
        iou = tracker._compute_iou(bbox1, bbox2)
        assert iou == 0.0

    def test_iou_calculation_partial_overlap(self):
        """Test IoU calculation with partially overlapping boxes."""
        tracker = SubjectTracker()
        bbox1 = np.array([0, 0, 20, 20], dtype=np.float32)
        bbox2 = np.array([10, 10, 30, 30], dtype=np.float32)
        iou = tracker._compute_iou(bbox1, bbox2)

        # Intersection = 10x10 = 100
        # Union = 400 + 400 - 100 = 700
        # IoU = 100/700 = 0.142...
        assert iou == pytest.approx(100/700, abs=1e-6)

    def test_get_bbox_first_frame_with_initial(self, sample_bbox):
        """Test getting bbox for first frame with initial bbox."""
        tracker = SubjectTracker(initial_bbox=sample_bbox)
        bbox = tracker.get_bbox_for_frame(0)

        assert np.array_equal(bbox, sample_bbox)
        assert np.array_equal(tracker.previous_bbox, sample_bbox)

    def test_get_bbox_first_frame_without_initial(self):
        """Test getting bbox for first frame without initial bbox."""
        tracker = SubjectTracker()
        bbox = tracker.get_bbox_for_frame(0)
        assert bbox is None

    def test_get_bbox_single_detection(self, sample_bbox):
        """Test getting bbox with single detection."""
        tracker = SubjectTracker()
        detections = np.array([sample_bbox])

        bbox = tracker.get_bbox_for_frame(1, detections)
        assert np.array_equal(bbox, sample_bbox)

    def test_get_bbox_multi_detection_temporal_consistency(self):
        """Test multi-detection selection based on temporal consistency."""
        # Previous bbox
        prev_bbox = np.array([100, 100, 200, 200], dtype=np.float32)
        tracker = SubjectTracker()
        tracker.previous_bbox = prev_bbox

        # Multiple detectections - one close, one far
        detections = np.array([
            [105, 105, 205, 205],  # Close to previous (high IoU)
            [300, 300, 400, 400],  # Far from previous (low IoU)
        ], dtype=np.float32)

        bbox = tracker.get_bbox_for_frame(1, detections)

        # Should select the one with high IoU
        assert np.array_equal(bbox, detections[0])

    def test_get_bbox_multi_detection_below_threshold(self):
        """Test multi-detection when all below IoU threshold."""
        prev_bbox = np.array([100, 100, 200, 200], dtype=np.float32)
        tracker = SubjectTracker(iou_threshold=0.9)  # High threshold
        tracker.previous_bbox = prev_bbox

        # Detections with low IoU
        detections = np.array([
            [150, 150, 250, 250],  # IoU ~0.14
        ], dtype=np.float32)

        bbox = tracker.get_bbox_for_frame(1, detections)

        # Should still return first detection as fallback
        assert np.array_equal(bbox, detections[0])


class TestCreateVideoProcessor:
    """Tests for create_video_processor convenience function."""

    def test_create_without_subject(self):
        """Test creating processor without subject tracking."""
        extractor, tracker = create_video_processor(target_fps=30.0)

        assert isinstance(extractor, VideoFrameExtractor)
        assert extractor.target_fps == 30.0
        assert tracker is None

    def test_create_with_bbox(self, sample_bbox):
        """Test creating processor with bounding box."""
        bbox_list = sample_bbox.tolist()
        extractor, tracker = create_video_processor(
            target_fps=60.0,
            bbox=bbox_list,
        )

        assert isinstance(extractor, VideoFrameExtractor)
        assert isinstance(tracker, SubjectTracker)
        assert tracker.initial_bbox is not None
        assert np.array_equal(tracker.initial_bbox, sample_bbox)

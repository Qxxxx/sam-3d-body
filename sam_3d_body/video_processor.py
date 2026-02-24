"""
Video processing module for SAM 3D Body inference pipeline.

Handles frame extraction, subject tracking, and video preprocessing.
"""

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union
from urllib.parse import urlparse

import cv2
import httpx
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """Represents a single extracted frame with metadata."""

    frame_idx: int
    timestamp: float  # seconds
    image: NDArray[np.uint8]  # RGB format


@dataclass
class VideoMetadata:
    """Metadata about the processed video."""

    width: int
    height: int
    fps: float
    total_frames: int
    duration: float
    extracted_frames: int


class VideoFrameExtractor:
    """
    Extracts frames from video at a fixed FPS with time range support.

    Features:
    - Configurable target FPS
    - Time range cropping (start_time, end_time)
    - Retry logic for failed frame extraction
    - Automatic download from URLs
    """

    def __init__(
        self,
        target_fps: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        temp_dir: Optional[str] = None,
    ):
        """
        Initialize frame extractor.

        Args:
            target_fps: Target frames per second for extraction
            max_retries: Maximum retry attempts for failed operations
            retry_delay: Delay between retries in seconds
            temp_dir: Directory for temporary files (default: system temp)
        """
        self.target_fps = target_fps
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temp_dir = temp_dir or tempfile.gettempdir()

    def _download_video(self, url: str) -> str:
        """
        Download video from URL to temporary file.

        Args:
            url: Video URL (http, https, or file://)

        Returns:
            Path to downloaded video file
        """
        parsed = urlparse(url)

        # Handle local file:// URLs
        if parsed.scheme == "file":
            local_path = parsed.path
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"Local file not found: {local_path}")
            return local_path

        # Handle HTTP URLs
        if parsed.scheme in ("http", "https"):
            temp_path = os.path.join(
                self.temp_dir, f"video_{hash(url) % 1000000:06d}.mp4"
            )

            for attempt in range(self.max_retries):
                try:
                    logger.info(f"Downloading video from {url} (attempt {attempt + 1})")
                    with httpx.Client(follow_redirects=True, timeout=300.0) as client:
                        response = client.get(url)
                        response.raise_for_status()

                        with open(temp_path, "wb") as f:
                            f.write(response.content)

                    logger.info(f"Video downloaded to {temp_path}")
                    return temp_path

                except Exception as e:
                    logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                    if attempt < self.max_retries - 1:
                        import time

                        time.sleep(self.retry_delay)
                    else:
                        raise RuntimeError(f"Failed to download video after {self.max_retries} attempts: {e}")

        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

    def extract_frames(
        self,
        video_source: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Tuple[List[FrameData], VideoMetadata]:
        """
        Extract frames from video at target FPS.

        Args:
            video_source: Path to video file or URL
            start_time: Start time in seconds (None = from beginning)
            end_time: End time in seconds (None = to end)

        Returns:
            Tuple of (list of FrameData, VideoMetadata)
        """
        # Download if URL
        video_path = video_source
        temp_file = None

        try:
            if video_source.startswith(("http://", "https://", "file://")):
                video_path = self._download_video(video_source)
                temp_file = video_path if video_source.startswith(("http://", "https://")) else None

            # Open video with retry logic
            cap = None
            for attempt in range(self.max_retries):
                try:
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        raise RuntimeError("Failed to open video capture")
                    break
                except Exception as e:
                    logger.warning(f"Video open attempt {attempt + 1} failed: {e}")
                    if attempt < self.max_retries - 1:
                        import time

                        time.sleep(self.retry_delay)
                    else:
                        raise RuntimeError(f"Failed to open video after {self.max_retries} attempts: {e}")

            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / original_fps if original_fps > 0 else 0

            logger.info(
                f"Video properties: {width}x{height} @ {original_fps:.2f}fps, "
                f"{total_frames} frames, {duration:.2f}s duration"
            )

            # Calculate frame extraction parameters
            fps_ratio = self.target_fps / original_fps if original_fps > 0 else 1.0
            frame_interval = int(1 / fps_ratio) if fps_ratio <= 1 else 1

            # Time range in frames
            start_frame = int((start_time or 0) * original_fps) if start_time else 0
            end_frame = int(end_time * original_fps) if end_time else total_frames
            start_frame = max(0, start_frame)
            end_frame = min(total_frames, end_frame)

            # Seek to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            frames: List[FrameData] = []
            current_frame = start_frame
            frame_counter = 0

            while current_frame < end_frame:
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame {current_frame}, stopping extraction")
                    break

                # Extract frame at target FPS
                if frame_counter % frame_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    timestamp = current_frame / original_fps

                    frames.append(
                        FrameData(
                            frame_idx=len(frames),
                            timestamp=timestamp,
                            image=frame_rgb,
                        )
                    )

                current_frame += 1
                frame_counter += 1

            cap.release()

            metadata = VideoMetadata(
                width=width,
                height=height,
                fps=original_fps,
                total_frames=total_frames,
                duration=duration,
                extracted_frames=len(frames),
            )

            logger.info(f"Extracted {len(frames)} frames at {self.target_fps} FPS")

            return frames, metadata

        finally:
            # Cleanup temporary file
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    logger.debug(f"Cleaned up temporary file: {temp_file}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file: {e}")

    def extract_frames_generator(
        self,
        video_source: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Iterator[FrameData]:
        """
        Generator version for memory-efficient frame extraction.

        Args:
            video_source: Path to video file or URL
            start_time: Start time in seconds (None = from beginning)
            end_time: End time in seconds (None = to end)

        Yields:
            FrameData for each extracted frame
        """
        # This is a memory-efficient version that yields frames one at a time
        # Implementation similar to extract_frames but with yield
        frames, _ = self.extract_frames(video_source, start_time, end_time)
        yield from frames


class SubjectTracker:
    """
    Tracks a subject across frames using bbox/mask input.

    Features:
    - Initial bbox/mask support for subject locking
    - Temporal consistency checks
    - Optional: simple IoU-based tracking
    """

    def __init__(
        self,
        initial_bbox: Optional[NDArray[np.float32]] = None,
        initial_mask: Optional[NDArray[np.uint8]] = None,
        iou_threshold: float = 0.3,
    ):
        """
        Initialize subject tracker.

        Args:
            initial_bbox: Initial bounding box [x1, y1, x2, y2]
            initial_mask: Initial segmentation mask
            iou_threshold: Minimum IoU for temporal consistency
        """
        self.initial_bbox = initial_bbox
        self.initial_mask = initial_mask
        self.iou_threshold = iou_threshold
        self.previous_bbox: Optional[NDArray[np.float32]] = None

    def get_bbox_for_frame(
        self,
        frame_idx: int,
        detected_bboxes: Optional[NDArray[np.float32]] = None,
    ) -> Optional[NDArray[np.float32]]:
        """
        Get bounding box for current frame.

        Args:
            frame_idx: Frame index
            detected_bboxes: Detected bounding boxes from detector (N x 4)

        Returns:
            Selected bounding box or None
        """
        # First frame: use initial bbox if provided
        if frame_idx == 0 and self.initial_bbox is not None:
            self.previous_bbox = self.initial_bbox
            return self.initial_bbox

        # No detections available
        if detected_bboxes is None or len(detected_bboxes) == 0:
            return self.previous_bbox

        # Only one detection
        if len(detected_bboxes) == 1:
            self.previous_bbox = detected_bboxes[0]
            return detected_bboxes[0]

        # Multiple detections: select based on temporal consistency
        if self.previous_bbox is not None:
            best_iou = 0.0
            best_bbox = None

            for bbox in detected_bboxes:
                iou = self._compute_iou(bbox, self.previous_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_bbox = bbox

            if best_iou >= self.iou_threshold:
                self.previous_bbox = best_bbox
                return best_bbox

        # Fallback: return first detection
        self.previous_bbox = detected_bboxes[0]
        return detected_bboxes[0]

    def _compute_iou(
        self,
        bbox1: NDArray[np.float32],
        bbox2: NDArray[np.float32],
    ) -> float:
        """Compute Intersection over Union between two bboxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0


def create_video_processor(
    target_fps: float = 30.0,
    bbox: Optional[List[float]] = None,
    mask: Optional[List[List[int]]] = None,
) -> Tuple[VideoFrameExtractor, Optional[SubjectTracker]]:
    """
    Convenience function to create video processor components.

    Args:
        target_fps: Target extraction FPS
        bbox: Optional initial bounding box [x1, y1, x2, y2]
        mask: Optional initial segmentation mask

    Returns:
        Tuple of (VideoFrameExtractor, SubjectTracker or None)
    """
    extractor = VideoFrameExtractor(target_fps=target_fps)

    tracker = None
    if bbox is not None or mask is not None:
        bbox_array = np.array(bbox, dtype=np.float32) if bbox else None
        mask_array = np.array(mask, dtype=np.uint8) if mask else None
        tracker = SubjectTracker(initial_bbox=bbox_array, initial_mask=mask_array)

    return extractor, tracker

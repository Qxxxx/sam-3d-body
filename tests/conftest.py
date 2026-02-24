"""
Pytest configuration and fixtures for SAM 3D Body tests.
"""

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def test_video_path():
    """Create a test video file and return its path."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        video_path = f.name

    # Create a simple test video (30fps, 3 seconds = 90 frames)
    width, height = 640, 480
    fps = 30
    duration = 3
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for i in range(fps * duration):
        # Create a frame with a moving circle
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        x = int((i / (fps * duration)) * width)
        y = height // 2
        cv2.circle(frame, (x, y), 20, (0, 0, 255), -1)
        out.write(frame)

    out.release()

    yield video_path

    # Cleanup
    if os.path.exists(video_path):
        os.remove(video_path)


@pytest.fixture
def sample_frame():
    """Return a sample RGB frame."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_bbox():
    """Return a sample bounding box [x1, y1, x2, y2]."""
    return np.array([100, 100, 300, 400], dtype=np.float32)


@pytest.fixture
def sample_skeleton_sequence():
    """Return a sample skeleton sequence (100 frames, 70 joints, 3D)."""
    np.random.seed(42)
    num_frames = 100
    num_joints = 70

    # Create smooth motion
    t = np.linspace(0, 2 * np.pi, num_frames)
    joints = np.zeros((num_frames, num_joints, 3), dtype=np.float32)

    for j in range(num_joints):
        joints[:, j, 0] = np.sin(t + j * 0.1) * 0.5 + j * 0.01  # x
        joints[:, j, 1] = np.cos(t + j * 0.1) * 0.3 + j * 0.005  # y
        joints[:, j, 2] = np.sin(2 * t + j * 0.05) * 0.2  # z

    timestamps = np.arange(num_frames, dtype=np.float32) / 30.0  # 30fps

    return joints, timestamps


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for output files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_video_url(test_video_path):
    """Return a file:// URL for the test video."""
    return f"file://{test_video_path}"

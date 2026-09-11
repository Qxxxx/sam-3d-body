from __future__ import annotations

import io
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

import cv2
import numpy as np
import pytest

from sam_3d_body.video_processor import (
    VideoExtractionConfig,
    extract_skeleton_sequence_from_video,
)


def _write_dummy_video(path: Path, fps: float, num_frames: int) -> None:
    width, height = 120, 80
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Failed to create test video")

    try:
        for frame_idx in range(num_frames):
            color = int((frame_idx * 17) % 255)
            frame = np.full((height, width, 3), color, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


class _SinglePersonEstimator:
    def __init__(self) -> None:
        self.call_count = 0

    def process_one_image(self, _frame_rgb: np.ndarray, **_kwargs: Any) -> list[dict[str, Any]]:
        value = float(self.call_count)
        self.call_count += 1
        keypoints = np.array(
            [
                [value, 0.0, 0.0],
                [value + 1.0, 0.0, 0.0],
                [value + 2.0, 0.0, 0.0],
                [value + 3.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        return [
            {
                "bbox": np.array([10, 10, 60, 60], dtype=np.float32),
                "pred_keypoints_3d": keypoints,
                "cam_intrinsics": np.array(
                    [[100.0, 0.0, 60.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]],
                    dtype=np.float32,
                ),
                "camera_source": "moge2",
            }
        ]


class _TwoPersonEstimator:
    def process_one_image(self, _frame_rgb: np.ndarray, **_kwargs: Any) -> list[dict[str, Any]]:
        left = {
            "bbox": np.array([0, 0, 40, 40], dtype=np.float32),
            "pred_keypoints_3d": np.array(
                [[1.0, 0.0, 0.0], [1.1, 0.0, 0.0], [1.2, 0.0, 0.0]],
                dtype=np.float32,
            ),
        }
        right = {
            "bbox": np.array([80, 0, 119, 40], dtype=np.float32),
            "pred_keypoints_3d": np.array(
                [[9.0, 0.0, 0.0], [9.1, 0.0, 0.0], [9.2, 0.0, 0.0]],
                dtype=np.float32,
            ),
        }
        return [left, right]


def test_extract_skeleton_sequence_samples_by_target_fps(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    _write_dummy_video(video_path, fps=10.0, num_frames=12)

    sequence = extract_skeleton_sequence_from_video(
        video_path=video_path,
        estimator=_SinglePersonEstimator(),  # type: ignore[arg-type]
        config=VideoExtractionConfig(
            target_fps=5.0,  # should sample every 2 frames from source fps 10
            max_frames=4,
        ),
    )

    assert sequence.num_frames == 4
    assert sequence.num_joints == 4
    assert np.allclose(sequence.timestamps, np.array([0.0, 0.2, 0.4, 0.6], dtype=np.float32), atol=0.05)


def test_extract_skeleton_sequence_honors_selection_bbox(tmp_path: Path) -> None:
    video_path = tmp_path / "selection.mp4"
    _write_dummy_video(video_path, fps=8.0, num_frames=4)

    sequence = extract_skeleton_sequence_from_video(
        video_path=video_path,
        estimator=_TwoPersonEstimator(),  # type: ignore[arg-type]
        config=VideoExtractionConfig(target_fps=8.0, max_frames=2),
        selection_bbox_xyxy=(80.0, 0.0, 119.0, 40.0),
    )

    assert sequence.num_frames == 2
    # First joint should come from the right-side person's synthetic keypoints.
    assert float(sequence.keypoints_3d[0, 0, 0]) == 9.0


def test_extract_skeleton_sequence_supports_http_video_path(
    tmp_path: Path, monkeypatch: Any
) -> None:
    source_video = tmp_path / "remote-source.mp4"
    _write_dummy_video(source_video, fps=10.0, num_frames=8)
    source_bytes = source_video.read_bytes()

    class _FakeResponse:
        def __init__(self, payload: bytes) -> None:
            self._buffer = io.BytesIO(payload)

        def read(self, size: int = -1) -> bytes:
            return self._buffer.read(size)

        def close(self) -> None:
            self._buffer.close()

        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, *_args: Any) -> bool:
            self.close()
            return False

    def _fake_urlopen(url: str, timeout: int = 30) -> _FakeResponse:
        assert url == "https://example.com/test-video.mp4"
        assert timeout == 30
        return _FakeResponse(source_bytes)

    monkeypatch.setattr("sam_3d_body.video_processor.urlopen", _fake_urlopen)

    sequence = extract_skeleton_sequence_from_video(
        video_path="https://example.com/test-video.mp4",
        estimator=_SinglePersonEstimator(),  # type: ignore[arg-type]
        config=VideoExtractionConfig(
            target_fps=5.0,
            max_frames=3,
        ),
    )

    assert sequence.num_frames == 3
    assert sequence.num_joints == 4


def test_extract_skeleton_sequence_surfaces_remote_http_errors(monkeypatch: Any) -> None:
    def _fake_urlopen(url: str, timeout: int = 30):
        assert url == "https://example.com/forbidden.mp4"
        assert timeout == 30
        raise HTTPError(url, 403, "Forbidden", hdrs=None, fp=None)

    monkeypatch.setattr("sam_3d_body.video_processor.urlopen", _fake_urlopen)

    with pytest.raises(ConnectionError, match="HTTP 403"):
        extract_skeleton_sequence_from_video(
            video_path="https://example.com/forbidden.mp4",
            estimator=_SinglePersonEstimator(),  # type: ignore[arg-type]
            config=VideoExtractionConfig(
                target_fps=5.0,
                max_frames=3,
            ),
        )


def test_extract_skeleton_sequence_returns_camera_metadata(tmp_path: Path) -> None:
    video_path = tmp_path / "camera.mp4"
    _write_dummy_video(video_path, fps=10.0, num_frames=10)

    sequence, camera = extract_skeleton_sequence_from_video(
        video_path=video_path,
        estimator=_SinglePersonEstimator(),  # type: ignore[arg-type]
        config=VideoExtractionConfig(
            target_fps=5.0,
            max_frames=3,
        ),
        return_camera_metadata=True,
    )

    assert sequence.num_frames == 3
    assert camera is not None
    assert camera["source"] == "moge2"
    assert len(camera["horizontalFovDeg"]) == 3
    assert len(camera["timestamps"]) == 3


@pytest.mark.parametrize("failure", ["timeout", "partial", "503"])
def test_remote_video_retries_and_discards_partial_download(monkeypatch: Any, failure: str) -> None:
    from sam_3d_body.video_processor import _resolve_video_file
    calls = []
    class PartialResponse(io.BytesIO):
        def read(self, size=-1):
            if self.tell() > 0:
                raise ConnectionResetError("disconnected")
            return super().read(3)
    def open_url(url, timeout):
        calls.append(url)
        if len(calls) == 1:
            if failure == "timeout":
                raise TimeoutError("network timeout")
            if failure == "503":
                raise HTTPError(url, 503, "Unavailable", None, None)
            return PartialResponse(b"partial")
        return io.BytesIO(b"complete-video")
    monkeypatch.setattr("sam_3d_body.video_processor.urlopen", open_url)
    monkeypatch.setattr("sam_3d_body.video_processor.time.sleep", lambda _: None)
    with _resolve_video_file("https://example.com/video.mp4?signature=secret") as resolved:
        assert resolved.read_bytes() == b"complete-video"
    assert not resolved.exists()
    assert len(calls) == 2
    assert calls[0] == calls[1]


@pytest.mark.parametrize("failure,attempts", [(403, 1), (404, 1), (503, 3), ("timeout", 3)])
def test_remote_video_bounds_retries_and_redacts_errors(monkeypatch: Any, failure: Any, attempts: int) -> None:
    from sam_3d_body.video_processor import _resolve_video_file
    calls = []
    def open_url(url, timeout):
        calls.append(url)
        if failure == "timeout":
            raise TimeoutError(f"Timed out: {url}")
        raise HTTPError(url, failure, "Failure", None, None)
    monkeypatch.setattr("sam_3d_body.video_processor.urlopen", open_url)
    monkeypatch.setattr("sam_3d_body.video_processor.time.sleep", lambda _: None)
    with pytest.raises((ConnectionError, FileNotFoundError, TimeoutError)) as error:
        with _resolve_video_file("https://example.com/video.mp4?signature=secret"):
            pass
    assert len(calls) == attempts
    assert "secret" not in str(error.value)
    assert "https://" not in str(error.value)

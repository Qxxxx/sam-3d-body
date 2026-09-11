from __future__ import annotations

import io
from urllib.error import HTTPError, URLError

import numpy as np
import pytest

from sam_3d_body.technique_alignment import (
    AlignmentConfig,
    NormalizationConfig,
    SkeletonSequence,
    align_sequences,
    build_alignment_report,
    load_skeleton_sequence_npz,
    normalize_skeleton_sequence,
    resolve_npz_file,
)


def _make_sequence(frames: np.ndarray) -> SkeletonSequence:
    timestamps = np.arange(frames.shape[0], dtype=np.float32) / 10.0
    joint_names = tuple(f"joint_{idx}" for idx in range(frames.shape[1]))
    return SkeletonSequence(
        keypoints_3d=frames.astype(np.float32),
        timestamps=timestamps,
        joint_names=joint_names,
    )


def test_normalization_keeps_root_at_origin_and_aligns_hips() -> None:
    frames = np.array(
        [
            [
                [10.0, 5.0, 3.0],   # root
                [9.0, 5.0, 3.0],    # left hip
                [11.0, 5.0, 3.0],   # right hip
                [10.0, 6.0, 3.0],   # up joint
            ],
            [
                [20.0, 15.0, 8.0],  # translated + scaled
                [18.0, 15.0, 8.0],
                [22.0, 15.0, 8.0],
                [20.0, 17.0, 8.0],
            ],
        ],
        dtype=np.float32,
    )
    sequence = _make_sequence(frames)
    normalized = normalize_skeleton_sequence(
        sequence,
        NormalizationConfig(smooth_window=1),
    )

    assert normalized.shape == frames.shape
    assert np.allclose(normalized[:, 0, :], 0.0, atol=1e-5)

    # Hips should be primarily aligned with x-axis after orientation normalization.
    hips_vec = normalized[0, 2] - normalized[0, 1]
    assert hips_vec[0] > 0.5
    assert abs(hips_vec[1]) < 1e-3
    assert abs(hips_vec[2]) < 1e-3


def test_align_sequences_falls_back_to_exact_dtw() -> None:
    user_frames = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]],
            [[0.2, 0.0, 0.0], [1.2, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    reference_frames = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.05, 0.0, 0.0], [1.05, 0.0, 0.0]],
            [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]],
            [[0.2, 0.0, 0.0], [1.2, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    distance, path, algorithm = align_sequences(
        user_frames,
        reference_frames,
        use_fastdtw=False,
    )

    assert algorithm == "exact_dtw"
    assert distance >= 0
    assert path[0] == (0, 0)
    assert path[-1] == (2, 3)


def test_build_alignment_report_outputs_expected_sections() -> None:
    user_frames = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [2.0, 0.2, 0.0], [3.0, 0.3, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [2.0, 0.3, 0.0], [3.0, 0.4, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.3, 0.0], [2.0, 0.4, 0.0], [3.0, 0.5, 0.0]],
        ],
        dtype=np.float32,
    )
    reference_frames = user_frames * 1.05
    report = build_alignment_report(
        _make_sequence(user_frames),
        _make_sequence(reference_frames),
        config=AlignmentConfig(use_fastdtw=False),
    )

    assert report["algorithm"] == "exact_dtw"
    assert report["summary"]["numAlignedPairs"] >= 3
    assert len(report["alignmentPath"]) == report["summary"]["numAlignedPairs"]
    assert len(report["frameErrors"]) == report["summary"]["numAlignedPairs"]
    assert len(report["jointErrors"]) == user_frames.shape[1]
    assert report["frameErrors"][0]["phase"] in {
        "preparation",
        "acceleration",
        "contact",
        "follow_through",
    }


def test_load_skeleton_sequence_npz_supports_remote_url(tmp_path, monkeypatch) -> None:
    local_npz = tmp_path / "reference.npz"
    expected_keypoints = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            [[0.1, 0.0, 0.0], [1.1, 1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    expected_timestamps = np.array([0.0, 0.1], dtype=np.float32)
    expected_joint_names = np.array(["hip", "wrist"], dtype=object)
    np.savez(
        local_npz,
        keypoints_3d=expected_keypoints,
        timestamps=expected_timestamps,
        joint_names=expected_joint_names,
    )
    payload = local_npz.read_bytes()

    class _FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()
            return False

    def _fake_urlopen(url: str, timeout: int):
        assert url == "https://example.com/reference.npz"
        assert timeout == 60
        return _FakeResponse(payload)

    monkeypatch.setattr("sam_3d_body.technique_alignment.urlopen", _fake_urlopen)

    sequence = load_skeleton_sequence_npz("https://example.com/reference.npz")
    assert sequence.keypoints_3d.shape == (2, 2, 3)
    assert np.allclose(sequence.keypoints_3d, expected_keypoints)
    assert np.allclose(sequence.timestamps, expected_timestamps)
    assert sequence.joint_names == ("hip", "wrist")


def test_load_skeleton_sequence_npz_surfaces_remote_http_errors(monkeypatch) -> None:
    def _fake_urlopen(url: str, timeout: int):
        assert url == "https://example.com/forbidden.npz"
        assert timeout == 60
        raise HTTPError(url, 403, "Forbidden", hdrs=None, fp=None)

    monkeypatch.setattr("sam_3d_body.technique_alignment.urlopen", _fake_urlopen)

    with pytest.raises(ConnectionError, match="HTTP 403"):
        load_skeleton_sequence_npz("https://example.com/forbidden.npz")


@pytest.mark.parametrize("failure", [
    TimeoutError("secret URL"),
    URLError(TimeoutError("secret URL")),
    ConnectionResetError("secret URL"),
    HTTPError("https://example.com/private?signature=secret", 503, "Unavailable", None, None),
])
def test_remote_asset_retries_transient_failure(monkeypatch, failure) -> None:
    attempts = []
    delays = []

    def fetch(url, timeout):
        attempts.append(timeout)
        if len(attempts) == 1:
            raise failure
        return io.BytesIO(b"complete asset")

    monkeypatch.setattr("sam_3d_body.technique_alignment.urlopen", fetch)
    monkeypatch.setattr("sam_3d_body.technique_alignment.time.sleep", delays.append)
    with resolve_npz_file("https://example.com/reference.render.npz") as path:
        assert path.read_bytes() == b"complete asset"
    assert not path.exists()
    assert attempts == [60, 60]
    assert delays == [1]


def test_remote_asset_restarts_partial_download(monkeypatch) -> None:
    calls = []

    class PartialResponse(io.BytesIO):
        def read(self, size=-1):
            if self.tell():
                raise TimeoutError("interrupted")
            return super().read(size)

    def fetch(url, timeout):
        calls.append(url)
        return PartialResponse(b"partial junk") if len(calls) == 1 else io.BytesIO(b"ok")

    monkeypatch.setattr("sam_3d_body.technique_alignment.urlopen", fetch)
    monkeypatch.setattr("sam_3d_body.technique_alignment.time.sleep", lambda _: None)
    with resolve_npz_file("https://example.com/reference.render.npz") as path:
        assert path.read_bytes() == b"ok"
    assert len(calls) == 2
    assert not path.exists()


def test_remote_asset_exhausts_retries_without_exposing_signed_url(monkeypatch, tmp_path) -> None:
    calls = []
    signed_url = "https://example.com/private.npz?X-Amz-Signature=secret"

    def fetch(url, timeout):
        calls.append(url)
        raise URLError(TimeoutError(signed_url))

    monkeypatch.setattr("sam_3d_body.technique_alignment.urlopen", fetch)
    monkeypatch.setattr("sam_3d_body.technique_alignment.time.sleep", lambda _: None)
    monkeypatch.setattr("sam_3d_body.technique_alignment.tempfile.tempdir", str(tmp_path))
    with pytest.raises(TimeoutError) as caught:
        with resolve_npz_file(signed_url):
            pytest.fail("must not yield an incomplete asset")
    assert len(calls) == 3
    assert str(caught.value) == "Timed out fetching 3D reference asset"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("status,exception", [(403, ConnectionError), (404, FileNotFoundError)])
def test_remote_asset_does_not_retry_permanent_http_failure(monkeypatch, status, exception) -> None:
    calls = []

    def fetch(url, timeout):
        calls.append(url)
        raise HTTPError(url, status, "secret", None, None)

    monkeypatch.setattr("sam_3d_body.technique_alignment.urlopen", fetch)
    with pytest.raises(exception) as caught:
        with resolve_npz_file("https://example.com/private?signature=secret"):
            pytest.fail("must not yield")
    assert len(calls) == 1
    assert "secret" not in str(caught.value)
    assert "https" not in str(caught.value)


def test_remote_asset_preserves_consumer_error_without_retry(monkeypatch) -> None:
    calls = []

    def fetch(url, timeout):
        calls.append(url)
        return io.BytesIO(b"asset")

    monkeypatch.setattr("sam_3d_body.technique_alignment.urlopen", fetch)
    failure = ValueError("invalid render fields")
    with pytest.raises(ValueError) as caught:
        with resolve_npz_file("https://example.com/reference.render.npz") as path:
            raise failure
    assert caught.value is failure
    assert len(calls) == 1
    assert not path.exists()

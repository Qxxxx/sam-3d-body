from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import time
from typing import Any

import cv2
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException
from pydantic import ValidationError

from api.config import ApiSettings, load_api_settings
from api.main import (
    TechniqueTraceContext,
    _emit_inference_failure_event,
    _emit_trace_event,
    _run_video_inference_sync,
    create_app,
)
from api.models import VideoInferenceRequest


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
            color = int((frame_idx * 19) % 255)
            frame = np.full((height, width, 3), color, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


class _DummyEstimator:
    def __init__(self) -> None:
        self.call_count = 0

    def process_one_image(
        self, _frame_rgb: np.ndarray, **_kwargs: Any
    ) -> list[dict[str, Any]]:
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


def _route_endpoint(app: Any, path: str, method: str = "POST") -> Any:
    for route in app.routes:
        if getattr(route, "path", None) == path and method in getattr(
            route, "methods", set()
        ):
            return route.endpoint
    raise RuntimeError(f"Route not found: {method} {path}")


def _build_async_job_request(video_path: str) -> dict[str, Any]:
    return {
        "videoPath": video_path,
        "selection": {"bbox": [10, 20, 60, 70]},
        "videoConfig": {"targetFps": 5.0, "maxFrames": 3},
        "assetConfig": {
            "assetId": "user_bundle",
            "actionType": "smash",
            "handedness": "right",
            "metadata": {
                "taskId": "tech-task-unit-test",
                "userId": "user-unit-test",
                "traceId": "tech-trace-unit-test",
                "clientRunId": "tech-client-run-unit-test",
                "matchId": "match-unit-test",
                "referenceAssetId": "ref-unit-test",
            },
        },
        "storage": {
            "mode": "direct_upload",
            "prefix": "unit-tests",
            "uploads": {
                "skeleton": {
                    "putUrl": "https://uploads.example/skeleton.npz",
                    "fetchUrl": "r2://test-bucket/technique/user-assets/user_bundle/skeleton.npz",
                    "contentType": "application/octet-stream",
                },
                "render": {
                    "putUrl": "https://uploads.example/render.npz",
                    "fetchUrl": "r2://test-bucket/technique/user-assets/user_bundle/render.npz",
                    "contentType": "application/octet-stream",
                },
                "metadata": {
                    "putUrl": "https://uploads.example/metadata.json",
                    "fetchUrl": "r2://test-bucket/technique/user-assets/user_bundle/metadata.json",
                    "contentType": "application/json",
                },
            },
        },
        "callback": {
            "url": "https://callback.example/internal/service-callback",
            "token": "callback-token",
            "taskId": "tech-task-unit-test",
            "traceId": "tech-trace-unit-test",
            "clientRunId": "tech-client-run-unit-test",
        },
    }


def _build_async_job_result(asset_id: str) -> dict[str, Any]:
    return {
        "assetId": asset_id,
        "summary": {
            "numFrames": 3,
            "numJoints": 4,
            "firstTimestamp": 0.0,
            "lastTimestamp": 0.4,
            "durationSec": 0.4,
            "sourceFps": 10.0,
            "imageSizeHw": [80, 120],
            "frameIndices": [0, 2, 4],
        },
        "files": {
            "skeleton": {
                "path": "/tmp/skeleton.npz",
                "fetchUrl": "r2://test-bucket/technique/user-assets/user_bundle/skeleton.npz",
                "sizeBytes": 101,
            },
            "render": {
                "path": "/tmp/render.npz",
                "fetchUrl": "r2://test-bucket/technique/user-assets/user_bundle/render.npz",
                "sizeBytes": 202,
            },
            "metadata": {
                "path": "/tmp/metadata.json",
                "fetchUrl": "r2://test-bucket/technique/user-assets/user_bundle/metadata.json",
                "sizeBytes": 303,
            },
        },
        "manifest": {
            "assetCount": 1,
            "assets": [{"selectionBbox": [10.0, 20.0, 60.0, 70.0]}],
        },
    }


def test_health_endpoint_returns_service_status() -> None:
    app = create_app(estimator=_DummyEstimator())
    health_endpoint = _route_endpoint(app, "/health", method="GET")

    payload = health_endpoint()
    assert payload["status"] == "ok"
    assert payload["modelLoaded"] is True
    assert payload["modelLoadError"] is None


def test_load_api_settings_defaults_artifact_root_to_system_tmp(
    monkeypatch: Any,
) -> None:
    monkeypatch.delenv("SAM3DBODY_ARTIFACT_ROOT", raising=False)

    settings = load_api_settings()

    assert (
        Path(settings.artifact_root) == Path(tempfile.gettempdir()) / "sam3d-body-api"
    )


def test_run_video_inference_sync_returns_asset_manifest_and_local_files(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "user.mp4"
    _write_dummy_video(video_path, fps=10.0, num_frames=10)
    artifact_root = tmp_path / "artifacts"

    app = create_app(
        estimator=_DummyEstimator(),
        settings=ApiSettings(
            checkpoint_path="/tmp/model.ckpt",
            mhr_path="/tmp/mhr_model.pt",
            device="cpu",
            fov_name="moge2",
            fov_path="",
            artifact_root=str(artifact_root),
        ),
    )
    request = VideoInferenceRequest.model_validate(
        {
            "videoPath": str(video_path),
            "selection": {"bbox": [10, 20, 60, 70]},
            "videoConfig": {"targetFps": 5.0, "maxFrames": 3},
            "assetConfig": {
                "assetId": "user_bundle",
                "actionType": "smash",
                "handedness": "right",
            },
            "storage": {
                "mode": "local",
                "prefix": "unit-tests",
            },
        }
    )

    payload = _run_video_inference_sync(
        app.state.service_state, request, TechniqueTraceContext()
    )
    assert payload["assetId"] == "user_bundle"
    assert payload["summary"]["numFrames"] == 3
    assert payload["summary"]["numJoints"] == 4
    assert payload["summary"]["sourceFps"] == pytest.approx(10.0)
    assert payload["summary"]["frameIndices"] == [0, 2, 4]
    assert payload["camera"]["source"] == "moge2"
    assert len(payload["camera"]["horizontalFovDeg"]) == 3
    assert len(payload["camera"]["timestamps"]) == 3
    assert payload["manifest"]["assetCount"] == 1
    assert payload["manifest"]["assets"][0]["selectionBbox"] == [10.0, 20.0, 60.0, 70.0]

    skeleton_path = Path(payload["files"]["skeleton"]["path"])
    render_path = Path(payload["files"]["render"]["path"])
    metadata_path = Path(payload["files"]["metadata"]["path"])
    assert skeleton_path.exists()
    assert render_path.exists()
    assert metadata_path.exists()
    assert payload["files"]["skeleton"]["fetchUrl"] is not None
    assert payload["files"]["render"]["fetchUrl"] is not None
    assert payload["files"]["metadata"]["fetchUrl"] is not None

    client = TestClient(app)
    render_response = client.get(payload["files"]["render"]["fetchUrl"])
    assert render_response.status_code == 200
    assert len(render_response.content) > 0


def test_run_video_inference_sync_uploads_generated_files_for_direct_upload(
    tmp_path: Path, monkeypatch: Any
) -> None:
    video_path = tmp_path / "user.mp4"
    _write_dummy_video(video_path, fps=10.0, num_frames=10)
    artifact_root = tmp_path / "artifacts"
    uploaded_requests: list[tuple[str, bytes, dict[str, str]]] = []

    def _fake_httpx_put(
        url: str,
        *,
        content: bytes,
        headers: dict[str, str],
        follow_redirects: bool,
        timeout: float,
    ) -> httpx.Response:
        uploaded_requests.append((url, bytes(content), dict(headers)))
        assert follow_redirects is True
        assert timeout == 120.0
        return httpx.Response(200, request=httpx.Request("PUT", url))

    monkeypatch.setattr("api.main.httpx.put", _fake_httpx_put)

    app = create_app(
        estimator=_DummyEstimator(),
        settings=ApiSettings(
            checkpoint_path="/tmp/model.ckpt",
            mhr_path="/tmp/mhr_model.pt",
            device="cpu",
            fov_name="moge2",
            fov_path="",
            artifact_root=str(artifact_root),
        ),
    )
    request = VideoInferenceRequest.model_validate(
        {
            "videoPath": str(video_path),
            "selection": {"bbox": [10, 20, 60, 70]},
            "videoConfig": {"targetFps": 5.0, "maxFrames": 3},
            "assetConfig": {
                "assetId": "user_bundle",
                "actionType": "smash",
                "handedness": "right",
            },
            "storage": {
                "mode": "direct_upload",
                "prefix": "unit-tests",
                "uploads": {
                    "skeleton": {
                        "putUrl": "https://uploads.example/skeleton.npz",
                        "fetchUrl": "r2://test-bucket/technique/user-assets/user_bundle/skeleton.npz",
                        "contentType": "application/octet-stream",
                    },
                    "render": {
                        "putUrl": "https://uploads.example/render.npz",
                        "fetchUrl": "r2://test-bucket/technique/user-assets/user_bundle/render.npz",
                        "contentType": "application/octet-stream",
                    },
                    "metadata": {
                        "putUrl": "https://uploads.example/metadata.json",
                        "fetchUrl": "r2://test-bucket/technique/user-assets/user_bundle/metadata.json",
                        "contentType": "application/json",
                    },
                },
            },
        }
    )

    payload = _run_video_inference_sync(
        app.state.service_state, request, TechniqueTraceContext()
    )
    assert payload["assetId"] == "user_bundle"
    assert payload["files"]["skeleton"]["fetchUrl"] == (
        "r2://test-bucket/technique/user-assets/user_bundle/skeleton.npz"
    )
    assert payload["files"]["render"]["fetchUrl"] == (
        "r2://test-bucket/technique/user-assets/user_bundle/render.npz"
    )
    assert payload["files"]["metadata"]["fetchUrl"] == (
        "r2://test-bucket/technique/user-assets/user_bundle/metadata.json"
    )
    assert len(uploaded_requests) == 3
    assert [request_url for request_url, _, _ in uploaded_requests] == [
        "https://uploads.example/skeleton.npz",
        "https://uploads.example/render.npz",
        "https://uploads.example/metadata.json",
    ]
    assert uploaded_requests[0][2]["Content-Type"] == "application/octet-stream"
    assert uploaded_requests[1][2]["Content-Type"] == "application/octet-stream"
    assert uploaded_requests[2][2]["Content-Type"] == "application/json"
    assert all(len(body) > 0 for _, body, _ in uploaded_requests)
    assert not Path(payload["files"]["skeleton"]["path"]).exists()
    assert not Path(payload["files"]["render"]["path"]).exists()
    assert not Path(payload["files"]["metadata"]["path"]).exists()
    assert not (artifact_root / "unit-tests" / "user_bundle").exists()


def test_run_video_inference_sync_uploads_cropped_video_when_requested(
    tmp_path: Path, monkeypatch: Any
) -> None:
    video_path = tmp_path / "user.mp4"
    _write_dummy_video(video_path, fps=10.0, num_frames=10)
    artifact_root = tmp_path / "artifacts"
    uploaded_requests: list[tuple[str, bytes, dict[str, str]]] = []

    def _fake_httpx_put(
        url: str,
        *,
        content: bytes,
        headers: dict[str, str],
        follow_redirects: bool,
        timeout: float,
    ) -> httpx.Response:
        uploaded_requests.append((url, bytes(content), dict(headers)))
        assert follow_redirects is True
        assert timeout == 120.0
        return httpx.Response(200, request=httpx.Request("PUT", url))

    monkeypatch.setattr("api.main.httpx.put", _fake_httpx_put)

    app = create_app(
        estimator=_DummyEstimator(),
        settings=ApiSettings(
            checkpoint_path="/tmp/model.ckpt",
            mhr_path="/tmp/mhr_model.pt",
            device="cpu",
            fov_name="moge2",
            fov_path="",
            artifact_root=str(artifact_root),
        ),
    )
    request = VideoInferenceRequest.model_validate(
        {
            "videoPath": str(video_path),
            "selection": {"bbox": [10, 20, 60, 70]},
            "videoConfig": {"targetFps": 5.0, "maxFrames": 3},
            "assetConfig": {
                "assetId": "user_bundle",
                "actionType": "smash",
                "handedness": "right",
            },
            "storage": {
                "mode": "direct_upload",
                "prefix": "unit-tests",
                "uploads": {
                    "skeleton": {
                        "putUrl": "https://uploads.example/skeleton.npz",
                        "fetchUrl": "r2://test-bucket/technique/user-assets/user_bundle/skeleton.npz",
                        "contentType": "application/octet-stream",
                    },
                    "render": {
                        "putUrl": "https://uploads.example/render.npz",
                        "fetchUrl": "r2://test-bucket/technique/user-assets/user_bundle/render.npz",
                        "contentType": "application/octet-stream",
                    },
                    "metadata": {
                        "putUrl": "https://uploads.example/metadata.json",
                        "fetchUrl": "r2://test-bucket/technique/user-assets/user_bundle/metadata.json",
                        "contentType": "application/json",
                    },
                    "croppedVideo": {
                        "putUrl": "https://uploads.example/source.mp4",
                        "fetchUrl": "r2://test-bucket/technique/user-assets/user_bundle/source.mp4",
                        "contentType": "video/mp4",
                    },
                },
            },
        }
    )

    payload = _run_video_inference_sync(
        app.state.service_state, request, TechniqueTraceContext()
    )
    assert payload["files"]["croppedVideo"]["fetchUrl"] == (
        "r2://test-bucket/technique/user-assets/user_bundle/source.mp4"
    )
    assert len(uploaded_requests) == 4
    assert [request_url for request_url, _, _ in uploaded_requests] == [
        "https://uploads.example/skeleton.npz",
        "https://uploads.example/render.npz",
        "https://uploads.example/metadata.json",
        "https://uploads.example/source.mp4",
    ]
    assert uploaded_requests[3][2]["Content-Type"] == "video/mp4"
    assert all(len(body) > 0 for _, body, _ in uploaded_requests)
    assert not Path(payload["files"]["croppedVideo"]["path"]).exists()
    assert not (artifact_root / "unit-tests" / "user_bundle").exists()


def test_run_video_inference_sync_emits_trace_events_with_request_context(
    tmp_path: Path, monkeypatch: Any
) -> None:
    video_path = tmp_path / "user.mp4"
    _write_dummy_video(video_path, fps=10.0, num_frames=10)
    artifact_root = tmp_path / "artifacts"
    trace_events: list[tuple[str, str]] = []

    def _fake_httpx_put(
        url: str,
        *,
        content: bytes,
        headers: dict[str, str],
        follow_redirects: bool,
        timeout: float,
    ) -> httpx.Response:
        assert follow_redirects is True
        assert timeout == 120.0
        return httpx.Response(200, request=httpx.Request("PUT", url))

    def _capture_trace_event(_settings: Any, _context: Any, **kwargs: Any) -> None:
        trace_events.append((kwargs["stage"], kwargs["message"]))

    monkeypatch.setattr("api.main.httpx.put", _fake_httpx_put)
    monkeypatch.setattr("api.main._emit_trace_event", _capture_trace_event)

    app = create_app(
        estimator=_DummyEstimator(),
        settings=ApiSettings(
            checkpoint_path="/tmp/model.ckpt",
            mhr_path="/tmp/mhr_model.pt",
            device="cpu",
            fov_name="moge2",
            fov_path="",
            artifact_root=str(artifact_root),
        ),
    )
    request = VideoInferenceRequest.model_validate(
        {
            "videoPath": str(video_path),
            "selection": {"bbox": [10, 20, 60, 70]},
            "videoConfig": {"targetFps": 5.0, "maxFrames": 3},
            "assetConfig": {
                "assetId": "user_bundle",
                "actionType": "smash",
                "handedness": "right",
                "metadata": {
                    "taskId": "tech-task-unit-test",
                    "userId": "user-unit-test",
                    "traceId": "tech-trace-unit-test",
                    "clientRunId": "tech-client-run-unit-test",
                    "matchId": "match-unit-test",
                    "referenceAssetId": "ref-unit-test",
                },
            },
            "storage": {
                "mode": "direct_upload",
                "prefix": "unit-tests",
                "uploads": {
                    "skeleton": {
                        "putUrl": "https://uploads.example/skeleton.npz",
                        "fetchUrl": "r2://test-bucket/technique/user-assets/user_bundle/skeleton.npz",
                        "contentType": "application/octet-stream",
                    },
                    "render": {
                        "putUrl": "https://uploads.example/render.npz",
                        "fetchUrl": "r2://test-bucket/technique/user-assets/user_bundle/render.npz",
                        "contentType": "application/octet-stream",
                    },
                    "metadata": {
                        "putUrl": "https://uploads.example/metadata.json",
                        "fetchUrl": "r2://test-bucket/technique/user-assets/user_bundle/metadata.json",
                        "contentType": "application/json",
                    },
                },
            },
        }
    )
    _run_video_inference_sync(
        app.state.service_state,
        request,
        TechniqueTraceContext(
            trace_id="tech-trace-unit-test",
            task_id="tech-task-unit-test",
            user_id="user-unit-test",
            client_run_id="tech-client-run-unit-test",
            match_id="match-unit-test",
            technique_type="smash",
            reference_asset_id="ref-unit-test",
        ),
    )

    assert [stage for stage, _message in trace_events] == [
        "request_received",
        "artifact_output_dir_resolved",
        "inference_started",
        "artifact_written",
        "direct_upload_started",
        "direct_upload_completed",
        "response_sent",
    ]


def test_emit_trace_event_uses_explicit_user_agent_for_ingest(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    class _FakeResponse:
        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            return False

        def read(self) -> bytes:
            return b'{"code":0,"message":"ok"}'

    def _fake_urlopen(request_obj: Any, timeout: int = 30) -> _FakeResponse:
        captured["url"] = request_obj.full_url
        captured["timeout"] = timeout
        captured["headers"] = {
            key.lower(): value for key, value in request_obj.header_items()
        }
        return _FakeResponse()

    monkeypatch.setattr("api.main.urlopen", _fake_urlopen)

    settings = ApiSettings(
        checkpoint_path="/tmp/model.ckpt",
        mhr_path="/tmp/mhr_model.pt",
        device="cpu",
        fov_name="moge2",
        fov_path="",
        artifact_root="/tmp/artifacts",
        technique_trace_ingest_url=(
            "https://api-staging.duolian.cc/api/v1/analysis/technique/observability/events/internal"
        ),
        technique_trace_ingest_token="trace-token",
        runtime_env="staging-gpu",
    )
    context = TechniqueTraceContext(
        trace_id="tech-trace-unit-test",
        task_id="tech-task-unit-test",
        user_id="user-unit-test",
        client_run_id="run-unit-test",
        match_id="match-unit-test",
        technique_type="smash",
        reference_asset_id="ref-unit-test",
    )

    _emit_trace_event(
        settings,
        context,
        stage="probe",
        message="probe",
    )

    assert captured["url"] == settings.technique_trace_ingest_url
    assert captured["timeout"] == 5
    assert captured["headers"]["x-technique-trace-ingest-token"] == "trace-token"
    assert captured["headers"]["user-agent"] == "curl/8.7.1"


def test_emit_trace_event_preserves_cf_backend_compatible_payload(
    monkeypatch: Any,
) -> None:
    captured: dict[str, Any] = {}

    class _FakeResponse:
        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            return False

        def read(self) -> bytes:
            return b'{"code":0,"message":"ok"}'

    def _fake_urlopen(request_obj: Any, timeout: int = 30) -> _FakeResponse:
        captured["timeout"] = timeout
        captured["body"] = json.loads(request_obj.data.decode("utf-8"))
        return _FakeResponse()

    monkeypatch.setattr("api.main.urlopen", _fake_urlopen)

    settings = ApiSettings(
        checkpoint_path="/tmp/model.ckpt",
        mhr_path="/tmp/mhr_model.pt",
        device="cpu",
        fov_name="moge2",
        fov_path="",
        artifact_root="/tmp/artifacts",
        technique_trace_ingest_url=(
            "https://api-staging.duolian.cc/api/v1/analysis/technique/observability/events/internal"
        ),
        technique_trace_ingest_token="trace-token",
        runtime_env="staging-gpu",
    )
    context = TechniqueTraceContext(
        trace_id="tech-trace-unit-test",
        task_id="tech-task-unit-test",
        user_id="user-unit-test",
        client_run_id="run-unit-test",
        match_id="match-unit-test",
        technique_type="smash",
        reference_asset_id="ref-unit-test",
    )

    _emit_trace_event(
        settings,
        context,
        stage="probe",
        message="probe",
        level="warn",
        meta={"attempt": 1, "path": "/tmp/probe"},
    )

    assert captured["timeout"] == 5
    assert captured["body"]["traceId"] == "tech-trace-unit-test"
    assert captured["body"]["taskId"] == "tech-task-unit-test"
    assert captured["body"]["userId"] == "user-unit-test"
    assert captured["body"]["service"] == "sam-3d-body"
    assert captured["body"]["runtimeEnv"] == "staging-gpu"
    assert captured["body"]["level"] == "warn"
    assert captured["body"]["stage"] == "probe"
    assert captured["body"]["message"] == "probe"
    assert captured["body"]["matchId"] == "match-unit-test"
    assert captured["body"]["clientRunId"] == "run-unit-test"
    assert captured["body"]["techniqueType"] == "smash"
    assert captured["body"]["referenceAssetId"] == "ref-unit-test"
    assert captured["body"]["meta"] == {"attempt": 1, "path": "/tmp/probe"}
    assert isinstance(captured["body"]["createdAt"], int)


def test_run_video_inference_sync_emits_estimator_load_trace_events(
    tmp_path: Path, monkeypatch: Any
) -> None:
    video_path = tmp_path / "user.mp4"
    _write_dummy_video(video_path, fps=10.0, num_frames=8)
    artifact_root = tmp_path / "artifacts"
    trace_events: list[tuple[str, dict[str, Any] | None]] = []

    def _capture_trace_event(_settings: Any, _context: Any, **kwargs: Any) -> None:
        trace_events.append((kwargs["stage"], kwargs.get("meta")))

    monkeypatch.setattr("api.main._build_estimator", lambda _settings: _DummyEstimator())
    monkeypatch.setattr("api.main._emit_trace_event", _capture_trace_event)

    app = create_app(
        estimator=None,
        settings=ApiSettings(
            checkpoint_path="/tmp/model.ckpt",
            mhr_path="/tmp/mhr_model.pt",
            device="cpu",
            fov_name="moge2",
            fov_path="",
            artifact_root=str(artifact_root),
        ),
    )
    request = VideoInferenceRequest.model_validate(
        {
            "videoPath": str(video_path),
            "selection": {"bbox": [10, 20, 60, 70]},
            "videoConfig": {"targetFps": 5.0, "maxFrames": 3},
            "assetConfig": {
                "assetId": "user_bundle",
                "actionType": "smash",
                "handedness": "right",
            },
            "storage": {
                "mode": "local",
                "prefix": "unit-tests",
            },
        }
    )

    _run_video_inference_sync(
        app.state.service_state,
        request,
        TechniqueTraceContext(trace_id="tech-trace-unit-test"),
    )

    stages = [stage for stage, _meta in trace_events]
    assert "estimator_load_started" in stages
    assert "estimator_load_succeeded" in stages
    started_meta = next(meta for stage, meta in trace_events if stage == "estimator_load_started")
    assert started_meta is not None
    assert started_meta["device"] == "cpu"
    assert started_meta["fovName"] == "moge2"


def test_run_video_inference_sync_emits_artifact_upload_failed_trace_event(
    tmp_path: Path, monkeypatch: Any
) -> None:
    video_path = tmp_path / "user.mp4"
    _write_dummy_video(video_path, fps=10.0, num_frames=8)
    artifact_root = tmp_path / "artifacts"
    trace_events: list[tuple[str, dict[str, Any] | None]] = []

    def _fake_httpx_put(
        url: str,
        *,
        content: bytes,
        headers: dict[str, str],
        follow_redirects: bool,
        timeout: float,
    ) -> httpx.Response:
        raise httpx.TimeoutException("upload timed out")

    def _capture_trace_event(_settings: Any, _context: Any, **kwargs: Any) -> None:
        trace_events.append((kwargs["stage"], kwargs.get("meta")))

    monkeypatch.setattr("api.main.httpx.put", _fake_httpx_put)
    monkeypatch.setattr("api.main._emit_trace_event", _capture_trace_event)

    app = create_app(
        estimator=_DummyEstimator(),
        settings=ApiSettings(
            checkpoint_path="/tmp/model.ckpt",
            mhr_path="/tmp/mhr_model.pt",
            device="cpu",
            fov_name="moge2",
            fov_path="",
            artifact_root=str(artifact_root),
        ),
    )
    request = VideoInferenceRequest.model_validate(
        {
            "videoPath": str(video_path),
            "selection": {"bbox": [10, 20, 60, 70]},
            "videoConfig": {"targetFps": 5.0, "maxFrames": 3},
            "assetConfig": {
                "assetId": "user_bundle",
                "actionType": "smash",
                "handedness": "right",
            },
            "storage": {
                "mode": "direct_upload",
                "prefix": "unit-tests",
                "uploads": {
                    "skeleton": {
                        "putUrl": "https://uploads.example/skeleton.npz",
                        "fetchUrl": "r2://test-bucket/technique/user-assets/user_bundle/skeleton.npz",
                        "contentType": "application/octet-stream",
                    },
                    "render": {
                        "putUrl": "https://uploads.example/render.npz",
                        "fetchUrl": "r2://test-bucket/technique/user-assets/user_bundle/render.npz",
                        "contentType": "application/octet-stream",
                    },
                    "metadata": {
                        "putUrl": "https://uploads.example/metadata.json",
                        "fetchUrl": "r2://test-bucket/technique/user-assets/user_bundle/metadata.json",
                        "contentType": "application/json",
                    },
                },
            },
        }
    )

    with pytest.raises(TimeoutError, match="Timed out uploading generated artifact"):
        _run_video_inference_sync(
            app.state.service_state,
            request,
            TechniqueTraceContext(trace_id="tech-trace-unit-test"),
        )

    stages = [stage for stage, _meta in trace_events]
    assert "direct_upload_started" in stages
    assert "artifact_upload_failed" in stages
    upload_failed_meta = next(
        meta for stage, meta in trace_events if stage == "artifact_upload_failed"
    )
    assert upload_failed_meta is not None
    assert upload_failed_meta["artifactType"] == "skeleton"
    assert upload_failed_meta["fetchUrl"] == (
        "r2://test-bucket/technique/user-assets/user_bundle/skeleton.npz"
    )


def test_run_video_inference_sync_maps_remote_fetch_failures_to_bad_gateway(
    tmp_path: Path, monkeypatch: Any
) -> None:
    artifact_root = tmp_path / "artifacts"
    app = create_app(
        estimator=_DummyEstimator(),
        settings=ApiSettings(
            checkpoint_path="/tmp/model.ckpt",
            mhr_path="/tmp/mhr_model.pt",
            device="cpu",
            fov_name="moge2",
            fov_path="",
            artifact_root=str(artifact_root),
        ),
    )
    request = VideoInferenceRequest.model_validate(
        {
            "videoPath": "https://example.com/forbidden.mp4",
            "assetConfig": {
                "assetId": "user_bundle",
                "actionType": "smash",
                "handedness": "right",
            },
            "storage": {
                "mode": "local",
                "prefix": "unit-tests",
            },
        }
    )

    def _fake_urlopen(_url: str, timeout: int = 30):
        raise ConnectionError(
            f"Failed to fetch video: https://example.com/forbidden.mp4 (HTTP 403)"
        )

    monkeypatch.setattr("sam_3d_body.video_processor.urlopen", _fake_urlopen)

    with pytest.raises(HTTPException) as exc_info:
        try:
            _run_video_inference_sync(
                app.state.service_state, request, TechniqueTraceContext()
            )
        except Exception as exc:
            raise _emit_inference_failure_event(
                app.state.service_state.settings,
                TechniqueTraceContext(),
                exc,
            ) from exc

    assert exc_info.value.status_code == 502
    assert "Failed to fetch video" in str(exc_info.value.detail)


def test_video_inference_request_uses_selection_bbox() -> None:
    payload = VideoInferenceRequest.model_validate(
        {
            "videoPath": "/tmp/video.mp4",
            "selection": {"bbox": [10, 20, 30, 40]},
        }
    )
    assert payload.selection is not None
    assert payload.selection.bbox_xyxy == (10.0, 20.0, 30.0, 40.0)


def test_video_inference_request_defaults_target_fps_to_30() -> None:
    payload = VideoInferenceRequest.model_validate({"videoPath": "/tmp/video.mp4"})

    assert payload.video_config.target_fps == 30.0


def test_video_inference_request_rejects_legacy_fields() -> None:
    with pytest.raises(ValidationError):
        VideoInferenceRequest.model_validate(
            {
                "videoPath": "/tmp/video.mp4",
                "jointNames": ["a", "b", "c"],
            }
        )

    with pytest.raises(ValidationError):
        VideoInferenceRequest.model_validate(
            {
                "videoPath": "/tmp/video.mp4",
                "selectionBbox": [10, 20, 30, 40],
            }
        )

    with pytest.raises(ValidationError):
        VideoInferenceRequest.model_validate(
            {
                "videoPath": "/tmp/video.mp4",
                "saveNpzPath": "/tmp/out.npz",
            }
        )


def test_infer_video_jobs_endpoint_returns_accepted_response_without_waiting_for_completion(
    tmp_path: Path, monkeypatch: Any
) -> None:
    artifact_root = tmp_path / "artifacts"
    started_event = threading.Event()
    release_event = threading.Event()
    callback_requests: list[dict[str, Any]] = []

    def _fake_run_video_inference_sync(
        _state: Any,
        payload: Any,
        _trace_context: Any,
    ) -> dict[str, Any]:
        started_event.set()
        assert payload.asset_config.asset_id == "user_bundle"
        assert release_event.wait(timeout=2.0) is True
        return _build_async_job_result("user_bundle")

    def _fake_httpx_post(
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
        follow_redirects: bool,
        timeout: float,
    ) -> httpx.Response:
        callback_requests.append(
            {
                "url": url,
                "json": json,
                "headers": headers,
                "follow_redirects": follow_redirects,
                "timeout": timeout,
            }
        )
        return httpx.Response(202, request=httpx.Request("POST", url))

    monkeypatch.setattr(
        "api.main._run_video_inference_sync", _fake_run_video_inference_sync
    )
    monkeypatch.setattr("api.main.httpx.post", _fake_httpx_post)
    monkeypatch.setattr("api.main.JOB_CALLBACK_LOOP_INTERVAL_SECONDS", 0.01)

    app = create_app(
        estimator=_DummyEstimator(),
        settings=ApiSettings(
            checkpoint_path="/tmp/model.ckpt",
            mhr_path="/tmp/mhr_model.pt",
            device="cpu",
            fov_name="moge2",
            fov_path="",
            artifact_root=str(artifact_root),
        ),
    )

    with TestClient(app) as client:
        started_at = time.monotonic()
        response = client.post(
            "/infer/video/jobs", json=_build_async_job_request("/tmp/video.mp4")
        )
        elapsed = time.monotonic() - started_at

        assert response.status_code == 202
        payload = response.json()
        assert payload["status"] == "queued"
        assert payload["jobId"].startswith("infer-video-job-")
        assert elapsed < 0.5

        job_response = client.get(f"/infer/video/jobs/{payload['jobId']}")
        assert job_response.status_code == 200
        assert job_response.json()["status"] in {"queued", "running"}

        assert started_event.wait(timeout=1.0) is True
        release_event.set()

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            final_response = client.get(f"/infer/video/jobs/{payload['jobId']}")
            final_payload = final_response.json()
            if (
                final_payload["status"] == "succeeded"
                and final_payload["callbackDelivery"]["status"] == "delivered"
            ):
                break
            time.sleep(0.02)
        else:
            raise AssertionError("async inference job did not finish in time")

        assert final_payload["result"]["assetId"] == "user_bundle"
        assert callback_requests[0]["json"]["status"] == "succeeded"
        assert callback_requests[0]["json"]["taskId"] == "tech-task-unit-test"
        assert (
            callback_requests[0]["headers"]["x-technique-service-callback-token"]
            == "callback-token"
        )
        job_record = json.loads(
            (artifact_root / "jobs" / payload["jobId"] / "job.json").read_text(
                encoding="utf-8"
            )
        )
        assert job_record["request"] == {
            "callback": {
                "taskId": "tech-task-unit-test",
                "traceId": "tech-trace-unit-test",
                "clientRunId": "tech-client-run-unit-test",
            }
        }


def test_infer_video_jobs_retry_callback_and_recover_after_restart(
    tmp_path: Path, monkeypatch: Any
) -> None:
    artifact_root = tmp_path / "artifacts"
    callback_attempts: list[str] = []

    def _fake_run_video_inference_sync(
        _state: Any,
        _payload: Any,
        _trace_context: Any,
    ) -> dict[str, Any]:
        return _build_async_job_result("user_bundle")

    def _failing_httpx_post(
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
        follow_redirects: bool,
        timeout: float,
    ) -> httpx.Response:
        callback_attempts.append(json["status"])
        assert headers["x-technique-service-callback-token"] == "callback-token"
        return httpx.Response(500, request=httpx.Request("POST", url))

    monkeypatch.setattr(
        "api.main._run_video_inference_sync", _fake_run_video_inference_sync
    )
    monkeypatch.setattr("api.main.httpx.post", _failing_httpx_post)
    monkeypatch.setattr("api.main.JOB_CALLBACK_LOOP_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr("api.main.JOB_CALLBACK_BASE_BACKOFF_SECONDS", 0.01)
    monkeypatch.setattr("api.main.JOB_CALLBACK_MAX_BACKOFF_SECONDS", 0.05)

    app = create_app(
        estimator=_DummyEstimator(),
        settings=ApiSettings(
            checkpoint_path="/tmp/model.ckpt",
            mhr_path="/tmp/mhr_model.pt",
            device="cpu",
            fov_name="moge2",
            fov_path="",
            artifact_root=str(artifact_root),
        ),
    )

    job_id = ""
    with TestClient(app) as client:
        create_response = client.post(
            "/infer/video/jobs",
            json=_build_async_job_request("/tmp/video.mp4"),
        )
        assert create_response.status_code == 202
        job_id = create_response.json()["jobId"]

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            job_response = client.get(f"/infer/video/jobs/{job_id}")
            payload = job_response.json()
            if (
                payload["status"] == "succeeded"
                and payload["callbackDelivery"]["attempts"] >= 1
                and payload["callbackDelivery"]["status"] == "pending"
            ):
                break
            time.sleep(0.02)
        else:
            raise AssertionError(
                "callback retry state was not observed before shutdown"
            )

    job_record = json.loads(
        (artifact_root / "jobs" / job_id / "job.json").read_text(encoding="utf-8")
    )
    assert job_record["request"] == {
        "callback": {
            "url": "https://callback.example/internal/service-callback",
            "token": "callback-token",
            "taskId": "tech-task-unit-test",
            "traceId": "tech-trace-unit-test",
            "clientRunId": "tech-client-run-unit-test",
        }
    }
    assert callback_attempts

    delivered_attempts: list[str] = []

    def _successful_httpx_post(
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
        follow_redirects: bool,
        timeout: float,
    ) -> httpx.Response:
        delivered_attempts.append(json["status"])
        assert headers["x-technique-service-callback-token"] == "callback-token"
        return httpx.Response(202, request=httpx.Request("POST", url))

    monkeypatch.setattr("api.main.httpx.post", _successful_httpx_post)

    recovered_app = create_app(
        estimator=_DummyEstimator(),
        settings=ApiSettings(
            checkpoint_path="/tmp/model.ckpt",
            mhr_path="/tmp/mhr_model.pt",
            device="cpu",
            fov_name="moge2",
            fov_path="",
            artifact_root=str(artifact_root),
        ),
    )

    with TestClient(recovered_app) as client:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            job_response = client.get(f"/infer/video/jobs/{job_id}")
            payload = job_response.json()
            if payload["callbackDelivery"]["status"] == "delivered":
                break
            time.sleep(0.02)
        else:
            raise AssertionError("recovered callback delivery did not complete")

        assert payload["status"] == "succeeded"
        assert payload["callbackDelivery"]["attempts"] >= 2
        assert payload["callbackDelivery"]["deliveredAt"] is not None
        assert delivered_attempts == ["succeeded"]

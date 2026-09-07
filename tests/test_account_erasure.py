from dataclasses import replace
import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.account_erasure import AccountErasure
from api.config import load_api_settings
from api.main import create_app, AsyncInferenceJobManager, ServiceState, TechniqueTraceContext
from api.models import VideoInferenceJobRequest
from fastapi import HTTPException


def job(root: Path, user: str, name: str, prefix: str | None = None) -> Path:
    path = root / "jobs" / name / "job.json"
    path.parent.mkdir(parents=True)
    prefix = prefix or f"technique-analysis/{name}"
    path.write_text(json.dumps({"traceContext": {"userId": user},
                                "request": {"storage": {"prefix": prefix}}}))
    output = root / prefix
    output.mkdir(parents=True, exist_ok=True)
    (output / "pose.npz").write_bytes(b"synthetic")
    return path


def test_erasure_keeps_other_accounts_and_shared_assets(tmp_path):
    owned = job(tmp_path, "a", "a-job")
    other = job(tmp_path, "b", "b-job")
    shared = tmp_path / "reference-assets" / "keep.npz"
    shared.parent.mkdir()
    shared.write_bytes(b"shared")
    registry = AccountErasure(str(tmp_path))
    assert registry.erase("a")
    assert not owned.exists()
    assert not (tmp_path / "technique-analysis/a-job").exists()
    assert other.exists() and shared.exists()
    assert registry.erase("a")
    with pytest.raises(ValueError, match="account_deleted"):
        with AccountErasure(str(tmp_path)).processing("a"):
            pass


def test_waits_for_running_work_in_another_registry_instance(tmp_path):
    owned = job(tmp_path, "a", "a-job")
    running = AccountErasure(str(tmp_path))
    eraser = AccountErasure(str(tmp_path))
    with running.processing("a"):
        assert not eraser.erase("a")
        assert owned.exists()
    assert eraser.erase("a")
    assert not owned.exists()


def test_unknown_or_shared_output_ownership_stays_pending(tmp_path):
    owned = job(tmp_path, "a", "a-job", "reference-assets/common")
    registry = AccountErasure(str(tmp_path))
    assert not registry.erase("a")
    assert owned.exists()
    assert (tmp_path / "reference-assets/common/pose.npz").exists()


def test_endpoint_requires_dedicated_secret_and_never_serves_receipts(tmp_path):
    settings = replace(load_api_settings(), artifact_root=str(tmp_path),
                       account_deletion_token="synthetic-erasure-test-secret")
    app = create_app(settings=settings)
    # No lifespan/model worker is needed for deterministic filesystem erasure.
    client = TestClient(app)
    assert client.post("/internal/account-erasure", json={"userId": "a"}).status_code == 401
    assert not app.state.service_state.account_erasure.is_deleted("a")
    response = client.post("/internal/account-erasure", json={"userId": "a", "acceptedAt": 1},
                           headers={"Authorization": "Bearer synthetic-erasure-test-secret"})
    assert response.status_code == 200
    assert response.json() == {"userId": "a", "status": "complete"}
    receipt = app.state.service_state.account_erasure._receipt("a")
    assert client.get(f"/artifacts/account-erasure/{receipt.name}").status_code == 404


def test_terminal_job_keeps_erasure_namespace_and_cannot_be_recreated(tmp_path):
    settings = replace(load_api_settings(), artifact_root=str(tmp_path))
    state = ServiceState(settings=settings)
    manager = AsyncInferenceJobManager(state)
    payload = VideoInferenceJobRequest.model_validate({
        "videoPath": "/tmp/synthetic-video.mp4",
        "storage": {"prefix": "technique-analysis/task-a"},
        "callback": {"url": "https://callback.test", "token": "synthetic", "taskId": "task-a"},
    })

    async def scenario():
        record = await manager.create_job(payload, TechniqueTraceContext(user_id="a", task_id="task-a"))
        record["status"] = "succeeded"
        record["callbackDelivery"]["status"] = "delivered"
        assert await manager._write_job_record(record["jobId"], record)
        stored = await manager.get_job_record(record["jobId"])
        assert "storage" not in stored["request"]
        assert stored["erasureStorage"] == {"prefix": "technique-analysis/task-a", "managed": True}
        assert state.account_erasure.erase("a")
        assert not await manager._write_job_record(record["jobId"], record)
        with pytest.raises(HTTPException) as error:
            await manager.create_job(payload, TechniqueTraceContext(user_id="a"))
        assert error.value.status_code == 410
        assert list((tmp_path / "jobs").glob("*/job.json")) == []

    asyncio.run(scenario())

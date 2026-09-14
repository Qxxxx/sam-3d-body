from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import threading

import pytest

from api.reference_cache import _cache_key, prepare_reference_asset


@pytest.fixture
def downloads(monkeypatch, tmp_path):
    calls = []

    @contextmanager
    def download(source):
        calls.append(source)
        path = tmp_path / f"download-{len(calls)}.npz"
        path.write_bytes(b"valid reference")
        try:
            yield path
        finally:
            path.unlink(missing_ok=True)

    monkeypatch.setattr("api.reference_cache.resolve_npz_file", download)
    return calls


def validate(path):
    assert path.read_bytes() == b"valid reference"


def prepare(root, source="https://r2.example/bucket/ref.npz?X-Amz-Signature=secret", version="ref:1", validator=validate):
    return prepare_reference_asset(source, cache_root=root, version=version, kind="skeleton", validate=validator)


def test_cache_survives_contexts_and_signature_rotation_without_network(tmp_path, downloads, monkeypatch):
    root = tmp_path / "cache"
    with prepare(root) as first:
        validate(first)
    assert first.exists()
    assert len(downloads) == 1
    def offline(*args):
        raise AssertionError("cache hit must not access network")
    monkeypatch.setattr("api.reference_cache.resolve_npz_file", offline)
    with prepare(root, source="https://r2.example/bucket/ref.npz?X-Amz-Signature=rotated") as second:
        assert first == second
        validate(second)
    assert "secret" not in "".join(p.read_text() for p in root.glob("*.json"))


def test_version_change_downloads_new_entry_and_preserves_previous(tmp_path, downloads):
    with prepare(tmp_path / "cache") as old:
        pass
    with prepare(tmp_path / "cache", version="ref:2") as new:
        assert new != old
        assert old.exists()
    assert len(downloads) == 2


@pytest.mark.parametrize("changed", [
    "https://other.example/bucket/ref.npz",
    "https://r2.example/other/ref.npz",
    "https://r2.example/bucket/ref.npz?versionId=2",
])
def test_cache_key_keeps_resource_identity(changed):
    assert _cache_key(changed, "v1", "render") != _cache_key("https://r2.example/bucket/ref.npz", "v1", "render")


def test_concurrent_requests_download_only_once(tmp_path, downloads):
    barrier = threading.Barrier(4)
    def worker():
        barrier.wait()
        with prepare(tmp_path / "cache") as path:
            validate(path)
            return path
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(lambda _: worker(), range(4)))
    assert len(set(paths)) == 1
    assert len(downloads) == 1


def test_corrupt_cached_file_is_redownloaded(tmp_path, downloads):
    with prepare(tmp_path / "cache") as path:
        pass
    path.write_bytes(b"corrupt")
    with prepare(tmp_path / "cache") as repaired:
        validate(repaired)
    assert len(downloads) == 2


def test_invalid_download_is_never_published(tmp_path, downloads):
    def invalid(path):
        raise ValueError("invalid reference")
    root = tmp_path / "cache"
    with pytest.raises(ValueError, match="invalid reference"):
        with prepare(root, validator=invalid):
            pytest.fail("invalid data must not be used")
    assert not list(root.glob("*.npz"))
    assert not list(root.glob("*.part"))
    with prepare(root) as path:
        validate(path)
    assert len(downloads) == 2


def test_failed_download_does_not_poison_next_attempt(tmp_path, downloads, monkeypatch):
    original = __import__("api.reference_cache", fromlist=["resolve_npz_file"]).resolve_npz_file
    @contextmanager
    def failing(source):
        raise TimeoutError("Timed out fetching 3D reference asset")
        yield
    root = tmp_path / "cache"
    monkeypatch.setattr("api.reference_cache.resolve_npz_file", failing)
    with pytest.raises(TimeoutError):
        with prepare(root):
            pass
    assert not list(root.glob("*.npz"))
    monkeypatch.setattr("api.reference_cache.resolve_npz_file", original)
    with prepare(root) as path:
        validate(path)


def test_unversioned_legacy_requests_prefetch_without_persistent_reuse(tmp_path, downloads):
    root = tmp_path / "cache"
    for _ in range(2):
        with prepare(root, version=None) as path:
            validate(path)
        assert not path.exists()
    assert len(downloads) == 2
    assert not root.exists()


def test_reference_failure_prevents_video_inference(monkeypatch, tmp_path):
    from api.main import ServiceState, TechniqueTraceContext, _run_video_inference_sync
    from api.config import ApiSettings
    from api.models import VideoInferenceRequest

    payload = VideoInferenceRequest.model_validate({
        "videoPath": "https://example.com/user.mp4",
        "assetConfig": {"assetId": "user-test", "actionType": "smash"},
        "storage": {"mode": "direct_upload", "uploads": {
            name: {"putUrl": "https://example.com/put", "fetchUrl": "https://example.com/get", "contentType": "application/octet-stream"}
            for name in ("skeleton", "render", "metadata")
        }},
        "viewerComparison": {
            "referenceSkeletonPath": "https://example.com/ref.npz",
            "referenceRenderPath": "https://example.com/ref.render.npz",
            "referenceAssetVersion": "ref:1",
            "uploads": {"viewer-data.json": {"putUrl": "https://example.com/put", "fetchUrl": "https://example.com/get", "contentType": "application/json"}},
        },
    })
    settings = ApiSettings(checkpoint_path="unused", mhr_path="unused", device="cpu", fov_name="", fov_path="", artifact_root=str(tmp_path), reference_cache_root=str(tmp_path / "cache"))
    stages = []
    monkeypatch.setattr("api.main._emit_trace_event", lambda *args, **kwargs: stages.append(kwargs["stage"]))
    def forbidden(*args):
        pytest.fail("must not start inference before both references are ready")
    monkeypatch.setattr("api.main._run_video_inference_sync_impl", forbidden)
    @contextmanager
    def prefetch(source, **kwargs):
        if kwargs["kind"] == "render":
            raise TimeoutError("reference download timeout")
        yield tmp_path / "skeleton.npz"
    monkeypatch.setattr("api.main.prepare_reference_asset", prefetch)
    with pytest.raises(TimeoutError):
        _run_video_inference_sync(ServiceState(settings), payload, TechniqueTraceContext())
    assert stages == ["reference_assets_prepare_started"]


def test_shared_reference_version_contract():
    from api.models import ViewerComparisonConfigModel
    fixture = Path(__file__).resolve().parents[2] / "harness/contracts/fixtures/technique-viewer-comparison.request.sample.json"
    payload = ViewerComparisonConfigModel.model_validate(json.loads(fixture.read_text()))
    assert payload.reference_asset_version == "forehand-reference:1789394400000"
    assert payload.model_dump(by_alias=True)["referenceAssetVersion"] == payload.reference_asset_version


def test_preflight_passes_local_paths_to_inference_without_mutating_request(monkeypatch, tmp_path, downloads):
    from api.main import ServiceState, TechniqueTraceContext, _run_video_inference_sync
    from api.config import ApiSettings
    from api.models import VideoInferenceRequest

    fixture = Path(__file__).resolve().parents[2] / "harness/contracts/fixtures/technique-viewer-comparison.request.sample.json"
    viewer = json.loads(fixture.read_text())
    payload = VideoInferenceRequest.model_validate({
        "videoPath": "https://example.com/user.mp4",
        "viewerComparison": viewer,
        "storage": {"mode": "direct_upload", "uploads": {
            name: {"putUrl": "https://example.com/put", "fetchUrl": "https://example.com/get", "contentType": "application/octet-stream"}
            for name in ("skeleton", "render", "metadata")
        }},
    })
    settings = ApiSettings(checkpoint_path="unused", mhr_path="unused", device="cpu", fov_name="", fov_path="", artifact_root=str(tmp_path), reference_cache_root=str(tmp_path / "cache"))
    stages = []
    monkeypatch.setattr("api.main._emit_trace_event", lambda *args, **kwargs: stages.append(kwargs["stage"]))
    monkeypatch.setattr("api.main.load_skeleton_sequence_npz", validate)
    monkeypatch.setattr("api.main._load_render_asset", validate)
    def infer(state, prepared, trace):
        assert stages[-1] == "reference_assets_ready"
        assert prepared.video_path == payload.video_path
        for value in (prepared.viewer_comparison.reference_skeleton_path, prepared.viewer_comparison.reference_render_path):
            validate(Path(value))
        return {"ok": True}
    monkeypatch.setattr("api.main._run_video_inference_sync_impl", infer)
    assert _run_video_inference_sync(ServiceState(settings), payload, TechniqueTraceContext()) == {"ok": True}
    assert payload.viewer_comparison.reference_skeleton_path == viewer["referenceSkeletonPath"]
    assert len(downloads) == 2
    assert _run_video_inference_sync(ServiceState(settings), payload, TechniqueTraceContext()) == {"ok": True}
    assert len(downloads) == 2

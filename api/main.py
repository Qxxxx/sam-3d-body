from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from copy import deepcopy
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import shutil
from threading import Lock
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import quote
from urllib.request import Request as UrlRequest, urlopen
from uuid import uuid4

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from sam_3d_body import __version__
from sam_3d_body.reference_assets import (
    DEFAULT_CROPPED_VIDEO_FILENAME,
    DEFAULT_METADATA_FILENAME,
    ReferenceVideoEntry,
    build_reference_asset_bundle,
    build_reference_assets_metadata,
    save_reference_assets_metadata,
)

from .config import ApiSettings, load_api_settings
from .models import (
    GeneratedAssetFileModel,
    GeneratedAssetFilesModel,
    HealthResponse,
    VideoInferenceCallbackConfigModel,
    VideoInferenceCameraModel,
    VideoInferenceJobAcceptedResponse,
    VideoInferenceJobRequest,
    VideoInferenceJobStatusResponse,
    VideoInferenceRequest,
    VideoInferenceResponse,
    dump_alias_model,
)

if TYPE_CHECKING:
    from sam_3d_body import SAM3DBodyEstimator


LOGGER = logging.getLogger("sam3d.api")
TRACE_INGEST_USER_AGENT = "curl/8.7.1"
JOB_CALLBACK_USER_AGENT = "sam3d-body-job-runner/1.0"
TECHNIQUE_SERVICE_CALLBACK_TOKEN_HEADER = "x-technique-service-callback-token"
JOB_CALLBACK_LOOP_INTERVAL_SECONDS = 1.0
JOB_CALLBACK_TIMEOUT_SECONDS = 10.0
JOB_CALLBACK_BASE_BACKOFF_SECONDS = 2.0
JOB_CALLBACK_MAX_BACKOFF_SECONDS = 60.0


@dataclass
class ServiceState:
    settings: ApiSettings
    estimator: "SAM3DBodyEstimator | Any | None" = None
    estimator_load_error: str | None = None
    estimator_lock: Lock = field(default_factory=Lock)
    job_manager: "AsyncInferenceJobManager | None" = None


@dataclass(frozen=True)
class TechniqueTraceContext:
    trace_id: str | None = None
    task_id: str | None = None
    user_id: str | None = None
    client_run_id: str | None = None
    match_id: str | None = None
    technique_type: str | None = None
    reference_asset_id: str | None = None


def _build_estimator(settings: ApiSettings) -> "SAM3DBodyEstimator":
    from tools.build_fov_estimator import FOVEstimator
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

    checkpoint_path = Path(settings.checkpoint_path)
    mhr_path = Path(settings.mhr_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not mhr_path.exists():
        raise FileNotFoundError(f"MHR model not found: {mhr_path}")
    if not settings.fov_name.strip():
        raise ValueError("SAM3DBODY_FOV_NAME must be configured")

    model, model_cfg = load_sam_3d_body(
        checkpoint_path=str(checkpoint_path),
        device=settings.device,
        mhr_path=str(mhr_path),
    )
    fov_estimator = FOVEstimator(
        name=settings.fov_name.strip(),
        device=settings.device,
        path=settings.fov_path.strip(),
    )
    return SAM3DBodyEstimator(
        model,
        model_cfg,
        fov_estimator=fov_estimator,
    )


def _map_service_exception(exc: Exception) -> HTTPException:
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, TimeoutError):
        return HTTPException(status_code=504, detail=str(exc))
    if isinstance(exc, ConnectionError):
        return HTTPException(status_code=502, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=f"Internal server error: {exc}")


def _normalize_optional_string(value: Any, *, max_length: int = 512) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return normalized[:max_length]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _resolve_trace_context(
    payload: VideoInferenceRequest,
    request: Request | None,
) -> TechniqueTraceContext:
    metadata = payload.asset_config.metadata or {}
    headers = request.headers if request is not None else {}
    return TechniqueTraceContext(
        trace_id=_normalize_optional_string(headers.get("x-duolian-trace-id"))
        or _normalize_optional_string(metadata.get("traceId"), max_length=128),
        task_id=_normalize_optional_string(headers.get("x-duolian-task-id"))
        or _normalize_optional_string(metadata.get("taskId"), max_length=128),
        user_id=_normalize_optional_string(metadata.get("userId"), max_length=128),
        client_run_id=_normalize_optional_string(headers.get("x-duolian-client-run-id"))
        or _normalize_optional_string(metadata.get("clientRunId"), max_length=128),
        match_id=_normalize_optional_string(metadata.get("matchId"), max_length=128),
        technique_type=_normalize_optional_string(
            payload.asset_config.action_type,
            max_length=64,
        ),
        reference_asset_id=_normalize_optional_string(
            metadata.get("referenceAssetId"),
            max_length=128,
        ),
    )


def _build_job_trace_context(
    payload: VideoInferenceRequest,
    callback: VideoInferenceCallbackConfigModel,
    request: Request | None,
) -> TechniqueTraceContext:
    resolved = _resolve_trace_context(payload, request)
    return TechniqueTraceContext(
        trace_id=resolved.trace_id
        or _normalize_optional_string(callback.trace_id, max_length=128),
        task_id=resolved.task_id
        or _normalize_optional_string(callback.task_id, max_length=128),
        user_id=resolved.user_id,
        client_run_id=resolved.client_run_id
        or _normalize_optional_string(callback.client_run_id, max_length=128),
        match_id=resolved.match_id,
        technique_type=resolved.technique_type,
        reference_asset_id=resolved.reference_asset_id,
    )


def _emit_trace_event(
    settings: ApiSettings,
    context: TechniqueTraceContext,
    *,
    stage: str,
    message: str,
    level: str = "info",
    meta: dict[str, Any] | None = None,
) -> None:
    safe_meta = _json_safe(meta or {})
    log_payload = {
        "event": "technique_trace",
        "service": "sam-3d-body",
        "runtimeEnv": settings.runtime_env,
        "level": level,
        "stage": stage,
        "message": message,
        "traceId": context.trace_id,
        "taskId": context.task_id,
        "userId": context.user_id,
        "clientRunId": context.client_run_id,
        "matchId": context.match_id,
        "techniqueType": context.technique_type,
        "referenceAssetId": context.reference_asset_id,
        "meta": safe_meta,
    }

    logger_method = LOGGER.info
    if level == "warn":
        logger_method = LOGGER.warning
    elif level == "error":
        logger_method = LOGGER.error
    logger_method(json.dumps(log_payload, ensure_ascii=True))

    if (
        not context.trace_id
        or not settings.technique_trace_ingest_url
        or not settings.technique_trace_ingest_token
    ):
        return

    body = json.dumps(
        {
            "traceId": context.trace_id,
            "taskId": context.task_id,
            "userId": context.user_id,
            "service": "sam-3d-body",
            "runtimeEnv": settings.runtime_env,
            "level": level,
            "stage": stage,
            "message": message,
            "matchId": context.match_id,
            "clientRunId": context.client_run_id,
            "techniqueType": context.technique_type,
            "referenceAssetId": context.reference_asset_id,
            "meta": safe_meta,
            "createdAt": int(time.time() * 1000),
        },
        ensure_ascii=True,
    ).encode("utf-8")
    request_obj = UrlRequest(
        settings.technique_trace_ingest_url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-technique-trace-ingest-token": settings.technique_trace_ingest_token,
            "User-Agent": TRACE_INGEST_USER_AGENT,
        },
        method="POST",
    )
    try:
        with urlopen(request_obj, timeout=5) as response:
            response.read()
    except (
        Exception
    ) as exc:  # pragma: no cover - network failure depends on runtime env
        LOGGER.warning(
            json.dumps(
                {
                    "event": "technique_trace_ingest_failed",
                    "traceId": context.trace_id,
                    "taskId": context.task_id,
                    "error": str(exc),
                },
                ensure_ascii=True,
            )
        )


def _ensure_estimator(state: ServiceState) -> "SAM3DBodyEstimator | Any":
    if state.estimator is not None:
        return state.estimator

    with state.estimator_lock:
        if state.estimator is not None:
            return state.estimator
        try:
            state.estimator = _build_estimator(state.settings)
            state.estimator_load_error = None
        except Exception as exc:  # pragma: no cover - depends on runtime environment
            state.estimator_load_error = str(exc)
            raise HTTPException(
                status_code=503,
                detail=f"Model is unavailable: {exc}",
            ) from exc
    return state.estimator


def _resolve_asset_output_dir(
    *,
    settings: ApiSettings,
    asset_id: str,
    storage_output_dir: str | None,
    storage_prefix: str,
) -> Path:
    if storage_output_dir is not None:
        return Path(storage_output_dir).expanduser().resolve()

    artifact_root = Path(settings.artifact_root).expanduser().resolve()
    prefix = storage_prefix.strip().strip("/")
    if prefix:
        return (artifact_root / prefix / asset_id).resolve()
    return (artifact_root / asset_id).resolve()


def _build_local_file_descriptor(
    *,
    settings: ApiSettings,
    path: Path,
) -> GeneratedAssetFileModel:
    resolved_path = path.resolve()
    artifact_root = Path(settings.artifact_root).expanduser().resolve()
    relative_path: str | None = None
    fetch_url: str | None = None
    try:
        relative_path = resolved_path.relative_to(artifact_root).as_posix()
        fetch_url = f"/artifacts/{quote(relative_path, safe='/')}"
    except ValueError:
        relative_path = None
        fetch_url = None

    return GeneratedAssetFileModel(
        path=str(resolved_path),
        relativePath=relative_path,
        fetchUrl=fetch_url,
        sizeBytes=int(resolved_path.stat().st_size),
    )


def _upload_file_to_target(
    *,
    path: Path,
    put_url: str,
    content_type: str | None,
) -> None:
    resolved_path = path.resolve()
    headers: dict[str, str] = {}
    if content_type is not None:
        headers["Content-Type"] = content_type

    try:
        response = httpx.put(
            put_url,
            content=resolved_path.read_bytes(),
            headers=headers,
            follow_redirects=True,
            timeout=120.0,
        )
        response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise TimeoutError(
            f"Timed out uploading generated artifact to {put_url}"
        ) from exc
    except httpx.HTTPError as exc:
        raise ConnectionError(
            f"Failed to upload generated artifact to {put_url}: {exc}"
        ) from exc


def _build_uploaded_file_descriptor(
    *,
    path: Path,
    fetch_url: str,
) -> GeneratedAssetFileModel:
    resolved_path = path.resolve()
    return GeneratedAssetFileModel(
        path=str(resolved_path),
        relativePath=None,
        fetchUrl=fetch_url,
        sizeBytes=int(resolved_path.stat().st_size),
    )


def _cleanup_managed_artifact_output_dir(
    *,
    settings: ApiSettings,
    output_dir: Path,
    storage_output_dir: str | None,
) -> None:
    if storage_output_dir is not None:
        return

    artifact_root = Path(settings.artifact_root).expanduser().resolve()
    resolved_output_dir = output_dir.expanduser().resolve()
    try:
        resolved_output_dir.relative_to(artifact_root)
    except ValueError:
        return

    if not resolved_output_dir.exists():
        return

    try:
        shutil.rmtree(resolved_output_dir)
    except FileNotFoundError:
        return
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.warning(
            "Failed to clean managed artifact output dir %s: %s",
            resolved_output_dir,
            exc,
        )
        return

    current = resolved_output_dir.parent
    while current != artifact_root:
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def _build_asset_entry(payload: VideoInferenceRequest) -> ReferenceVideoEntry:
    selection_bbox_xyxy = None
    selection_point_px = None
    if payload.selection is not None:
        selection_bbox_xyxy = payload.selection.bbox_xyxy
        selection_point_px = payload.selection.point_px

    return ReferenceVideoEntry(
        video_path=payload.video_path,
        action_type=(payload.asset_config.action_type or "unknown"),
        reference_id=payload.asset_config.asset_id,
        asset_role=payload.asset_config.asset_role,
        camera_view=payload.asset_config.camera_view,
        handedness=payload.asset_config.handedness,
        selection_bbox_xyxy=selection_bbox_xyxy,
        selection_point_px=selection_point_px,
        phase_annotations_file=payload.asset_config.phase_annotations_file,
        video_config=payload.video_config.to_domain(),
        metadata=dict(payload.asset_config.metadata),
    )


def _emit_inference_failure_event(
    settings: ApiSettings,
    trace_context: TechniqueTraceContext,
    exc: Exception,
) -> HTTPException:
    mapped_exc = _map_service_exception(exc)
    _emit_trace_event(
        settings,
        trace_context,
        stage="request_failed",
        message="Technique inference request failed",
        level="error",
        meta={
            "statusCode": mapped_exc.status_code,
            "detail": mapped_exc.detail,
            "errorType": exc.__class__.__name__,
        },
    )
    return mapped_exc


def _run_video_inference_sync(
    state: ServiceState,
    payload: VideoInferenceRequest,
    trace_context: TechniqueTraceContext,
) -> dict[str, Any]:
    _emit_trace_event(
        state.settings,
        trace_context,
        stage="request_received",
        message="Technique inference request received",
        meta={
            "assetId": payload.asset_config.asset_id,
            "assetRole": payload.asset_config.asset_role,
            "storagePrefix": payload.storage.prefix,
            "storageMode": payload.storage.mode,
            "videoPathType": (
                "remote"
                if payload.video_path.startswith(("http://", "https://"))
                else "local"
            ),
            "hasSelection": payload.selection is not None,
        },
    )
    entry = _build_asset_entry(payload)
    output_dir = _resolve_asset_output_dir(
        settings=state.settings,
        asset_id=entry.reference_id or "asset",
        storage_output_dir=payload.storage.output_dir,
        storage_prefix=payload.storage.prefix,
    )
    _emit_trace_event(
        state.settings,
        trace_context,
        stage="artifact_output_dir_resolved",
        message="Technique artifact output directory resolved",
        meta={
            "assetId": entry.reference_id or "asset",
            "outputDir": str(output_dir),
            "storagePrefix": payload.storage.prefix,
        },
    )
    _emit_trace_event(
        state.settings,
        trace_context,
        stage="inference_started",
        message="Technique inference started",
        meta={
            "assetId": entry.reference_id or "asset",
            "outputDir": str(output_dir),
        },
    )
    bundle = build_reference_asset_bundle(
        entry,
        estimator=_ensure_estimator(state),
        output_dir=output_dir,
        cropped_video_output_path=(
            output_dir / DEFAULT_CROPPED_VIDEO_FILENAME
            if payload.storage.uploads is not None
            and payload.storage.uploads.cropped_video is not None
            else None
        ),
        render_asset_float_dtype=payload.asset_config.render_float_dtype,
        render_include_masks=payload.asset_config.render_include_masks,
        overwrite=True,
    )
    manifest = build_reference_assets_metadata(
        [bundle.asset_metadata],
        skeleton_version=payload.asset_config.skeleton_version,
        fov_estimator_name=state.settings.fov_name.strip() or None,
        fov_estimator_path=state.settings.fov_path.strip() or None,
        render_asset_float_dtype=payload.asset_config.render_float_dtype,
        render_include_masks=payload.asset_config.render_include_masks,
    )
    metadata_path = output_dir / DEFAULT_METADATA_FILENAME
    save_reference_assets_metadata(manifest, metadata_path)
    _emit_trace_event(
        state.settings,
        trace_context,
        stage="artifact_written",
        message="Technique inference artifacts written",
        meta={
            "assetId": entry.reference_id or "asset",
            "outputDir": str(output_dir),
            "skeletonPath": str(bundle.skeleton_path),
            "renderPath": str(bundle.render_path),
            "metadataPath": str(metadata_path),
            "croppedVideoPath": (
                str(bundle.cropped_video_path)
                if bundle.cropped_video_path is not None
                else None
            ),
        },
    )

    if payload.storage.mode == "direct_upload":
        if payload.storage.uploads is None:
            raise ValueError(
                "storage.uploads is required when storage.mode is direct_upload."
            )

        _emit_trace_event(
            state.settings,
            trace_context,
            stage="direct_upload_started",
            message="Technique artifact direct upload started",
            meta={
                "assetId": entry.reference_id or "asset",
                "skeletonFetchUrl": payload.storage.uploads.skeleton.fetch_url,
                "renderFetchUrl": payload.storage.uploads.render.fetch_url,
                "metadataFetchUrl": payload.storage.uploads.metadata.fetch_url,
                "croppedVideoFetchUrl": (
                    payload.storage.uploads.cropped_video.fetch_url
                    if payload.storage.uploads.cropped_video is not None
                    else None
                ),
            },
        )
        _upload_file_to_target(
            path=bundle.skeleton_path,
            put_url=payload.storage.uploads.skeleton.put_url,
            content_type=payload.storage.uploads.skeleton.content_type,
        )
        _upload_file_to_target(
            path=bundle.render_path,
            put_url=payload.storage.uploads.render.put_url,
            content_type=payload.storage.uploads.render.content_type,
        )
        _upload_file_to_target(
            path=metadata_path,
            put_url=payload.storage.uploads.metadata.put_url,
            content_type=payload.storage.uploads.metadata.content_type,
        )
        if (
            bundle.cropped_video_path is not None
            and payload.storage.uploads.cropped_video is not None
        ):
            _upload_file_to_target(
                path=bundle.cropped_video_path,
                put_url=payload.storage.uploads.cropped_video.put_url,
                content_type=payload.storage.uploads.cropped_video.content_type,
            )
        _emit_trace_event(
            state.settings,
            trace_context,
            stage="direct_upload_completed",
            message="Technique artifact direct upload completed",
            meta={
                "assetId": entry.reference_id or "asset",
                "skeletonFetchUrl": payload.storage.uploads.skeleton.fetch_url,
                "renderFetchUrl": payload.storage.uploads.render.fetch_url,
                "metadataFetchUrl": payload.storage.uploads.metadata.fetch_url,
                "croppedVideoFetchUrl": (
                    payload.storage.uploads.cropped_video.fetch_url
                    if payload.storage.uploads.cropped_video is not None
                    else None
                ),
            },
        )

        generated_files = GeneratedAssetFilesModel(
            skeleton=_build_uploaded_file_descriptor(
                path=bundle.skeleton_path,
                fetch_url=payload.storage.uploads.skeleton.fetch_url,
            ),
            render=_build_uploaded_file_descriptor(
                path=bundle.render_path,
                fetch_url=payload.storage.uploads.render.fetch_url,
            ),
            metadata=_build_uploaded_file_descriptor(
                path=metadata_path,
                fetch_url=payload.storage.uploads.metadata.fetch_url,
            ),
            croppedVideo=(
                _build_uploaded_file_descriptor(
                    path=bundle.cropped_video_path,
                    fetch_url=payload.storage.uploads.cropped_video.fetch_url,
                )
                if bundle.cropped_video_path is not None
                and payload.storage.uploads.cropped_video is not None
                else None
            ),
        )
        _cleanup_managed_artifact_output_dir(
            settings=state.settings,
            output_dir=output_dir,
            storage_output_dir=payload.storage.output_dir,
        )
    else:
        generated_files = GeneratedAssetFilesModel(
            skeleton=_build_local_file_descriptor(
                settings=state.settings,
                path=bundle.skeleton_path,
            ),
            render=_build_local_file_descriptor(
                settings=state.settings,
                path=bundle.render_path,
            ),
            metadata=_build_local_file_descriptor(
                settings=state.settings,
                path=metadata_path,
            ),
            croppedVideo=(
                _build_local_file_descriptor(
                    settings=state.settings,
                    path=bundle.cropped_video_path,
                )
                if bundle.cropped_video_path is not None
                else None
            ),
        )

    timestamps = bundle.sequence.timestamps.astype(np.float32).tolist()
    first_timestamp = float(timestamps[0])
    last_timestamp = float(timestamps[-1])
    response = VideoInferenceResponse(
        assetId=entry.reference_id or "asset",
        summary={
            "numFrames": bundle.sequence.num_frames,
            "numJoints": bundle.sequence.num_joints,
            "firstTimestamp": first_timestamp,
            "lastTimestamp": last_timestamp,
            "durationSec": (
                float(last_timestamp - first_timestamp)
                if bundle.sequence.num_frames > 1
                else 0.0
            ),
            "sourceFps": float(bundle.asset_metadata["sourceFps"]),
            "imageSizeHw": list(bundle.asset_metadata["imageSizeHw"]),
            "frameIndices": list(bundle.asset_metadata["frameIndices"]),
        },
        camera=(
            VideoInferenceCameraModel.model_validate(
                {
                    "source": bundle.render_asset["cameraSource"],
                    "horizontalFovDeg": bundle.render_asset["horizontalFovDeg"],
                    "timestamps": bundle.render_asset["timestamps"],
                }
            )
            if bundle.render_asset.get("cameraSource") is not None
            and bundle.render_asset.get("horizontalFovDeg") is not None
            and bundle.render_asset.get("timestamps") is not None
            else None
        ),
        files=generated_files,
        manifest=manifest,
    )
    response_payload = dump_alias_model(response)
    _emit_trace_event(
        state.settings,
        trace_context,
        stage="response_sent",
        message="Technique inference response sent",
        meta={
            "assetId": entry.reference_id or "asset",
            "numFrames": bundle.sequence.num_frames,
            "numJoints": bundle.sequence.num_joints,
            "metadataFetchUrl": response_payload["files"]["metadata"]["fetchUrl"],
        },
    )
    return response_payload


def _jobs_root(settings: ApiSettings) -> Path:
    return Path(settings.artifact_root).expanduser().resolve() / "jobs"


def _job_record_path(settings: ApiSettings, job_id: str) -> Path:
    return _jobs_root(settings) / job_id / "job.json"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp-{uuid4().hex}")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _list_job_record_paths(settings: ApiSettings) -> list[Path]:
    jobs_root = _jobs_root(settings)
    if not jobs_root.exists():
        return []
    return sorted(path for path in jobs_root.glob("*/job.json") if path.is_file())


def _normalize_job_callback_delivery(
    record: dict[str, Any], *, now_ms: int
) -> dict[str, Any]:
    callback_delivery = record.get("callbackDelivery")
    if not isinstance(callback_delivery, dict):
        callback_delivery = {}
    return {
        "status": (
            callback_delivery.get("status")
            if callback_delivery.get("status") in {"pending", "delivering", "delivered"}
            else "pending"
        ),
        "attempts": int(callback_delivery.get("attempts") or 0),
        "lastAttemptAt": (
            callback_delivery.get("lastAttemptAt")
            if isinstance(callback_delivery.get("lastAttemptAt"), int)
            else None
        ),
        "nextAttemptAt": (
            callback_delivery.get("nextAttemptAt")
            if isinstance(callback_delivery.get("nextAttemptAt"), int)
            else now_ms
        ),
        "deliveredAt": (
            callback_delivery.get("deliveredAt")
            if isinstance(callback_delivery.get("deliveredAt"), int)
            else None
        ),
        "lastError": _normalize_optional_string(callback_delivery.get("lastError")),
    }


def _build_job_response_payload(record: dict[str, Any]) -> dict[str, Any]:
    request_payload = (
        record.get("request") if isinstance(record.get("request"), dict) else {}
    )
    callback_payload = (
        request_payload.get("callback")
        if isinstance(request_payload.get("callback"), dict)
        else {}
    )
    payload = {
        "jobId": record["jobId"],
        "taskId": callback_payload.get("taskId") or "",
        "status": record["status"],
        "acceptedAt": record["acceptedAt"],
        "startedAt": record.get("startedAt"),
        "completedAt": record.get("completedAt"),
        "callbackDelivery": {
            "status": record["callbackDelivery"]["status"],
            "attempts": record["callbackDelivery"]["attempts"],
            "lastAttemptAt": record["callbackDelivery"]["lastAttemptAt"],
            "nextAttemptAt": record["callbackDelivery"]["nextAttemptAt"],
            "deliveredAt": record["callbackDelivery"]["deliveredAt"],
            "lastError": record["callbackDelivery"]["lastError"],
        },
    }
    if isinstance(record.get("result"), dict):
        payload["result"] = record["result"]
    if isinstance(record.get("error"), dict):
        payload["error"] = record["error"]
    return payload


def _build_persisted_job_request(record: dict[str, Any]) -> dict[str, Any]:
    request_payload = (
        record.get("request") if isinstance(record.get("request"), dict) else {}
    )
    if not request_payload:
        return {}

    if record.get("status") in {"queued", "running"}:
        return deepcopy(request_payload)

    callback_payload = (
        request_payload.get("callback")
        if isinstance(request_payload.get("callback"), dict)
        else {}
    )
    persisted_callback: dict[str, Any] = {}
    for key in ("taskId", "traceId", "clientRunId"):
        value = callback_payload.get(key)
        if value is not None:
            persisted_callback[key] = value

    callback_delivery = (
        record.get("callbackDelivery")
        if isinstance(record.get("callbackDelivery"), dict)
        else {}
    )
    if callback_delivery.get("status") != "delivered":
        for key in ("url", "token"):
            value = callback_payload.get(key)
            if value is not None:
                persisted_callback[key] = value

    return {"callback": persisted_callback} if persisted_callback else {}


def _job_callback_backoff_ms(attempts: int) -> int:
    if attempts <= 0:
        return int(JOB_CALLBACK_BASE_BACKOFF_SECONDS * 1000)
    seconds = min(
        JOB_CALLBACK_MAX_BACKOFF_SECONDS,
        JOB_CALLBACK_BASE_BACKOFF_SECONDS * (2 ** max(0, attempts - 1)),
    )
    return int(seconds * 1000)


def _extract_trace_context_from_job_record(
    record: dict[str, Any],
) -> TechniqueTraceContext:
    raw_trace_context = (
        record.get("traceContext")
        if isinstance(record.get("traceContext"), dict)
        else {}
    )
    return TechniqueTraceContext(
        trace_id=_normalize_optional_string(
            raw_trace_context.get("traceId"), max_length=128
        ),
        task_id=_normalize_optional_string(
            raw_trace_context.get("taskId"), max_length=128
        ),
        user_id=_normalize_optional_string(
            raw_trace_context.get("userId"), max_length=128
        ),
        client_run_id=_normalize_optional_string(
            raw_trace_context.get("clientRunId"),
            max_length=128,
        ),
        match_id=_normalize_optional_string(
            raw_trace_context.get("matchId"), max_length=128
        ),
        technique_type=_normalize_optional_string(
            raw_trace_context.get("techniqueType"),
            max_length=64,
        ),
        reference_asset_id=_normalize_optional_string(
            raw_trace_context.get("referenceAssetId"),
            max_length=128,
        ),
    )


def _post_callback_request(
    *,
    url: str,
    token: str,
    body: dict[str, Any],
) -> tuple[bool, str | None]:
    try:
        response = httpx.post(
            url,
            json=body,
            headers={
                "Content-Type": "application/json",
                TECHNIQUE_SERVICE_CALLBACK_TOKEN_HEADER: token,
                "User-Agent": JOB_CALLBACK_USER_AGENT,
            },
            follow_redirects=True,
            timeout=JOB_CALLBACK_TIMEOUT_SECONDS,
        )
        if 200 <= response.status_code < 300:
            return True, None

        try:
            response_payload = response.json()
        except Exception:
            response_payload = None

        detail = None
        if isinstance(response_payload, dict):
            detail = _normalize_optional_string(response_payload.get("message"))
        if not detail:
            detail = f"Callback returned HTTP {response.status_code}"
        return False, detail
    except Exception as exc:
        return False, str(exc)


class AsyncInferenceJobManager:
    def __init__(self, state: ServiceState) -> None:
        self._state = state
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._queued_job_ids: set[str] = set()
        self._lock = asyncio.Lock()
        self._worker_tasks: list[asyncio.Task[None]] = []
        self._callback_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        for worker_index in range(max(1, self._state.settings.job_concurrency)):
            self._worker_tasks.append(
                asyncio.create_task(
                    self._worker_loop(worker_index),
                    name=f"sam3d-job-worker-{worker_index}",
                )
            )
        self._callback_task = asyncio.create_task(
            self._callback_loop(),
            name="sam3d-job-callback-loop",
        )
        await self._recover_jobs()

    async def stop(self) -> None:
        tasks = [*self._worker_tasks]
        if self._callback_task is not None:
            tasks.append(self._callback_task)
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError):
                await task
        self._worker_tasks.clear()
        self._callback_task = None

    async def create_job(
        self,
        payload: VideoInferenceJobRequest,
        trace_context: TechniqueTraceContext,
    ) -> dict[str, Any]:
        job_id = f"infer-video-job-{uuid4().hex}"
        accepted_at = int(time.time() * 1000)
        record = {
            "jobId": job_id,
            "status": "queued",
            "acceptedAt": accepted_at,
            "startedAt": None,
            "completedAt": None,
            "request": payload.model_dump(by_alias=True),
            "traceContext": {
                "traceId": trace_context.trace_id,
                "taskId": trace_context.task_id,
                "userId": trace_context.user_id,
                "clientRunId": trace_context.client_run_id,
                "matchId": trace_context.match_id,
                "techniqueType": trace_context.technique_type,
                "referenceAssetId": trace_context.reference_asset_id,
            },
            "callbackDelivery": {
                "status": "pending",
                "attempts": 0,
                "lastAttemptAt": None,
                "nextAttemptAt": accepted_at,
                "deliveredAt": None,
                "lastError": None,
            },
            "result": None,
            "error": None,
        }
        await self._write_job_record(job_id, record)
        _emit_trace_event(
            self._state.settings,
            trace_context,
            stage="job_accepted",
            message="Technique async inference job accepted",
            meta={
                "jobId": job_id,
                "acceptedAt": accepted_at,
            },
        )
        await self.enqueue(job_id)
        return record

    async def get_job_record(self, job_id: str) -> dict[str, Any] | None:
        record = await asyncio.to_thread(
            _read_json,
            _job_record_path(self._state.settings, job_id),
        )
        if not isinstance(record, dict):
            return None
        record["callbackDelivery"] = _normalize_job_callback_delivery(
            record,
            now_ms=int(time.time() * 1000),
        )
        return record

    async def enqueue(self, job_id: str) -> None:
        async with self._lock:
            if job_id in self._queued_job_ids:
                return
            self._queued_job_ids.add(job_id)
        await self._queue.put(job_id)

    async def _write_job_record(self, job_id: str, record: dict[str, Any]) -> None:
        record_to_store = deepcopy(record)
        record_to_store["request"] = _build_persisted_job_request(record_to_store)
        await asyncio.to_thread(
            _write_json_atomic,
            _job_record_path(self._state.settings, job_id),
            record_to_store,
        )

    async def _recover_jobs(self) -> None:
        for job_path in await asyncio.to_thread(
            _list_job_record_paths, self._state.settings
        ):
            record = await asyncio.to_thread(_read_json, job_path)
            if not isinstance(record, dict):
                continue
            now_ms = int(time.time() * 1000)
            record["callbackDelivery"] = _normalize_job_callback_delivery(
                record, now_ms=now_ms
            )
            trace_context = _extract_trace_context_from_job_record(record)
            if record.get("status") in {"queued", "running"}:
                _emit_trace_event(
                    self._state.settings,
                    trace_context,
                    stage="job_restart_recovery",
                    message="Recovered unfinished async inference job after process restart",
                    meta={
                        "jobId": record.get("jobId"),
                        "status": record.get("status"),
                    },
                )
                await self.enqueue(str(record.get("jobId")))
                continue

            if (
                record.get("status") in {"succeeded", "failed"}
                and record["callbackDelivery"]["status"] != "delivered"
            ):
                if record["callbackDelivery"]["nextAttemptAt"] is None:
                    record["callbackDelivery"]["nextAttemptAt"] = now_ms
                    await self._write_job_record(str(record.get("jobId")), record)
                _emit_trace_event(
                    self._state.settings,
                    trace_context,
                    stage="job_restart_recovery",
                    message="Recovered async inference job awaiting callback delivery",
                    meta={
                        "jobId": record.get("jobId"),
                        "status": record.get("status"),
                        "callbackStatus": record["callbackDelivery"]["status"],
                    },
                )

    async def _worker_loop(self, _worker_index: int) -> None:
        while True:
            job_id = await self._queue.get()
            async with self._lock:
                self._queued_job_ids.discard(job_id)
            try:
                await self._run_job(job_id)
            except Exception as exc:  # pragma: no cover - defensive logging
                LOGGER.exception(
                    "Unhandled async inference job error for %s: %s", job_id, exc
                )
            finally:
                self._queue.task_done()

    async def _run_job(self, job_id: str) -> None:
        record = await self.get_job_record(job_id)
        if record is None:
            return
        if record.get("status") not in {"queued", "running"}:
            return

        trace_context = _extract_trace_context_from_job_record(record)
        request_payload = VideoInferenceJobRequest.model_validate(record["request"])
        inference_payload = VideoInferenceRequest.model_validate(
            request_payload.model_dump(by_alias=True, exclude={"callback"})
        )
        now_ms = int(time.time() * 1000)
        record["status"] = "running"
        record["startedAt"] = record.get("startedAt") or now_ms
        record["callbackDelivery"] = _normalize_job_callback_delivery(
            record, now_ms=now_ms
        )
        await self._write_job_record(job_id, record)
        _emit_trace_event(
            self._state.settings,
            trace_context,
            stage="job_execution_started",
            message="Technique async inference job execution started",
            meta={
                "jobId": job_id,
            },
        )

        try:
            result = await asyncio.to_thread(
                _run_video_inference_sync,
                self._state,
                inference_payload,
                trace_context,
            )
            completed_at = int(time.time() * 1000)
            record = await self.get_job_record(job_id)
            if record is None:
                return
            record["status"] = "succeeded"
            record["completedAt"] = completed_at
            record["result"] = result
            record["error"] = None
            callback_delivery = _normalize_job_callback_delivery(
                record, now_ms=completed_at
            )
            callback_delivery["status"] = "pending"
            callback_delivery["nextAttemptAt"] = completed_at
            callback_delivery["lastError"] = None
            record["callbackDelivery"] = callback_delivery
            await self._write_job_record(job_id, record)
            _emit_trace_event(
                self._state.settings,
                trace_context,
                stage="job_execution_succeeded",
                message="Technique async inference job execution succeeded",
                meta={
                    "jobId": job_id,
                },
            )
        except Exception as exc:
            mapped_exc = _emit_inference_failure_event(
                self._state.settings, trace_context, exc
            )
            completed_at = int(time.time() * 1000)
            record = await self.get_job_record(job_id)
            if record is None:
                return
            record["status"] = "failed"
            record["completedAt"] = completed_at
            record["result"] = None
            record["error"] = {
                "message": str(mapped_exc.detail),
                "errorType": exc.__class__.__name__,
                "statusCode": mapped_exc.status_code,
            }
            callback_delivery = _normalize_job_callback_delivery(
                record, now_ms=completed_at
            )
            callback_delivery["status"] = "pending"
            callback_delivery["nextAttemptAt"] = completed_at
            callback_delivery["lastError"] = None
            record["callbackDelivery"] = callback_delivery
            await self._write_job_record(job_id, record)
            _emit_trace_event(
                self._state.settings,
                trace_context,
                stage="job_execution_failed",
                message="Technique async inference job execution failed",
                level="error",
                meta={
                    "jobId": job_id,
                    "statusCode": mapped_exc.status_code,
                    "detail": mapped_exc.detail,
                },
            )

    async def _callback_loop(self) -> None:
        while True:
            try:
                await self._deliver_due_callbacks()
                await asyncio.sleep(JOB_CALLBACK_LOOP_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - defensive logging
                LOGGER.exception("Async inference callback loop error: %s", exc)

    async def _deliver_due_callbacks(self) -> None:
        now_ms = int(time.time() * 1000)
        for job_path in await asyncio.to_thread(
            _list_job_record_paths, self._state.settings
        ):
            record = await asyncio.to_thread(_read_json, job_path)
            if not isinstance(record, dict):
                continue
            if record.get("status") not in {"succeeded", "failed"}:
                continue
            callback_delivery = _normalize_job_callback_delivery(record, now_ms=now_ms)
            if callback_delivery["status"] == "delivered":
                continue
            next_attempt_at = callback_delivery["nextAttemptAt"] or 0
            if next_attempt_at > now_ms:
                continue
            record["callbackDelivery"] = callback_delivery
            await self._deliver_callback(record)

    async def _deliver_callback(self, record: dict[str, Any]) -> None:
        trace_context = _extract_trace_context_from_job_record(record)
        request_payload = (
            record.get("request") if isinstance(record.get("request"), dict) else {}
        )
        callback_payload = (
            request_payload.get("callback")
            if isinstance(request_payload.get("callback"), dict)
            else {}
        )
        callback_config = VideoInferenceCallbackConfigModel.model_validate(
            callback_payload
        )
        callback_delivery = _normalize_job_callback_delivery(
            record,
            now_ms=int(time.time() * 1000),
        )
        attempts = callback_delivery["attempts"] + 1
        last_attempt_at = int(time.time() * 1000)
        callback_delivery["status"] = "delivering"
        callback_delivery["attempts"] = attempts
        callback_delivery["lastAttemptAt"] = last_attempt_at
        callback_delivery["lastError"] = None
        record["callbackDelivery"] = callback_delivery
        await self._write_job_record(str(record["jobId"]), record)

        _emit_trace_event(
            self._state.settings,
            trace_context,
            stage="callback_delivery_started",
            message="Technique async inference callback delivery started",
            meta={
                "jobId": record["jobId"],
                "attempts": attempts,
                "callbackUrl": callback_config.url,
            },
        )

        callback_body: dict[str, Any] = {
            "jobId": record["jobId"],
            "taskId": callback_config.task_id,
            "status": "succeeded" if record["status"] == "succeeded" else "failed",
            "traceId": trace_context.trace_id,
            "clientRunId": trace_context.client_run_id,
        }
        if record["status"] == "succeeded":
            callback_body["result"] = record.get("result")
        else:
            callback_body["error"] = record.get("error")

        ok, error_message = await asyncio.to_thread(
            _post_callback_request,
            url=callback_config.url,
            token=callback_config.token,
            body=callback_body,
        )

        record = await self.get_job_record(str(record["jobId"]))
        if record is None:
            return
        callback_delivery = _normalize_job_callback_delivery(
            record,
            now_ms=int(time.time() * 1000),
        )
        callback_delivery["attempts"] = attempts
        callback_delivery["lastAttemptAt"] = last_attempt_at

        if ok:
            delivered_at = int(time.time() * 1000)
            callback_delivery["status"] = "delivered"
            callback_delivery["deliveredAt"] = delivered_at
            callback_delivery["nextAttemptAt"] = None
            callback_delivery["lastError"] = None
            record["callbackDelivery"] = callback_delivery
            await self._write_job_record(str(record["jobId"]), record)
            _emit_trace_event(
                self._state.settings,
                trace_context,
                stage="callback_delivery_succeeded",
                message="Technique async inference callback delivery succeeded",
                meta={
                    "jobId": record["jobId"],
                    "attempts": attempts,
                },
            )
            return

        callback_delivery["status"] = "pending"
        callback_delivery["lastError"] = (
            error_message or "Unknown callback delivery error"
        )[:500]
        callback_delivery["nextAttemptAt"] = int(
            time.time() * 1000
        ) + _job_callback_backoff_ms(attempts)
        record["callbackDelivery"] = callback_delivery
        await self._write_job_record(str(record["jobId"]), record)
        _emit_trace_event(
            self._state.settings,
            trace_context,
            stage="callback_delivery_failed",
            message="Technique async inference callback delivery failed",
            level="warn",
            meta={
                "jobId": record["jobId"],
                "attempts": attempts,
                "error": callback_delivery["lastError"],
                "nextAttemptAt": callback_delivery["nextAttemptAt"],
            },
        )


def create_app(
    *,
    settings: ApiSettings | None = None,
    estimator: "SAM3DBodyEstimator | Any | None" = None,
) -> FastAPI:
    resolved_settings = settings or load_api_settings()
    service_state = ServiceState(
        settings=resolved_settings,
        estimator=estimator,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        service_state.job_manager = AsyncInferenceJobManager(service_state)
        await service_state.job_manager.start()
        try:
            if (
                service_state.settings.eager_model_load
                and service_state.estimator is None
            ):
                try:
                    _ensure_estimator(service_state)
                except HTTPException:
                    pass
            yield
        finally:
            if service_state.job_manager is not None:
                await service_state.job_manager.stop()
                service_state.job_manager = None

    app = FastAPI(
        title="sam-3d-body service",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.service_state = service_state

    @app.get("/health", response_model=HealthResponse)
    def health() -> dict[str, Any]:
        state: ServiceState = app.state.service_state
        response = HealthResponse(
            status="ok",
            version=__version__,
            modelLoaded=state.estimator is not None,
            modelLoadError=state.estimator_load_error,
        )
        return dump_alias_model(response)

    @app.post(
        "/infer/video/jobs",
        response_model=VideoInferenceJobAcceptedResponse,
        status_code=202,
    )
    async def create_infer_video_job(
        payload: VideoInferenceJobRequest,
        request: Request,
    ) -> dict[str, Any]:
        state: ServiceState = app.state.service_state
        if state.job_manager is None:
            raise HTTPException(
                status_code=503, detail="Async inference job manager is unavailable."
            )
        trace_context = _build_job_trace_context(
            VideoInferenceRequest.model_validate(
                payload.model_dump(by_alias=True, exclude={"callback"})
            ),
            payload.callback,
            request,
        )
        record = await state.job_manager.create_job(payload, trace_context)
        response = VideoInferenceJobAcceptedResponse(
            jobId=str(record["jobId"]),
            status="queued",
            acceptedAt=int(record["acceptedAt"]),
        )
        return dump_alias_model(response)

    @app.get(
        "/infer/video/jobs/{job_id}", response_model=VideoInferenceJobStatusResponse
    )
    async def get_infer_video_job(job_id: str) -> dict[str, Any]:
        state: ServiceState = app.state.service_state
        if state.job_manager is None:
            raise HTTPException(
                status_code=503, detail="Async inference job manager is unavailable."
            )
        record = await state.job_manager.get_job_record(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        response = VideoInferenceJobStatusResponse.model_validate(
            _build_job_response_payload(record)
        )
        return dump_alias_model(response)

    @app.get("/artifacts/{artifact_path:path}")
    def get_artifact(artifact_path: str) -> FileResponse:
        root = (
            Path(app.state.service_state.settings.artifact_root).expanduser().resolve()
        )
        resolved_path = (root / artifact_path).resolve()
        try:
            resolved_path.relative_to(root)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="Artifact not found.") from exc
        if not resolved_path.exists() or not resolved_path.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found.")
        return FileResponse(resolved_path)

    return app


app = create_app()

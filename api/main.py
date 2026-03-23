from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from threading import Lock
import time
from typing import TYPE_CHECKING, Any
from urllib.request import Request as UrlRequest, urlopen
from urllib.parse import quote

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from sam_3d_body import (
    __version__,
)
from sam_3d_body.reference_assets import (
    DEFAULT_METADATA_FILENAME,
    ReferenceVideoEntry,
    build_reference_asset_bundle,
    build_reference_assets_metadata,
    save_reference_assets_metadata,
)

from .config import ApiSettings, load_api_settings
from .models import (
    HealthResponse,
    GeneratedAssetFileModel,
    GeneratedAssetFilesModel,
    VideoInferenceCameraModel,
    VideoInferenceRequest,
    VideoInferenceResponse,
    dump_alias_model,
)

if TYPE_CHECKING:
    from sam_3d_body import SAM3DBodyEstimator


LOGGER = logging.getLogger("sam3d.api")


@dataclass
class ServiceState:
    settings: ApiSettings
    estimator: "SAM3DBodyEstimator | Any | None" = None
    estimator_load_error: str | None = None
    estimator_lock: Lock = field(default_factory=Lock)


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
        },
        method="POST",
    )
    try:
        with urlopen(request_obj, timeout=5) as response:
            response.read()
    except Exception as exc:  # pragma: no cover - network failure depends on runtime env
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
        raise TimeoutError(f"Timed out uploading generated artifact to {put_url}") from exc
    except httpx.HTTPError as exc:
        raise ConnectionError(f"Failed to upload generated artifact to {put_url}: {exc}") from exc


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
        if (
            service_state.settings.eager_model_load
            and service_state.estimator is None
        ):
            try:
                _ensure_estimator(service_state)
            except HTTPException:
                # Keep process up so /health can expose model load errors.
                pass
        yield

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

    @app.post("/infer/video", response_model=VideoInferenceResponse)
    def infer_video(
        payload: VideoInferenceRequest,
        request: Request | None = None,
    ) -> dict[str, Any]:
        state: ServiceState = app.state.service_state
        trace_context = _resolve_trace_context(payload, request)
        try:
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
            _emit_trace_event(
                state.settings,
                trace_context,
                stage="response_sent",
                message="Technique inference response sent",
                meta={
                    "assetId": entry.reference_id or "asset",
                    "numFrames": bundle.sequence.num_frames,
                    "numJoints": bundle.sequence.num_joints,
                    "metadataFetchUrl": dump_alias_model(response)["files"]["metadata"]["fetchUrl"],
                },
            )
            return dump_alias_model(response)
        except Exception as exc:
            mapped_exc = _map_service_exception(exc)
            _emit_trace_event(
                state.settings,
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
            raise mapped_exc from exc

    @app.get("/artifacts/{artifact_path:path}")
    def get_artifact(artifact_path: str) -> FileResponse:
        root = Path(app.state.service_state.settings.artifact_root).expanduser().resolve()
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

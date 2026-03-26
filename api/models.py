from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from sam_3d_body.video_processor import VideoExtractionConfig


def _validate_bbox_xyxy(
    bbox: tuple[float, float, float, float],
    *,
    field_name: str,
) -> None:
    x1, y1, x2, y2 = (float(value) for value in bbox)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{field_name} must satisfy x2 > x1 and y2 > y1.")


def _validate_point_xy(
    point: tuple[float, float],
    *,
    field_name: str,
) -> None:
    x, y = (float(value) for value in point)
    if not np.isfinite(x) or not np.isfinite(y):
        raise ValueError(f"{field_name} must contain finite values.")


class VideoExtractionConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    target_fps: float = Field(default=30.0, alias="targetFps", gt=0.0)
    start_time_sec: float = Field(default=0.0, alias="startTimeSec", ge=0.0)
    end_time_sec: float | None = Field(default=None, alias="endTimeSec", ge=0.0)
    max_frames: int = Field(default=240, alias="maxFrames", ge=1)
    bbox_thr: float = Field(default=0.5, alias="bboxThr", ge=0.0, le=1.0)
    use_mask: bool = Field(default=False, alias="useMask")
    inference_type: str = Field(default="body", alias="inferenceType")
    sample_every_frame: bool = Field(default=False, alias="sampleEveryFrame")

    def to_domain(self) -> VideoExtractionConfig:
        return VideoExtractionConfig(
            target_fps=self.target_fps,
            start_time_sec=self.start_time_sec,
            end_time_sec=self.end_time_sec,
            max_frames=self.max_frames,
            bbox_thr=self.bbox_thr,
            use_mask=self.use_mask,
            inference_type=self.inference_type,
            sample_every_frame=self.sample_every_frame,
        )


class VideoSelectionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    bbox_xyxy: tuple[float, float, float, float] | None = Field(default=None, alias="bbox")
    point_px: tuple[float, float] | None = Field(default=None, alias="pointPx")

    @model_validator(mode="after")
    def validate_exactly_one_selector(self) -> "VideoSelectionModel":
        num_selectors = int(self.bbox_xyxy is not None) + int(self.point_px is not None)
        if num_selectors != 1:
            raise ValueError("Exactly one of bbox or pointPx must be provided.")
        if self.bbox_xyxy is not None:
            _validate_bbox_xyxy(self.bbox_xyxy, field_name="selection.bbox")
        if self.point_px is not None:
            _validate_point_xy(self.point_px, field_name="selection.pointPx")
        return self


class VideoAssetConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    asset_id: str | None = Field(default=None, alias="assetId")
    asset_role: str = Field(default="user", alias="assetRole")
    action_type: str | None = Field(default=None, alias="actionType")
    camera_view: str | None = Field(default=None, alias="cameraView")
    handedness: str | None = None
    phase_annotations_file: str | None = Field(default=None, alias="phaseAnnotationsFile")
    skeleton_version: str = Field(default="sam3db_v1", alias="skeletonVersion")
    render_float_dtype: Literal["float16", "float32"] = Field(
        default="float16",
        alias="renderFloatDtype",
    )
    render_include_masks: bool = Field(default=False, alias="renderIncludeMasks")
    metadata: dict[str, Any] = Field(default_factory=dict)


class VideoAssetUploadTargetModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    put_url: str = Field(alias="putUrl")
    fetch_url: str = Field(alias="fetchUrl")
    content_type: str | None = Field(default=None, alias="contentType")

    @model_validator(mode="after")
    def validate_required_fields(self) -> "VideoAssetUploadTargetModel":
        if not self.put_url.strip():
            raise ValueError("storage.uploads.*.putUrl is required.")
        if not self.fetch_url.strip():
            raise ValueError("storage.uploads.*.fetchUrl is required.")
        if self.content_type is not None and not self.content_type.strip():
            raise ValueError("storage.uploads.*.contentType must be non-empty when provided.")
        return self


class VideoAssetUploadTargetsModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    skeleton: VideoAssetUploadTargetModel
    render: VideoAssetUploadTargetModel
    metadata: VideoAssetUploadTargetModel


class VideoAssetStorageModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    mode: Literal["local", "direct_upload"] = "local"
    output_dir: str | None = Field(default=None, alias="outputDir")
    prefix: str = ""
    uploads: VideoAssetUploadTargetsModel | None = None

    @model_validator(mode="after")
    def validate_storage_mode(self) -> "VideoAssetStorageModel":
        if self.mode == "direct_upload" and self.uploads is None:
            raise ValueError("storage.uploads is required when storage.mode is direct_upload.")
        if self.mode == "local" and self.uploads is not None:
            raise ValueError("storage.uploads is only supported when storage.mode is direct_upload.")
        return self


class VideoInferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    video_path: str = Field(alias="videoPath")
    selection: VideoSelectionModel | None = None
    video_config: VideoExtractionConfigModel = Field(
        default_factory=VideoExtractionConfigModel,
        alias="videoConfig",
    )
    asset_config: VideoAssetConfigModel = Field(
        default_factory=VideoAssetConfigModel,
        alias="assetConfig",
    )
    storage: VideoAssetStorageModel = Field(default_factory=VideoAssetStorageModel)


class VideoInferenceCallbackConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    url: str
    token: str
    task_id: str = Field(alias="taskId")
    trace_id: str | None = Field(default=None, alias="traceId")
    client_run_id: str | None = Field(default=None, alias="clientRunId")

    @model_validator(mode="after")
    def validate_required_fields(self) -> "VideoInferenceCallbackConfigModel":
        if not self.url.strip():
            raise ValueError("callback.url is required.")
        if not self.token.strip():
            raise ValueError("callback.token is required.")
        if not self.task_id.strip():
            raise ValueError("callback.taskId is required.")
        return self


class VideoInferenceJobRequest(VideoInferenceRequest):
    callback: VideoInferenceCallbackConfigModel


class VideoInferenceSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    num_frames: int = Field(alias="numFrames")
    num_joints: int = Field(alias="numJoints")
    first_timestamp: float = Field(alias="firstTimestamp")
    last_timestamp: float = Field(alias="lastTimestamp")
    duration_sec: float = Field(alias="durationSec")
    source_fps: float = Field(alias="sourceFps")
    image_size_hw: list[int] = Field(alias="imageSizeHw")
    frame_indices: list[int] = Field(alias="frameIndices")


class VideoInferenceCameraModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    source: str
    horizontal_fov_deg: list[float] = Field(alias="horizontalFovDeg")
    timestamps: list[float]

    @model_validator(mode="after")
    def validate_lengths(self) -> "VideoInferenceCameraModel":
        if len(self.horizontal_fov_deg) != len(self.timestamps):
            raise ValueError("horizontalFovDeg length must match timestamps length.")
        return self


class GeneratedAssetFileModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    path: str
    relative_path: str | None = Field(default=None, alias="relativePath")
    fetch_url: str | None = Field(default=None, alias="fetchUrl")
    size_bytes: int = Field(alias="sizeBytes")


class GeneratedAssetFilesModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    skeleton: GeneratedAssetFileModel
    render: GeneratedAssetFileModel
    metadata: GeneratedAssetFileModel


class VideoInferenceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    asset_id: str = Field(alias="assetId")
    summary: VideoInferenceSummary
    camera: VideoInferenceCameraModel | None = None
    files: GeneratedAssetFilesModel
    manifest: dict[str, Any]


class VideoInferenceJobAcceptedResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    job_id: str = Field(alias="jobId")
    status: Literal["queued"]
    accepted_at: int = Field(alias="acceptedAt")


class VideoInferenceJobCallbackDeliveryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    status: Literal["pending", "delivering", "delivered"]
    attempts: int
    last_attempt_at: int | None = Field(default=None, alias="lastAttemptAt")
    next_attempt_at: int | None = Field(default=None, alias="nextAttemptAt")
    delivered_at: int | None = Field(default=None, alias="deliveredAt")
    last_error: str | None = Field(default=None, alias="lastError")


class VideoInferenceJobErrorModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    message: str
    error_type: str | None = Field(default=None, alias="errorType")
    status_code: int | None = Field(default=None, alias="statusCode")


class VideoInferenceJobStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    job_id: str = Field(alias="jobId")
    task_id: str = Field(alias="taskId")
    status: Literal["queued", "running", "succeeded", "failed"]
    accepted_at: int = Field(alias="acceptedAt")
    started_at: int | None = Field(default=None, alias="startedAt")
    completed_at: int | None = Field(default=None, alias="completedAt")
    callback_delivery: VideoInferenceJobCallbackDeliveryModel = Field(alias="callbackDelivery")
    result: dict[str, Any] | None = None
    error: VideoInferenceJobErrorModel | None = None


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    status: str
    version: str
    model_loaded: bool = Field(alias="modelLoaded")
    model_load_error: str | None = Field(default=None, alias="modelLoadError")


def dump_alias_model(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(by_alias=True)

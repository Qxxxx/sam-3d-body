from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable
from urllib.parse import urlparse

import cv2
import numpy as np

from .technique_alignment import SkeletonSequence, normalize_skeleton_sequence
from .video_processor import (
    VideoExtractionConfig,
    _horizontal_fov_deg_from_intrinsics,
    _is_remote_video_path,
    _normalize_cam_intrinsics,
    _resolve_video_file,
    _select_person_output,
    _validate_selection_bbox_xyxy,
)

if TYPE_CHECKING:
    from .sam_3d_body_estimator import SAM3DBodyEstimator


DEFAULT_METADATA_FILENAME = "metadata.json"
DEFAULT_CROPPED_VIDEO_FILENAME = "source.mp4"
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
RENDER_ASSET_SCHEMA_VERSION = "technique_reference_render.v1"
REFERENCE_PHASES_SCHEMA_VERSION = "technique_reference_phases.v1"
REFERENCE_PHASE_IDS = (
    "preparatory_phase",
    "backswing_phase",
    "power_generation_phase",
    "followthrough_phase",
)


@dataclass
class ReferenceExtractionResult:
    sequence: SkeletonSequence
    selected_outputs: list[dict[str, Any]]
    frame_indices: np.ndarray
    source_fps: float
    image_size_hw: tuple[int, int]


@dataclass(frozen=True)
class ReferencePhaseAnnotation:
    id: str
    name: str
    description: str
    start_frame: int
    end_frame: int

    def to_metadata(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "startFrame": self.start_frame,
            "endFrame": self.end_frame,
        }


@dataclass(frozen=True)
class ReferenceAssetBundle:
    entry: "ReferenceVideoEntry"
    sequence: SkeletonSequence
    skeleton_path: Path
    render_path: Path
    cropped_video_path: Path | None
    asset_metadata: dict[str, Any]
    render_asset: dict[str, Any]


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    normalized = normalized.strip("_")
    return normalized or "reference"


def _normalize_video_path(video_path: str | Path) -> str | Path:
    raw_path = str(video_path).strip()
    if _is_remote_video_path(raw_path):
        return raw_path
    return Path(raw_path)


def _derive_reference_id(video_path: str | Path) -> str:
    raw_path = str(video_path).strip()
    if _is_remote_video_path(raw_path):
        return _slugify(Path(urlparse(raw_path).path).stem)
    return _slugify(Path(raw_path).stem)


def _base_video_stem(video_path: str | Path) -> str:
    raw_path = str(video_path).strip()
    stem = Path(urlparse(raw_path).path).stem if _is_remote_video_path(raw_path) else Path(raw_path).stem
    lowered = stem.lower()
    if lowered.endswith("_left"):
        return stem[:-5]
    if lowered.endswith("_right"):
        return stem[:-6]
    return stem


def _default_phase_annotations_path(video_path: str | Path) -> Path | None:
    raw_path = str(video_path).strip()
    if _is_remote_video_path(raw_path):
        return None
    path = Path(raw_path)
    base_stem = _base_video_stem(video_path)
    candidates = []
    if path.parent.name == "Videos":
        candidates.append(path.parent.parent / "PhaseAnnotations" / f"{base_stem}.json")
    candidates.extend(
        [
            path.with_name(f"{base_stem}.json"),
            path.with_name(f"{path.stem}.json"),
            path.with_name(f"{path.stem}.phase.json"),
            path.with_name(f"{base_stem}.phase.json"),
        ]
    )
    deduped_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped_candidates.append(candidate)
    for candidate in deduped_candidates:
        if candidate.exists():
            return candidate
    return deduped_candidates[0] if deduped_candidates else None


def _normalize_optional_path(path_value: str | Path | None) -> str | Path | None:
    if path_value is None:
        return None
    raw_path = str(path_value).strip()
    if not raw_path:
        return None
    if _is_remote_video_path(raw_path):
        return raw_path
    return Path(raw_path)


@dataclass(frozen=True)
class ReferenceVideoEntry:
    video_path: str | Path
    action_type: str
    reference_id: str | None = None
    asset_role: str = "reference"
    athlete_name: str | None = None
    camera_view: str | None = None
    handedness: str | None = None
    selection_bbox_xyxy: tuple[float, float, float, float] | None = None
    selection_point_px: tuple[float, float] | None = None
    phase_annotations_file: str | Path | None = None
    video_config: VideoExtractionConfig = field(default_factory=VideoExtractionConfig)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "video_path", _normalize_video_path(self.video_path))
        cleaned_action_type = self.action_type.strip()
        if not cleaned_action_type:
            raise ValueError("action_type must not be empty.")
        object.__setattr__(self, "action_type", cleaned_action_type)

        if self.reference_id is None or not self.reference_id.strip():
            object.__setattr__(
                self,
                "reference_id",
                _derive_reference_id(self.video_path),
            )
        else:
            object.__setattr__(self, "reference_id", _slugify(self.reference_id))

        cleaned_asset_role = self.asset_role.strip().lower()
        if cleaned_asset_role not in {"reference", "user"}:
            raise ValueError("asset_role must be one of: reference, user.")
        object.__setattr__(self, "asset_role", cleaned_asset_role)

        normalized_phase_annotations_file = _normalize_optional_path(
            self.phase_annotations_file
        )
        if normalized_phase_annotations_file is None and cleaned_asset_role == "reference":
            normalized_phase_annotations_file = _default_phase_annotations_path(self.video_path)
        object.__setattr__(
            self,
            "phase_annotations_file",
            normalized_phase_annotations_file,
        )

        if self.selection_bbox_xyxy is not None:
            bbox = _validate_selection_bbox_xyxy(self.selection_bbox_xyxy)
            object.__setattr__(
                self,
                "selection_bbox_xyxy",
                tuple(float(value) for value in bbox.tolist()),
            )

        if self.selection_bbox_xyxy is not None and self.selection_point_px is not None:
            raise ValueError("selection_bbox_xyxy and selection_point_px are mutually exclusive.")

        if cleaned_asset_role == "reference" and not self.video_config.sample_every_frame:
            object.__setattr__(
                self,
                "video_config",
                replace(self.video_config, sample_every_frame=True),
            )


def _parse_video_config(
    raw_config: dict[str, Any] | None,
    default: VideoExtractionConfig | None = None,
) -> VideoExtractionConfig:
    base = default or VideoExtractionConfig()
    if raw_config is None:
        return VideoExtractionConfig(
            target_fps=base.target_fps,
            start_time_sec=base.start_time_sec,
            end_time_sec=base.end_time_sec,
            max_frames=base.max_frames,
            bbox_thr=base.bbox_thr,
            use_mask=base.use_mask,
            inference_type=base.inference_type,
            sample_every_frame=base.sample_every_frame,
        )

    return VideoExtractionConfig(
        target_fps=float(raw_config.get("targetFps", base.target_fps)),
        start_time_sec=float(raw_config.get("startTimeSec", base.start_time_sec)),
        end_time_sec=(
            None
            if raw_config.get("endTimeSec", base.end_time_sec) is None
            else float(raw_config.get("endTimeSec", base.end_time_sec))
        ),
        max_frames=int(raw_config.get("maxFrames", base.max_frames)),
        bbox_thr=float(raw_config.get("bboxThr", base.bbox_thr)),
        use_mask=bool(raw_config.get("useMask", base.use_mask)),
        inference_type=str(raw_config.get("inferenceType", base.inference_type)),
        sample_every_frame=bool(
            raw_config.get("sampleEveryFrame", base.sample_every_frame)
        ),
    )


def discover_reference_videos(
    input_dir: str | Path,
    *,
    action_type: str,
    athlete_name: str | None = None,
    camera_view: str | None = None,
    handedness: str | None = None,
    video_config: VideoExtractionConfig | None = None,
) -> list[ReferenceVideoEntry]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")
    if not root.is_dir():
        raise ValueError(f"Input path must be a directory: {root}")

    entries: list[ReferenceVideoEntry] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            continue
        entries.append(
            ReferenceVideoEntry(
                video_path=path,
                action_type=action_type,
                asset_role="reference",
                athlete_name=athlete_name,
                camera_view=camera_view,
                handedness=handedness,
                video_config=_parse_video_config(None, default=video_config),
            )
        )
    return entries


def load_reference_manifest(
    manifest_path: str | Path,
    *,
    default_action_type: str | None = None,
    default_athlete_name: str | None = None,
    default_camera_view: str | None = None,
    default_handedness: str | None = None,
    default_video_config: VideoExtractionConfig | None = None,
) -> list[ReferenceVideoEntry]:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        raw_items = payload
    elif isinstance(payload, dict) and isinstance(payload.get("assets"), list):
        raw_items = payload["assets"]
    else:
        raise ValueError("Manifest must be a JSON array or an object with an 'assets' array.")

    entries: list[ReferenceVideoEntry] = []
    for idx, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            raise ValueError(f"Manifest item at index {idx} must be an object.")
        video_path_raw = raw.get("videoPath")
        if not isinstance(video_path_raw, str) or not video_path_raw.strip():
            raise ValueError(f"Manifest item at index {idx} is missing valid 'videoPath'.")

        action_type = str(raw.get("actionType") or default_action_type or "").strip()
        if not action_type:
            raise ValueError(
                f"Manifest item at index {idx} is missing 'actionType', and no default_action_type was provided."
            )

        selection_point_px_raw = raw.get("selectionPointPx")
        selection_point_px: tuple[float, float] | None = None
        if selection_point_px_raw is not None:
            if (
                not isinstance(selection_point_px_raw, (list, tuple))
                or len(selection_point_px_raw) != 2
            ):
                raise ValueError(
                    f"Manifest item at index {idx} has invalid 'selectionPointPx'. Expected [x, y]."
                )
            selection_point_px = (
                float(selection_point_px_raw[0]),
                float(selection_point_px_raw[1]),
            )

        selection_bbox_xyxy_raw = raw.get("selectionBbox")
        selection_bbox_xyxy: tuple[float, float, float, float] | None = None
        if selection_bbox_xyxy_raw is not None:
            if (
                not isinstance(selection_bbox_xyxy_raw, (list, tuple))
                or len(selection_bbox_xyxy_raw) != 4
            ):
                raise ValueError(
                    f"Manifest item at index {idx} has invalid 'selectionBbox'. Expected [x1, y1, x2, y2]."
                )
            selection_bbox_xyxy = (
                float(selection_bbox_xyxy_raw[0]),
                float(selection_bbox_xyxy_raw[1]),
                float(selection_bbox_xyxy_raw[2]),
                float(selection_bbox_xyxy_raw[3]),
            )

        entries.append(
            ReferenceVideoEntry(
                video_path=video_path_raw,
                action_type=action_type,
                asset_role=str(raw.get("assetRole") or "reference"),
                reference_id=raw.get("referenceId"),
                athlete_name=raw.get("athleteName", default_athlete_name),
                camera_view=raw.get("cameraView", default_camera_view),
                handedness=raw.get("handedness", default_handedness),
                selection_bbox_xyxy=selection_bbox_xyxy,
                selection_point_px=selection_point_px,
                phase_annotations_file=raw.get("phaseAnnotationsFile"),
                video_config=_parse_video_config(
                    raw.get("videoConfig"),
                    default=default_video_config,
                ),
                metadata=(
                    dict(raw["metadata"])
                    if isinstance(raw.get("metadata"), dict)
                    else {}
                ),
            )
        )

    return entries


def save_skeleton_sequence_npz(
    sequence: SkeletonSequence,
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "keypoints_3d": sequence.keypoints_3d,
        "timestamps": sequence.timestamps,
    }
    if sequence.joint_names is not None:
        payload["joint_names"] = np.asarray(sequence.joint_names, dtype=object)
    np.savez(path, **payload)


def _load_phase_annotations(
    entry: ReferenceVideoEntry,
) -> list[ReferencePhaseAnnotation]:
    if entry.asset_role != "reference":
        return []
    if entry.phase_annotations_file is None:
        raise ValueError(
            f"Reference asset '{entry.reference_id}' requires a phase annotations file."
        )

    phase_file = entry.phase_annotations_file
    if isinstance(phase_file, str) and _is_remote_video_path(phase_file):
        raise ValueError("phase_annotations_file must point to a local JSON file.")

    phase_path = Path(phase_file).expanduser()
    if not phase_path.exists():
        raise FileNotFoundError(
            f"Phase annotations file not found for '{entry.reference_id}': {phase_path}"
        )

    with phase_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("Phase annotations file must contain a JSON object.")

    raw_annotations = payload.get("phaseAnnotations")
    if isinstance(raw_annotations, list):
        schema_version = str(payload.get("schemaVersion") or "").strip()
        if schema_version != REFERENCE_PHASES_SCHEMA_VERSION:
            raise ValueError(
                f"Phase annotations schemaVersion must be {REFERENCE_PHASES_SCHEMA_VERSION}."
            )

        reference_id = _slugify(str(payload.get("referenceId") or ""))
        if reference_id != entry.reference_id:
            raise ValueError(
                f"Phase annotations referenceId '{reference_id}' does not match '{entry.reference_id}'."
            )

        action_type = str(payload.get("actionType") or "").strip()
        if action_type != entry.action_type:
            raise ValueError(
                f"Phase annotations actionType '{action_type}' does not match '{entry.action_type}'."
            )
    else:
        video_id = _slugify(str(payload.get("videoId") or ""))
        expected_video_id = _slugify(_base_video_stem(entry.video_path))
        if not video_id:
            raise ValueError(
                "Phase annotations file must include either phaseAnnotations or iOS-style videoId/phases."
            )
        if video_id != expected_video_id:
            raise ValueError(
                f"Phase annotations videoId '{video_id}' does not match source video '{expected_video_id}'."
            )
        technique_type = str(payload.get("techniqueType") or "").strip()
        if not technique_type:
            raise ValueError("iOS phase annotations file must include a non-empty techniqueType.")
        raw_annotations = payload.get("phases")
        if not isinstance(raw_annotations, list):
            raise ValueError("iOS phase annotations file must include a phases array.")

    if len(raw_annotations) != len(REFERENCE_PHASE_IDS):
        raise ValueError(
            f"Phase annotations must contain exactly {len(REFERENCE_PHASE_IDS)} entries."
        )

    annotations: list[ReferencePhaseAnnotation] = []
    previous_end_frame: int | None = None
    for index, expected_id in enumerate(REFERENCE_PHASE_IDS):
        raw_annotation = raw_annotations[index]
        if not isinstance(raw_annotation, dict):
            raise ValueError(f"phaseAnnotations[{index}] must be an object.")

        phase_id = str(raw_annotation.get("id") or "").strip()
        if phase_id != expected_id:
            raise ValueError(
                f"phaseAnnotations[{index}].id must be '{expected_id}', got '{phase_id}'."
            )

        name = str(raw_annotation.get("name") or "").strip()
        description = str(raw_annotation.get("description") or "").strip()
        if not name:
            raise ValueError(f"phaseAnnotations[{index}].name must not be empty.")
        if not description:
            raise ValueError(f"phaseAnnotations[{index}].description must not be empty.")

        start_frame = raw_annotation.get("startFrame")
        end_frame = raw_annotation.get("endFrame")
        if not isinstance(start_frame, int) or not isinstance(end_frame, int):
            raise ValueError(
                f"phaseAnnotations[{index}] startFrame/endFrame must be integers."
            )
        if start_frame < 0 or end_frame < start_frame:
            raise ValueError(
                f"phaseAnnotations[{index}] must satisfy 0 <= startFrame <= endFrame."
            )
        if previous_end_frame is not None and start_frame < previous_end_frame:
            raise ValueError(
                "phaseAnnotations must be ordered and may only share a boundary frame."
            )

        annotations.append(
            ReferencePhaseAnnotation(
                id=phase_id,
                name=name,
                description=description,
                start_frame=start_frame,
                end_frame=end_frame,
            )
        )
        previous_end_frame = end_frame

    return annotations


def _validate_phase_annotations_against_extraction(
    annotations: list[ReferencePhaseAnnotation],
    extraction: ReferenceExtractionResult,
) -> None:
    if not annotations:
        return
    frame_indices = {int(frame_index) for frame_index in extraction.frame_indices.tolist()}
    if not frame_indices:
        raise ValueError("Reference extraction returned no frame indices.")

    max_frame_index = max(frame_indices)
    for annotation in annotations:
        if annotation.start_frame > max_frame_index or annotation.end_frame > max_frame_index:
            raise ValueError(
                f"Phase annotation '{annotation.id}' extends past extracted frames "
                f"(max extracted frame={max_frame_index})."
            )
        if annotation.start_frame not in frame_indices:
            raise ValueError(
                f"Phase annotation '{annotation.id}' startFrame {annotation.start_frame} "
                "does not exactly match extracted frameIndices."
            )
        if annotation.end_frame not in frame_indices:
            raise ValueError(
                f"Phase annotation '{annotation.id}' endFrame {annotation.end_frame} "
                "does not exactly match extracted frameIndices."
            )


def _extract_reference_result(
    *,
    entry: ReferenceVideoEntry,
    estimator: SAM3DBodyEstimator,
    resolved_video_path: str | Path | None = None,
) -> ReferenceExtractionResult:
    selection_bbox = (
        _validate_selection_bbox_xyxy(entry.selection_bbox_xyxy)
        if entry.selection_bbox_xyxy is not None
        else None
    )
    if resolved_video_path is None:
        with _resolve_video_file(entry.video_path) as video_file:
            return _extract_reference_result(
                entry=entry,
                estimator=estimator,
                resolved_video_path=video_file,
            )

    cap = cv2.VideoCapture(str(resolved_video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {resolved_video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if source_fps <= 0:
        source_fps = max(1.0, entry.video_config.target_fps)
    sample_every_n_frames = (
        1
        if entry.video_config.sample_every_frame
        else max(1, int(round(source_fps / max(entry.video_config.target_fps, 1e-6))))
    )

    start_frame = max(0, int(round(entry.video_config.start_time_sec * source_fps)))
    end_frame = (
        int(round(entry.video_config.end_time_sec * source_fps))
        if entry.video_config.end_time_sec is not None
        else None
    )

    keypoints_sequence: list[np.ndarray] = []
    timestamps: list[float] = []
    frame_indices: list[int] = []
    selected_outputs: list[dict[str, Any]] = []

    frame_index = 0
    previous_bbox: np.ndarray | None = None
    image_size_hw: tuple[int, int] | None = None
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_index < start_frame:
                frame_index += 1
                continue
            if end_frame is not None and frame_index > end_frame:
                break
            if ((frame_index - start_frame) % sample_every_n_frames) != 0:
                frame_index += 1
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if image_size_hw is None:
                image_size_hw = (int(frame_rgb.shape[0]), int(frame_rgb.shape[1]))
            outputs = estimator.process_one_image(
                frame_rgb,
                bbox_thr=entry.video_config.bbox_thr,
                use_mask=entry.video_config.use_mask,
                inference_type=entry.video_config.inference_type,
            )

            selected = _select_person_output(
                outputs,
                previous_bbox,
                selection_bbox,
                entry.selection_point_px,
            )
            if selected is None:
                frame_index += 1
                continue
            if "pred_keypoints_3d" not in selected:
                frame_index += 1
                continue

            keypoints = np.asarray(selected["pred_keypoints_3d"], dtype=np.float32)
            if keypoints.ndim != 2 or keypoints.shape[1] != 3:
                frame_index += 1
                continue

            keypoints_sequence.append(keypoints)
            timestamps.append(frame_index / source_fps)
            frame_indices.append(frame_index)
            selected_outputs.append(selected)

            previous_bbox = np.asarray(selected["bbox"], dtype=np.float32)
            if len(keypoints_sequence) >= entry.video_config.max_frames:
                break
            frame_index += 1
    finally:
        cap.release()

    if not keypoints_sequence:
        raise ValueError("No valid skeleton frames extracted from video")
    if image_size_hw is None:
        raise ValueError("Failed to capture image size from video")

    return ReferenceExtractionResult(
        sequence=SkeletonSequence(
            keypoints_3d=np.stack(keypoints_sequence, axis=0),
            timestamps=np.asarray(timestamps, dtype=np.float32),
            joint_names=None,
        ),
        selected_outputs=selected_outputs,
        frame_indices=np.asarray(frame_indices, dtype=np.int32),
        source_fps=float(source_fps),
        image_size_hw=image_size_hw,
    )


def _rolling_median_1d(values: np.ndarray, window_size: int = 5) -> np.ndarray:
    if values.ndim != 1:
        raise ValueError("values must be a 1D array")
    if values.size <= 1 or window_size <= 1:
        return values.astype(np.float32, copy=True)

    radius = max(0, window_size // 2)
    smoothed = np.empty_like(values, dtype=np.float32)
    for index in range(values.shape[0]):
        start = max(0, index - radius)
        end = min(values.shape[0], index + radius + 1)
        smoothed[index] = float(np.median(values[start:end]))
    return smoothed


def _moving_average_1d(values: np.ndarray, window_size: int = 9) -> np.ndarray:
    if values.ndim != 1:
        raise ValueError("values must be a 1D array")
    if values.size <= 1 or window_size <= 1:
        return values.astype(np.float32, copy=True)

    radius = max(0, window_size // 2)
    kernel_size = radius * 2 + 1
    if kernel_size <= 1:
        return values.astype(np.float32, copy=True)

    padded = np.pad(values.astype(np.float32, copy=False), (radius, radius), mode="edge")
    kernel = np.full(kernel_size, 1.0 / float(kernel_size), dtype=np.float32)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _smooth_track_1d(
    values: np.ndarray,
    *,
    outlier_window: int = 5,
    motion_window: int = 9,
) -> np.ndarray:
    # First reject detector spikes, then smooth the remaining motion so the crop
    # center glides frame-to-frame instead of stepping between medians.
    filtered = _rolling_median_1d(values, outlier_window)
    return _moving_average_1d(filtered, motion_window)


def _force_even_size(size: int, maximum: int) -> int:
    clamped = max(2, min(size, maximum))
    if clamped % 2 == 1:
        if clamped == maximum and clamped > 2:
            clamped -= 1
        else:
            clamped += 1
            clamped = min(clamped, maximum)
            if clamped % 2 == 1 and clamped > 2:
                clamped -= 1
    return max(2, clamped)


def _build_smoothed_crop_track(
    extraction: ReferenceExtractionResult,
    *,
    horizontal_padding_ratio: float = 0.20,
    vertical_padding_ratio: float = 0.30,
    outlier_window: int = 5,
    motion_window: int = 9,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    if not extraction.selected_outputs:
        raise ValueError("No selected outputs available for cropped video generation")

    bbox_xyxy = np.stack(
        [
            np.asarray(output["bbox"], dtype=np.float32).reshape(4)
            for output in extraction.selected_outputs
        ],
        axis=0,
    )
    x1 = bbox_xyxy[:, 0]
    y1 = bbox_xyxy[:, 1]
    x2 = bbox_xyxy[:, 2]
    y2 = bbox_xyxy[:, 3]

    center_x = _smooth_track_1d(
        (x1 + x2) * 0.5,
        outlier_window=outlier_window,
        motion_window=motion_window,
    )
    center_y = _smooth_track_1d(
        (y1 + y2) * 0.5,
        outlier_window=outlier_window,
        motion_window=motion_window,
    )
    widths = _rolling_median_1d(x2 - x1, outlier_window)
    heights = _rolling_median_1d(y2 - y1, outlier_window)

    image_height, image_width = extraction.image_size_hw
    padded_width = int(
        np.ceil(float(np.max(widths)) * (1.0 + horizontal_padding_ratio * 2.0))
    )
    padded_height = int(
        np.ceil(float(np.max(heights)) * (1.0 + vertical_padding_ratio * 2.0))
    )
    if padded_width >= image_width or padded_height >= image_height:
        return center_x, center_y, image_width, image_height

    crop_width = _force_even_size(
        padded_width,
        image_width,
    )
    crop_height = _force_even_size(
        padded_height,
        image_height,
    )
    return center_x, center_y, crop_width, crop_height


def _resolve_crop_bounds(
    *,
    center_x: float,
    center_y: float,
    crop_width: int,
    crop_height: int,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    max_x1 = max(0, frame_width - crop_width)
    max_y1 = max(0, frame_height - crop_height)
    x1 = min(max(int(round(center_x - crop_width * 0.5)), 0), max_x1)
    y1 = min(max(int(round(center_y - crop_height * 0.5)), 0), max_y1)
    x2 = x1 + crop_width
    y2 = y1 + crop_height
    return x1, y1, x2, y2


def _copy_video_stream_without_reencode(
    *,
    video_path: str | Path,
    output_path: str | Path,
) -> bool:
    ffmpeg_command = shutil.which("ffmpeg")
    if ffmpeg_command is None:
        return False

    result = subprocess.run(
        [
            ffmpeg_command,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(video_path),
            "-map",
            "0:v:0",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
    )
    return result.returncode == 0 and Path(output_path).exists()


def _probe_video_stream(video_path: str | Path) -> dict[str, Any] | None:
    ffprobe_command = shutil.which("ffprobe")
    if ffprobe_command is None:
        return None

    result = subprocess.run(
        [
            ffprobe_command,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,color_space,color_transfer,color_primaries,color_range,width,height,avg_frame_rate",
            "-of",
            "json",
            str(video_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        return None

    try:
        payload = json.loads(result.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None

    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        return None
    stream = streams[0]
    return stream if isinstance(stream, dict) else None


def _format_ffmpeg_number(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    formatted = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return formatted or "0"


def _escape_ffmpeg_expression(value: str) -> str:
    return value.replace("\\", "\\\\").replace(",", "\\,")


def _build_piecewise_linear_expression(
    *,
    frame_indices: np.ndarray,
    values: np.ndarray,
) -> str:
    frames = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
    samples = np.asarray(values, dtype=np.float32).reshape(-1)
    if frames.size == 0 or samples.size == 0 or frames.size != samples.size:
        raise ValueError("frame_indices and values must be non-empty arrays with matching lengths")

    expression = _format_ffmpeg_number(float(samples[-1]))
    for index in range(samples.size - 2, -1, -1):
        left_frame = int(frames[index])
        right_frame = int(frames[index + 1])
        left_value = float(samples[index])
        right_value = float(samples[index + 1])
        if right_frame <= left_frame:
            expression = _format_ffmpeg_number(left_value)
            continue

        delta_value = right_value - left_value
        segment_expression = (
            f"({_format_ffmpeg_number(left_value)}+"
            f"((n-{left_frame})*({_format_ffmpeg_number(delta_value)})/{right_frame - left_frame}))"
        )
        expression = f"if(lte(n,{right_frame}),{segment_expression},{expression})"

    first_frame = int(frames[0])
    if first_frame > 0:
        expression = (
            f"if(lt(n,{first_frame}),{_format_ffmpeg_number(float(samples[0]))},{expression})"
        )

    return expression


def _build_crop_origin_expression(
    *,
    frame_indices: np.ndarray,
    center_values: np.ndarray,
    crop_size: int,
    frame_size: int,
) -> str:
    max_origin = max(0, frame_size - crop_size)
    if max_origin == 0:
        return "0"

    center_expression = _build_piecewise_linear_expression(
        frame_indices=frame_indices,
        values=center_values,
    )
    half_crop = crop_size * 0.5
    return (
        f"min(max(floor(({center_expression})-{_format_ffmpeg_number(half_crop)}+0.5),0),{max_origin})"
    )


def _build_cropped_follow_filter(
    *,
    frame_indices: np.ndarray,
    center_x: np.ndarray,
    center_y: np.ndarray,
    crop_width: int,
    crop_height: int,
    frame_width: int,
    frame_height: int,
    start_frame: int,
    end_frame: int | None,
) -> str:
    relative_frame_indices = np.asarray(frame_indices, dtype=np.int32) - int(start_frame)
    if relative_frame_indices.size == 0:
        raise ValueError("frame_indices must not be empty for ffmpeg crop rendering")

    trim_parts: list[str] = []
    if start_frame > 0 or end_frame is not None:
        trim_expression = f"trim=start_frame={start_frame}"
        if end_frame is not None:
            trim_expression += f":end_frame={end_frame + 1}"
        trim_parts.append(trim_expression)
    trim_parts.append("setpts=PTS-STARTPTS")

    x_expression = _build_crop_origin_expression(
        frame_indices=relative_frame_indices,
        center_values=center_x,
        crop_size=crop_width,
        frame_size=frame_width,
    )
    y_expression = _build_crop_origin_expression(
        frame_indices=relative_frame_indices,
        center_values=center_y,
        crop_size=crop_height,
        frame_size=frame_height,
    )

    trim_parts.append(
        "crop="
        f"w={crop_width}:"
        f"h={crop_height}:"
        f"x={_escape_ffmpeg_expression(x_expression)}:"
        f"y={_escape_ffmpeg_expression(y_expression)}:"
        "exact=1"
    )
    return ",".join(trim_parts)


def _build_color_metadata_args(source_stream: dict[str, Any] | None) -> list[str]:
    if source_stream is None:
        return []

    args: list[str] = []
    field_map = (
        ("color_space", "-colorspace"),
        ("color_primaries", "-color_primaries"),
        ("color_transfer", "-color_trc"),
        ("color_range", "-color_range"),
    )
    for field_name, argument_name in field_map:
        raw_value = source_stream.get(field_name)
        if not isinstance(raw_value, str):
            continue
        value = raw_value.strip()
        if not value or value == "unknown":
            continue
        args.extend([argument_name, value])
    return args


def _build_ffmpeg_video_encode_candidates(
    source_stream: dict[str, Any] | None,
) -> list[list[str]]:
    pix_fmt = ""
    if source_stream is not None:
        raw_pix_fmt = source_stream.get("pix_fmt")
        if isinstance(raw_pix_fmt, str):
            pix_fmt = raw_pix_fmt.strip().lower()

    high_bit_depth = any(token in pix_fmt for token in ("10", "12", "14", "16"))

    candidates: list[list[str]] = []
    if high_bit_depth:
        candidates.append(
            [
                "-c:v",
                "hevc_nvenc",
                "-preset",
                "p6",
                "-tune",
                "hq",
                "-rc",
                "vbr",
                "-cq",
                "18",
                "-b:v",
                "0",
                "-pix_fmt",
                "p010le",
                "-profile:v",
                "main10",
                "-tag:v",
                "hvc1",
            ]
        )
    else:
        candidates.append(
            [
                "-c:v",
                "hevc_nvenc",
                "-preset",
                "p6",
                "-tune",
                "hq",
                "-rc",
                "vbr",
                "-cq",
                "18",
                "-b:v",
                "0",
                "-pix_fmt",
                "yuv420p",
                "-profile:v",
                "main",
                "-tag:v",
                "hvc1",
            ]
        )
        candidates.append(
            [
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p6",
                "-tune",
                "hq",
                "-rc",
                "vbr",
                "-cq",
                "18",
                "-b:v",
                "0",
                "-pix_fmt",
                "yuv420p",
            ]
        )

    candidates.append(
        [
            "-c:v",
            "mpeg4",
            "-q:v",
            "2",
            "-pix_fmt",
            "yuv420p",
        ]
    )
    return candidates


def _render_cropped_follow_video_with_ffmpeg(
    *,
    video_path: str | Path,
    output_path: str | Path,
    filter_expression: str,
    source_stream: dict[str, Any] | None,
) -> bool:
    ffmpeg_command = shutil.which("ffmpeg")
    if ffmpeg_command is None:
        return False

    base_args = [
        ffmpeg_command,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        filter_expression,
        "-an",
    ]
    base_args.extend(_build_color_metadata_args(source_stream))

    for encode_args in _build_ffmpeg_video_encode_candidates(source_stream):
        candidate_args = [
            *base_args,
            *encode_args,
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        result = subprocess.run(
            candidate_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=120,
        )
        if result.returncode == 0 and Path(output_path).exists():
            return True

    return False


def save_cropped_follow_video(
    *,
    video_path: str | Path,
    extraction: ReferenceExtractionResult,
    video_config: VideoExtractionConfig,
    output_path: str | Path,
) -> Path:
    center_x, center_y, crop_width, crop_height = _build_smoothed_crop_track(extraction)
    sampled_frame_indices = extraction.frame_indices.astype(np.float32)
    image_height, image_width = extraction.image_size_hw

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if (
        crop_width == image_width
        and crop_height == image_height
        and video_config.start_time_sec <= 0
        and video_config.end_time_sec is None
        and _copy_video_stream_without_reencode(
            video_path=video_path,
            output_path=path,
        )
    ):
        return path

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video for cropped output: {video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if source_fps <= 0:
        source_fps = max(1.0, extraction.source_fps)
    start_frame = max(0, int(round(video_config.start_time_sec * source_fps)))
    end_frame = (
        int(round(video_config.end_time_sec * source_fps))
        if video_config.end_time_sec is not None
        else None
    )
    source_stream = _probe_video_stream(video_path)
    cap.release()

    ffmpeg_filter = _build_cropped_follow_filter(
        frame_indices=extraction.frame_indices,
        center_x=center_x,
        center_y=center_y,
        crop_width=crop_width,
        crop_height=crop_height,
        frame_width=image_width,
        frame_height=image_height,
        start_frame=start_frame,
        end_frame=end_frame,
    )

    if _render_cropped_follow_video_with_ffmpeg(
        video_path=video_path,
        output_path=path,
        filter_expression=ffmpeg_filter,
        source_stream=source_stream,
    ):
        return path

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to reopen video for fallback cropped output: {video_path}")

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (crop_width, crop_height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create cropped video writer: {path}")

    try:
        frame_index = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_index < start_frame:
                frame_index += 1
                continue
            if end_frame is not None and frame_index > end_frame:
                break

            frame_height, frame_width = frame_bgr.shape[:2]
            interpolated_center_x = float(np.interp(frame_index, sampled_frame_indices, center_x))
            interpolated_center_y = float(np.interp(frame_index, sampled_frame_indices, center_y))
            x1, y1, x2, y2 = _resolve_crop_bounds(
                center_x=interpolated_center_x,
                center_y=interpolated_center_y,
                crop_width=crop_width,
                crop_height=crop_height,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            writer.write(frame_bgr[y1:y2, x1:x2])
            frame_index += 1
    finally:
        writer.release()
        cap.release()

    return path


def _stack_optional_field(
    selected_outputs: list[dict[str, Any]],
    key: str,
) -> np.ndarray | None:
    values: list[np.ndarray] = []
    expected_shape: tuple[int, ...] | None = None
    for output in selected_outputs:
        if key not in output or output[key] is None:
            return None
        value = np.asarray(output[key])
        if expected_shape is None:
            expected_shape = value.shape
        elif value.shape != expected_shape:
            raise ValueError(
                f"Inconsistent shape for field '{key}': {value.shape} vs {expected_shape}"
            )
        values.append(value)
    if not values:
        return None
    return np.stack(values, axis=0)


def _cast_float_array(array: np.ndarray, float_dtype: str) -> np.ndarray:
    if not np.issubdtype(array.dtype, np.floating):
        return array
    target_dtype = np.float16 if float_dtype == "float16" else np.float32
    if array.dtype == target_dtype:
        return array
    return array.astype(target_dtype)


def save_render_asset_npz(
    *,
    extraction: ReferenceExtractionResult,
    estimator: SAM3DBodyEstimator,
    output_path: str | Path,
    float_dtype: str = "float16",
    include_masks: bool = False,
) -> dict[str, Any]:
    if float_dtype not in {"float16", "float32"}:
        raise ValueError("float_dtype must be one of: float16, float32")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    image_height, image_width = extraction.image_size_hw
    principal_point_xy = np.asarray(
        [image_width * 0.5, image_height * 0.5], dtype=np.float32
    )

    payload: dict[str, Any] = {
        "schema_version": np.asarray(RENDER_ASSET_SCHEMA_VERSION),
        "keypoints_3d": extraction.sequence.keypoints_3d,
        "keypoints_3d_normalized": normalize_skeleton_sequence(extraction.sequence),
        "timestamps": extraction.sequence.timestamps,
        "frame_indices": extraction.frame_indices,
        "source_fps": np.asarray(extraction.source_fps, dtype=np.float32),
        "image_size_hw": np.asarray(extraction.image_size_hw, dtype=np.int32),
        "principal_point_xy": principal_point_xy,
    }
    if extraction.sequence.joint_names is not None:
        payload["joint_names"] = np.asarray(extraction.sequence.joint_names, dtype=object)

    optional_field_map = {
        "bbox_xyxy": "bbox",
        "keypoints_2d": "pred_keypoints_2d",
        "vertices_3d": "pred_vertices",
        "cam_t": "pred_cam_t",
        "cam_intrinsics": "cam_intrinsics",
        "focal_length": "focal_length",
        "global_rot": "global_rot",
        "body_pose_params": "body_pose_params",
        "hand_pose_params": "hand_pose_params",
        "shape_params": "shape_params",
        "scale_params": "scale_params",
        "pred_joint_coords": "pred_joint_coords",
        "pred_global_rots": "pred_global_rots",
        "mhr_model_params": "mhr_model_params",
    }
    for payload_key, output_key in optional_field_map.items():
        stacked = _stack_optional_field(extraction.selected_outputs, output_key)
        if stacked is None:
            continue
        payload[payload_key] = stacked

    horizontal_fov_deg: np.ndarray | None = None
    camera_source: str | None = None
    if "cam_intrinsics" in payload:
        cam_intrinsics = np.asarray(payload["cam_intrinsics"], dtype=np.float32)
        if cam_intrinsics.ndim == 4 and cam_intrinsics.shape[1] == 1:
            cam_intrinsics = cam_intrinsics[:, 0]
        if cam_intrinsics.ndim == 3 and cam_intrinsics.shape[1:] == (3, 3):
            payload["cam_intrinsics"] = cam_intrinsics
            hfov_values: list[float] = []
            for frame_intrinsics in cam_intrinsics:
                normalized_intrinsics = _normalize_cam_intrinsics(frame_intrinsics)
                hfov = _horizontal_fov_deg_from_intrinsics(
                    normalized_intrinsics,
                    float(image_width),
                )
                if hfov is None:
                    hfov_values = []
                    break
                hfov_values.append(hfov)
            if len(hfov_values) == int(cam_intrinsics.shape[0]):
                horizontal_fov_deg = np.asarray(hfov_values, dtype=np.float32)
                payload["horizontal_fov_deg"] = horizontal_fov_deg
        else:
            payload.pop("cam_intrinsics", None)

    camera_source_values = _stack_optional_field(
        extraction.selected_outputs,
        "camera_source",
    )
    if camera_source_values is not None:
        raw_sources = np.asarray(camera_source_values).reshape(-1).tolist()
        normalized_sources = [str(item).strip() for item in raw_sources if str(item).strip()]
        if normalized_sources:
            camera_source = normalized_sources[0]
            payload["camera_source"] = np.asarray(camera_source)

    if include_masks:
        masks = _stack_optional_field(extraction.selected_outputs, "mask")
        if masks is not None:
            payload["masks"] = masks.astype(np.uint8)

    if hasattr(estimator, "faces") and getattr(estimator, "faces") is not None:
        payload["faces"] = np.asarray(getattr(estimator, "faces"), dtype=np.int32)

    for key, value in list(payload.items()):
        payload[key] = _cast_float_array(np.asarray(value), float_dtype)

    np.savez_compressed(path, **payload)
    return {
        "path": str(path),
        "schemaVersion": RENDER_ASSET_SCHEMA_VERSION,
        "floatDtype": float_dtype,
        "fields": sorted(payload.keys()),
        "cameraSource": camera_source,
        "horizontalFovDeg": (
            horizontal_fov_deg.astype(np.float32).tolist() if horizontal_fov_deg is not None else None
        ),
        "timestamps": (
            extraction.sequence.timestamps.astype(np.float32).tolist()
            if horizontal_fov_deg is not None
            else None
        ),
        "horizontalFovDegCount": (
            int(horizontal_fov_deg.shape[0]) if horizontal_fov_deg is not None else 0
        ),
        "horizontalFovDegRange": (
            [
                float(np.min(horizontal_fov_deg)),
                float(np.max(horizontal_fov_deg)),
            ]
            if horizontal_fov_deg is not None and horizontal_fov_deg.size > 0
            else None
        ),
    }


def _sequence_duration_seconds(sequence: SkeletonSequence) -> float:
    if sequence.num_frames <= 1:
        return 0.0
    return float(sequence.timestamps[-1] - sequence.timestamps[0])


def _validate_unique_reference_ids(entries: Iterable[ReferenceVideoEntry]) -> None:
    seen: set[str] = set()
    for entry in entries:
        if entry.reference_id in seen:
            raise ValueError(f"Duplicate reference_id detected: {entry.reference_id}")
        seen.add(entry.reference_id or "")


def _build_asset_metadata(
    entry: ReferenceVideoEntry,
    sequence: SkeletonSequence,
    output_npz: Path,
    extraction: ReferenceExtractionResult,
    render_asset: dict[str, Any],
    phase_annotations: list[ReferencePhaseAnnotation],
) -> dict[str, Any]:
    return {
        "referenceId": entry.reference_id,
        "actionType": entry.action_type,
        "assetRole": entry.asset_role,
        "athleteName": entry.athlete_name,
        "cameraView": entry.camera_view,
        "handedness": entry.handedness,
        "sourceVideoPath": str(entry.video_path),
        "skeletonPath": str(output_npz),
        "renderAssetPath": render_asset["path"],
        "renderAssetSchemaVersion": render_asset["schemaVersion"],
        "renderAssetFloatDtype": render_asset["floatDtype"],
        "renderAssetFields": render_asset["fields"],
        "cameraSource": render_asset.get("cameraSource"),
        "horizontalFovDegCount": render_asset.get("horizontalFovDegCount"),
        "horizontalFovDegRange": render_asset.get("horizontalFovDegRange"),
        "selectionPointPx": (
            [entry.selection_point_px[0], entry.selection_point_px[1]]
            if entry.selection_point_px is not None
            else None
        ),
        "selectionBbox": (
            [
                entry.selection_bbox_xyxy[0],
                entry.selection_bbox_xyxy[1],
                entry.selection_bbox_xyxy[2],
                entry.selection_bbox_xyxy[3],
            ]
            if entry.selection_bbox_xyxy is not None
            else None
        ),
        "videoConfig": {
            "targetFps": entry.video_config.target_fps,
            "startTimeSec": entry.video_config.start_time_sec,
            "endTimeSec": entry.video_config.end_time_sec,
            "maxFrames": entry.video_config.max_frames,
            "bboxThr": entry.video_config.bbox_thr,
            "useMask": entry.video_config.use_mask,
            "inferenceType": entry.video_config.inference_type,
            "sampleEveryFrame": entry.video_config.sample_every_frame,
        },
        "numFrames": sequence.num_frames,
        "numJoints": sequence.num_joints,
        "durationSec": _sequence_duration_seconds(sequence),
        "sourceFps": extraction.source_fps,
        "imageSizeHw": [extraction.image_size_hw[0], extraction.image_size_hw[1]],
        "frameIndices": extraction.frame_indices.tolist(),
        "jointNames": list(sequence.joint_names) if sequence.joint_names is not None else None,
        "phaseAnnotations": [item.to_metadata() for item in phase_annotations],
        "metadata": entry.metadata,
    }


def build_reference_assets_metadata(
    asset_entries: list[dict[str, Any]],
    *,
    skeleton_version: str = "sam3db_v1",
    fov_estimator_name: str | None = None,
    fov_estimator_path: str | None = None,
    render_asset_float_dtype: str = "float16",
    render_include_masks: bool = False,
) -> dict[str, Any]:
    return {
        "schemaVersion": "technique_reference_assets.v2",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "skeletonVersion": skeleton_version,
        "fovEstimator": {
            "name": fov_estimator_name,
            "path": fov_estimator_path,
        }
        if fov_estimator_name is not None
        else None,
        "jointUnit": "model_space",
        "renderAssetEnabled": True,
        "renderAssetSchemaVersion": RENDER_ASSET_SCHEMA_VERSION,
        "renderAssetFloatDtype": render_asset_float_dtype,
        "renderAssetIncludeMasks": render_include_masks,
        "assetCount": len(asset_entries),
        "assets": asset_entries,
    }


def save_reference_assets_metadata(
    metadata: dict[str, Any],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def build_reference_asset_bundle(
    entry: ReferenceVideoEntry,
    *,
    estimator: SAM3DBodyEstimator,
    output_dir: str | Path,
    cropped_video_output_path: str | Path | None = None,
    render_asset_float_dtype: str = "float16",
    render_include_masks: bool = False,
    overwrite: bool = False,
) -> ReferenceAssetBundle:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if not _is_remote_video_path(str(entry.video_path)) and not Path(entry.video_path).exists():
        raise FileNotFoundError(f"Reference video not found: {entry.video_path}")

    output_npz = output_root / f"{entry.reference_id}.npz"
    if output_npz.exists() and not overwrite:
        raise FileExistsError(
            f"Reference output already exists: {output_npz}. Set overwrite=True to replace."
        )

    render_output_npz = output_root / f"{entry.reference_id}.render.npz"
    if render_output_npz.exists() and not overwrite:
        raise FileExistsError(
            f"Reference render output already exists: {render_output_npz}. Set overwrite=True to replace."
        )

    cropped_video_path = (
        Path(cropped_video_output_path)
        if cropped_video_output_path is not None
        else None
    )
    if cropped_video_path is not None and cropped_video_path.exists() and not overwrite:
        raise FileExistsError(
            f"Cropped video output already exists: {cropped_video_path}. Set overwrite=True to replace."
        )

    with _resolve_video_file(entry.video_path) as video_file:
        extraction = _extract_reference_result(
            entry=entry,
            estimator=estimator,
            resolved_video_path=video_file,
        )
        phase_annotations = _load_phase_annotations(entry)
        _validate_phase_annotations_against_extraction(phase_annotations, extraction)
        sequence = extraction.sequence
        save_skeleton_sequence_npz(sequence, output_npz)
        render_asset = save_render_asset_npz(
            extraction=extraction,
            estimator=estimator,
            output_path=render_output_npz,
            float_dtype=render_asset_float_dtype,
            include_masks=render_include_masks,
        )
        if cropped_video_path is not None:
            save_cropped_follow_video(
                video_path=video_file,
                extraction=extraction,
                video_config=entry.video_config,
                output_path=cropped_video_path,
            )
        asset_metadata = _build_asset_metadata(
            entry,
            sequence,
            output_npz,
            extraction,
            render_asset,
            phase_annotations,
        )

    return ReferenceAssetBundle(
        entry=entry,
        sequence=sequence,
        skeleton_path=output_npz,
        render_path=render_output_npz,
        cropped_video_path=cropped_video_path,
        asset_metadata=asset_metadata,
        render_asset=render_asset,
    )


def build_reference_assets(
    entries: list[ReferenceVideoEntry],
    *,
    estimator: SAM3DBodyEstimator,
    output_dir: str | Path,
    skeleton_version: str = "sam3db_v1",
    fov_estimator_name: str | None = None,
    fov_estimator_path: str | None = None,
    metadata_filename: str = DEFAULT_METADATA_FILENAME,
    render_asset_float_dtype: str = "float16",
    render_include_masks: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not entries:
        raise ValueError("No reference videos provided.")

    _validate_unique_reference_ids(entries)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    metadata_path = output_root / metadata_filename
    if metadata_path.exists() and not overwrite:
        raise FileExistsError(
            f"Metadata file already exists: {metadata_path}. Set overwrite=True to replace."
        )

    bundles = [
        build_reference_asset_bundle(
            entry,
            estimator=estimator,
            output_dir=output_root,
            render_asset_float_dtype=render_asset_float_dtype,
            render_include_masks=render_include_masks,
            overwrite=overwrite,
        )
        for entry in entries
    ]
    metadata = build_reference_assets_metadata(
        [bundle.asset_metadata for bundle in bundles],
        skeleton_version=skeleton_version,
        fov_estimator_name=fov_estimator_name,
        fov_estimator_path=fov_estimator_path,
        render_asset_float_dtype=render_asset_float_dtype,
        render_include_masks=render_include_masks,
    )
    save_reference_assets_metadata(metadata, metadata_path)
    return metadata

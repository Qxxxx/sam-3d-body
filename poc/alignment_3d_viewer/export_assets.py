#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from sam_3d_body.technique_alignment import resolve_npz_file


# SAM-3D-Body's persisted 70-keypoint payload follows metadata/mhr70.py.
# It has no explicit pelvis keypoint, so the canonical root is the hip midpoint.
LEFT_HIP_INDEX = 9
RIGHT_HIP_INDEX = 10
UP_JOINT_INDEX = 69  # neck
EPSILON = 1e-6
# Shared scene size for the skeletal chain, not the posed mesh's Y extent.
TARGET_BODY_SCALE = 2.4
TEMPORAL_FILTER_COEFFICIENTS = (0.2, 0.6, 0.2)
TEMPORAL_FILTER_PASSES = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export two SAM-3D-Body render assets and a DTW alignment report "
            "into browser-friendly mesh buffers."
        )
    )
    parser.add_argument("--user-render-npz", required=True)
    parser.add_argument("--reference-render-npz", required=True)
    parser.add_argument("--alignment-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--user-label", default="IMG_3194")
    parser.add_argument("--reference-label", default="Fan Zhendong")
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def _normalize_vector(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm < EPSILON:
        return np.zeros_like(value)
    return value / norm


def temporal_smooth_frames(
    values: np.ndarray,
    *,
    passes: int = TEMPORAL_FILTER_PASSES,
) -> np.ndarray:
    """Apply a short symmetric low-pass filter without temporal phase lag."""
    output = np.asarray(values, dtype=np.float32).copy()
    if output.ndim < 1:
        raise ValueError("Temporal filter input must have a frame dimension")
    if passes < 0:
        raise ValueError("Temporal filter passes must be non-negative")
    if output.shape[0] < 3 or passes == 0:
        return output

    previous_weight, center_weight, next_weight = TEMPORAL_FILTER_COEFFICIENTS
    for _ in range(passes):
        previous = np.concatenate([output[:1], output[:-1]], axis=0)
        following = np.concatenate([output[1:], output[-1:]], axis=0)
        output = (
            previous_weight * previous
            + center_weight * output
            + next_weight * following
        ).astype(np.float32, copy=False)
    return output


def _body_basis(keypoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    root = (keypoints[LEFT_HIP_INDEX] + keypoints[RIGHT_HIP_INDEX]) * 0.5
    centered = keypoints - root

    x_axis = _normalize_vector(
        centered[RIGHT_HIP_INDEX] - centered[LEFT_HIP_INDEX]
    )
    if float(np.linalg.norm(x_axis)) < EPSILON:
        x_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

    y_hint = centered[UP_JOINT_INDEX]
    y_axis = _normalize_vector(y_hint - np.dot(y_hint, x_axis) * x_axis)
    if float(np.linalg.norm(y_axis)) < EPSILON:
        y_axis = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

    z_axis = _normalize_vector(np.cross(x_axis, y_axis))
    if float(np.linalg.norm(z_axis)) < EPSILON:
        z_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    y_axis = _normalize_vector(np.cross(z_axis, x_axis))

    basis = np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float32)
    return root.astype(np.float32), basis


def sequence_body_scale(keypoints: np.ndarray) -> float:
    """Return one pose-invariant body scale for an entire MHR70 sequence."""
    keypoints = np.asarray(keypoints, dtype=np.float32)
    pelvis = (keypoints[:, LEFT_HIP_INDEX] + keypoints[:, RIGHT_HIP_INDEX]) * 0.5
    torso = np.linalg.norm(keypoints[:, UP_JOINT_INDEX] - pelvis, axis=1)
    head = np.linalg.norm(keypoints[:, 0] - keypoints[:, UP_JOINT_INDEX], axis=1)
    left_leg = np.linalg.norm(keypoints[:, 9] - keypoints[:, 11], axis=1) + np.linalg.norm(
        keypoints[:, 11] - keypoints[:, 13], axis=1
    )
    right_leg = np.linalg.norm(
        keypoints[:, 10] - keypoints[:, 12], axis=1
    ) + np.linalg.norm(keypoints[:, 12] - keypoints[:, 14], axis=1)
    chain_length = torso + head + (left_leg + right_leg) * 0.5
    valid = chain_length[np.isfinite(chain_length) & (chain_length > EPSILON)]
    if valid.size == 0:
        raise ValueError("Unable to derive a valid body scale from MHR70 keypoints")
    return max(float(np.median(valid)), EPSILON)


def canonicalize_mesh_frames(
    vertices: np.ndarray,
    keypoints: np.ndarray,
    source_frame_indices: list[int],
    *,
    body_scale: float | None = None,
) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float32)
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if vertices.ndim != 3 or vertices.shape[2] != 3:
        raise ValueError("vertices_3d must have shape [frames, vertices, 3]")
    if keypoints.ndim != 3 or keypoints.shape[2] != 3:
        raise ValueError("keypoints_3d must have shape [frames, joints, 3]")
    if vertices.shape[0] != keypoints.shape[0]:
        raise ValueError("vertices_3d and keypoints_3d frame counts must match")
    if keypoints.shape[1] <= UP_JOINT_INDEX:
        raise ValueError("keypoints_3d does not contain the required body basis joints")

    stable_scale = sequence_body_scale(keypoints) if body_scale is None else body_scale
    if not np.isfinite(stable_scale) or stable_scale <= EPSILON:
        raise ValueError("body_scale must be finite and positive")

    output = np.empty(
        (len(source_frame_indices), vertices.shape[1], 3), dtype=np.float32
    )
    for output_index, source_index in enumerate(source_frame_indices):
        if source_index < 0 or source_index >= vertices.shape[0]:
            raise IndexError(f"Source frame index out of bounds: {source_index}")
        root, basis = _body_basis(keypoints[source_index])
        output[output_index] = ((vertices[source_index] - root) @ basis) / stable_scale
    return output


def root_center_mesh_frames(
    vertices: np.ndarray,
    keypoints: np.ndarray,
    source_frame_indices: list[int],
    *,
    body_scale: float,
    target_body_scale: float,
) -> np.ndarray:
    """Keep the capture orientation while removing translation and body size."""
    vertices = np.asarray(vertices, dtype=np.float32)
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if vertices.ndim != 3 or vertices.shape[2] != 3:
        raise ValueError("vertices_3d must have shape [frames, vertices, 3]")
    if keypoints.ndim != 3 or keypoints.shape[2] != 3:
        raise ValueError("keypoints_3d must have shape [frames, joints, 3]")
    if not np.isfinite(body_scale) or body_scale <= EPSILON:
        raise ValueError("body_scale must be finite and positive")
    if not np.isfinite(target_body_scale) or target_body_scale <= EPSILON:
        raise ValueError("target_body_scale must be finite and positive")

    output = np.empty(
        (len(source_frame_indices), vertices.shape[1], 3), dtype=np.float32
    )
    for output_index, source_index in enumerate(source_frame_indices):
        if source_index < 0 or source_index >= vertices.shape[0]:
            raise IndexError(f"Source frame index out of bounds: {source_index}")
        root = (
            keypoints[source_index, LEFT_HIP_INDEX]
            + keypoints[source_index, RIGHT_HIP_INDEX]
        ) * 0.5
        output[output_index] = (
            (vertices[source_index] - root) / body_scale * target_body_scale
        )
    return output


def body_basis_frames(
    keypoints: np.ndarray, source_frame_indices: list[int]
) -> np.ndarray:
    """Return row-vector bases that rotate raw frames into canonical view."""
    keypoints = np.asarray(keypoints, dtype=np.float32)
    bases = np.empty((len(source_frame_indices), 3, 3), dtype=np.float32)
    for output_index, source_index in enumerate(source_frame_indices):
        if source_index < 0 or source_index >= keypoints.shape[0]:
            raise IndexError(f"Source frame index out of bounds: {source_index}")
        _, bases[output_index] = _body_basis(keypoints[source_index])
    return bases


def _load_render_asset(path: str | Path) -> dict[str, np.ndarray]:
    with resolve_npz_file(path) as resolved_path, np.load(
        resolved_path, allow_pickle=False
    ) as data:
        required = (
            "vertices_3d",
            "faces",
            "keypoints_3d",
            "timestamps",
            "cam_t",
            "cam_intrinsics",
            "image_size_hw",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Missing render fields in {path}: {missing}")
        asset = {key: np.asarray(data[key]) for key in required}

    frame_count = int(asset["vertices_3d"].shape[0])
    if asset["cam_t"].shape != (frame_count, 3):
        raise ValueError(f"cam_t in {path} must have shape [frames, 3]")
    if asset["cam_intrinsics"].shape != (frame_count, 3, 3):
        raise ValueError(
            f"cam_intrinsics in {path} must have shape [frames, 3, 3]"
        )
    if asset["timestamps"].shape != (frame_count,):
        raise ValueError(f"timestamps in {path} must have shape [frames]")
    if asset["image_size_hw"].shape != (2,):
        raise ValueError(f"image_size_hw in {path} must have shape [2]")
    if np.any(np.asarray(asset["image_size_hw"]) <= 0):
        raise ValueError(f"image_size_hw in {path} must be positive")
    return asset


def _write_float_buffer(path: Path, values: np.ndarray) -> None:
    np.asarray(values, dtype="<f4").tofile(path)


def _write_index_buffer(path: Path, values: np.ndarray) -> None:
    np.asarray(values, dtype="<u4").tofile(path)


def _bounds(*position_sets: np.ndarray) -> dict[str, list[float]]:
    all_positions = np.concatenate(
        [positions.reshape(-1, 3) for positions in position_sets], axis=0
    )
    return {
        "min": np.min(all_positions, axis=0).astype(float).tolist(),
        "max": np.max(all_positions, axis=0).astype(float).tolist(),
    }


def export_assets(
    *,
    user_render_path: Path,
    reference_render_path: Path,
    alignment_path: Path,
    output_dir: Path,
    user_label: str,
    reference_label: str,
    fps: float,
) -> dict[str, Any]:
    if fps <= 0:
        raise ValueError("fps must be positive")

    user = _load_render_asset(user_render_path)
    reference = _load_render_asset(reference_render_path)
    report = json.loads(alignment_path.read_text(encoding="utf-8"))
    steps = report.get("alignmentPath")
    if not isinstance(steps, list) or not steps:
        raise ValueError("Alignment report must contain a non-empty alignmentPath")

    user_faces = np.asarray(user["faces"], dtype=np.uint32)
    reference_faces = np.asarray(reference["faces"], dtype=np.uint32)
    if user_faces.shape != reference_faces.shape or not np.array_equal(
        user_faces, reference_faces
    ):
        raise ValueError("User and reference meshes must use identical topology")

    user_indices = [int(step["userFrameIndex"]) for step in steps]
    reference_indices = [int(step["referenceFrameIndex"]) for step in steps]
    user_vertices = temporal_smooth_frames(user["vertices_3d"])
    reference_vertices = temporal_smooth_frames(reference["vertices_3d"])
    user_keypoints = temporal_smooth_frames(user["keypoints_3d"])
    reference_keypoints = temporal_smooth_frames(reference["keypoints_3d"])
    user_camera_translation = temporal_smooth_frames(user["cam_t"])
    reference_camera_translation = temporal_smooth_frames(reference["cam_t"])
    user_body_scale = sequence_body_scale(user["keypoints_3d"])
    reference_body_scale = sequence_body_scale(reference["keypoints_3d"])
    user_positions = canonicalize_mesh_frames(
        user_vertices,
        user_keypoints,
        user_indices,
        body_scale=user_body_scale,
    )
    reference_positions = canonicalize_mesh_frames(
        reference_vertices,
        reference_keypoints,
        reference_indices,
        body_scale=reference_body_scale,
    )
    # Never fit each person's posed height: crouching or raising a hand must
    # not change their overall size. Both use the same scene-space chain size.
    user_positions *= TARGET_BODY_SCALE
    reference_positions *= TARGET_BODY_SCALE
    user_raw_positions = root_center_mesh_frames(
        user_vertices,
        user_keypoints,
        user_indices,
        body_scale=user_body_scale,
        target_body_scale=TARGET_BODY_SCALE,
    )
    reference_raw_positions = root_center_mesh_frames(
        reference_vertices,
        reference_keypoints,
        reference_indices,
        body_scale=reference_body_scale,
        target_body_scale=TARGET_BODY_SCALE,
    )
    user_bases = body_basis_frames(user_keypoints, user_indices)
    reference_bases = body_basis_frames(
        reference_keypoints, reference_indices
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    user_buffer_name = "user.positions.f32.bin"
    reference_buffer_name = "reference.positions.f32.bin"
    user_raw_buffer_name = "user.raw.positions.f32.bin"
    reference_raw_buffer_name = "reference.raw.positions.f32.bin"
    user_basis_buffer_name = "user.basis.f32.bin"
    reference_basis_buffer_name = "reference.basis.f32.bin"
    user_camera_position_buffer_name = "user.camera.positions.f32.bin"
    reference_camera_position_buffer_name = "reference.camera.positions.f32.bin"
    user_camera_translation_buffer_name = "user.camera.translation.f32.bin"
    reference_camera_translation_buffer_name = (
        "reference.camera.translation.f32.bin"
    )
    user_camera_intrinsics_buffer_name = "user.camera.intrinsics.f32.bin"
    reference_camera_intrinsics_buffer_name = "reference.camera.intrinsics.f32.bin"
    index_buffer_name = "mesh.indices.u32.bin"
    _write_float_buffer(output_dir / user_buffer_name, user_positions)
    _write_float_buffer(output_dir / reference_buffer_name, reference_positions)
    _write_float_buffer(output_dir / user_raw_buffer_name, user_raw_positions)
    _write_float_buffer(output_dir / reference_raw_buffer_name, reference_raw_positions)
    _write_float_buffer(output_dir / user_basis_buffer_name, user_bases)
    _write_float_buffer(output_dir / reference_basis_buffer_name, reference_bases)
    _write_float_buffer(
        output_dir / user_camera_position_buffer_name, user_vertices
    )
    _write_float_buffer(
        output_dir / reference_camera_position_buffer_name,
        reference_vertices,
    )
    _write_float_buffer(
        output_dir / user_camera_translation_buffer_name, user_camera_translation
    )
    _write_float_buffer(
        output_dir / reference_camera_translation_buffer_name,
        reference_camera_translation,
    )
    _write_float_buffer(
        output_dir / user_camera_intrinsics_buffer_name,
        user["cam_intrinsics"],
    )
    _write_float_buffer(
        output_dir / reference_camera_intrinsics_buffer_name,
        reference["cam_intrinsics"],
    )
    _write_index_buffer(output_dir / index_buffer_name, user_faces)

    compact_steps = [
        {
            "userFrameIndex": int(step["userFrameIndex"]),
            "referenceFrameIndex": int(step["referenceFrameIndex"]),
            "userTimestamp": float(step["userTimestamp"]),
            "referenceTimestamp": float(step["referenceTimestamp"]),
            "distance": float(step["distance"]),
        }
        for step in steps
    ]
    metadata: dict[str, Any] = {
        "schemaVersion": "sam3dbody.alignment-viewer.v1",
        "labels": {"user": user_label, "reference": reference_label},
        "mesh": {
            "vertexCount": int(user_positions.shape[1]),
            "triangleCount": int(user_faces.shape[0]),
            "stepCount": len(compact_steps),
            "positionComponents": 3,
            "positionEncoding": "float32-le",
            "indexEncoding": "uint32-le",
        },
        "playback": {
            "fps": float(fps),
            "durationSec": len(compact_steps) / float(fps),
        },
        "alignment": {
            "algorithm": report.get("algorithm"),
            "distance": float(report.get("distance", 0.0)),
            "summary": report.get("summary") or {},
            "steps": compact_steps,
        },
        "normalization": {
            "method": "skeletal-chain-v1",
            "root": "midpoint(leftHip, rightHip)",
            "leftHipIndex": LEFT_HIP_INDEX,
            "rightHipIndex": RIGHT_HIP_INDEX,
            "upJointIndex": UP_JOINT_INDEX,
            "scale": "median(neck-to-pelvis + neck-to-nose + average leg-chain length)",
            "userBodyScale": user_body_scale,
            "referenceBodyScale": reference_body_scale,
            "targetBodyScale": TARGET_BODY_SCALE,
            "userScaleFactor": TARGET_BODY_SCALE / user_body_scale,
            "referenceScaleFactor": TARGET_BODY_SCALE / reference_body_scale,
            "description": "Per-frame root translation and body-basis rotation with one uniform scale per person derived from the sequence median skeletal chain. Both share the same target chain size; posed mesh height and body proportions are preserved.",
        },
        "temporalFilter": {
            "type": "symmetric-three-tap-low-pass",
            "coefficients": list(TEMPORAL_FILTER_COEFFICIENTS),
            "passes": TEMPORAL_FILTER_PASSES,
            "phaseLagFrames": 0,
            "appliedTo": [
                "vertices_3d",
                "keypoints_3d",
                "cam_t",
                "body_basis",
            ],
            "description": "Short offline bidirectional smoothing reduces frame-to-frame prediction jitter without adding playback delay.",
        },
        "cameraOverlay": {
            "projection": "sam3dbody-pinhole-camera-v1",
            "description": "Original mesh vertices projected with each source frame's SAM-3D-Body camera translation and intrinsics. No body-size normalization is applied in video overlay mode.",
            "user": {
                "frameCount": int(user["vertices_3d"].shape[0]),
                "timestamps": np.asarray(user["timestamps"], dtype=float).tolist(),
                "imageSizeHW": np.asarray(
                    user["image_size_hw"], dtype=int
                ).tolist(),
            },
            "reference": {
                "frameCount": int(reference["vertices_3d"].shape[0]),
                "timestamps": np.asarray(
                    reference["timestamps"], dtype=float
                ).tolist(),
                "imageSizeHW": np.asarray(
                    reference["image_size_hw"], dtype=int
                ).tolist(),
            },
        },
        "bounds": _bounds(user_positions, reference_positions),
        "files": {
            "userPositions": user_buffer_name,
            "referencePositions": reference_buffer_name,
            "userRawPositions": user_raw_buffer_name,
            "referenceRawPositions": reference_raw_buffer_name,
            "userBasis": user_basis_buffer_name,
            "referenceBasis": reference_basis_buffer_name,
            "userCameraPositions": user_camera_position_buffer_name,
            "referenceCameraPositions": reference_camera_position_buffer_name,
            "userCameraTranslation": user_camera_translation_buffer_name,
            "referenceCameraTranslation": reference_camera_translation_buffer_name,
            "userCameraIntrinsics": user_camera_intrinsics_buffer_name,
            "referenceCameraIntrinsics": reference_camera_intrinsics_buffer_name,
            "indices": index_buffer_name,
        },
    }
    (output_dir / "viewer-data.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> None:
    args = parse_args()
    metadata = export_assets(
        user_render_path=Path(args.user_render_npz).expanduser().resolve(),
        reference_render_path=Path(args.reference_render_npz).expanduser().resolve(),
        alignment_path=Path(args.alignment_json).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        user_label=args.user_label,
        reference_label=args.reference_label,
        fps=args.fps,
    )
    print(
        json.dumps(
            {
                "outputDir": str(Path(args.output_dir).expanduser().resolve()),
                "vertexCount": metadata["mesh"]["vertexCount"],
                "triangleCount": metadata["mesh"]["triangleCount"],
                "stepCount": metadata["mesh"]["stepCount"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

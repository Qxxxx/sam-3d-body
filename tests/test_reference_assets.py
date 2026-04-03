from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from sam_3d_body.reference_assets import (
    ReferenceVideoEntry,
    _smooth_track_1d,
    build_reference_asset_bundle,
    build_reference_assets,
    discover_reference_videos,
    load_reference_manifest,
)
from sam_3d_body.video_processor import VideoExtractionConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_IMG_1966_RIGHT_VIDEO = (
    REPO_ROOT / "duolian" / "duolian" / "Resources" / "Videos" / "IMG_1966_right.MOV"
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
            color = int((frame_idx * 13) % 255)
            frame = np.full((height, width, 3), color, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def _write_phase_annotations(
    path: Path,
    *,
    video_id: str,
    technique_type: str,
    final_frame: int,
) -> None:
    if final_frame < 3:
        raise ValueError("final_frame must be >= 3")
    phase_lengths = [
        final_frame // 4,
        final_frame // 2,
        (final_frame * 3) // 4,
        final_frame,
    ]
    payload = {
        "videoId": video_id,
        "techniqueType": technique_type,
        "phases": [
            {
                "id": "preparatory_phase",
                "name": "Preparatory Phase",
                "description": "Load posture and prepare to swing.",
                "startFrame": 0,
                "endFrame": phase_lengths[0],
            },
            {
                "id": "backswing_phase",
                "name": "Backswing Phase",
                "description": "Draw the racket back.",
                "startFrame": phase_lengths[0] + 1,
                "endFrame": phase_lengths[1],
            },
            {
                "id": "power_generation_phase",
                "name": "Power Generation Phase",
                "description": "Accelerate into the shuttle.",
                "startFrame": phase_lengths[1] + 1,
                "endFrame": phase_lengths[2],
            },
            {
                "id": "followthrough_phase",
                "name": "Follow-through Phase",
                "description": "Finish the swing and recover.",
                "startFrame": phase_lengths[2] + 1,
                "endFrame": phase_lengths[3],
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class _DummyEstimator:
    def __init__(self) -> None:
        self.call_count = 0
        self.faces = np.array([[0, 1, 2], [2, 3, 4]], dtype=np.int32)

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
        keypoints_2d = keypoints[:, :2] * 4.0 + 16.0
        vertices = np.array(
            [
                [value, 0.0, 0.0],
                [value + 0.2, 0.1, 0.0],
                [value + 0.4, 0.3, 0.0],
                [value + 0.6, 0.4, 0.1],
                [value + 0.8, 0.5, 0.2],
            ],
            dtype=np.float32,
        )
        return [
            {
                "bbox": np.array([8, 8, 80, 72], dtype=np.float32),
                "pred_keypoints_3d": keypoints,
                "pred_keypoints_2d": keypoints_2d,
                "pred_vertices": vertices,
                "pred_cam_t": np.array([0.1, -0.05, 3.2], dtype=np.float32),
                "focal_length": np.float32(1200.0),
                "global_rot": np.array([0.0, 0.1, 0.2], dtype=np.float32),
                "body_pose_params": np.linspace(0.0, 1.0, 12, dtype=np.float32),
                "hand_pose_params": np.linspace(0.0, 1.0, 6, dtype=np.float32),
                "shape_params": np.linspace(-1.0, 1.0, 10, dtype=np.float32),
                "scale_params": np.array([1.0], dtype=np.float32),
                "pred_joint_coords": keypoints.copy(),
                "pred_global_rots": np.tile(np.eye(3, dtype=np.float32), (4, 1, 1)),
                "mhr_model_params": np.linspace(0.0, 1.0, 16, dtype=np.float32),
                "cam_intrinsics": np.array(
                    [
                        [800.0, 0.0, 60.0],
                        [0.0, 800.0, 40.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
                "camera_source": "moge2",
                "mask": np.zeros((80, 120, 1), dtype=np.uint8),
            }
        ]


class _MovingBBoxEstimator(_DummyEstimator):
    def process_one_image(self, frame_rgb: np.ndarray, **kwargs: Any) -> list[dict[str, Any]]:
        outputs = super().process_one_image(frame_rgb, **kwargs)
        frame_idx = self.call_count - 1
        outputs[0]["bbox"] = np.array(
            [10.0 + frame_idx * 4.0, 12.0, 30.0 + frame_idx * 4.0, 42.0],
            dtype=np.float32,
        )
        return outputs


class _LargeBBoxEstimator(_DummyEstimator):
    def process_one_image(self, frame_rgb: np.ndarray, **kwargs: Any) -> list[dict[str, Any]]:
        outputs = super().process_one_image(frame_rgb, **kwargs)
        outputs[0]["bbox"] = np.array([2.0, 2.0, 118.0, 78.0], dtype=np.float32)
        return outputs


def test_discover_reference_videos_filters_and_sorts(tmp_path: Path) -> None:
    video_dir = tmp_path / "videos"
    video_dir.mkdir(parents=True)
    (video_dir / "b_ref.mp4").touch()
    (video_dir / "a_ref.mov").touch()
    (video_dir / "ignore.txt").touch()

    entries = discover_reference_videos(
        video_dir,
        action_type="smash",
        athlete_name="lin",
        camera_view="side",
        video_config=VideoExtractionConfig(target_fps=9.0, max_frames=99),
    )

    assert [entry.reference_id for entry in entries] == ["a_ref", "b_ref"]
    assert all(entry.action_type == "smash" for entry in entries)
    assert all(entry.athlete_name == "lin" for entry in entries)
    assert all(entry.camera_view == "side" for entry in entries)
    assert all(entry.video_config.target_fps == 9.0 for entry in entries)
    assert all(entry.video_config.max_frames == 99 for entry in entries)


def test_load_reference_manifest_applies_defaults(tmp_path: Path) -> None:
    video_path = tmp_path / "ref_clip.mp4"
    video_path.touch()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "assets": [
                    {
                        "videoPath": str(video_path),
                        "referenceId": "Smash Pro 01",
                        "selectionPointPx": [123, 45],
                        "metadata": {"source": "youtube"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    entries = load_reference_manifest(
        manifest_path,
        default_action_type="smash",
        default_athlete_name="pro_player",
        default_camera_view="back",
        default_handedness="right",
        default_video_config=VideoExtractionConfig(target_fps=15.0, max_frames=200),
    )

    assert len(entries) == 1
    entry = entries[0]
    assert entry.reference_id == "smash_pro_01"
    assert entry.action_type == "smash"
    assert entry.asset_role == "reference"
    assert entry.athlete_name == "pro_player"
    assert entry.camera_view == "back"
    assert entry.handedness == "right"
    assert entry.selection_point_px == (123.0, 45.0)
    assert entry.video_config.target_fps == 15.0
    assert entry.video_config.max_frames == 200
    assert entry.video_config.sample_every_frame is True
    assert entry.metadata["source"] == "youtube"


def test_reference_video_entry_preserves_remote_video_url() -> None:
    entry = ReferenceVideoEntry(
        video_path="https://example.com/library/smash_remote_ref.mp4",
        action_type="smash",
    )

    assert entry.video_path == "https://example.com/library/smash_remote_ref.mp4"
    assert entry.reference_id == "smash_remote_ref"


def test_load_reference_manifest_preserves_remote_video_url(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "assets": [
                    {
                        "videoPath": "https://example.com/library/smash_remote_ref.mp4",
                        "referenceId": "remote_ref",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    entries = load_reference_manifest(
        manifest_path,
        default_action_type="smash",
    )

    assert len(entries) == 1
    assert entries[0].video_path == "https://example.com/library/smash_remote_ref.mp4"
    assert entries[0].reference_id == "remote_ref"


def test_build_reference_assets_writes_npz_and_metadata(tmp_path: Path) -> None:
    video1 = tmp_path / "smash_1.mp4"
    video2 = tmp_path / "smash_2.mp4"
    _write_dummy_video(video1, fps=10.0, num_frames=8)
    _write_dummy_video(video2, fps=12.0, num_frames=12)
    _write_phase_annotations(
        tmp_path / "smash_1.json",
        video_id="smash_1",
        technique_type="smash",
        final_frame=7,
    )
    _write_phase_annotations(
        tmp_path / "smash_2.json",
        video_id="smash_2",
        technique_type="smash",
        final_frame=11,
    )

    entries = [
        ReferenceVideoEntry(
            video_path=video1,
            action_type="smash",
            reference_id="smash_pro_001",
            athlete_name="athlete_a",
            camera_view="side",
            handedness="right",
            selection_point_px=(20.0, 20.0),
            video_config=VideoExtractionConfig(target_fps=5.0),
        ),
        ReferenceVideoEntry(
            video_path=video2,
            action_type="smash",
            reference_id="smash_pro_002",
            athlete_name="athlete_b",
            camera_view="back",
            handedness="left",
            video_config=VideoExtractionConfig(target_fps=6.0),
            metadata={"source": "manual_pick"},
        ),
    ]

    output_dir = tmp_path / "out_assets"
    metadata = build_reference_assets(
        entries,
        estimator=_DummyEstimator(),  # type: ignore[arg-type]
        output_dir=output_dir,
        skeleton_version="test_v1",
    )

    assert metadata["schemaVersion"] == "technique_reference_assets.v2"
    assert metadata["skeletonVersion"] == "test_v1"
    assert metadata["fovEstimator"] is None
    assert metadata["renderAssetEnabled"] is True
    assert metadata["renderAssetSchemaVersion"] == "technique_reference_render.v1"
    assert metadata["assetCount"] == 2
    assert len(metadata["assets"]) == 2

    metadata_file = output_dir / "metadata.json"
    assert metadata_file.exists()
    file_metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    assert file_metadata["assetCount"] == 2

    for asset in file_metadata["assets"]:
        npz_path = Path(asset["skeletonPath"])
        assert npz_path.exists()
        npz_data = np.load(npz_path, allow_pickle=True)
        assert "keypoints_3d" in npz_data
        assert "timestamps" in npz_data
        assert npz_data["keypoints_3d"].shape[2] == 3
        assert asset["cameraSource"] == "moge2"
        assert len(asset["phaseAnnotations"]) == 4
        assert asset["horizontalFovDegCount"] == int(npz_data["timestamps"].shape[0])
        assert asset["horizontalFovDegRange"] is not None
        assert asset["renderAssetPath"] is not None
        render_npz_path = Path(asset["renderAssetPath"])
        assert render_npz_path.exists()
        render_data = np.load(render_npz_path, allow_pickle=True)
        assert render_data["schema_version"].item() == "technique_reference_render.v1"
        assert "vertices_3d" in render_data
        assert "keypoints_2d" in render_data
        assert "faces" in render_data
        assert "frame_indices" in render_data
        assert "image_size_hw" in render_data
        assert "cam_intrinsics" in render_data
        assert "horizontal_fov_deg" in render_data
        assert "camera_source" in render_data
        assert render_data["vertices_3d"].shape[2] == 3


@pytest.mark.skipif(
    not GOLDEN_IMG_1966_RIGHT_VIDEO.exists(),
    reason="Golden IMG_1966_right.MOV fixture not available",
)
def test_build_reference_asset_bundle_preserves_all_decodable_frames_for_img_1966_right(
    tmp_path: Path,
) -> None:
    phase_file = tmp_path / "IMG_1966.json"
    _write_phase_annotations(
        phase_file,
        video_id="IMG_1966",
        technique_type="jump-smash",
        final_frame=100,
    )
    entry = ReferenceVideoEntry(
        video_path=GOLDEN_IMG_1966_RIGHT_VIDEO,
        action_type="smash",
        reference_id="img_1966_right_golden",
        phase_annotations_file=phase_file,
        handedness="right",
        camera_view="side",
        video_config=VideoExtractionConfig(target_fps=30.0, max_frames=240),
    )

    bundle = build_reference_assets(
        [entry],
        estimator=_DummyEstimator(),  # type: ignore[arg-type]
        output_dir=tmp_path / "golden_out",
        skeleton_version="test_v1",
    )

    asset = bundle["assets"][0]
    assert asset["numFrames"] == 101
    assert asset["frameIndices"] == list(range(0, 101))
    assert asset["videoConfig"]["targetFps"] == 30.0
    assert asset["videoConfig"]["startTimeSec"] == 0.0
    assert asset["videoConfig"]["endTimeSec"] is None
    assert asset["videoConfig"]["sampleEveryFrame"] is True


def test_build_reference_assets_requires_phase_annotations_for_reference_assets(
    tmp_path: Path,
) -> None:
    video = tmp_path / "smash_missing_phase.mp4"
    _write_dummy_video(video, fps=10.0, num_frames=8)
    entry = ReferenceVideoEntry(
        video_path=video,
        action_type="smash",
        reference_id="smash_missing_phase",
    )

    with pytest.raises(FileNotFoundError, match="Phase annotations file not found"):
        build_reference_assets(
            [entry],
            estimator=_DummyEstimator(),  # type: ignore[arg-type]
            output_dir=tmp_path / "out_assets",
        )


def test_build_reference_assets_rejects_truncated_phase_annotations(
    tmp_path: Path,
) -> None:
    video = tmp_path / "smash_truncated.mp4"
    _write_dummy_video(video, fps=10.0, num_frames=8)
    phase_file = tmp_path / "smash_truncated.json"
    _write_phase_annotations(
        phase_file,
        video_id="smash_truncated",
        technique_type="smash",
        final_frame=7,
    )
    entry = ReferenceVideoEntry(
        video_path=video,
        action_type="smash",
        reference_id="smash_truncated",
        phase_annotations_file=phase_file,
        video_config=VideoExtractionConfig(max_frames=4),
    )

    with pytest.raises(ValueError, match="extends past extracted frames"):
        build_reference_assets(
            [entry],
            estimator=_DummyEstimator(),  # type: ignore[arg-type]
            output_dir=tmp_path / "out_assets",
        )


def test_build_reference_assets_accepts_shared_phase_boundary_frames(
    tmp_path: Path,
) -> None:
    video = tmp_path / "smash_shared_boundary.mp4"
    _write_dummy_video(video, fps=10.0, num_frames=8)
    phase_file = tmp_path / "smash_shared_boundary.json"
    phase_file.write_text(
        json.dumps(
            {
                "videoId": "smash_shared_boundary",
                "techniqueType": "smash",
                "phases": [
                    {
                        "id": "preparatory_phase",
                        "name": "Preparatory Phase",
                        "description": "Load posture and prepare to swing.",
                        "startFrame": 0,
                        "endFrame": 1,
                    },
                    {
                        "id": "backswing_phase",
                        "name": "Backswing Phase",
                        "description": "Draw the racket back.",
                        "startFrame": 2,
                        "endFrame": 3,
                    },
                    {
                        "id": "power_generation_phase",
                        "name": "Power Generation Phase",
                        "description": "Accelerate into the shuttle.",
                        "startFrame": 4,
                        "endFrame": 4,
                    },
                    {
                        "id": "followthrough_phase",
                        "name": "Follow-through Phase",
                        "description": "Finish the swing and recover.",
                        "startFrame": 4,
                        "endFrame": 7,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    entry = ReferenceVideoEntry(
        video_path=video,
        action_type="smash",
        reference_id="smash_shared_boundary",
        phase_annotations_file=phase_file,
    )

    metadata = build_reference_assets(
        [entry],
        estimator=_DummyEstimator(),  # type: ignore[arg-type]
        output_dir=tmp_path / "out_assets",
    )

    asset = metadata["assets"][0]
    assert asset["phaseAnnotations"][2]["endFrame"] == 4
    assert asset["phaseAnnotations"][3]["startFrame"] == 4


def test_build_reference_assets_rejects_true_overlapping_phase_ranges(
    tmp_path: Path,
) -> None:
    video = tmp_path / "smash_overlapping.mp4"
    _write_dummy_video(video, fps=10.0, num_frames=8)
    phase_file = tmp_path / "smash_overlapping.json"
    phase_file.write_text(
        json.dumps(
            {
                "videoId": "smash_overlapping",
                "techniqueType": "smash",
                "phases": [
                    {
                        "id": "preparatory_phase",
                        "name": "Preparatory Phase",
                        "description": "Load posture and prepare to swing.",
                        "startFrame": 0,
                        "endFrame": 1,
                    },
                    {
                        "id": "backswing_phase",
                        "name": "Backswing Phase",
                        "description": "Draw the racket back.",
                        "startFrame": 2,
                        "endFrame": 4,
                    },
                    {
                        "id": "power_generation_phase",
                        "name": "Power Generation Phase",
                        "description": "Accelerate into the shuttle.",
                        "startFrame": 3,
                        "endFrame": 5,
                    },
                    {
                        "id": "followthrough_phase",
                        "name": "Follow-through Phase",
                        "description": "Finish the swing and recover.",
                        "startFrame": 6,
                        "endFrame": 7,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    entry = ReferenceVideoEntry(
        video_path=video,
        action_type="smash",
        reference_id="smash_overlapping",
        phase_annotations_file=phase_file,
    )

    with pytest.raises(
        ValueError,
        match="phaseAnnotations must be ordered and may only share a boundary frame",
    ):
        build_reference_assets(
            [entry],
            estimator=_DummyEstimator(),  # type: ignore[arg-type]
            output_dir=tmp_path / "out_assets",
        )

def test_build_reference_assets_rejects_duplicate_reference_id(tmp_path: Path) -> None:
    video = tmp_path / "dup.mp4"
    _write_dummy_video(video, fps=10.0, num_frames=4)
    entries = [
        ReferenceVideoEntry(video_path=video, action_type="smash", reference_id="dup_id"),
        ReferenceVideoEntry(video_path=video, action_type="smash", reference_id="dup_id"),
    ]

    with pytest.raises(ValueError, match="Duplicate reference_id"):
        build_reference_assets(
            entries,
            estimator=_DummyEstimator(),  # type: ignore[arg-type]
            output_dir=tmp_path / "out",
        )


def test_build_reference_asset_bundle_writes_cropped_follow_video(tmp_path: Path) -> None:
    video = tmp_path / "cropped.mp4"
    _write_dummy_video(video, fps=10.0, num_frames=8)
    phase_file = tmp_path / "cropped.json"
    _write_phase_annotations(
        phase_file,
        video_id="cropped",
        technique_type="smash",
        final_frame=3,
    )

    bundle = build_reference_asset_bundle(
        ReferenceVideoEntry(
            video_path=video,
            action_type="smash",
            asset_role="user",
            reference_id="cropped_user",
            phase_annotations_file=phase_file,
        ),
        estimator=_DummyEstimator(),  # type: ignore[arg-type]
        output_dir=tmp_path / "out",
        cropped_video_output_path=tmp_path / "out" / "source.mp4",
        overwrite=True,
    )

    assert bundle.cropped_video_path is not None
    assert bundle.cropped_video_path.exists()

    capture = cv2.VideoCapture(str(bundle.cropped_video_path))
    ok, frame = capture.read()
    capture.release()

    assert ok
    assert frame.shape[0] > 0
    assert frame.shape[1] > 0
    assert frame.shape[0] % 2 == 0
    assert frame.shape[1] % 2 == 0


def test_build_reference_asset_bundle_cropped_video_uses_smoothed_output_track(
    tmp_path: Path,
) -> None:
    video = tmp_path / "tracked.mp4"
    _write_dummy_video(video, fps=10.0, num_frames=6)

    bundle = build_reference_asset_bundle(
        ReferenceVideoEntry(
            video_path=video,
            action_type="smash",
            asset_role="user",
            reference_id="tracked_user",
            selection_bbox_xyxy=(0.0, 0.0, 100.0, 70.0),
        ),
        estimator=_MovingBBoxEstimator(),  # type: ignore[arg-type]
        output_dir=tmp_path / "out",
        cropped_video_output_path=tmp_path / "out" / "tracked_source.mp4",
        overwrite=True,
    )

    assert bundle.cropped_video_path is not None
    capture = cv2.VideoCapture(str(bundle.cropped_video_path))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 28
        assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 48
    finally:
        capture.release()


def test_smooth_track_reduces_single_frame_center_jump() -> None:
    centers = np.array([20.0, 24.0, 60.0, 28.0, 32.0], dtype=np.float32)

    smoothed = _smooth_track_1d(centers, outlier_window=3, motion_window=5)

    assert smoothed.shape == centers.shape
    assert float(np.max(np.abs(np.diff(smoothed)))) < float(np.max(np.abs(np.diff(centers))))
    assert abs(float(smoothed[2] - smoothed[1])) < abs(float(centers[2] - centers[1]))
    assert abs(float(smoothed[3] - smoothed[2])) < abs(float(centers[3] - centers[2]))


def test_build_reference_asset_bundle_uses_full_frame_when_padded_crop_exceeds_source(
    tmp_path: Path,
) -> None:
    video = tmp_path / "full_frame.mp4"
    _write_dummy_video(video, fps=10.0, num_frames=4)

    bundle = build_reference_asset_bundle(
        ReferenceVideoEntry(
            video_path=video,
            action_type="smash",
            asset_role="user",
            reference_id="full_frame_user",
            selection_bbox_xyxy=(0.0, 0.0, 120.0, 80.0),
        ),
        estimator=_LargeBBoxEstimator(),  # type: ignore[arg-type]
        output_dir=tmp_path / "out",
        cropped_video_output_path=tmp_path / "out" / "full_frame_source.mp4",
        overwrite=True,
    )

    assert bundle.cropped_video_path is not None
    capture = cv2.VideoCapture(str(bundle.cropped_video_path))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 120
        assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 80
    finally:
        capture.release()

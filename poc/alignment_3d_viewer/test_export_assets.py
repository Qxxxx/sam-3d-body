from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from export_assets import (
    TARGET_BODY_SCALE,
    _load_render_asset,
    body_basis_frames,
    canonicalize_mesh_frames,
    export_assets,
    root_center_mesh_frames,
    sequence_body_scale,
    temporal_smooth_frames,
)


class CanonicalMeshExportTests(unittest.TestCase):
    def test_export_assets_writes_complete_viewer_contract(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            keypoints = np.zeros((3, 70, 3), dtype=np.float32)
            keypoints[:, 0] = [0.0, 1.5, 0.0]
            keypoints[:, 9] = [-0.5, 0.0, 0.0]
            keypoints[:, 10] = [0.5, 0.0, 0.0]
            keypoints[:, 11] = [-0.5, -0.5, 0.0]
            keypoints[:, 12] = [0.5, -0.5, 0.0]
            keypoints[:, 13] = [-0.5, -1.0, 0.0]
            keypoints[:, 14] = [0.5, -1.0, 0.0]
            keypoints[:, 69] = [0.0, 1.0, 0.0]
            vertices = np.repeat(
                np.asarray(
                    [[[-0.5, -1.0, 0.0], [0.5, -1.0, 0.0], [0.0, 1.0, 0.0]]],
                    dtype=np.float32,
                ),
                3,
                axis=0,
            )
            faces = np.asarray([[0, 1, 2]], dtype=np.int32)
            intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], 3, axis=0)
            for name in ("user", "reference"):
                np.savez(
                    root / f"{name}.render.npz",
                    vertices_3d=vertices,
                    faces=faces,
                    keypoints_3d=keypoints,
                    timestamps=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
                    cam_t=np.asarray(
                        [[0.0, 0.0, 3.0], [0.0, 0.0, 3.0], [0.0, 0.0, 3.0]],
                        dtype=np.float32,
                    ),
                    cam_intrinsics=intrinsics,
                    image_size_hw=np.asarray([720, 1280]),
                )
            alignment_path = root / "alignment.json"
            alignment_path.write_text(
                json.dumps(
                    {
                        "algorithm": "dtw",
                        "distance": 0.2,
                        "summary": {"meanFrameError": 0.1},
                        "alignmentPath": [
                            {
                                "userFrameIndex": index,
                                "referenceFrameIndex": index,
                                "userTimestamp": index * 0.5,
                                "referenceTimestamp": index * 0.5,
                                "distance": (index + 1) * 0.1,
                            }
                            for index in range(3)
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "output"

            metadata = export_assets(
                user_render_path=root / "user.render.npz",
                reference_render_path=root / "reference.render.npz",
                alignment_path=alignment_path,
                output_dir=output,
                user_label="User",
                reference_label="Reference",
                fps=3.0,
            )

            self.assertEqual(metadata["mesh"]["stepCount"], 3)
            self.assertEqual(metadata["labels"], {"user": "User", "reference": "Reference"})
            self.assertEqual(metadata["temporalFilter"]["phaseLagFrames"], 0)
            self.assertTrue((output / "viewer-data.json").is_file())
            for filename in metadata["files"].values():
                with self.subTest(filename=filename):
                    self.assertGreater((output / filename).stat().st_size, 0)

    def test_temporal_filter_smooths_impulse_without_frame_delay(self) -> None:
        values = np.asarray([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        smoothed = temporal_smooth_frames(values)

        np.testing.assert_allclose(smoothed, [0.0, 0.2, 0.6, 0.2, 0.0])
        self.assertEqual(int(np.argmax(smoothed)), 2)

    def test_temporal_filter_preserves_constant_sequence(self) -> None:
        values = np.full((5, 2, 3), 4.25, dtype=np.float32)

        smoothed = temporal_smooth_frames(values)

        np.testing.assert_array_equal(smoothed, values)

    def test_render_asset_preserves_source_camera_contract(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "render.npz"
            np.savez(
                path,
                vertices_3d=np.zeros((2, 4, 3), dtype=np.float32),
                faces=np.zeros((2, 3), dtype=np.int32),
                keypoints_3d=np.zeros((2, 70, 3), dtype=np.float32),
                timestamps=np.asarray([0.0, 0.5], dtype=np.float32),
                cam_t=np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                cam_intrinsics=np.repeat(
                    np.asarray(
                        [
                            [
                                [900.0, 0.0, 360.0],
                                [0.0, 900.0, 640.0],
                                [0.0, 0.0, 1.0],
                            ]
                        ]
                    ),
                    2,
                    axis=0,
                ),
                image_size_hw=np.asarray([1280, 720]),
            )

            asset = _load_render_asset(path)

            self.assertEqual(asset["cam_t"].shape, (2, 3))
            self.assertEqual(asset["cam_intrinsics"].shape, (2, 3, 3))
            np.testing.assert_array_equal(asset["image_size_hw"], [1280, 720])

    def test_canonicalization_removes_translation_and_scale(self) -> None:
        keypoints = np.zeros((1, 70, 3), dtype=np.float32)
        keypoints[:, 0] = [0.0, 1.5, 0.0]
        keypoints[:, 9] = [-0.5, 0.0, 0.0]
        keypoints[:, 10] = [0.5, 0.0, 0.0]
        keypoints[:, 11] = [-0.5, -0.5, 0.0]
        keypoints[:, 12] = [0.5, -0.5, 0.0]
        keypoints[:, 13] = [-0.5, -1.0, 0.0]
        keypoints[:, 14] = [0.5, -1.0, 0.0]
        keypoints[:, 69] = [0.0, 1.0, 0.0]
        vertices = np.asarray(
            [[[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 1.0, 0.0]]],
            dtype=np.float32,
        )
        scaled_keypoints = keypoints * 2.0 + np.asarray([10.0, 4.0, -2.0])
        scaled_vertices = vertices * 2.0 + np.asarray([10.0, 4.0, -2.0])

        canonical = canonicalize_mesh_frames(vertices, keypoints, [0])
        scaled_canonical = canonicalize_mesh_frames(
            scaled_vertices, scaled_keypoints, [0]
        )

        np.testing.assert_allclose(canonical, scaled_canonical, atol=1e-6)
        np.testing.assert_allclose(
            canonical[0],
            np.asarray(
                [[-0.2, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.4, 0.0]],
                dtype=np.float32,
            ),
            atol=1e-6,
        )

    def test_sequence_body_scale_uses_median_bone_chain(self) -> None:
        keypoints = np.zeros((3, 70, 3), dtype=np.float32)
        keypoints[:, 0, 1] = 1.5
        keypoints[:, 69, 1] = 1.0
        keypoints[:, 9] = [-0.5, 0.0, 0.0]
        keypoints[:, 10] = [0.5, 0.0, 0.0]
        keypoints[:, 11] = [-0.5, -0.5, 0.0]
        keypoints[:, 12] = [0.5, -0.5, 0.0]
        keypoints[:, 13] = [-0.5, -1.0, 0.0]
        keypoints[:, 14] = [0.5, -1.0, 0.0]
        keypoints[2] *= 10.0

        self.assertAlmostEqual(sequence_body_scale(keypoints), 2.5)

    def test_raw_frames_and_basis_reproduce_canonical_view(self) -> None:
        keypoints = np.zeros((1, 70, 3), dtype=np.float32)
        keypoints[:, 0] = [0.0, 1.5, 0.0]
        keypoints[:, 9] = [0.0, 0.0, 0.5]
        keypoints[:, 10] = [0.0, 0.0, -0.5]
        keypoints[:, 11] = [0.0, -0.5, 0.5]
        keypoints[:, 12] = [0.0, -0.5, -0.5]
        keypoints[:, 13] = [0.0, -1.0, 0.5]
        keypoints[:, 14] = [0.0, -1.0, -0.5]
        keypoints[:, 69] = [0.0, 1.0, 0.0]
        vertices = np.asarray(
            [[[0.0, 0.0, 0.5], [0.0, 0.0, -0.5], [0.0, 1.0, 0.0]]],
            dtype=np.float32,
        )
        body_scale = sequence_body_scale(keypoints)
        canonical = canonicalize_mesh_frames(
            vertices, keypoints, [0], body_scale=body_scale
        )
        raw = root_center_mesh_frames(
            vertices,
            keypoints,
            [0],
            body_scale=body_scale,
            target_body_scale=1.0,
        )
        basis = body_basis_frames(keypoints, [0])

        np.testing.assert_allclose(raw[0] @ basis[0], canonical[0], atol=1e-6)
        np.testing.assert_allclose(basis[0].T @ basis[0], np.eye(3), atol=1e-6)

    def test_relative_basis_rotates_user_into_reference_capture(self) -> None:
        user_keypoints = np.zeros((1, 70, 3), dtype=np.float32)
        user_keypoints[:, 9] = [-0.5, 0.0, 0.0]
        user_keypoints[:, 10] = [0.5, 0.0, 0.0]
        user_keypoints[:, 69] = [0.0, 1.0, 0.0]
        reference_keypoints = user_keypoints.copy()
        quarter_turn = np.asarray(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        reference_keypoints[0] = reference_keypoints[0] @ quarter_turn.T
        user_basis = body_basis_frames(user_keypoints, [0])[0]
        reference_basis = body_basis_frames(reference_keypoints, [0])[0]
        relative_column_rotation = reference_basis @ user_basis.T
        canonical_vector = np.asarray([0.4, 0.7, -0.2], dtype=np.float32)
        user_raw = canonical_vector @ user_basis.T
        reference_raw = canonical_vector @ reference_basis.T

        np.testing.assert_allclose(
            relative_column_rotation @ user_raw,
            reference_raw,
            atol=1e-6,
        )


class SkeletalScaleExportTests(unittest.TestCase):
    @staticmethod
    def standing_pose() -> np.ndarray:
        points = np.zeros((3, 70, 3), dtype=np.float32)
        points[:, 0] = [0.0, 1.5, 0.0]
        points[:, 69] = [0.0, 1.0, 0.0]
        points[:, 5] = [-0.6, 0.9, 0.0]
        points[:, 6] = [0.6, 0.9, 0.0]
        for hip, knee, ankle, x in ((9, 11, 13, -0.3), (10, 12, 14, 0.3)):
            points[:, hip] = [x, 0.0, 0.0]
            points[:, knee] = [x, -0.5, 0.0]
            points[:, ankle] = [x, -1.0, 0.0]
        return points

    def export_pair(self, user: np.ndarray, reference: np.ndarray) -> tuple:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name, points in (("user", user), ("reference", reference)):
                # Use joint positions as synthetic mesh probes so a hand or
                # ankle can alter mesh bounds without changing body length.
                np.savez(
                    root / f"{name}.npz",
                    vertices_3d=points,
                    keypoints_3d=points,
                    faces=np.asarray([[0, 5, 6]], dtype=np.int32),
                    timestamps=np.arange(3, dtype=np.float32),
                    cam_t=np.tile([0.0, 0.0, 3.0], (3, 1)),
                    cam_intrinsics=np.tile(np.eye(3), (3, 1, 1)),
                    image_size_hw=np.asarray([720, 1280]),
                )
            alignment = root / "alignment.json"
            alignment.write_text(
                json.dumps({"alignmentPath": [
                    {
                        "userFrameIndex": i,
                        "referenceFrameIndex": i,
                        "userTimestamp": float(i),
                        "referenceTimestamp": float(i),
                        "distance": 0.0,
                    }
                    for i in range(3)
                ]}),
                encoding="utf-8",
            )
            metadata = export_assets(
                user_render_path=root / "user.npz",
                reference_render_path=root / "reference.npz",
                alignment_path=alignment,
                output_dir=root / "output",
                user_label="User",
                reference_label="Reference",
                fps=1.0,
            )
            buffers = {
                key: np.fromfile(
                    root / "output" / metadata["files"][key], dtype="<f4"
                ).reshape(3, 70, 3)
                for key in (
                    "userPositions", "referencePositions", "userRawPositions",
                    "referenceRawPositions", "userCameraPositions",
                    "referenceCameraPositions",
                )
            }
            for subject in ("user", "reference"):
                for suffix in ("CameraTranslation", "CameraIntrinsics"):
                    key = subject + suffix
                    buffers[key] = np.fromfile(
                        root / "output" / metadata["files"][key], dtype="<f4"
                    )
            return metadata, buffers

    def test_unified_scale_cannot_change_video_overlay_buffers(self) -> None:
        reference = self.standing_pose()
        user = reference * 1.5
        user[:, 7, 1] = [1.0, 3.0, 4.0]
        user += np.asarray([1.0, 2.0, 0.5], dtype=np.float32)
        original_metadata, original = self.export_pair(user, reference)

        with patch("export_assets.TARGET_BODY_SCALE", TARGET_BODY_SCALE * 2):
            enlarged_metadata, enlarged = self.export_pair(user, reference)

        for subject in ("user", "reference"):
            for suffix in ("CameraPositions", "CameraTranslation", "CameraIntrinsics"):
                key = subject + suffix
                self.assertEqual(original[key].tobytes(), enlarged[key].tobytes())
            for suffix in ("Positions", "RawPositions"):
                key = subject + suffix
                np.testing.assert_allclose(enlarged[key], original[key] * 2)
        self.assertEqual(
            original_metadata["cameraOverlay"], enlarged_metadata["cameraOverlay"]
        )
        np.testing.assert_array_equal(
            original["userCameraPositions"], temporal_smooth_frames(user)
        )
        np.testing.assert_array_equal(
            original["referenceCameraPositions"], temporal_smooth_frames(reference)
        )

    def test_crouching_does_not_enlarge_body(self) -> None:
        standing = self.standing_pose()
        crouching = standing.copy()
        for knee, ankle, x in ((11, 13, -0.3), (12, 14, 0.3)):
            crouching[:, knee] = [x, -0.3, 0.4]
            crouching[:, ankle] = [x, -0.6, 0.0]

        metadata, buffers = self.export_pair(crouching, standing)

        normalization = metadata["normalization"]
        self.assertEqual(normalization["method"], "skeletal-chain-v1")
        self.assertAlmostEqual(normalization["userBodyScale"], 2.5)
        self.assertAlmostEqual(normalization["referenceBodyScale"], 2.5)
        self.assertAlmostEqual(
            normalization["userScaleFactor"], TARGET_BODY_SCALE / 2.5
        )
        self.assertNotIn("targetVisualHeight", normalization)
        for suffix in ("Positions", "RawPositions"):
            user, reference = buffers["user" + suffix], buffers["reference" + suffix]
            np.testing.assert_allclose(
                user[:, [0, 5, 6, 69]], reference[:, [0, 5, 6, 69]]
            )
            self.assertTrue(np.all(
                np.ptp(user[:, :, 1], axis=1) < np.ptp(reference[:, :, 1], axis=1)
            ))
        np.testing.assert_allclose(buffers["userCameraPositions"], crouching)

    def test_raised_hand_does_not_shrink_body_or_pulse_between_frames(self) -> None:
        standing = self.standing_pose()
        raised = standing.copy()
        raised[:, 7] = [[0.7, 1.0, 0.0], [0.7, 3.0, 0.0], [0.7, 4.0, 0.0]]

        _, buffers = self.export_pair(raised, standing)

        for suffix in ("Positions", "RawPositions"):
            user, reference = buffers["user" + suffix], buffers["reference" + suffix]
            np.testing.assert_allclose(
                user[:, [0, 5, 6, 69]], reference[:, [0, 5, 6, 69]]
            )
            np.testing.assert_allclose(user[0, [5, 6]], user[2, [5, 6]])
            self.assertGreater(float(user[2, 7, 1]), float(reference[2, 0, 1]))

    def test_different_sizes_match_but_body_proportions_are_preserved(self) -> None:
        reference = self.standing_pose()
        user = reference * 1.5
        user[:, [5, 6], 0] *= 1.2  # A larger person with proportionally wider shoulders.
        user += np.asarray([10.0, 4.0, -2.0], dtype=np.float32)

        metadata, buffers = self.export_pair(user, reference)

        self.assertAlmostEqual(metadata["normalization"]["userBodyScale"], 3.75)
        for suffix in ("Positions", "RawPositions"):
            u, r = buffers["user" + suffix], buffers["reference" + suffix]
            np.testing.assert_allclose(
                u[:, [0, 9, 10, 11, 12, 13, 14, 69]],
                r[:, [0, 9, 10, 11, 12, 13, 14, 69]],
                atol=1e-6,
            )
            np.testing.assert_allclose(
                u[:, 6, 0] - u[:, 5, 0],
                1.2 * (r[:, 6, 0] - r[:, 5, 0]),
                atol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()

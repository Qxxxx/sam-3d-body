from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from export_assets import (
    TARGET_VISUAL_HEIGHT,
    _load_render_asset,
    body_basis_frames,
    canonicalize_mesh_frames,
    export_assets,
    normalize_visual_height,
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

    def test_visual_height_normalization_uses_one_sequence_scale(self) -> None:
        positions = np.asarray(
            [
                [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, -2.0, 0.0], [0.0, 2.0, 0.0]],
                [[0.0, -1.5, 0.0], [0.0, 1.5, 0.0]],
            ],
            dtype=np.float32,
        )

        normalized, median_height, visual_scale = normalize_visual_height(positions)

        self.assertAlmostEqual(median_height, 3.0)
        self.assertAlmostEqual(visual_scale, TARGET_VISUAL_HEIGHT / 3.0)
        self.assertAlmostEqual(
            float(np.median(np.ptp(normalized[:, :, 1], axis=1))),
            TARGET_VISUAL_HEIGHT,
            places=6,
        )

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
            visual_scale=1.0,
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


if __name__ == "__main__":
    unittest.main()

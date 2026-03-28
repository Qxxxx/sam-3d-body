from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from sam_3d_body import sam_3d_body_estimator as estimator_module
from sam_3d_body.sam_3d_body_estimator import SAM3DBodyEstimator


class _DummyHeadPose:
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)


class _DummyModel:
    def __init__(self) -> None:
        self.device = "cpu"
        self.head_pose = _DummyHeadPose()

    def _initialize_batch(self, _batch: dict[str, Any]) -> None:
        return None

    def run_inference(
        self,
        _img: np.ndarray,
        _batch: dict[str, Any],
        **_kwargs: Any,
    ) -> dict[str, dict[str, np.ndarray]]:
        return {
            "mhr": {
                "focal_length": np.array([1200.0], dtype=np.float32),
                "pred_keypoints_3d": np.zeros((1, 4, 3), dtype=np.float32),
                "pred_keypoints_2d": np.zeros((1, 4, 2), dtype=np.float32),
                "pred_vertices": np.zeros((1, 5, 3), dtype=np.float32),
                "pred_cam_t": np.zeros((1, 3), dtype=np.float32),
                "pred_pose_raw": np.zeros((1, 3), dtype=np.float32),
                "global_rot": np.zeros((1, 3), dtype=np.float32),
                "body_pose": np.zeros((1, 6), dtype=np.float32),
                "hand": np.zeros((1, 6), dtype=np.float32),
                "scale": np.ones((1, 1), dtype=np.float32),
                "shape": np.zeros((1, 10), dtype=np.float32),
                "face": np.zeros((1, 4), dtype=np.float32),
                "pred_joint_coords": np.zeros((1, 4, 3), dtype=np.float32),
                "joint_global_rots": np.zeros((1, 4, 3, 3), dtype=np.float32),
                "mhr_model_params": np.zeros((1, 8), dtype=np.float32),
            }
        }


class _DummyFOVEstimator:
    name = "moge2"

    def get_cam_intrinsics(self, _input_image: Any) -> torch.Tensor:
        return torch.eye(3, dtype=torch.float32).unsqueeze(0)


def test_sam3d_body_estimator_emits_fov_runtime_notice_once(
    monkeypatch: Any, caplog: Any
) -> None:
    def _fake_prepare_batch(
        _img: np.ndarray,
        _transform: Any,
        _boxes: np.ndarray,
        _masks: Any,
        _masks_score: Any,
    ) -> dict[str, Any]:
        return {
            "img": torch.zeros((1, 1, 3, 4, 4), dtype=torch.float32),
            "img_ori": [torch.zeros((4, 4, 3), dtype=torch.float32)],
            "bbox": torch.tensor([[[0.0, 0.0, 4.0, 4.0]]], dtype=torch.float32),
            "cam_int": torch.eye(3, dtype=torch.float32).unsqueeze(0),
        }

    monkeypatch.setattr(estimator_module, "prepare_batch", _fake_prepare_batch)
    monkeypatch.setattr(estimator_module, "recursive_to", lambda value, *_args: value)
    monkeypatch.setattr(estimator_module.torch.cuda, "empty_cache", lambda: None)

    runtime_logger = estimator_module.logger
    runtime_logger.addHandler(caplog.handler)
    previous_level = runtime_logger.level
    runtime_logger.setLevel(logging.DEBUG)

    try:
        estimator = SAM3DBodyEstimator(
            _DummyModel(),
            SimpleNamespace(MODEL=SimpleNamespace(IMAGE_SIZE=(256, 256))),
            fov_estimator=_DummyFOVEstimator(),
        )

        input_frame = np.zeros((4, 4, 3), dtype=np.uint8)
        estimator.process_one_image(input_frame, inference_type="body")
        estimator.process_one_image(input_frame, inference_type="body")
    finally:
        runtime_logger.setLevel(previous_level)
        runtime_logger.removeHandler(caplog.handler)

    messages = [record.getMessage() for record in caplog.records]
    assert messages.count(
        "Running FOV estimator for frames without provided camera intrinsics"
    ) == 1

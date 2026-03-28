# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import Optional, Union

import cv2

import numpy as np
import torch

from sam_3d_body.data.transforms import (
    Compose,
    GetBBoxCenterScale,
    TopdownAffine,
    VisionTransformWrapper,
)

from sam_3d_body.data.utils.io import load_image
from sam_3d_body.data.utils.prepare_batch import prepare_batch
from sam_3d_body.utils import recursive_to
from sam_3d_body.utils.logging import get_pylogger
from torchvision.transforms import ToTensor


logger = get_pylogger(__name__)


class SAM3DBodyEstimator:
    def __init__(
        self,
        sam_3d_body_model,
        model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=None,
    ):
        self.device = sam_3d_body_model.device
        self.model, self.cfg = sam_3d_body_model, model_cfg
        self.detector = human_detector
        self.sam = human_segmentor
        self.fov_estimator = fov_estimator
        self.thresh_wrist_angle = 1.4
        self._emitted_runtime_notices: set[str] = set()

        # For mesh visualization
        self.faces = self.model.head_pose.faces.cpu().numpy()

        if self.detector is None:
            logger.info(
                "Human detector is not configured; falling back to full-frame inference",
                extra={"component": "estimator"},
            )
        if self.sam is None:
            logger.info(
                "Human segmentor is not configured; mask-conditioned inference is disabled",
                extra={"component": "estimator"},
            )
        if self.fov_estimator is None:
            logger.info(
                "FOV estimator is not configured; using default camera intrinsics",
                extra={"component": "estimator"},
            )

        self.transform = Compose(
            [
                GetBBoxCenterScale(),
                TopdownAffine(input_size=self.cfg.MODEL.IMAGE_SIZE, use_udp=False),
                VisionTransformWrapper(ToTensor()),
            ]
        )
        self.transform_hand = Compose(
            [
                GetBBoxCenterScale(padding=0.9),
                TopdownAffine(input_size=self.cfg.MODEL.IMAGE_SIZE, use_udp=False),
                VisionTransformWrapper(ToTensor()),
            ]
        )

    def _log_once(
        self,
        key: str,
        level: str,
        message: str,
        **extra: object,
    ) -> None:
        if key in self._emitted_runtime_notices:
            return
        self._emitted_runtime_notices.add(key)
        getattr(logger, level)(message, extra=extra)

    @torch.no_grad()
    def process_one_image(
        self,
        img: Union[str, np.ndarray],
        bboxes: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        cam_int: Optional[np.ndarray] = None,
        det_cat_id: int = 0,
        bbox_thr: float = 0.5,
        nms_thr: float = 0.3,
        use_mask: bool = False,
        inference_type: str = "full",
    ):
        """
        Perform model prediction in top-down format: assuming input is a full image.

        Args:
            img: Input image (path or numpy array)
            bboxes: Optional pre-computed bounding boxes
            masks: Optional pre-computed masks (numpy array). If provided, SAM2 will be skipped.
            det_cat_id: Detection category ID
            bbox_thr: Bounding box threshold
            nms_thr: NMS threshold
            inference_type:
                - full: full-body inference with both body and hand decoders
                - body: inference with body decoder only (still full-body output)
                - hand: inference with hand decoder only (only hand output)
        """

        # clear all cached results
        self.batch = None
        self.image_embeddings = None
        self.output = None
        self.prev_prompt = []
        torch.cuda.empty_cache()

        if type(img) == str:
            img = load_image(img, backend="cv2", image_format="bgr")
            image_format = "bgr"
        else:
            self._log_once(
                "rgb_input_notice",
                "warning",
                (
                    "Input ndarray is assumed to be RGB. "
                    "If your source is OpenCV BGR, convert with "
                    "cv2.cvtColor(..., cv2.COLOR_BGR2RGB)."
                ),
                component="estimator",
            )
            image_format = "rgb"
        height, width = img.shape[:2]

        if bboxes is not None:
            boxes = bboxes.reshape(-1, 4)
            self.is_crop = True
        elif self.detector is not None:
            if image_format == "rgb":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                image_format = "bgr"
            logger.debug(
                "Running object detector",
                extra={
                    "component": "estimator",
                    "bboxThreshold": float(bbox_thr),
                    "nmsThreshold": float(nms_thr),
                },
            )
            boxes = self.detector.run_human_detection(
                img,
                det_cat_id=det_cat_id,
                bbox_thr=bbox_thr,
                nms_thr=nms_thr,
                default_to_full_image=False,
            )
            logger.debug(
                "Object detector completed",
                extra={
                    "component": "estimator",
                    "bboxCount": int(len(boxes)),
                },
            )
            self.is_crop = True
        else:
            boxes = np.array([0, 0, width, height]).reshape(1, 4)
            self.is_crop = False

        # If there are no detected humans, don't run prediction
        if len(boxes) == 0:
            return []

        # The following models expect RGB images instead of BGR
        if image_format == "bgr":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Handle masks - either provided externally or generated via SAM2
        masks_score = None
        if masks is not None:
            # Use provided masks - ensure they match the number of detected boxes
            self._log_once(
                "provided_masks_notice",
                "debug",
                "Using provided masks for inference",
                component="estimator",
                maskShape=list(masks.shape),
            )
            assert (
                bboxes is not None
            ), "Mask-conditioned inference requires bboxes input!"
            masks = masks.reshape(-1, height, width, 1).astype(np.uint8)
            masks_score = np.ones(
                len(masks), dtype=np.float32
            )  # Set high confidence for provided masks
            use_mask = True
        elif use_mask and self.sam is not None:
            logger.debug(
                "Running SAM mask generation from bounding boxes",
                extra={"component": "estimator"},
            )
            # Generate masks using SAM2
            masks, masks_score = self.sam.run_sam(img, boxes)
        else:
            masks, masks_score = None, None

        #################### Construct batch data samples ####################
        batch = prepare_batch(img, self.transform, boxes, masks, masks_score)

        #################### Run model inference on an image ####################
        batch = recursive_to(batch, "cuda")
        self.model._initialize_batch(batch)

        # Handle camera intrinsics
        # - either provided externally or generated via default FOV estimator
        camera_source = "default"
        if cam_int is not None:
            self._log_once(
                "provided_cam_intrinsics_notice",
                "debug",
                "Using provided camera intrinsics",
                component="estimator",
            )
            cam_int = cam_int.to(batch["img"])
            batch["cam_int"] = cam_int.clone()
            camera_source = "provided"
        elif self.fov_estimator is not None:
            self._log_once(
                "fov_estimator_runtime_notice",
                "debug",
                "Running FOV estimator for frames without provided camera intrinsics",
                component="estimator",
                estimatorName=str(getattr(self.fov_estimator, "name", "fov_estimator")),
            )
            input_image = batch["img_ori"][0].data
            cam_int = self.fov_estimator.get_cam_intrinsics(input_image).to(
                batch["img"]
            )
            batch["cam_int"] = cam_int.clone()
            camera_source = str(getattr(self.fov_estimator, "name", "fov_estimator"))
        else:
            cam_int = batch["cam_int"].clone()

        cam_intrinsics_np: np.ndarray | None = None
        if isinstance(cam_int, torch.Tensor):
            cam_intrinsics_np = cam_int.detach().cpu().numpy()
        elif isinstance(cam_int, np.ndarray):
            cam_intrinsics_np = cam_int.copy()
        if cam_intrinsics_np is not None and cam_intrinsics_np.ndim == 3 and cam_intrinsics_np.shape[0] == 1:
            cam_intrinsics_np = cam_intrinsics_np[0]

        outputs = self.model.run_inference(
            img,
            batch,
            inference_type=inference_type,
            transform_hand=self.transform_hand,
            thresh_wrist_angle=self.thresh_wrist_angle,
        )
        if inference_type == "full":
            pose_output, batch_lhand, batch_rhand, _, _ = outputs
        else:
            pose_output = outputs

        out = pose_output["mhr"]
        out = recursive_to(out, "cpu")
        out = recursive_to(out, "numpy")
        all_out = []
        for idx in range(batch["img"].shape[1]):
            all_out.append(
                {
                    "bbox": batch["bbox"][0, idx].cpu().numpy(),
                    "focal_length": out["focal_length"][idx],
                    "pred_keypoints_3d": out["pred_keypoints_3d"][idx],
                    "pred_keypoints_2d": out["pred_keypoints_2d"][idx],
                    "pred_vertices": out["pred_vertices"][idx],
                    "pred_cam_t": out["pred_cam_t"][idx],
                    "pred_pose_raw": out["pred_pose_raw"][idx],
                    "global_rot": out["global_rot"][idx],
                    "body_pose_params": out["body_pose"][idx],
                    "hand_pose_params": out["hand"][idx],
                    "scale_params": out["scale"][idx],
                    "shape_params": out["shape"][idx],
                    "expr_params": out["face"][idx],
                    "mask": masks[idx] if masks is not None else None,
                    "pred_joint_coords": out["pred_joint_coords"][idx],
                    "pred_global_rots": out["joint_global_rots"][idx],
                    "mhr_model_params": out["mhr_model_params"][idx],
                    "cam_intrinsics": (
                        cam_intrinsics_np.copy()
                        if cam_intrinsics_np is not None
                        else None
                    ),
                    "camera_source": camera_source,
                }
            )

            if inference_type == "full":
                all_out[-1]["lhand_bbox"] = np.array(
                    [
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][0]
                            - batch_lhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][1]
                            - batch_lhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][0]
                            + batch_lhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][1]
                            + batch_lhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                    ]
                )
                all_out[-1]["rhand_bbox"] = np.array(
                    [
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][0]
                            - batch_rhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][1]
                            - batch_rhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][0]
                            + batch_rhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][1]
                            + batch_rhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                    ]
                )

        return all_out

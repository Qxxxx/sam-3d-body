"""
FastAPI application for SAM 3D Body inference service.

Protocol v1.0.0 compliant - Matches cf-backend/docs/pose-correction-protocol.md

Provides endpoints for:
- Health check
- Video inference (/infer/video)
- Alignment and scoring (/infer/alignment)
"""

import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import SAM 3D Body components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from sam_3d_body.video_processor import VideoFrameExtractor, SubjectTracker, create_video_processor
from sam_3d_body.skeleton_processing import (
    SkeletonSequence,
    SkeletonNormalizer,
    TemporalSmoother,
    SkeletonSerializer,
    process_skeleton_sequence,
    COCO17_JOINT_NAMES,
)
from sam_3d_body.alignment_scoring import (
    FastDTWAligner,
    PoseScorer,
    align_and_score,
    PoseCorrectionResult,
)

from api.models import (
    # Enums
    TechniqueType,
    TaskState,
    CameraAngle,
    ErrorSeverity,
    TipCategory,
    # Requests
    VideoInferenceRequest,
    AlignmentRequest,
    BoundingBox,
    # Responses
    HealthResponse,
    VideoInferenceResponse,
    VideoMetadataResponse,
    SkeletonOutput,
    AlignmentResponse,
    PoseCorrectionResponse,
    AlignmentPath,
    FrameErrorDetail,
    JointErrorDetail,
    PhaseScoreDetail,
    CoachingTipDetail,
    KeyFrameDetail,
    ProcessingInfoDetail,
    ErrorResponse,
    FeatureFlagsResponse,
)
from api.config import get_config, get_device, get_gpu_info, FEATURE_FLAGS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
config = get_config()
app = FastAPI(
    title=config.api_title,
    version=config.api_version,
    description="SAM 3D Body Inference API - Protocol v1.0.0 compliant",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance (loaded on startup)
estimator: Optional[SAM3DBodyEstimator] = None
model_loaded: bool = False

# MHR70 to COCO17 joint mapping
# SAM 3D Body outputs 70 joints (MHR70), but Protocol v1.0.0 expects 17 (COCO)
MHR70_TO_COCO17_INDICES = {
    # COCO index: MHR70 index
    0: 0,    # nose -> nose
    1: 1,    # left_eye -> left_eye
    2: 2,    # right_eye -> right_eye
    3: 3,    # left_ear -> left_ear
    4: 4,    # right_ear -> right_ear
    5: 5,    # left_shoulder -> left_shoulder
    6: 6,    # right_shoulder -> right_shoulder
    7: 7,    # left_elbow -> left_elbow
    8: 8,    # right_elbow -> right_elbow
    9: 62,   # left_wrist -> left_wrist (MHR70: 62)
    10: 41,  # right_wrist -> right_wrist (MHR70: 41)
    11: 9,   # left_hip -> left_hip
    12: 10,  # right_hip -> right_hip
    13: 11,  # left_knee -> left_knee
    14: 12,  # right_knee -> right_knee
    15: 13,  # left_ankle -> left_ankle
    16: 14,  # right_ankle -> right_ankle
}


def convert_mhr70_to_coco17(mhr70_joints: np.ndarray) -> np.ndarray:
    """
    Convert MHR70 joints to COCO17 format.

    Args:
        mhr70_joints: Shape (70, 3) or (num_frames, 70, 3)

    Returns:
        COCO17 joints: Shape (17, 3) or (num_frames, 17, 3)
    """
    if mhr70_joints.ndim == 2:
        # Single frame
        coco17 = np.zeros((17, 3), dtype=np.float32)
        for coco_idx, mhr_idx in MHR70_TO_COCO17_INDICES.items():
            coco17[coco_idx] = mhr70_joints[mhr_idx]
        return coco17
    else:
        # Sequence
        num_frames = mhr70_joints.shape[0]
        coco17 = np.zeros((num_frames, 17, 3), dtype=np.float32)
        for coco_idx, mhr_idx in MHR70_TO_COCO17_INDICES.items():
            coco17[:, coco_idx, :] = mhr70_joints[:, mhr_idx, :]
        return coco17


@app.on_event("startup")
async def startup_event():
    """Load SAM 3D Body model on startup."""
    global estimator, model_loaded

    logger.info("Loading SAM 3D Body model...")
    start_time = time.time()

    try:
        device = get_device()
        logger.info(f"Using device: {device}")

        # Check if checkpoint exists
        checkpoint_path = Path(config.model_checkpoint_path)
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found at {checkpoint_path}. Model will not be loaded.")
            logger.warning("Please download checkpoints using: hf download facebook/sam-3d-body-dinov3")
            return

        # Load model
        model, model_cfg = load_sam_3d_body(
            config.model_checkpoint_path,
            device=device,
            mhr_path=config.mhr_model_path if Path(config.mhr_model_path).exists() else None,
        )

        # Initialize detector if available
        human_detector = None
        if config.detector_name:
            try:
                from tools.build_detector import HumanDetector
                human_detector = HumanDetector(
                    name=config.detector_name,
                    device=device,
                    path=config.detector_path if Path(config.detector_path).exists() else "",
                )
                logger.info(f"Loaded detector: {config.detector_name}")
            except Exception as e:
                logger.warning(f"Failed to load detector: {e}")

        # Create estimator
        estimator = SAM3DBodyEstimator(
            sam_3d_body_model=model,
            model_cfg=model_cfg,
            human_detector=human_detector,
        )

        model_loaded = True
        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f}s")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("API will start without model - endpoints will return errors")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down SAM 3D Body API")


# ==================== Endpoints ====================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns service status, GPU information, and model status.
    """
    gpu_info = get_gpu_info()

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        version=config.api_version,
        timestamp=datetime.utcnow(),
        gpu_available=gpu_info["available"],
        gpu_name=gpu_info.get("name"),
        gpu_memory_mb=gpu_info.get("memory_mb"),
        model_loaded=model_loaded,
        model_version="sam-3d-body-1.0.0" if model_loaded else None,
    )


@app.get("/feature-flags", response_model=FeatureFlagsResponse)
async def get_feature_flags():
    """Get current feature flags configuration."""
    return FeatureFlagsResponse(
        enable_smash_analysis=FEATURE_FLAGS["ENABLE_SMASH_ANALYSIS"],
        supported_techniques=FEATURE_FLAGS["SUPPORTED_ACTIONS"],
        support_left_handed=FEATURE_FLAGS["SUPPORT_LEFT_HANDED"],
        support_multi_camera=FEATURE_FLAGS["SUPPORT_MULTI_CAMERA"],
        enable_phase_detection=FEATURE_FLAGS["ENABLE_PHASE_DETECTION"],
        max_video_duration=FEATURE_FLAGS["MAX_VIDEO_DURATION"],
        max_file_size_mb=FEATURE_FLAGS["MAX_FILE_SIZE_MB"],
    )


@app.post("/infer/video", response_model=VideoInferenceResponse)
async def infer_video(request: VideoInferenceRequest):
    """
    Process a video and extract 3D skeleton sequences.

    This endpoint:
    1. Downloads/extracts frames from the video
    2. Runs SAM 3D Body inference on each frame
    3. Converts MHR70 joints to COCO17 format
    4. Applies temporal smoothing and normalization
    5. Returns skeleton data in NPZ + JSON format

    **Supported techniques (MVP):** `smash` only
    """
    if not model_loaded or estimator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs.",
        )

    # Check feature flags
    if request.technique_type.value not in FEATURE_FLAGS["SUPPORTED_ACTIONS"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Technique '{request.technique_type.value}' not supported in MVP. Supported: {FEATURE_FLAGS['SUPPORTED_ACTIONS']}",
        )

    task_id = request.request_id or str(uuid.uuid4())
    start_time = time.time()

    logger.info(f"[Task {task_id}] Starting video inference: {request.video_url}")

    try:
        # Step 1: Extract frames from video
        logger.info(f"[Task {task_id}] Extracting frames at {request.fps} FPS")
        extractor = VideoFrameExtractor(
            target_fps=request.fps,
            temp_dir=config.temp_dir,
        )

        # Convert ms to seconds for time range
        start_sec = request.time_range_start / 1000.0 if request.time_range_start else None
        end_sec = request.time_range_end / 1000.0 if request.time_range_end else None

        frames, video_metadata = extractor.extract_frames(
            request.video_url,
            start_time=start_sec,
            end_time=end_sec,
        )

        logger.info(f"[Task {task_id}] Extracted {len(frames)} frames")

        if len(frames) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="NO_SUBJECT_DETECTED: No frames could be extracted from video",
            )

        # Step 2: Initialize subject tracker if bbox provided
        tracker = None
        if request.subject_bbox:
            # Convert normalized bbox to pixel coordinates
            x = request.subject_bbox.x * video_metadata.width
            y = request.subject_bbox.y * video_metadata.height
            w = request.subject_bbox.width * video_metadata.width
            h = request.subject_bbox.height * video_metadata.height
            bbox = np.array([x, y, x + w, y + h], dtype=np.float32)

            tracker = SubjectTracker(initial_bbox=bbox)
            logger.info(f"[Task {task_id}] Using provided bbox for subject locking")

        # Step 3: Run inference on each frame
        logger.info(f"[Task {task_id}] Running SAM 3D Body inference")

        all_keypoints_3d = []
        timestamps = []

        for i, frame_data in enumerate(frames):
            try:
                # Get bbox for this frame
                bbox = None
                if tracker:
                    # TODO: Run detector and use tracker to select bbox
                    pass

                # Run inference
                outputs = estimator.process_one_image(
                    frame_data.image,
                    bboxes=bbox,
                    bbox_thr=0.5,
                )

                if not outputs:
                    logger.warning(f"[Task {task_id}] No person detected in frame {i}")
                    continue

                # Use first detected person
                person = outputs[0]
                mhr70_keypoints = person["pred_keypoints_3d"]  # (70, 3)

                # Convert to COCO17
                coco17_keypoints = convert_mhr70_to_coco17(mhr70_keypoints)

                all_keypoints_3d.append(coco17_keypoints)
                timestamps.append(frame_data.timestamp)

            except Exception as e:
                logger.warning(f"[Task {task_id}] Failed to process frame {i}: {e}")
                continue

        if len(all_keypoints_3d) == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="NO_SUBJECT_DETECTED: No skeletons could be extracted from video",
            )

        # Stack into sequence
        joints = np.stack(all_keypoints_3d, axis=0).astype(np.float32)
        timestamps_array = np.array(timestamps, dtype=np.float32)

        logger.info(f"[Task {task_id}] Processed {len(all_keypoints_3d)} frames successfully")

        # Step 4: Create skeleton sequence and apply processing
        sequence = SkeletonSequence(
            joints=joints,
            timestamps=timestamps_array,
            joint_names=COCO17_JOINT_NAMES,
        )

        # Apply temporal smoothing
        smoother = TemporalSmoother(
            method="gaussian",
            gaussian_sigma=1.0,
        )
        sequence = smoother.smooth(sequence)

        # Apply normalization
        normalizer = SkeletonNormalizer(version="norm-2024-v1")
        sequence, norm_params = normalizer.normalize(sequence)

        # Step 5: Save outputs
        output_base = Path(config.output_dir) / f"{task_id}"
        serializer = SkeletonSerializer()
        npz_path, json_path = serializer.save(
            sequence,
            output_base,
            source_video=request.video_url,
            extraction_model="sam-3d-body-dinov3",
        )

        logger.info(f"[Task {task_id}] Saved outputs to {npz_path}, {json_path}")

        # Calculate processing time
        processing_time = time.time() - start_time

        return VideoInferenceResponse(
            task_id=task_id,
            status=TaskState.SUCCEEDED,
            technique_type=request.technique_type,
            video_metadata=VideoMetadataResponse(
                width=video_metadata.width,
                height=video_metadata.height,
                fps=video_metadata.fps,
                total_frames=video_metadata.total_frames,
                duration=video_metadata.duration,
                extracted_frames=video_metadata.extracted_frames,
            ),
            skeleton=SkeletonOutput(
                npz_url=f"file://{npz_path}",
                json_url=f"file://{json_path}",
                num_frames=sequence.num_frames,
                num_joints=17,
                joint_names=sequence.joint_names,
                timestamps=sequence.timestamps.tolist(),
            ),
            normalization_version=norm_params.version,
            processing_time_seconds=processing_time,
            created_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Task {task_id}] Video inference failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"INFERENCE_FAILED: {str(e)}",
        )


@app.post("/infer/alignment", response_model=AlignmentResponse)
async def infer_alignment(request: AlignmentRequest):
    """
    Align user skeleton sequence with reference and generate similarity scores.

    This endpoint:
    1. Loads user and reference skeleton sequences
    2. Aligns them using FastDTW
    3. Generates similarity scores and error analysis per Protocol v1.0.0
    4. Returns improvement tips
    """
    task_id = str(uuid.uuid4())
    start_time = time.time()

    logger.info(f"[Task {task_id}] Starting alignment")

    try:
        # Step 1: Load skeleton sequences
        logger.info(f"[Task {task_id}] Loading skeleton sequences")

        serializer = SkeletonSerializer()

        # Load user skeleton
        user_npz = request.video_url.replace("file://", "").replace(".mp4", ".npz")  # Hack for testing
        if not user_npz.endswith(".npz"):
            user_npz = user_npz + ".npz"
        user_json = user_npz.replace(".npz", ".json")

        # In production, these would be actual skeleton URLs
        # For now, assume they're already processed
        if not Path(user_npz).exists():
            # Return mock result for testing
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="VIDEO_NOT_FOUND: User skeleton not found. Process video first.",
            )

        user_sequence = serializer.load(user_npz, user_json)

        # Load reference skeleton
        ref_npz = request.reference_skeleton_url.replace("file://", "")
        ref_json = ref_npz.replace(".npz", ".json")
        reference_sequence = serializer.load(ref_npz, ref_json)

        logger.info(
            f"[Task {task_id}] Loaded user={user_sequence.num_frames}frames, "
            f"reference={reference_sequence.num_frames}frames"
        )

        # Step 2: Align and score
        logger.info(f"[Task {task_id}] Running alignment and scoring")
        result: PoseCorrectionResult = align_and_score(
            user_sequence,
            reference_sequence,
            timestamps=user_sequence.timestamps,
            detect_phases=request.detect_phases,
        )

        # Convert to Pydantic models
        alignment_path = AlignmentPath(
            type=result.alignment_path.type,
            pairs=result.alignment_path.pairs,
            total_cost=result.alignment_path.total_cost,
            average_distance=result.alignment_path.average_distance,
        )

        frame_errors = [
            FrameErrorDetail(
                user_frame_index=fe.user_frame_index,
                ref_frame_index=fe.ref_frame_index,
                timestamp=fe.timestamp,
                error_score=fe.error_score,
                max_deviation_joint=fe.max_deviation_joint,
                max_deviation_distance=fe.max_deviation_distance,
                is_significant=fe.is_significant,
            )
            for fe in result.frame_errors
        ]

        joint_errors = [
            JointErrorDetail(
                joint_name=je.joint_name,
                joint_index=je.joint_index,
                mean_error=je.mean_error,
                max_error=je.max_error,
                error_by_phase=je.error_by_phase,
                ranking=je.ranking,
                severity=ErrorSeverity(je.severity) if je.severity in ["critical", "moderate", "minor", "good"] else ErrorSeverity.MINOR,
            )
            for je in result.joint_errors
        ]

        phase_scores = [
            PhaseScoreDetail(
                phase=ps.phase,
                start_frame=ps.start_frame,
                end_frame=ps.end_frame,
                score=ps.score,
                weight=ps.weight,
                key_joints=ps.key_joints,
                feedback=ps.feedback,
            )
            for ps in result.phase_scores
        ]

        tips = [
            CoachingTipDetail(
                id=tip.id,
                priority=tip.priority,
                category=TipCategory(tip.category) if tip.category in ["timing", "posture", "range_of_motion", "coordination", "balance"] else TipCategory.POSTURE,
                title=tip.title,
                description=tip.description,
                affected_joints=tip.affected_joints,
                affected_phases=tip.affected_phases,
                suggested_drill=tip.suggested_drill,
                reference_frame_index=tip.reference_frame_index,
                user_frame_index=tip.user_frame_index,
            )
            for tip in result.tips
        ]

        key_frames = [
            KeyFrameDetail(
                frame_index=kf.frame_index,
                timestamp=kf.timestamp,
                type=kf.type,
                description=kf.description,
            )
            for kf in result.key_frames
        ]

        processing_info = ProcessingInfoDetail(
            processed_at=result.processing_info.processed_at,
            processing_time_ms=result.processing_info.processing_time_ms,
            model_version=result.processing_info.model_version,
            normalization_version=result.processing_info.normalization_version,
        )

        processing_time = time.time() - start_time

        logger.info(f"[Task {task_id}] Alignment completed: score={result.overall_score:.1f}")

        return AlignmentResponse(
            task_id=task_id,
            status=TaskState.SUCCEEDED,
            result=PoseCorrectionResponse(
                overall_score=result.overall_score,
                similarity_score=result.similarity_score,
                star_rating=result.star_rating,
                alignment_path=alignment_path,
                user_frame_count=result.user_frame_count,
                reference_frame_count=result.reference_frame_count,
                frame_errors=frame_errors,
                joint_errors=joint_errors,
                phase_scores=phase_scores,
                tips=tips,
                key_frames=key_frames,
                processing_info=processing_info,
            ),
            processing_time_seconds=processing_time,
            created_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Task {task_id}] Alignment failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ALIGNMENT_FAILED: {str(e)}",
        )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            code=exc.status_code,
            error_code="HTTP_ERROR",
            message=exc.detail,
            request_id=str(uuid.uuid4()),
            retryable=exc.status_code >= 500,
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            code=500,
            error_code="INTERNAL_ERROR",
            message="An unexpected error occurred",
            request_id=str(uuid.uuid4()),
            retryable=True,
        ).model_dump(),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=config.api_host,
        port=config.api_port,
        workers=config.api_workers,
        reload=False,
    )

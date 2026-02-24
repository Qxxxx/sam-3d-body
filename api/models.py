"""
Pydantic models for the SAM 3D Body inference API.

Protocol v1.0.0 compliant - Matches cf-backend/docs/pose-correction-protocol.md
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, HttpUrl


# ==================== Enums ====================

class TechniqueType(str, Enum):
    """Supported badminton technique types (Protocol v1.0.0)."""
    SMASH = "smash"           # 杀球
    CLEAR = "clear"           # 高远球
    DROP = "drop"             # 吊球
    NET = "net"               # 网前球
    DRIVE = "drive"           # 平抽球
    LIFT = "lift"             # 挑球


class CameraAngle(str, Enum):
    """Camera angle options (Protocol v1.0.0)."""
    BACK = "back"    # Behind player (primary)
    SIDE = "side"    # Side view (secondary)
    FRONT = "front"  # Front view
    AUTO = "auto"    # Auto-detect


class TaskState(str, Enum):
    """Task states (Protocol v1.0.0)."""
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ProcessingStage(str, Enum):
    """Processing stages (Protocol v1.0.0)."""
    WAITING = "waiting"
    DOWNLOADING = "downloading"
    PREPROCESSING = "preprocessing"
    INFERENCE = "inference"
    ALIGNMENT = "alignment"
    SCORING = "scoring"
    GENERATING_TIPS = "generating_tips"
    STORING_RESULTS = "storing_results"


class ErrorSeverity(str, Enum):
    """Severity of pose errors."""
    CRITICAL = "critical"
    MODERATE = "moderate"
    MINOR = "minor"
    GOOD = "good"


class TipCategory(str, Enum):
    """Coaching tip categories (Protocol v1.0.0)."""
    TIMING = "timing"           # 时机问题
    POSTURE = "posture"         # 姿势问题
    RANGE_OF_MOTION = "range_of_motion"  # 动作幅度
    COORDINATION = "coordination"  # 身体协调
    BALANCE = "balance"         # 平衡问题


# ==================== Request Models ====================

class BoundingBox(BaseModel):
    """Bounding box (Protocol v1.0.0)."""
    x: float = Field(..., ge=0, le=1, description="Normalized x coordinate (top-left)")
    y: float = Field(..., ge=0, le=1, description="Normalized y coordinate (top-left)")
    width: float = Field(..., ge=0, le=1, description="Normalized width")
    height: float = Field(..., ge=0, le=1, description="Normalized height")


class VideoInferenceRequest(BaseModel):
    """Request model for video inference endpoint (Protocol v1.0.0)."""

    video_url: str = Field(
        ...,
        description="URL to the video file (http, https, or file://)",
        examples=["file:///data/videos/smash_demo.mp4"],
    )

    # Technique specification
    technique_type: TechniqueType = Field(
        default=TechniqueType.SMASH,
        description="Type of badminton technique being analyzed",
    )

    # Time range in milliseconds (Protocol v1.0.0)
    time_range_start: Optional[float] = Field(
        default=None,
        ge=0,
        description="Start time in milliseconds (None = from beginning)",
    )
    time_range_end: Optional[float] = Field(
        default=None,
        ge=0,
        description="End time in milliseconds (None = to end)",
    )

    # Frame extraction
    fps: float = Field(
        default=30.0,
        gt=0,
        le=60,
        description="Target FPS for processing",
    )

    # Subject locking (Protocol v1.0.0)
    subject_bbox: Optional[BoundingBox] = Field(
        default=None,
        description="Initial subject bounding box",
    )
    subject_tracking_id: Optional[int] = Field(
        default=None,
        description="Tracking ID for multi-person videos",
    )

    # Camera angle
    camera_angle: CameraAngle = Field(
        default=CameraAngle.BACK,
        description="Camera position hint",
    )

    # Request ID for idempotency
    request_id: Optional[str] = Field(
        default=None,
        description="Client-provided idempotency key",
    )

    # Processing options
    detect_phases: bool = Field(
        default=False,
        description="Detect and score motion phases",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "video_url": "file:///data/videos/smash_demo.mp4",
                "technique_type": "smash",
                "time_range_start": 2000,
                "time_range_end": 5000,
                "fps": 30,
                "camera_angle": "back",
            }
        }


class AlignmentRequest(BaseModel):
    """Request model for alignment/scoring endpoint (Protocol v1.0.0)."""

    video_url: str = Field(
        ...,
        description="Presigned URL to user video",
    )
    reference_skeleton_url: str = Field(
        ...,
        description="Presigned URL to reference skeleton (.npz/.json)",
    )

    technique_type: TechniqueType = Field(
        default=TechniqueType.SMASH,
        description="Technique being analyzed",
    )

    fps: float = Field(
        default=30.0,
        description="Target FPS for processing",
    )

    time_range_start: Optional[float] = Field(
        default=None,
        description="Start frame or ms",
    )
    time_range_end: Optional[float] = Field(
        default=None,
        description="End frame or ms",
    )

    subject_bbox: Optional[BoundingBox] = Field(
        default=None,
        description="Initial subject location",
    )

    normalization_params: Optional[Dict] = Field(
        default=None,
        description="Versioned normalization parameters",
    )

    detect_phases: bool = Field(
        default=False,
        description="Detect and score motion phases",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "video_url": "https://example.com/user_video.mp4",
                "reference_skeleton_url": "https://example.com/ref_skeleton.npz",
                "technique_type": "smash",
                "detect_phases": True,
            }
        }


# ==================== Response Models ====================

class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(..., description="Current server timestamp")

    # GPU info
    gpu_available: bool = Field(..., description="Whether GPU is available")
    gpu_name: Optional[str] = Field(None, description="GPU name")
    gpu_memory_mb: Optional[float] = Field(None, description="Available GPU memory in MB")

    # Model status
    model_loaded: bool = Field(..., description="Whether SAM 3D Body model is loaded")
    model_version: Optional[str] = Field(None, description="Model version")


class VideoMetadataResponse(BaseModel):
    """Video metadata in inference response."""

    width: int = Field(..., description="Video width in pixels")
    height: int = Field(..., description="Video height in pixels")
    fps: float = Field(..., description="Original FPS")
    total_frames: int = Field(..., description="Total frames in video")
    duration: float = Field(..., description="Duration in seconds")
    extracted_frames: int = Field(..., description="Number of frames extracted")


class SkeletonOutput(BaseModel):
    """Skeleton output information."""

    npz_url: Optional[str] = Field(None, description="URL to NPZ file with dense data")
    json_url: Optional[str] = Field(None, description="URL to JSON file with metadata")
    num_frames: int = Field(..., description="Number of frames")
    num_joints: int = Field(..., description="Number of joints (17 for Protocol v1.0.0)")
    joint_names: List[str] = Field(..., description="Joint names in order")
    timestamps: List[float] = Field(..., description="Frame timestamps")


class AlignmentPath(BaseModel):
    """DTW alignment path (Protocol v1.0.0)."""

    type: str = Field(default="dtw", description="Alignment type")
    pairs: List[Dict[str, int]] = Field(..., description="Frame pairs {userFrame, refFrame}")
    total_cost: float = Field(..., description="Total DTW cost")
    average_distance: float = Field(..., description="Average distance")


class FrameErrorDetail(BaseModel):
    """Error per aligned frame pair (Protocol v1.0.0)."""

    user_frame_index: int = Field(..., description="User sequence frame index")
    ref_frame_index: int = Field(..., description="Reference sequence frame index")
    timestamp: float = Field(..., description="Seconds from video start")
    error_score: float = Field(..., description="0-1 (0 = perfect match)")
    max_deviation_joint: str = Field(..., description="Joint with highest error")
    max_deviation_distance: float = Field(..., description="Meters")
    is_significant: bool = Field(..., description="Above threshold for highlighting")


class JointErrorDetail(BaseModel):
    """Error per joint (Protocol v1.0.0)."""

    joint_name: str = Field(..., description="Joint name")
    joint_index: int = Field(..., description="Joint index (0-16)")
    mean_error: float = Field(..., description="Average Euclidean distance (meters)")
    max_error: float = Field(..., description="Maximum error (meters)")
    error_by_phase: Dict[str, float] = Field(default_factory=dict, description="Per-phase breakdown")
    ranking: int = Field(..., description="1-17 rank (1 = most problematic)")
    severity: ErrorSeverity = Field(..., description="Error severity")


class PhaseScoreDetail(BaseModel):
    """Score per motion phase (Protocol v1.0.0)."""

    phase: str = Field(..., description="Motion phase name")
    start_frame: int = Field(..., description="Start frame index")
    end_frame: int = Field(..., description="End frame index")
    score: float = Field(..., ge=0, le=100, description="0-100")
    weight: float = Field(..., description="Importance weight")
    key_joints: List[str] = Field(..., description="Joints critical to this phase")
    feedback: str = Field(..., description="Phase-specific feedback")


class CoachingTipDetail(BaseModel):
    """Coaching feedback (Protocol v1.0.0)."""

    id: str = Field(..., description="Tip ID")
    priority: int = Field(..., ge=1, le=10, description="10 = highest")
    category: TipCategory = Field(..., description="Tip category")
    title: str = Field(..., description="Short title (Chinese)")
    description: str = Field(..., description="Detailed explanation")
    affected_joints: List[str] = Field(..., description="Affected joints")
    affected_phases: List[str] = Field(..., description="Affected phases")
    suggested_drill: Optional[str] = Field(None, description="Practice recommendation")
    reference_frame_index: Optional[int] = Field(None, description="Reference frame")
    user_frame_index: Optional[int] = Field(None, description="User frame showing mistake")


class KeyFrameDetail(BaseModel):
    """Significant frames (Protocol v1.0.0)."""

    frame_index: int = Field(..., description="Frame index")
    timestamp: float = Field(..., description="Timestamp in seconds")
    type: str = Field(..., description="impact | max_error | phase_transition | reference_pose")
    description: str = Field(..., description="Human-readable description")


class ProcessingInfoDetail(BaseModel):
    """Processing metadata (Protocol v1.0.0)."""

    processed_at: str = Field(..., description="ISO 8601 timestamp")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_version: str = Field(..., description="Model version")
    normalization_version: str = Field(..., description="Normalization version")


class PoseCorrectionResponse(BaseModel):
    """Complete pose correction result (Protocol v1.0.0)."""

    # Overall scores
    overall_score: float = Field(..., ge=0, le=100, description="0-100 composite score")
    similarity_score: float = Field(..., ge=0, le=1, description="0-1 DTW-based similarity")
    star_rating: float = Field(..., ge=1, le=5, description="1-5 stars")

    # Temporal alignment
    alignment_path: AlignmentPath = Field(..., description="DTW warping path")
    user_frame_count: int = Field(..., description="Number of user frames")
    reference_frame_count: int = Field(..., description="Number of reference frames")

    # Per-frame analysis
    frame_errors: List[FrameErrorDetail] = Field(..., description="Error per aligned frame pair")

    # Per-joint analysis
    joint_errors: List[JointErrorDetail] = Field(..., description="Error per joint")

    # Phase-based analysis
    phase_scores: List[PhaseScoreDetail] = Field(default_factory=list, description="Scores per phase")

    # Coaching feedback
    tips: List[CoachingTipDetail] = Field(..., description="Prioritized improvement suggestions")

    # Key frames
    key_frames: List[KeyFrameDetail] = Field(..., description="Significant frames")

    # Metadata
    processing_info: ProcessingInfoDetail = Field(..., description="Processing metadata")


class VideoInferenceResponse(BaseModel):
    """Response model for video inference endpoint (Protocol v1.0.0)."""

    task_id: str = Field(..., description="Unique task ID")
    status: TaskState = Field(..., description="Task status")

    # Input info
    technique_type: TechniqueType = Field(..., description="Technique type analyzed")

    # Video metadata
    video_metadata: VideoMetadataResponse = Field(..., description="Video metadata")

    # Output skeletons
    skeleton: SkeletonOutput = Field(..., description="Output skeleton data")

    # Normalization info
    normalization_version: Optional[str] = Field(None, description="Normalization version")

    # Processing info
    processing_time_seconds: float = Field(..., description="Total processing time")
    errors: List[str] = Field(default_factory=list, description="Any errors during processing")

    # Timestamps
    created_at: datetime = Field(..., description="Task creation time")
    completed_at: Optional[datetime] = Field(None, description="Task completion time")


class AlignmentResponse(BaseModel):
    """Response model for alignment/scoring endpoint (Protocol v1.0.0)."""

    task_id: str = Field(..., description="Unique task ID")
    status: TaskState = Field(..., description="Task status")

    # Full result
    result: PoseCorrectionResponse = Field(..., description="Complete pose correction result")

    # Processing info
    processing_time_seconds: float = Field(..., description="Processing time")

    # Timestamps
    created_at: datetime = Field(..., description="Task creation time")
    completed_at: Optional[datetime] = Field(None, description="Task completion time")


class ErrorResponse(BaseModel):
    """Error response (Protocol v1.0.0)."""

    code: int = Field(..., description="HTTP status code")
    error_code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable message (bilingual)")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    request_id: str = Field(..., description="For tracing")
    retryable: bool = Field(..., description="Can client retry")
    retry_after_ms: Optional[int] = Field(None, description="Suggested wait time")


# ==================== Feature Flags ====================

class FeatureFlagsResponse(BaseModel):
    """Feature flags configuration."""

    enable_smash_analysis: bool = Field(..., description="Enable smash technique analysis")
    supported_techniques: List[str] = Field(..., description="Supported technique types")
    support_left_handed: bool = Field(..., description="Support left-handed players")
    support_multi_camera: bool = Field(..., description="Support multiple camera angles")
    enable_phase_detection: bool = Field(..., description="Enable automatic phase detection")
    max_video_duration: float = Field(..., description="Maximum video duration in seconds")
    max_file_size_mb: int = Field(..., description="Maximum file size in MB")

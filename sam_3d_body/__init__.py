# Copyright (c) Meta Platforms, Inc. and affiliates.
__version__ = "1.0.0"

from .sam_3d_body_estimator import SAM3DBodyEstimator
from .build_models import load_sam_3d_body, load_sam_3d_body_hf

# Video processing components
try:
    from .video_processor import (
        VideoFrameExtractor,
        SubjectTracker,
        create_video_processor,
        FrameData,
        VideoMetadata,
    )
except ImportError:
    pass

# Skeleton processing components
try:
    from .skeleton_processing import (
        SkeletonSequence,
        SkeletonNormalizer,
        TemporalSmoother,
        SkeletonSerializer,
        process_skeleton_sequence,
        NormalizationParams,
        MHR70_JOINT_NAMES,
    )
except ImportError:
    pass

# Alignment and scoring components
try:
    from .alignment_scoring import (
        FastDTWAligner,
        PoseScorer,
        align_and_score,
        AlignmentResult,
        ScoringResult,
        JointError,
        PoseScore,
    )
except ImportError:
    pass

__all__ = [
    "__version__",
    "load_sam_3d_body",
    "load_sam_3d_body_hf",
    "SAM3DBodyEstimator",
]

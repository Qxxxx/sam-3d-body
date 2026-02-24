"""
Alignment and scoring module for pose correction.

Provides FastDTW-based sequence alignment and pose similarity scoring.
Protocol v1.0.0 compliant.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from fastdtw import fastdtw
from numpy.typing import NDArray
from scipy.spatial.distance import euclidean

from .skeleton_processing import SkeletonSequence, COCO17_JOINT_NAMES, COCO17_JOINT_INDICES

logger = logging.getLogger(__name__)


@dataclass
class AlignmentPath:
    """DTW alignment path (Protocol v1.0.0 format)."""
    type: str = "dtw"
    pairs: List[Dict[str, int]] = field(default_factory=list)
    total_cost: float = 0.0
    average_distance: float = 0.0


@dataclass
class AlignmentResult:
    """Result of sequence alignment."""

    # Alignment path: list of (query_idx, reference_idx) pairs
    path: List[Tuple[int, int]]

    # Warped sequences (aligned in time)
    query_warped: NDArray[np.float32]
    reference_warped: NDArray[np.float32]

    # DTW distance
    dtw_distance: float

    # Per-frame distances after alignment
    frame_distances: NDArray[np.float32]

    # Per-joint distances after alignment
    joint_distances: NDArray[np.float32]  # Shape: (num_aligned_frames, num_joints)

    def to_protocol_format(self) -> AlignmentPath:
        """Convert to Protocol v1.0.0 format."""
        pairs = [
            {"userFrame": q, "refFrame": r}
            for q, r in self.path
        ]
        return AlignmentPath(
            type="dtw",
            pairs=pairs,
            total_cost=self.dtw_distance,
            average_distance=float(np.mean(self.frame_distances)) if len(self.frame_distances) > 0 else 0.0,
        )


@dataclass
class JointError:
    """Error information for a specific joint (Protocol v1.0.0)."""

    joint_name: str
    joint_index: int
    mean_error: float  # Average Euclidean distance (meters)
    max_error: float
    error_by_phase: Dict[str, float]  # Per-phase breakdown
    ranking: int  # 1-17 rank (1 = most problematic)
    error_timeline: List[float]  # Error at each aligned frame
    severity: str  # "critical", "moderate", "minor", "good"


@dataclass
class FrameError:
    """Error per aligned frame pair (Protocol v1.0.0)."""

    user_frame_index: int
    ref_frame_index: int
    timestamp: float  # Seconds from video start
    error_score: float  # 0-1 (0 = perfect match)
    max_deviation_joint: str  # Joint with highest error
    max_deviation_distance: float  # Meters
    is_significant: bool  # Above threshold for highlighting


@dataclass
class PhaseScore:
    """Score for a specific motion phase (Protocol v1.0.0)."""

    phase: str  # MotionPhase
    start_frame: int
    end_frame: int
    score: float  # 0-100
    weight: float  # Importance weight
    key_joints: List[str]  # Joints critical to this phase
    feedback: str  # Phase-specific feedback


@dataclass
class CoachingTip:
    """Coaching feedback (Protocol v1.0.0)."""

    id: str
    priority: int  # 1-10 (10 = highest)
    category: str  # TipCategory
    title: str  # Short title (Chinese)
    description: str  # Detailed explanation
    affected_joints: List[str]
    affected_phases: List[str]
    suggested_drill: Optional[str] = None
    reference_frame_index: Optional[int] = None
    user_frame_index: Optional[int] = None


@dataclass
class KeyFrame:
    """Significant frames to highlight (Protocol v1.0.0)."""

    frame_index: int
    timestamp: float
    type: str  # 'impact' | 'max_error' | 'phase_transition' | 'reference_pose'
    description: str


@dataclass
class PoseScore:
    """Score for a specific pose aspect."""

    name: str
    score: float  # 0-100
    weight: float  # Contribution to overall score
    description: str


@dataclass
class ProcessingInfo:
    """Processing metadata (Protocol v1.0.0)."""

    processed_at: str  # ISO 8601
    processing_time_ms: float
    model_version: str
    normalization_version: str


@dataclass
class PoseCorrectionResult:
    """Complete scoring result for pose correction (Protocol v1.0.0)."""

    # Overall scores
    overall_score: float  # 0-100 composite score
    similarity_score: float  # 0-1 DTW-based similarity
    star_rating: float  # 1-5 stars

    # Temporal alignment
    alignment_path: AlignmentPath  # DTW warping path
    user_frame_count: int
    reference_frame_count: int

    # Per-frame analysis
    frame_errors: List[FrameError]  # Error per aligned frame pair

    # Per-joint analysis
    joint_errors: List[JointError]  # Error per joint across sequence

    # Phase-based analysis
    phase_scores: List[PhaseScore]  # Scores per motion phase

    # Coaching feedback
    tips: List[CoachingTip]  # Prioritized improvement suggestions

    # Key frames for visualization
    key_frames: List[KeyFrame]  # Significant frames to highlight

    # Metadata
    processing_info: ProcessingInfo


class FastDTWAligner:
    """
    Aligns two 3D skeleton sequences using FastDTW algorithm.

    FastDTW provides a good balance between accuracy and computational efficiency
    for sequences with temporal variations.
    """

    def __init__(
        self,
        radius: int = 1,
        distance_metric: str = "euclidean",
    ):
        """
        Initialize FastDTW aligner.

        Args:
            radius: Radius for FastDTW local constraint (higher = more accurate but slower)
            distance_metric: Distance metric to use
        """
        self.radius = radius
        self.distance_metric = distance_metric

    def align(
        self,
        query_sequence: SkeletonSequence,
        reference_sequence: SkeletonSequence,
        joint_weights: Optional[NDArray[np.float32]] = None,
    ) -> AlignmentResult:
        """
        Align query sequence to reference sequence.

        Args:
            query_sequence: User's skeleton sequence
            reference_sequence: Professional reference sequence
            joint_weights: Optional weights for each joint (num_joints,)

        Returns:
            AlignmentResult with warped sequences and distances
        """
        # Prepare sequences: flatten joints per frame for DTW
        query_flat = self._flatten_sequence(query_sequence.joints)
        reference_flat = self._flatten_sequence(reference_sequence.joints)

        logger.info(
            f"Aligning sequences: query={query_flat.shape}, reference={reference_flat.shape}"
        )

        # Compute FastDTW alignment
        distance, path = fastdtw(
            query_flat,
            reference_flat,
            radius=self.radius,
            dist=euclidean,
        )

        path = [(int(p[0]), int(p[1])) for p in path]

        # Warp sequences according to alignment path
        query_warped, reference_warped = self._warp_sequences(
            query_sequence.joints,
            reference_sequence.joints,
            path,
        )

        # Compute per-frame and per-joint distances
        frame_distances, joint_distances = self._compute_distances(
            query_warped,
            reference_warped,
            joint_weights,
        )

        return AlignmentResult(
            path=path,
            query_warped=query_warped,
            reference_warped=reference_warped,
            dtw_distance=distance,
            frame_distances=frame_distances,
            joint_distances=joint_distances,
        )

    def _flatten_sequence(self, joints: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Flatten joint dimensions for DTW.

        Args:
            joints: Shape (num_frames, num_joints, 3)

        Returns:
            Flattened array of shape (num_frames, num_joints * 3)
        """
        return joints.reshape(joints.shape[0], -1)

    def _warp_sequences(
        self,
        query_joints: NDArray[np.float32],
        reference_joints: NDArray[np.float32],
        path: List[Tuple[int, int]],
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Warp sequences according to alignment path.

        Args:
            query_joints: Shape (num_query_frames, num_joints, 3)
            reference_joints: Shape (num_ref_frames, num_joints, 3)
            path: Alignment path as list of (query_idx, ref_idx) pairs

        Returns:
            Tuple of (query_warped, reference_warped) with aligned time dimension
        """
        query_warped = []
        reference_warped = []

        for q_idx, r_idx in path:
            query_warped.append(query_joints[q_idx])
            reference_warped.append(reference_joints[r_idx])

        return (
            np.array(query_warped, dtype=np.float32),
            np.array(reference_warped, dtype=np.float32),
        )

    def _compute_distances(
        self,
        query_warped: NDArray[np.float32],
        reference_warped: NDArray[np.float32],
        joint_weights: Optional[NDArray[np.float32]] = None,
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Compute per-frame and per-joint distances.

        Args:
            query_warped: Warped query joints (num_aligned_frames, num_joints, 3)
            reference_warped: Warped reference joints (num_aligned_frames, num_joints, 3)
            joint_weights: Optional joint weights

        Returns:
            Tuple of (frame_distances, joint_distances)
        """
        diff = query_warped - reference_warped  # (num_frames, num_joints, 3)

        # Per-joint distances per frame
        joint_distances = np.linalg.norm(diff, axis=2)  # (num_frames, num_joints)

        # Apply weights if provided
        if joint_weights is not None:
            joint_distances = joint_distances * joint_weights[np.newaxis, :]

        # Per-frame distances (average across joints)
        frame_distances = np.mean(joint_distances, axis=1)

        return frame_distances, joint_distances


class PoseScorer:
    """
    Generates similarity scores and identifies errors from alignment results.
    Protocol v1.0.0 compliant.
    """

    # Error thresholds in meters (normalized skeleton units)
    CRITICAL_THRESHOLD = 0.15  # 15cm
    MODERATE_THRESHOLD = 0.08  # 8cm
    MINOR_THRESHOLD = 0.04  # 4cm

    # Joint importance weights for badminton smash (Protocol v1.0.0 - 17 joints)
    JOINT_WEIGHTS = {
        # Core body (high importance)
        "right_shoulder": 1.5,
        "left_shoulder": 1.2,
        "right_elbow": 1.5,
        "left_elbow": 1.2,
        "right_wrist": 1.8,
        "left_wrist": 1.2,
        "right_hip": 1.3,
        "left_hip": 1.3,
        "right_knee": 1.2,
        "left_knee": 1.2,
        "right_ankle": 1.0,
        "left_ankle": 1.0,
        # Head (lower importance)
        "nose": 0.5,
        "left_eye": 0.3,
        "right_eye": 0.3,
        "left_ear": 0.3,
        "right_ear": 0.3,
        # Default
        "_default": 0.7,
    }

    # Motion phases (Protocol v1.0.0)
    MOTION_PHASES = [
        "preparation",
        "backswing",
        "power_generation",
        "impact",
        "follow_through",
    ]

    def __init__(self):
        self.aligner = FastDTWAligner()

    def score(
        self,
        user_sequence: SkeletonSequence,
        reference_sequence: SkeletonSequence,
        timestamps: Optional[NDArray[np.float32]] = None,
        detect_phases: bool = False,
    ) -> PoseCorrectionResult:
        """
        Generate comprehensive scoring result per Protocol v1.0.0.

        Args:
            user_sequence: User's skeleton sequence
            reference_sequence: Professional reference sequence
            timestamps: Optional frame timestamps
            detect_phases: Whether to detect and score phases

        Returns:
            PoseCorrectionResult with all protocol fields
        """
        start_time = logging.getLogger().handlers[0].baseFilename if logging.getLogger().handlers else None
        import time
        processing_start = time.time()

        # Prepare joint weights array
        joint_weights = self._get_joint_weights_array(user_sequence.joint_names)

        # Align sequences
        alignment = self.aligner.align(user_sequence, reference_sequence, joint_weights)

        # Compute overall score
        overall_score = self._compute_overall_score(alignment.frame_distances)
        similarity_score = self._compute_similarity_score(alignment.dtw_distance)
        star_rating = self._compute_star_rating(overall_score)

        # Compute per-frame errors
        frame_errors = self._compute_frame_errors(
            alignment,
            timestamps if timestamps is not None else user_sequence.timestamps,
        )

        # Compute per-joint errors
        joint_errors = self._compute_joint_errors(
            alignment.joint_distances,
            user_sequence.joint_names,
        )

        # Detect phases if requested
        phase_scores = []
        if detect_phases:
            phase_scores = self._detect_phases_and_score(
                alignment,
                user_sequence,
                reference_sequence,
            )

        # Generate coaching tips
        tips = self._generate_coaching_tips(joint_errors, phase_scores)

        # Identify key frames
        key_frames = self._identify_key_frames(alignment, frame_errors)

        processing_time = (time.time() - processing_start) * 1000

        from datetime import datetime
        processing_info = ProcessingInfo(
            processed_at=datetime.utcnow().isoformat() + "Z",
            processing_time_ms=processing_time,
            model_version="sam-3d-body-1.0.0",
            normalization_version=user_sequence.normalization_params.version if user_sequence.normalization_params else "none",
        )

        return PoseCorrectionResult(
            overall_score=overall_score,
            similarity_score=similarity_score,
            star_rating=star_rating,
            alignment_path=alignment.to_protocol_format(),
            user_frame_count=user_sequence.num_frames,
            reference_frame_count=reference_sequence.num_frames,
            frame_errors=frame_errors,
            joint_errors=joint_errors,
            phase_scores=phase_scores,
            tips=tips,
            key_frames=key_frames,
            processing_info=processing_info,
        )

    def _get_joint_weights_array(self, joint_names: List[str]) -> NDArray[np.float32]:
        """Convert joint weight dictionary to array."""
        weights = np.array(
            [self.JOINT_WEIGHTS.get(name, self.JOINT_WEIGHTS["_default"]) for name in joint_names],
            dtype=np.float32,
        )
        # Normalize to sum to num_joints
        weights = weights * len(weights) / weights.sum()
        return weights

    def _compute_overall_score(self, frame_distances: NDArray[np.float32]) -> float:
        """Compute overall similarity score from frame distances."""
        # Convert distances to scores (inverse relationship)
        # Use exponential decay: score = 100 * exp(-distance / scale)
        mean_distance = np.mean(frame_distances)
        score = 100 * np.exp(-mean_distance / 0.1)  # 0.1m scale factor
        return float(np.clip(score, 0, 100))

    def _compute_similarity_score(self, dtw_distance: float) -> float:
        """Compute DTW-based similarity score (0-1)."""
        # Normalize DTW distance to similarity
        similarity = np.exp(-dtw_distance / 10.0)
        return float(np.clip(similarity, 0, 1))

    def _compute_star_rating(self, overall_score: float) -> float:
        """Convert score to 1-5 star rating."""
        if overall_score >= 90:
            return 5.0
        elif overall_score >= 75:
            return 4.0
        elif overall_score >= 60:
            return 3.0
        elif overall_score >= 40:
            return 2.0
        else:
            return 1.0

    def _compute_frame_errors(
        self,
        alignment: AlignmentResult,
        timestamps: NDArray[np.float32],
    ) -> List[FrameError]:
        """Compute per-frame error details."""
        errors = []
        threshold = np.mean(alignment.frame_distances) + np.std(alignment.frame_distances)

        for i, (q_idx, r_idx) in enumerate(alignment.path):
            joint_distances = alignment.joint_distances[i]
            max_joint_idx = int(np.argmax(joint_distances))
            max_distance = float(joint_distances[max_joint_idx])

            errors.append(FrameError(
                user_frame_index=q_idx,
                ref_frame_index=r_idx,
                timestamp=float(timestamps[q_idx]) if q_idx < len(timestamps) else 0.0,
                error_score=float(alignment.frame_distances[i]),
                max_deviation_joint=COCO17_JOINT_NAMES[max_joint_idx],
                max_deviation_distance=max_distance,
                is_significant=alignment.frame_distances[i] > threshold,
            ))

        return errors

    def _compute_joint_errors(
        self,
        joint_distances: NDArray[np.float32],
        joint_names: List[str],
    ) -> List[JointError]:
        """Compute per-joint error statistics."""
        errors = []

        for i, joint_name in enumerate(joint_names):
            distances = joint_distances[:, i]
            mean_error = float(np.mean(distances))
            max_error = float(np.max(distances))

            # Determine severity
            if mean_error > self.CRITICAL_THRESHOLD:
                severity = "critical"
            elif mean_error > self.MODERATE_THRESHOLD:
                severity = "moderate"
            elif mean_error > self.MINOR_THRESHOLD:
                severity = "minor"
            else:
                severity = "good"

            errors.append(
                JointError(
                    joint_name=joint_name,
                    joint_index=i,
                    mean_error=mean_error,
                    max_error=max_error,
                    error_by_phase={},  # Populated if phase detection enabled
                    ranking=0,  # Will be set after sorting
                    error_timeline=[float(d) for d in distances],
                    severity=severity,
                )
            )

        # Sort by mean error (descending) and set rankings
        errors.sort(key=lambda e: e.mean_error, reverse=True)
        for i, error in enumerate(errors):
            error.ranking = i + 1

        return errors

    def _detect_phases_and_score(
        self,
        alignment: AlignmentResult,
        user_sequence: SkeletonSequence,
        reference_sequence: SkeletonSequence,
    ) -> List[PhaseScore]:
        """
        Detect motion phases and score each phase.
        """
        # Compute velocity magnitude for phase detection
        user_velocities = self._compute_velocities(user_sequence.joints)

        phases = []
        try:
            # Find peak velocity frame (impact)
            peak_frame = int(np.argmax(user_velocities))
            total_frames = len(user_velocities)

            # Simple phase detection based on velocity profile
            phase_boundaries = self._estimate_phase_boundaries(peak_frame, total_frames)

            for phase_name, (start, end) in phase_boundaries.items():
                if start < end and end <= len(alignment.frame_distances):
                    phase_distances = alignment.frame_distances[start:end]
                    score = self._score_phase(phase_distances)

                    phases.append(PhaseScore(
                        phase=phase_name,
                        start_frame=start,
                        end_frame=end,
                        score=score,
                        weight=self._get_phase_weight(phase_name),
                        key_joints=self._get_phase_key_joints(phase_name),
                        feedback=self._get_phase_feedback(phase_name, score),
                    ))
        except Exception as e:
            logger.warning(f"Phase detection failed: {e}")

        return phases

    def _compute_velocities(self, joints: NDArray[np.float32]) -> NDArray[np.float32]:
        """Compute velocity magnitude for each frame."""
        velocities = np.zeros(joints.shape[0], dtype=np.float32)

        for i in range(1, joints.shape[0]):
            diff = joints[i] - joints[i - 1]
            velocities[i] = np.linalg.norm(diff)

        return velocities

    def _estimate_phase_boundaries(
        self,
        peak_frame: int,
        total_frames: int,
    ) -> Dict[str, Tuple[int, int]]:
        """Estimate phase boundaries based on impact frame."""
        # Simple heuristic-based phase boundaries
        preparation_end = int(peak_frame * 0.2)
        backswing_end = int(peak_frame * 0.5)
        power_end = peak_frame
        impact_end = min(peak_frame + 3, total_frames)
        follow_end = total_frames

        return {
            "preparation": (0, preparation_end),
            "backswing": (preparation_end, backswing_end),
            "power_generation": (backswing_end, power_end),
            "impact": (power_end, impact_end),
            "follow_through": (impact_end, follow_end),
        }

    def _score_phase(self, phase_distances: NDArray[np.float32]) -> float:
        """Compute score for a specific phase."""
        if len(phase_distances) == 0:
            return 50.0
        mean_dist = np.mean(phase_distances)
        score = 100 * np.exp(-mean_dist / 0.1)
        return float(np.clip(score, 0, 100))

    def _get_phase_weight(self, phase: str) -> float:
        """Get importance weight for a phase."""
        weights = {
            "preparation": 0.15,
            "backswing": 0.20,
            "power_generation": 0.25,
            "impact": 0.25,
            "follow_through": 0.15,
        }
        return weights.get(phase, 0.2)

    def _get_phase_key_joints(self, phase: str) -> List[str]:
        """Get critical joints for a phase."""
        phase_joints = {
            "preparation": ["left_hip", "right_hip", "left_knee", "right_knee"],
            "backswing": ["right_shoulder", "right_elbow", "right_wrist"],
            "power_generation": ["right_shoulder", "right_elbow", "right_wrist", "right_hip"],
            "impact": ["right_wrist", "right_elbow", "nose"],
            "follow_through": ["right_shoulder", "left_shoulder", "right_wrist"],
        }
        return phase_joints.get(phase, [])

    def _get_phase_feedback(self, phase: str, score: float) -> str:
        """Generate phase-specific feedback."""
        if score >= 80:
            return f"{phase} 阶段表现良好"
        elif score >= 60:
            return f"{phase} 阶段需要一些调整"
        else:
            return f"{phase} 阶段需要重点改进"

    def _generate_coaching_tips(
        self,
        joint_errors: List[JointError],
        phase_scores: List[PhaseScore],
    ) -> List[CoachingTip]:
        """Generate coaching tips based on errors."""
        tips = []
        tip_id = 0

        # Generate tips from critical errors
        critical_errors = [e for e in joint_errors if e.severity == "critical"][:3]
        for error in critical_errors:
            tip_id += 1
            tip = self._create_joint_tip(error, tip_id)
            if tip:
                tips.append(tip)

        # Generate tips from low phase scores
        if phase_scores:
            low_phases = [p for p in phase_scores if p.score < 60][:2]
            for phase in low_phases:
                tip_id += 1
                tips.append(CoachingTip(
                    id=f"tip_{tip_id:03d}",
                    priority=8,
                    category="timing" if phase.phase in ["impact", "power_generation"] else "posture",
                    title=f"{phase.phase} 阶段需要改进",
                    description=phase.feedback,
                    affected_joints=phase.key_joints,
                    affected_phases=[phase.phase],
                    suggested_drill=f"练习{phase.phase}阶段的动作",
                ))

        # Sort by priority
        tips.sort(key=lambda t: t.priority, reverse=True)

        return tips

    def _create_joint_tip(self, error: JointError, tip_id: int) -> Optional[CoachingTip]:
        """Create a coaching tip for a joint error."""
        # Joint-specific tip templates
        templates = {
            "right_elbow": {
                "title": "引拍时手肘角度过大",
                "description": "在引拍阶段，保持手肘在适当角度（约90度）可以产生更大的挥拍力量。尝试在准备姿势时调整手肘位置。",
                "category": "posture",
                "drill": "对着镜子练习引拍动作，观察手肘角度",
            },
            "right_wrist": {
                "title": "击球时手腕不够放松",
                "description": "击球瞬间保持手腕放松并自然内旋（甩腕）可以增加击球力量和控制。避免手腕过于僵硬。",
                "category": "coordination",
                "drill": "多球练习：专注于击球时的手腕动作",
            },
            "right_shoulder": {
                "title": "转体不充分",
                "description": "杀球需要充分利用身体的旋转力量。尝试在击球前更充分地转动肩膀和髋部。",
                "category": "range_of_motion",
                "drill": "无拍练习：专注于转体动作，感受力量的传递",
            },
            "right_knee": {
                "title": "起跳时机或姿势需要调整",
                "description": "杀球时的起跳姿势影响整体发力。确保起跳前膝盖适度弯曲，起跳时向上发力。",
                "category": "balance",
                "drill": "跳绳和深蹲练习，提高腿部爆发力",
            },
            "left_shoulder": {
                "title": "非持拍手位置需要调整",
                "description": "非持拍手（左手）在杀球时起到平衡作用。保持左手在身体前方可以帮助维持平衡。",
                "category": "balance",
                "drill": "练习时注意观察左手的位置",
            },
        }

        if error.joint_name not in templates:
            return None

        t = templates[error.joint_name]
        return CoachingTip(
            id=f"tip_{tip_id:03d}",
            priority=10 if error.severity == "critical" else 7,
            category=t["category"],
            title=t["title"],
            description=t["description"],
            affected_joints=[error.joint_name],
            affected_phases=["backswing", "power_generation", "impact"],
            suggested_drill=t["drill"],
        )

    def _identify_key_frames(
        self,
        alignment: AlignmentResult,
        frame_errors: List[FrameError],
    ) -> List[KeyFrame]:
        """Identify significant frames for visualization."""
        key_frames = []

        # Find impact frame (highest velocity/change)
        # For now, use middle of sequence as approximation
        if len(frame_errors) > 0:
            impact_idx = len(frame_errors) // 2
            key_frames.append(KeyFrame(
                frame_index=frame_errors[impact_idx].user_frame_index,
                timestamp=frame_errors[impact_idx].timestamp,
                type="impact",
                description="击球瞬间",
            ))

        # Find max error frame
        max_error_frame = max(frame_errors, key=lambda f: f.error_score)
        key_frames.append(KeyFrame(
            frame_index=max_error_frame.user_frame_index,
            timestamp=max_error_frame.timestamp,
            type="max_error",
            description=f"最大偏差位置: {max_error_frame.max_deviation_joint}",
        ))

        return key_frames


def align_and_score(
    user_sequence: SkeletonSequence,
    reference_sequence: SkeletonSequence,
    timestamps: Optional[NDArray[np.float32]] = None,
    detect_phases: bool = False,
) -> PoseCorrectionResult:
    """
    Convenience function for full alignment and scoring pipeline.

    Args:
        user_sequence: User's skeleton sequence
        reference_sequence: Professional reference sequence
        timestamps: Optional frame timestamps
        detect_phases: Whether to detect motion phases

    Returns:
        PoseCorrectionResult per Protocol v1.0.0
    """
    scorer = PoseScorer()
    return scorer.score(user_sequence, reference_sequence, timestamps, detect_phases)

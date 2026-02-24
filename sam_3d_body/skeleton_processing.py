"""
3D skeleton processing module for SAM 3D Body inference pipeline.

Handles normalization, temporal smoothing, and serialization of skeleton data.

Protocol v1.0.0 compliant - Uses 17-joint COCO-inspired format.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)

# 17-joint COCO-inspired format (Protocol v1.0.0)
COCO17_JOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# Joint indices for quick lookup
COCO17_JOINT_INDICES = {name: idx for idx, name in enumerate(COCO17_JOINT_NAMES)}


@dataclass
class NormalizationParams:
    """Parameters for skeleton normalization (versioned)."""

    version: str = "norm-2024-v1"
    translation: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    scale_factor: float = 1.0
    rotation_matrix: NDArray[np.float32] = field(
        default_factory=lambda: np.eye(3, dtype=np.float32)
    )

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "version": self.version,
            "translation": {
                "x": float(self.translation[0]),
                "y": float(self.translation[1]),
                "z": float(self.translation[2]),
            },
            "scaleFactor": float(self.scale_factor),
            "rotationMatrix": self.rotation_matrix.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "NormalizationParams":
        """Create from dictionary."""
        t = data["translation"]
        return cls(
            version=data["version"],
            translation=np.array([t["x"], t["y"], t["z"]], dtype=np.float32),
            scale_factor=data["scaleFactor"],
            rotation_matrix=np.array(data["rotationMatrix"], dtype=np.float32),
        )


@dataclass
class SkeletonSequence:
    """Container for a sequence of 3D skeletons.

    Protocol v1.0.0: Uses 17-joint COCO-inspired format.
    """

    joints: NDArray[np.float32]  # Shape: (num_frames, 17, 3)
    timestamps: NDArray[np.float32]  # Shape: (num_frames,)
    confidences: Optional[NDArray[np.float32]] = None  # Shape: (num_frames, 17)
    joint_names: List[str] = field(default_factory=lambda: COCO17_JOINT_NAMES.copy())
    metadata: Dict = field(default_factory=dict)
    normalization_params: Optional[NormalizationParams] = None

    def __post_init__(self):
        """Validate shapes."""
        num_frames, num_joints, dims = self.joints.shape
        assert dims == 3, f"Expected 3D joints, got {dims}D"
        assert num_joints == 17, f"Protocol v1.0.0 requires 17 joints, got {num_joints}"
        assert len(self.timestamps) == num_frames, "Timestamp count must match frame count"
        assert len(self.joint_names) == num_joints, "Joint name count must match joint count"

        if self.confidences is not None:
            assert self.confidences.shape == (num_frames, num_joints), \
                "Confidence shape must match joints shape"

    @property
    def num_frames(self) -> int:
        return self.joints.shape[0]

    @property
    def num_joints(self) -> int:
        return self.joints.shape[1]

    def get_joint(self, name: str) -> Optional[NDArray[np.float32]]:
        """Get trajectory for a specific joint by name."""
        if name not in self.joint_names:
            return None
        idx = self.joint_names.index(name)
        return self.joints[:, idx, :]

    def copy(self) -> "SkeletonSequence":
        """Create a deep copy."""
        return SkeletonSequence(
            joints=self.joints.copy(),
            timestamps=self.timestamps.copy(),
            confidences=self.confidences.copy() if self.confidences is not None else None,
            joint_names=self.joint_names.copy(),
            metadata=self.metadata.copy(),
            normalization_params=self.normalization_params,
        )

    def to_protocol_format(self) -> List[Dict]:
        """Convert to Protocol v1.0.0 JSON format."""
        frames = []
        for i in range(self.num_frames):
            keypoints = []
            for j, name in enumerate(self.joint_names):
                x, y, z = self.joints[i, j]
                c = float(self.confidences[i, j]) if self.confidences is not None else 1.0
                keypoints.extend([float(x), float(y), float(z), c])

            frames.append({
                "frameIndex": i,
                "timestamp": float(self.timestamps[i]),
                "keypoints": keypoints,  # [x0, y0, z0, c0, x1, y1, z1, c1, ...]
            })
        return frames


class SkeletonNormalizer:
    """
    Normalizes 3D skeleton sequences for consistent comparison.

    Protocol v1.0.0 Normalization:
    1. Translation: Center at pelvis origin (midpoint of hips)
    2. Scale: Normalize to unit height (pelvis to nose distance = 0.25 units)
    3. Rotation: Align facing direction to +Z axis

    Coordinate System:
    - Origin: Subject's pelvis midpoint (average of left_hip and right_hip)
    - X-axis: Right (subject's right is positive)
    - Y-axis: Up (toward sky is positive)
    - Z-axis: Forward (subject facing direction is positive)
    - Units: Meters (m)
    """

    def __init__(self, version: str = "norm-2024-v1"):
        self.version = version

    def normalize(
        self,
        sequence: SkeletonSequence,
        target_facing_direction: Optional[NDArray[np.float32]] = None,
    ) -> Tuple[SkeletonSequence, NormalizationParams]:
        """
        Normalize a skeleton sequence per Protocol v1.0.0.

        Args:
            sequence: Input skeleton sequence
            target_facing_direction: Target facing direction (default: +Z)

        Returns:
            Tuple of (normalized sequence, normalization parameters)
        """
        joints = sequence.joints.copy()

        # Step 1: Translation normalization (center at pelvis)
        left_hip_idx = COCO17_JOINT_INDICES["left_hip"]
        right_hip_idx = COCO17_JOINT_INDICES["right_hip"]
        pelvis = (joints[:, left_hip_idx, :] + joints[:, right_hip_idx, :]) / 2.0

        # Center all joints relative to pelvis
        joints = joints - pelvis[:, np.newaxis, :]
        translation = -pelvis[0]  # Store initial translation

        # Step 2: Scale normalization (unit height = 0.25)
        nose_idx = COCO17_JOINT_INDICES["nose"]
        nose = joints[:, nose_idx, :]
        pelvis_center = np.zeros((joints.shape[0], 3), dtype=np.float32)
        heights = np.linalg.norm(nose - pelvis_center, axis=1)
        mean_height = np.mean(heights)

        if mean_height > 1e-6:
            scale = 0.25 / mean_height  # Target: pelvis-to-nose = 0.25
            joints = joints * scale
        else:
            scale = 1.0

        # Step 3: Orientation normalization
        # Compute facing direction from shoulders
        left_shoulder_idx = COCO17_JOINT_INDICES["left_shoulder"]
        right_shoulder_idx = COCO17_JOINT_INDICES["right_shoulder"]

        left_shoulder = joints[:, left_shoulder_idx, :]
        right_shoulder = joints[:, right_shoulder_idx, :]

        # Shoulder vector points from right to left
        shoulder_vector = left_shoulder - right_shoulder
        mean_shoulder_vector = np.mean(shoulder_vector, axis=0)
        mean_shoulder_vector[1] = 0  # Project to XZ plane (ignore Y/height)

        # Current facing is perpendicular to shoulder vector
        # Left shoulder is on the left side, so facing is shoulder_vector × up
        up = np.array([0, 1, 0], dtype=np.float32)
        current_facing = np.cross(mean_shoulder_vector, up)

        # Normalize
        norm = np.linalg.norm(current_facing)
        if norm > 1e-6:
            current_facing = current_facing / norm
        else:
            current_facing = np.array([0, 0, 1], dtype=np.float32)

        # Target facing direction (+Z per protocol)
        if target_facing_direction is None:
            target_facing = np.array([0, 0, 1], dtype=np.float32)
        else:
            target_facing = target_facing_direction / np.linalg.norm(target_facing_direction)

        # Compute rotation to align facing directions
        rotation = self._compute_rotation(current_facing, target_facing)

        # Apply rotation to all joints
        for i in range(joints.shape[0]):
            joints[i] = (rotation @ joints[i].T).T

        # Create normalization parameters
        params = NormalizationParams(
            version=self.version,
            translation=translation,
            scale_factor=scale,
            rotation_matrix=rotation,
        )

        # Create normalized sequence
        normalized = SkeletonSequence(
            joints=joints,
            timestamps=sequence.timestamps.copy(),
            confidences=sequence.confidences.copy() if sequence.confidences is not None else None,
            joint_names=sequence.joint_names.copy(),
            metadata={
                **sequence.metadata,
                "normalized": True,
                "normalizationVersion": self.version,
            },
            normalization_params=params,
        )

        return normalized, params

    def _compute_rotation(
        self,
        from_vec: NDArray[np.float32],
        to_vec: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Compute rotation matrix to align from_vec to to_vec."""
        v = np.cross(from_vec, to_vec)
        s = np.linalg.norm(v)

        if s < 1e-6:
            # Vectors are parallel
            return np.eye(3, dtype=np.float32)

        c = np.dot(from_vec, to_vec)
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ], dtype=np.float32)

        rotation = np.eye(3, dtype=np.float32) + vx + vx @ vx * ((1 - c) / (s ** 2))
        return rotation

    def denormalize(
        self,
        sequence: SkeletonSequence,
        params: NormalizationParams,
    ) -> SkeletonSequence:
        """
        Reverse normalization (useful for visualization).

        Args:
            sequence: Normalized skeleton sequence
            params: Normalization parameters

        Returns:
            Denormalized sequence
        """
        joints = sequence.joints.copy()

        # Reverse rotation
        inv_rotation = params.rotation_matrix.T
        for i in range(joints.shape[0]):
            joints[i] = (inv_rotation @ joints[i].T).T

        # Reverse scale
        joints = joints / params.scale_factor

        # Reverse translation
        joints = joints - params.translation[np.newaxis, np.newaxis, :]

        return SkeletonSequence(
            joints=joints,
            timestamps=sequence.timestamps.copy(),
            confidences=sequence.confidences.copy() if sequence.confidences is not None else None,
            joint_names=sequence.joint_names.copy(),
            metadata={**sequence.metadata, "normalized": False},
            normalization_params=None,
        )


class TemporalSmoother:
    """
    Temporal smoothing for skeleton sequences to reduce jitter.

    Supports:
    - Gaussian smoothing (configurable sigma)
    - Savitzky-Golay filtering (configurable window, polynomial order)
    """

    def __init__(
        self,
        method: str = "gaussian",
        gaussian_sigma: float = 1.0,
        savgol_window: int = 7,
        savgol_polyorder: int = 3,
    ):
        """
        Initialize temporal smoother.

        Args:
            method: Smoothing method ("gaussian" or "savgol")
            gaussian_sigma: Sigma for Gaussian smoothing
            savgol_window: Window size for Savitzky-Golay (must be odd)
            savgol_polyorder: Polynomial order for Savitzky-Golay
        """
        self.method = method
        self.gaussian_sigma = gaussian_sigma
        self.savgol_window = savgol_window
        self.savgol_polyorder = savgol_polyorder

    def smooth(self, sequence: SkeletonSequence) -> SkeletonSequence:
        """
        Apply temporal smoothing to skeleton sequence.

        Args:
            sequence: Input skeleton sequence

        Returns:
            Smoothed skeleton sequence
        """
        joints = sequence.joints.copy()
        num_frames = joints.shape[0]

        if num_frames < 3:
            logger.warning("Sequence too short for smoothing, returning original")
            return sequence

        if self.method == "gaussian":
            smoothed = self._gaussian_smooth(joints)
        elif self.method == "savgol":
            smoothed = self._savgol_smooth(joints)
        else:
            raise ValueError(f"Unknown smoothing method: {self.method}")

        return SkeletonSequence(
            joints=smoothed,
            timestamps=sequence.timestamps.copy(),
            confidences=sequence.confidences.copy() if sequence.confidences is not None else None,
            joint_names=sequence.joint_names.copy(),
            metadata={**sequence.metadata, "smoothed": True, "smoothingMethod": self.method},
            normalization_params=sequence.normalization_params,
        )

    def _gaussian_smooth(self, joints: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply Gaussian smoothing along time axis."""
        smoothed = np.zeros_like(joints)
        num_joints = joints.shape[1]

        for j in range(num_joints):
            for d in range(3):  # x, y, z
                smoothed[:, j, d] = gaussian_filter1d(
                    joints[:, j, d],
                    sigma=self.gaussian_sigma,
                    mode="nearest",
                )

        return smoothed

    def _savgol_smooth(self, joints: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply Savitzky-Golay smoothing along time axis."""
        from scipy.signal import savgol_filter

        smoothed = np.zeros_like(joints)
        num_joints = joints.shape[1]

        # Adjust window size if sequence is too short
        window = min(self.savgol_window, joints.shape[0])
        if window % 2 == 0:
            window -= 1
        if window < self.savgol_polyorder + 2:
            window = self.savgol_polyorder + 2
            if window % 2 == 0:
                window += 1

        for j in range(num_joints):
            for d in range(3):  # x, y, z
                smoothed[:, j, d] = savgol_filter(
                    joints[:, j, d],
                    window_length=window,
                    polyorder=self.savgol_polyorder,
                    mode="nearest",
                )

        return smoothed


class SkeletonSerializer:
    """
    Serialize skeleton sequences to Protocol v1.0.0 format.

    Output formats:
    - NPZ: Dense numerical data (joints, timestamps, confidences)
    - JSON: Metadata and per-frame skeleton data
    """

    def __init__(self, joint_names: Optional[List[str]] = None):
        self.joint_names = joint_names or COCO17_JOINT_NAMES

    def save(
        self,
        sequence: SkeletonSequence,
        output_path: Union[str, Path],
        source_video: Optional[str] = None,
        extraction_model: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Save skeleton sequence to NPZ + JSON files.

        Args:
            sequence: Skeleton sequence to save
            output_path: Base output path (without extension)
            source_video: Source video identifier
            extraction_model: Model used for extraction

        Returns:
            Tuple of (npz_path, json_path)
        """
        output_path = Path(output_path)
        npz_path = output_path.with_suffix(".npz")
        json_path = output_path.with_suffix(".json")

        # Save dense data to NPZ
        save_dict = {
            "joints": sequence.joints,
            "timestamps": sequence.timestamps,
        }
        if sequence.confidences is not None:
            save_dict["confidences"] = sequence.confidences

        np.savez_compressed(npz_path, **save_dict)

        # Save metadata to JSON (Protocol v1.0.0 format)
        metadata = {
            "version": "1.0.0",
            "format": "3d_absolute",
            "fps": sequence.metadata.get("fps", 30.0),
            "frameCount": sequence.num_frames,
            "durationMs": float(sequence.timestamps[-1] * 1000) if len(sequence.timestamps) > 0 else 0,
            "jointCount": sequence.num_joints,
            "joints": sequence.to_protocol_format(),
            "metadata": {
                "sourceVideo": source_video or sequence.metadata.get("source_video", ""),
                "extractionModel": extraction_model or sequence.metadata.get("extraction_model", "sam-3d-body"),
                "extractedAt": sequence.metadata.get("extracted_at", ""),
                "normalizationVersion": sequence.metadata.get("normalizationVersion", ""),
            },
        }

        if sequence.normalization_params is not None:
            metadata["normalization"] = sequence.normalization_params.to_dict()

        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved skeleton sequence to {npz_path} and {json_path}")

        return str(npz_path), str(json_path)

    def load(self, npz_path: Union[str, Path], json_path: Union[str, Path]) -> SkeletonSequence:
        """
        Load skeleton sequence from NPZ + JSON files.

        Args:
            npz_path: Path to NPZ file
            json_path: Path to JSON file

        Returns:
            Loaded skeleton sequence
        """
        # Load metadata
        with open(json_path, "r") as f:
            metadata = json.load(f)

        # Load dense data
        data = np.load(npz_path)
        joints = data["joints"]
        timestamps = data["timestamps"]
        confidences = data["confidences"] if "confidences" in data else None

        # Parse normalization params
        norm_params = None
        if "normalization" in metadata:
            norm_params = NormalizationParams.from_dict(metadata["normalization"])

        return SkeletonSequence(
            joints=joints,
            timestamps=timestamps,
            confidences=confidences,
            joint_names=self.joint_names,
            metadata=metadata.get("metadata", {}),
            normalization_params=norm_params,
        )


def process_skeleton_sequence(
    joints: NDArray[np.float32],
    timestamps: NDArray[np.float32],
    confidences: Optional[NDArray[np.float32]] = None,
    normalize: bool = True,
    smooth: bool = True,
    smoothing_method: str = "gaussian",
    smoothing_sigma: float = 1.0,
) -> SkeletonSequence:
    """
    Convenience function for full skeleton processing pipeline.

    Args:
        joints: Joint positions (num_frames, 17, 3)
        timestamps: Frame timestamps (num_frames,)
        confidences: Joint confidences (num_frames, 17)
        normalize: Whether to apply normalization
        smooth: Whether to apply temporal smoothing
        smoothing_method: Smoothing method ("gaussian" or "savgol")
        smoothing_sigma: Sigma for Gaussian smoothing

    Returns:
        Processed skeleton sequence
    """
    # Create initial sequence
    sequence = SkeletonSequence(
        joints=joints,
        timestamps=timestamps,
        confidences=confidences,
    )

    # Apply temporal smoothing
    if smooth:
        smoother = TemporalSmoother(
            method=smoothing_method,
            gaussian_sigma=smoothing_sigma,
        )
        sequence = smoother.smooth(sequence)

    # Apply normalization
    if normalize:
        normalizer = SkeletonNormalizer()
        sequence, _ = normalizer.normalize(sequence)

    return sequence

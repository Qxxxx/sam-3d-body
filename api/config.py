"""
Configuration for SAM 3D Body inference API.
"""

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch
from pydantic import Field
from pydantic_settings import BaseSettings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class APIConfig(BaseSettings):
    """API configuration from environment variables."""

    # API Settings
    api_title: str = Field(default="SAM 3D Body Inference API", description="API title")
    api_version: str = Field(default="1.0.0", description="API version")
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_workers: int = Field(default=1, description="Number of worker processes")

    # Model Settings
    model_checkpoint_path: str = Field(
        default="./checkpoints/sam-3d-body-dinov3/model.ckpt",
        description="Path to SAM 3D Body checkpoint",
    )
    mhr_model_path: str = Field(
        default="./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt",
        description="Path to MHR model assets",
    )
    detector_name: str = Field(
        default="vitdet",
        description="Human detector to use (vitdet, sam3)",
    )
    detector_path: str = Field(
        default="",
        description="Path to detector model",
    )
    use_fov_estimator: bool = Field(
        default=True,
        description="Whether to use FOV estimator",
    )
    fov_estimator_name: str = Field(
        default="moge2",
        description="FOV estimator name",
    )

    # Processing Settings
    default_target_fps: float = Field(
        default=30.0,
        description="Default FPS for frame extraction",
    )
    max_video_duration: float = Field(
        default=30.0,
        description="Maximum video duration in seconds",
    )
    max_video_file_size_mb: int = Field(
        default=500,
        description="Maximum video file size in MB",
    )
    temp_dir: str = Field(
        default="/tmp/sam3db",
        description="Directory for temporary files",
    )
    output_dir: str = Field(
        default="./output",
        description="Directory for output files",
    )

    # GPU Settings
    device: str = Field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        description="Device for inference (cuda or cpu)",
    )
    gpu_memory_fraction: float = Field(
        default=0.9,
        description="Fraction of GPU memory to use",
    )

    # Feature Flags (MVP scope)
    enable_smash_analysis: bool = Field(
        default=True,
        description="Enable smash technique analysis",
    )
    support_left_handed: bool = Field(
        default=False,
        description="Support left-handed players",
    )
    support_multi_camera: bool = Field(
        default=False,
        description="Support multiple camera angles",
    )
    enable_phase_detection: bool = Field(
        default=False,
        description="Enable automatic phase detection",
    )

    # Alignment Settings
    dtw_radius: int = Field(
        default=1,
        description="Radius for FastDTW local constraint",
    )
    default_smoothing_sigma: float = Field(
        default=1.0,
        description="Default sigma for temporal smoothing",
    )

    class Config:
        env_prefix = "SAM3DB_"
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_config() -> APIConfig:
    """Get cached API configuration."""
    return APIConfig()


def get_device() -> torch.device:
    """Get the torch device for inference."""
    config = get_config()
    return torch.device(config.device)


def get_gpu_info() -> dict:
    """Get GPU information if available."""
    info = {
        "available": torch.cuda.is_available(),
        "name": None,
        "memory_mb": None,
    }

    if torch.cuda.is_available():
        info["name"] = torch.cuda.get_device_name(0)
        info["memory_mb"] = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)

    return info


# Feature flags for gradual rollout
FEATURE_FLAGS = {
    "ENABLE_SMASH_ANALYSIS": get_config().enable_smash_analysis,
    "SUPPORTED_ACTIONS": ["smash"],  # MVP: only smash
    "SUPPORT_LEFT_HANDED": get_config().support_left_handed,
    "SUPPORT_MULTI_CAMERA": get_config().support_multi_camera,
    "ENABLE_PHASE_DETECTION": get_config().enable_phase_detection,
    "MAX_VIDEO_DURATION": get_config().max_video_duration,
    "MAX_FILE_SIZE_MB": get_config().max_video_file_size_mb,
}

"""
HTTP Client for calling SAM 3D Body inference API.

This client is designed to be used by the cf-backend worker to call the
GPU inference service. It can be used as a reference for JavaScript/TypeScript
implementation or used directly if the worker is Python-based.

Protocol v1.0.0 compliant.
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urljoin

import httpx
from numpy.typing import NDArray

from api.models import (
    AlignmentResponse,
    CoachingTipDetail,
    ErrorResponse,
    HealthResponse,
    JointErrorDetail,
    PhaseScoreDetail,
    PoseCorrectionResponse,
    TaskState,
    TechniqueType,
    VideoInferenceResponse,
)

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for inference client."""

    base_url: str = "http://localhost:8000"
    timeout_seconds: float = 300.0  # 5 minutes for video processing
    max_retries: int = 3
    retry_delay_seconds: float = 5.0
    request_id_prefix: str = "worker"


class Sam3dBodyClient:
    """
    HTTP client for SAM 3D Body inference API.

    This client handles:
    - Health checks
    - Video inference requests
    - Alignment/scoring requests
    - Error handling and retries
    - Result extraction

    Usage:
        client = Sam3dBodyClient(InferenceConfig(base_url="http://gpu-service:8000"))

        # Health check
        health = await client.health_check()

        # Video inference
        result = await client.infer_video(
            video_url="https://r2.example.com/user_video.mp4",
            technique_type=TechniqueType.SMASH,
        )

        # Alignment
        alignment = await client.align_sequences(
            user_skeleton_url="https://r2.example.com/user_skeleton.npz",
            reference_skeleton_url="https://r2.example.com/ref_skeleton.npz",
        )
    """

    def __init__(self, config: Optional[InferenceConfig] = None):
        self.config = config or InferenceConfig()
        self.client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=self.config.timeout_seconds,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    async def _make_request(
        self,
        method: str,
        path: str,
        json_data: Optional[Dict] = None,
        retries: Optional[int] = None,
    ) -> Tuple[int, Dict]:
        """
        Make HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST)
            path: API path
            json_data: Request body for POST
            retries: Override max retries

        Returns:
            Tuple of (status_code, response_data)
        """
        max_retries = retries or self.config.max_retries
        last_error = None

        for attempt in range(max_retries):
            try:
                if method == "GET":
                    response = await self.client.get(path)
                elif method == "POST":
                    response = await self.client.post(path, json=json_data)
                else:
                    raise ValueError(f"Unsupported method: {method}")

                # Parse response
                try:
                    data = response.json()
                except Exception:
                    data = {"message": response.text}

                # Return on success or 4xx error (client error, don't retry)
                if response.status_code < 500:
                    return response.status_code, data

                # 5xx error - will retry
                last_error = f"HTTP {response.status_code}: {data.get('message', 'Unknown error')}"
                logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {last_error}")

            except httpx.TimeoutException as e:
                last_error = f"Timeout: {e}"
                logger.warning(f"Request timeout (attempt {attempt + 1}/{max_retries})")

            except httpx.NetworkError as e:
                last_error = f"Network error: {e}"
                logger.warning(f"Network error (attempt {attempt + 1}/{max_retries})")

            except Exception as e:
                last_error = f"Unexpected error: {e}"
                logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")

            # Wait before retry
            if attempt < max_retries - 1:
                wait_time = self.config.retry_delay_seconds * (attempt + 1)
                logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)

        # All retries exhausted
        raise InferenceError(f"Request failed after {max_retries} attempts: {last_error}")

    async def health_check(self) -> HealthResponse:
        """
        Check if the inference service is healthy.

        Returns:
            HealthResponse with service status

        Raises:
            InferenceError: If the service is unavailable
        """
        status_code, data = await self._make_request("GET", "/health")

        if status_code != 200:
            raise InferenceError(f"Health check failed: {data.get('message', 'Unknown error')}")

        return HealthResponse(**data)

    async def is_healthy(self) -> bool:
        """Quick check if service is healthy."""
        try:
            health = await self.health_check()
            return health.status == "healthy"
        except Exception:
            return False

    async def infer_video(
        self,
        video_url: str,
        technique_type: TechniqueType = TechniqueType.SMASH,
        time_range_start: Optional[float] = None,
        time_range_end: Optional[float] = None,
        fps: float = 30.0,
        subject_bbox: Optional[Dict[str, float]] = None,
        camera_angle: str = "back",
        request_id: Optional[str] = None,
        detect_phases: bool = True,
    ) -> VideoInferenceResponse:
        """
        Request video inference.

        Args:
            video_url: Presigned URL to user video
            technique_type: Type of technique being analyzed
            time_range_start: Start time in milliseconds
            time_range_end: End time in milliseconds
            fps: Target FPS for processing
            subject_bbox: Initial subject bounding box (normalized 0-1)
            camera_angle: Camera angle hint
            request_id: Unique request ID for idempotency
            detect_phases: Whether to detect motion phases

        Returns:
            VideoInferenceResponse with skeleton data

        Raises:
            InferenceError: If inference fails
        """
        # Build request
        request_data = {
            "video_url": video_url,
            "technique_type": technique_type.value,
            "fps": fps,
            "camera_angle": camera_angle,
            "detect_phases": detect_phases,
        }

        if time_range_start is not None:
            request_data["time_range_start"] = time_range_start
        if time_range_end is not None:
            request_data["time_range_end"] = time_range_end
        if subject_bbox:
            request_data["subject_bbox"] = subject_bbox
        if request_id:
            request_data["request_id"] = request_id

        # Make request with longer timeout for video processing
        status_code, data = await self._make_request(
            "POST",
            "/infer/video",
            json_data=request_data,
            retries=self.config.max_retries,
        )

        if status_code != 200:
            error_msg = data.get("message", "Video inference failed")
            logger.error(f"Video inference failed: {error_msg}")
            raise InferenceError(error_msg, status_code=status_code, details=data)

        return VideoInferenceResponse(**data)

    async def align_sequences(
        self,
        user_skeleton_url: str,
        reference_skeleton_url: str,
        technique_type: TechniqueType = TechniqueType.SMASH,
        detect_phases: bool = True,
    ) -> AlignmentResponse:
        """
        Align user skeleton with reference and generate scores.

        Args:
            user_skeleton_url: URL to user skeleton NPZ/JSON
            reference_skeleton_url: URL to reference skeleton NPZ/JSON
            technique_type: Type of technique
            detect_phases: Whether to detect motion phases

        Returns:
            AlignmentResponse with scores and tips

        Raises:
            InferenceError: If alignment fails
        """
        request_data = {
            "video_url": user_skeleton_url,  # Note: endpoint expects video_url for user skeleton
            "reference_skeleton_url": reference_skeleton_url,
            "technique_type": technique_type.value,
            "detect_phases": detect_phases,
        }

        status_code, data = await self._make_request(
            "POST",
            "/infer/alignment",
            json_data=request_data,
        )

        if status_code != 200:
            error_msg = data.get("message", "Alignment failed")
            logger.error(f"Alignment failed: {error_msg}")
            raise InferenceError(error_msg, status_code=status_code, details=data)

        return AlignmentResponse(**data)

    async def process_video_and_align(
        self,
        video_url: str,
        reference_skeleton_url: str,
        technique_type: TechniqueType = TechniqueType.SMASH,
        time_range_start: Optional[float] = None,
        time_range_end: Optional[float] = None,
        subject_bbox: Optional[Dict[str, float]] = None,
    ) -> Tuple[VideoInferenceResponse, AlignmentResponse]:
        """
        Full pipeline: extract skeleton from video and align with reference.

        Args:
            video_url: Presigned URL to user video
            reference_skeleton_url: URL to reference skeleton
            technique_type: Type of technique
            time_range_start: Start time in milliseconds
            time_range_end: End time in milliseconds
            subject_bbox: Initial subject bounding box

        Returns:
            Tuple of (VideoInferenceResponse, AlignmentResponse)
        """
        # Step 1: Extract skeleton from video
        logger.info("Step 1: Extracting skeleton from video")
        video_result = await self.infer_video(
            video_url=video_url,
            technique_type=technique_type,
            time_range_start=time_range_start,
            time_range_end=time_range_end,
            subject_bbox=subject_bbox,
            detect_phases=True,
        )

        if video_result.status != TaskState.SUCCEEDED:
            raise InferenceError(f"Video inference failed with status: {video_result.status}")

        # Step 2: Align with reference
        logger.info("Step 2: Aligning with reference skeleton")
        user_skeleton_url = video_result.skeleton.npz_url
        alignment_result = await self.align_sequences(
            user_skeleton_url=user_skeleton_url,
            reference_skeleton_url=reference_skeleton_url,
            technique_type=technique_type,
            detect_phases=True,
        )

        if alignment_result.status != TaskState.SUCCEEDED:
            raise InferenceError(f"Alignment failed with status: {alignment_result.status}")

        return video_result, alignment_result


class InferenceError(Exception):
    """Exception raised for inference service errors."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        details: Optional[Dict] = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}

    def to_error_response(self, request_id: str) -> ErrorResponse:
        """Convert to Protocol v1.0.0 error response."""
        return ErrorResponse(
            code=self.status_code or 500,
            error_code=self.details.get("error_code", "INFERENCE_FAILED"),
            message=self.message,
            details=self.details,
            request_id=request_id,
            retryable=self.status_code is None or self.status_code >= 500,
            retry_after_ms=5000 if self.status_code and self.status_code >= 500 else None,
        )


# Helper functions for result extraction

def extract_summary_for_d1(alignment_result: AlignmentResponse) -> Dict[str, Any]:
    """
    Extract summary data for D1 storage.

    Per Protocol v1.0.0 Section 6, D1 stores:
    - Task metadata, status
    - Result summary (scores, tips)
    """
    result = alignment_result.result

    return {
        "overallScore": result.overall_score,
        "similarityScore": result.similarity_score,
        "starRating": result.star_rating,
        "topTips": [
            {
                "priority": tip.priority,
                "title": tip.title,
                "category": tip.category.value,
            }
            for tip in result.tips[:3]  # Top 3 tips
        ],
        "phaseScores": [
            {
                "phase": ps.phase,
                "score": ps.score,
                "feedback": ps.feedback,
            }
            for ps in result.phase_scores
        ],
        "frameCount": result.user_frame_count,
        "processingTimeMs": result.processing_info.processing_time_ms,
    }


def extract_full_result_for_r2(
    video_result: VideoInferenceResponse,
    alignment_result: AlignmentResponse,
) -> Dict[str, Any]:
    """
    Extract full result data for R2 storage.

    Per Protocol v1.0.0 Section 6, R2 stores:
    - Full skeleton sequences
    - Alignment paths
    - Frame-by-frame errors
    """
    return {
        "version": "1.0.0",
        "taskId": alignment_result.task_id,
        "createdAt": alignment_result.created_at.isoformat(),
        "videoResult": video_result.model_dump(),
        "alignmentResult": alignment_result.model_dump(),
    }


def format_joint_errors_for_display(joint_errors: List[JointErrorDetail]) -> List[Dict[str, Any]]:
    """Format joint errors for user display."""
    return [
        {
            "joint": je.joint_name,
            "severity": je.severity.value,
            "meanErrorCm": round(je.mean_error * 100, 1),  # Convert to cm
            "ranking": je.ranking,
        }
        for je in sorted(joint_errors, key=lambda x: x.ranking)[:5]  # Top 5
    ]


def format_tips_for_display(tips: List[CoachingTipDetail]) -> List[Dict[str, Any]]:
    """Format coaching tips for user display."""
    return [
        {
            "id": tip.id,
            "priority": tip.priority,
            "title": tip.title,
            "description": tip.description,
            "category": tip.category.value,
            "suggestedDrill": tip.suggested_drill,
        }
        for tip in sorted(tips, key=lambda x: x.priority, reverse=True)
    ]


# Example usage
async def example_usage():
    """Example of how to use the client."""
    config = InferenceConfig(
        base_url="http://localhost:8000",
        timeout_seconds=300.0,
    )

    async with Sam3dBodyClient(config) as client:
        # Health check
        health = await client.health_check()
        print(f"Service status: {health.status}")
        print(f"GPU available: {health.gpu_available}")

        # Process video
        try:
            video_result, alignment_result = await client.process_video_and_align(
                video_url="https://r2.example.com/user_video.mp4",
                reference_skeleton_url="https://r2.example.com/ref_smash.npz",
                technique_type=TechniqueType.SMASH,
            )

            print(f"Overall score: {alignment_result.result.overall_score}")
            print(f"Star rating: {alignment_result.result.star_rating}")

            # Extract data for storage
            d1_summary = extract_summary_for_d1(alignment_result)
            r2_full_result = extract_full_result_for_r2(video_result, alignment_result)

            print("D1 Summary:", d1_summary)

        except InferenceError as e:
            print(f"Inference failed: {e.message}")
            error_response = e.to_error_response(request_id="req_123")
            print(f"Error response: {error_response}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())

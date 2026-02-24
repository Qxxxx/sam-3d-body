"""
Worker Integration Module for cf-backend to sam-3d-body communication.

This module provides the integration logic for the cf-backend worker
to call the GPU inference service and store results.

Protocol v1.0.0 compliant.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from api.client import (
    InferenceConfig,
    Sam3dBodyClient,
    InferenceError,
    extract_summary_for_d1,
    extract_full_result_for_r2,
)
from api.models import (
    TaskState,
    TechniqueType,
    ErrorResponse,
)

logger = logging.getLogger(__name__)


class WorkerIntegration:
    """
    Integration handler for cf-backend worker.

    This class manages:
    - Receiving tasks from queue
    - Calling GPU inference service
    - Storing results to R2/D1
    - Error handling and retry logic
    - Progress updates
    """

    def __init__(
        self,
        inference_config: InferenceConfig,
        r2_bucket: Any,  # Cloudflare R2 binding
        db: Any,  # D1 database binding
    ):
        self.inference_config = inference_config
        self.r2 = r2_bucket
        self.db = db

    async def process_technique_analysis_task(
        self,
        task_id: str,
        user_id: str,
        video_object_key: str,
        technique_type: str,
        reference_asset_id: str,
        time_range_start: Optional[float] = None,
        time_range_end: Optional[float] = None,
        subject_bbox: Optional[Dict[str, float]] = None,
        camera_angle: str = "back",
    ) -> Dict[str, Any]:
        """
        Process a technique analysis task end-to-end.

        This is the main entry point for the worker.

        Args:
            task_id: Unique task ID
            user_id: User ID
            video_object_key: R2 object key for user video
            technique_type: Technique type (smash, clear, etc.)
            reference_asset_id: Reference asset ID
            time_range_start: Start time in ms
            time_range_end: End time in ms
            subject_bbox: Subject bounding box (normalized)
            camera_angle: Camera angle hint

        Returns:
            Dict with processing result
        """
        logger.info(f"[Task {task_id}] Starting technique analysis")

        try:
            async with Sam3dBodyClient(self.inference_config) as client:
                # Step 1: Generate presigned URLs
                logger.info(f"[Task {task_id}] Generating presigned URLs")
                video_url = await self._get_presigned_url(video_object_key)
                ref_url = await self._get_reference_skeleton_url(reference_asset_id)

                # Step 2: Update progress - downloading
                await self._update_task_progress(
                    task_id=task_id,
                    state=TaskState.RUNNING,
                    progress_percent=10,
                    stage="downloading",
                )

                # Step 3: Call GPU inference service
                logger.info(f"[Task {task_id}] Calling GPU inference service")
                await self._update_task_progress(
                    task_id=task_id,
                    state=TaskState.RUNNING,
                    progress_percent=20,
                    stage="inference",
                )

                video_result = await client.infer_video(
                    video_url=video_url,
                    technique_type=TechniqueType(technique_type),
                    time_range_start=time_range_start,
                    time_range_end=time_range_end,
                    fps=30.0,
                    subject_bbox=subject_bbox,
                    camera_angle=camera_angle,
                    request_id=task_id,
                    detect_phases=True,
                )

                # Step 4: Store skeleton to R2
                logger.info(f"[Task {task_id}] Storing skeleton to R2")
                await self._update_task_progress(
                    task_id=task_id,
                    state=TaskState.RUNNING,
                    progress_percent=50,
                    stage="alignment",
                )

                skeleton_object_key = f"skeletons/{user_id}/{task_id}.npz"
                await self._copy_to_r2(video_result.skeleton.npz_url, skeleton_object_key)

                # Step 5: Call alignment
                logger.info(f"[Task {task_id}] Calling alignment")
                user_skeleton_url = await self._get_presigned_url(skeleton_object_key)

                alignment_result = await client.align_sequences(
                    user_skeleton_url=user_skeleton_url,
                    reference_skeleton_url=ref_url,
                    technique_type=TechniqueType(technique_type),
                    detect_phases=True,
                )

                # Step 6: Store full results to R2
                logger.info(f"[Task {task_id}] Storing full results to R2")
                await self._update_task_progress(
                    task_id=task_id,
                    state=TaskState.RUNNING,
                    progress_percent=80,
                    stage="storing_results",
                )

                result_object_key = f"results/{user_id}/{task_id}.json"
                full_result = extract_full_result_for_r2(video_result, alignment_result)
                await self._store_json_to_r2(full_result, result_object_key)

                # Step 7: Store summary to D1
                logger.info(f"[Task {task_id}] Storing summary to D1")
                summary = extract_summary_for_d1(alignment_result)
                await self._update_task_result(
                    task_id=task_id,
                    state=TaskState.SUCCEEDED,
                    progress_percent=100,
                    result_summary=summary,
                    result_object_key=result_object_key,
                )

                logger.info(f"[Task {task_id}] Completed successfully")

                return {
                    "success": True,
                    "task_id": task_id,
                    "overall_score": alignment_result.result.overall_score,
                    "star_rating": alignment_result.result.star_rating,
                }

        except InferenceError as e:
            logger.error(f"[Task {task_id}] Inference error: {e.message}")
            await self._handle_error(task_id, e)
            return {
                "success": False,
                "task_id": task_id,
                "error": e.message,
                "error_code": e.details.get("error_code", "INFERENCE_FAILED"),
            }

        except Exception as e:
            logger.error(f"[Task {task_id}] Unexpected error: {e}")
            await self._handle_error(
                task_id,
                InferenceError(
                    str(e),
                    status_code=500,
                    details={"error_code": "INTERNAL_ERROR"},
                ),
            )
            return {
                "success": False,
                "task_id": task_id,
                "error": str(e),
                "error_code": "INTERNAL_ERROR",
            }

    async def _get_presigned_url(self, object_key: str) -> str:
        """Generate presigned URL for R2 object."""
        # In production, this would use R2 API to generate presigned URL
        # For now, return a placeholder
        return f"https://r2.example.com/{object_key}"

    async def _get_reference_skeleton_url(self, asset_id: str) -> str:
        """Get reference skeleton URL from asset ID."""
        # Query D1 for reference asset
        # stmt = self.db.prepare("SELECT skeleton_object_key FROM technique_reference_assets WHERE id = ?")
        # result = await stmt.bind(asset_id).first()
        # return await self._get_presigned_url(result["skeleton_object_key"])
        return f"https://r2.example.com/references/{asset_id}.npz"

    async def _copy_to_r2(self, source_url: str, dest_object_key: str):
        """Copy file from source URL to R2."""
        # In production, download from source_url and upload to R2
        logger.info(f"Copying {source_url} to R2://{dest_object_key}")
        pass

    async def _store_json_to_r2(self, data: Dict, object_key: str):
        """Store JSON data to R2."""
        json_str = json.dumps(data)
        # await self.r2.put(object_key, json_str)
        logger.info(f"Stored JSON to R2://{object_key}")

    async def _update_task_progress(
        self,
        task_id: str,
        state: TaskState,
        progress_percent: float,
        stage: str,
    ):
        """Update task progress in D1."""
        logger.info(f"[Task {task_id}] Progress: {progress_percent}% ({stage})")
        # stmt = self.db.prepare("""
        #     UPDATE technique_analysis_tasks
        #     SET status = ?, progress_percent = ?, stage = ?, updated_at = ?
        #     WHERE id = ?
        # """)
        # await stmt.bind(
        #     state.value,
        #     progress_percent,
        #     stage,
        #     datetime.utcnow().isoformat(),
        #     task_id,
        # ).run()

    async def _update_task_result(
        self,
        task_id: str,
        state: TaskState,
        progress_percent: float,
        result_summary: Dict,
        result_object_key: str,
    ):
        """Update task with results in D1."""
        logger.info(f"[Task {task_id}] Storing results")
        # stmt = self.db.prepare("""
        #     UPDATE technique_analysis_tasks
        #     SET status = ?, progress_percent = ?, result_summary = ?,
        #         result_object_key = ?, completed_at = ?, updated_at = ?
        #     WHERE id = ?
        # """)
        # await stmt.bind(
        #     state.value,
        #     progress_percent,
        #     json.dumps(result_summary),
        #     result_object_key,
        #     datetime.utcnow().isoformat(),
        #     datetime.utcnow().isoformat(),
        #     task_id,
        # ).run()

    async def _handle_error(self, task_id: str, error: InferenceError):
        """Handle and store error."""
        logger.error(f"[Task {task_id}] Error: {error.message}")
        # stmt = self.db.prepare("""
        #     UPDATE technique_analysis_tasks
        #     SET status = ?, error_code = ?, error_message = ?, updated_at = ?
        #     WHERE id = ?
        # """)
        # await stmt.bind(
        #     TaskState.FAILED.value,
        #     error.details.get("error_code", "UNKNOWN"),
        #     error.message,
        #     datetime.utcnow().isoformat(),
        #     task_id,
        # ).run()


# Worker entry point
async def process_queue_message(
    message: Dict[str, Any],
    inference_config: InferenceConfig,
    r2_bucket: Any,
    db: Any,
) -> Dict[str, Any]:
    """
    Process a queue message from the technique analysis queue.

    This is the entry point called by the queue consumer.

    Args:
        message: Queue message with task details
        inference_config: GPU service configuration
        r2_bucket: R2 binding
        db: D1 database binding

    Returns:
        Processing result
    """
    worker = WorkerIntegration(inference_config, r2_bucket, db)

    return await worker.process_technique_analysis_task(
        task_id=message["taskId"],
        user_id=message["userId"],
        video_object_key=message["videoObjectKey"],
        technique_type=message["techniqueType"],
        reference_asset_id=message["referenceAssetId"],
        time_range_start=message.get("timeRangeStart"),
        time_range_end=message.get("timeRangeEnd"),
        subject_bbox=message.get("subjectBbox"),
        camera_angle=message.get("cameraAngle", "back"),
    )

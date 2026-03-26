from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


_DEFAULT_CHECKPOINT_DIR = (
    Path(__file__).resolve().parents[1]
    / "checkpoints"
    / "sam-3d-body-dinov3"
)


@dataclass(frozen=True)
class ApiSettings:
    checkpoint_path: str
    mhr_path: str
    device: str
    fov_name: str
    fov_path: str
    artifact_root: str
    eager_model_load: bool = False
    technique_trace_ingest_url: str | None = None
    technique_trace_ingest_token: str | None = None
    runtime_env: str = "gpu-server"
    job_concurrency: int = 1


def _read_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _read_positive_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value.strip())
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def load_api_settings() -> ApiSettings:
    default_checkpoint = _DEFAULT_CHECKPOINT_DIR / "model.ckpt"
    default_mhr = _DEFAULT_CHECKPOINT_DIR / "assets" / "mhr_model.pt"

    return ApiSettings(
        checkpoint_path=os.getenv("SAM3DBODY_CHECKPOINT_PATH", str(default_checkpoint)),
        mhr_path=os.getenv("SAM3DBODY_MHR_PATH", str(default_mhr)),
        device=os.getenv("SAM3DBODY_DEVICE", "cuda"),
        fov_name=os.getenv("SAM3DBODY_FOV_NAME", "moge2"),
        fov_path=os.getenv("SAM3DBODY_FOV_PATH", ""),
        artifact_root=os.getenv(
            "SAM3DBODY_ARTIFACT_ROOT",
            str(Path(__file__).resolve().parents[1] / "tmp_api_assets"),
        ),
        eager_model_load=_read_bool_env("SAM3DBODY_EAGER_MODEL_LOAD", False),
        technique_trace_ingest_url=(
            os.getenv("SAM3DBODY_TECHNIQUE_TRACE_INGEST_URL", "").strip() or None
        ),
        technique_trace_ingest_token=(
            os.getenv("SAM3DBODY_TECHNIQUE_TRACE_INGEST_TOKEN", "").strip() or None
        ),
        runtime_env=os.getenv("SAM3DBODY_RUNTIME_ENV", "gpu-server").strip() or "gpu-server",
        job_concurrency=_read_positive_int_env("SAM3DBODY_JOB_CONCURRENCY", 1),
    )

"""Runtime configuration for the PathSnap FastAPI inference service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class APISettings:
    """Environment-driven settings supporting single-laptop and two-laptop deployment."""

    inference_data_dir: Path
    model_path: Path
    hmm_config_path: Path
    pipeline_config_path: Path
    cors_origins: list[str]
    max_map_road_segments: int = 3000


def load_api_settings() -> APISettings:
    """Resolve settings from environment variables with local development defaults."""
    data_dir = Path(
        os.environ.get(
            "PATHSNAP_INFERENCE_DATA_DIR",
            str(PROJECT_ROOT / "data" / "raw" / "student_test"),
        )
    ).resolve()
    model_path = Path(
        os.environ.get(
            "PATHSNAP_MODEL_PATH",
            str(PROJECT_ROOT / "artifacts" / "hist_gradient_boosting_tuned.joblib"),
        )
    ).resolve()
    hmm_cfg_path = Path(
        os.environ.get(
            "PATHSNAP_HMM_CONFIG_PATH",
            str(PROJECT_ROOT / "configs" / "hmm_config.json"),
        )
    ).resolve()
    pipe_cfg_path = Path(
        os.environ.get(
            "PATHSNAP_PIPELINE_CONFIG_PATH",
            str(PROJECT_ROOT / "configs" / "pipeline_config.json"),
        )
    ).resolve()
    raw_origins = os.environ.get("PATHSNAP_CORS_ORIGINS", "*")
    cors_origins = [o.strip() for o in raw_origins.split(",") if o.strip()] or ["*"]

    return APISettings(
        inference_data_dir=data_dir,
        model_path=model_path,
        hmm_config_path=hmm_cfg_path,
        pipeline_config_path=pipe_cfg_path,
        cors_origins=cors_origins,
    )

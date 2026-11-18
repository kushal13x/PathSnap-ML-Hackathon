"""Preliminary baseline candidate classifiers for PathSnap Phase 1.

Includes:
1. NearestRoadDistanceBaseline: heuristic ranking by 1 / (1 + dist_to_seg_m).
2. SGDBaselineClassifier: StandardScaler + SGDClassifier(loss='log_loss') fitted
   strictly on training-fold trips with valid aligned labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class NearestRoadDistanceBaseline:
    """Simple distance-only baseline: predicts the nearest candidate road arc."""

    name = "nearest_road_distance"

    @staticmethod
    def predict_proba(dist_to_seg_m: np.ndarray) -> np.ndarray:
        """Return monotonic proximity score in (0, 1] for candidate ranking."""
        return (1.0 / (1.0 + np.maximum(dist_to_seg_m, 0.0))).astype(np.float64)


@dataclass
class SGDBaselineConfig:
    """Configuration for preliminary SGDClassifier baseline."""

    loss: str = "log_loss"
    penalty: str = "l2"
    alpha: float = 1e-4
    class_weight: str = "balanced"
    max_iter: int = 1000
    tol: float = 1e-3
    random_state: int = 42


def build_sgd_baseline_pipeline(cfg: SGDBaselineConfig) -> Pipeline:
    """Create StandardScaler + SGDClassifier(log_loss) pipeline."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                SGDClassifier(
                    loss=cfg.loss,
                    penalty=cfg.penalty,
                    alpha=cfg.alpha,
                    class_weight=cfg.class_weight,
                    max_iter=cfg.max_iter,
                    tol=cfg.tol,
                    random_state=cfg.random_state,
                ),
            ),
        ]
    )


def save_baseline_artifact(pipeline: Pipeline, metadata: dict, artifact_path: Path) -> None:
    """Persist fitted scaler+classifier pipeline and metadata via joblib."""
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, artifact_path, compress=3)


def load_baseline_artifact(artifact_path: Path) -> tuple[Pipeline, dict]:
    """Load persisted baseline artifact."""
    obj = joblib.load(artifact_path)
    return obj["pipeline"], obj["metadata"]

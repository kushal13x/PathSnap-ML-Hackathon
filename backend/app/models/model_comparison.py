"""Phase 2 five-model comparison builders, memory-monitored training, and sampling helpers."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.tree import DecisionTreeClassifier

from backend.app.features.feature_engineering import FEATURE_COLUMNS

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "nearest_road_baseline": "Nearest-Road Distance",
    "sgd_logistic": "SGD Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "hist_gradient_boosting": "HistGradientBoosting",
    "xgboost_gpu": "XGBoost (CUDA GPU)",
}

MODEL_FAMILIES: dict[str, str] = {
    "nearest_road_baseline": "Heuristic",
    "sgd_logistic": "Linear",
    "decision_tree": "Tree-based",
    "random_forest": "Ensemble (Bagging)",
    "extra_trees": "Ensemble (Randomized Trees)",
    "hist_gradient_boosting": "Gradient Boosting",
    "xgboost_gpu": "Gradient Boosting (GPU)",
}


class PeakRSSMonitor:
    """Context manager that polls process RSS every 50 ms to capture true peak RAM."""

    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self.proc = psutil.Process()
        self.start_rss_mb: float = 0.0
        self.peak_rss_mb: float = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll(self) -> None:
        while not self._stop_event.is_set():
            rss = self.proc.memory_info().rss / (1024.0 * 1024.0)
            if rss > self.peak_rss_mb:
                self.peak_rss_mb = rss
            time.sleep(self.interval_s)

    def __enter__(self) -> PeakRSSMonitor:
        self.start_rss_mb = self.proc.memory_info().rss / (1024.0 * 1024.0)
        self.peak_rss_mb = self.start_rss_mb
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        final_rss = self.proc.memory_info().rss / (1024.0 * 1024.0)
        if final_rss > self.peak_rss_mb:
            self.peak_rss_mb = final_rss


def load_observation_grouped_dataset(
    cand_dir: Path,
    trip_ids: list[str],
    obs_fraction: float = 1.0,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load valid labeled candidate rows from Parquet partitions.

    If `obs_fraction < 1.0`, samples GPS observations (preserving EVERY candidate
    for each selected observation) rather than randomly dropping individual candidates.
    """
    rng = np.random.default_rng(random_state)
    x_chunks = []
    y_chunks = []
    group_chunks = []

    total_raw_rows = 0
    excluded_invalid_rows = 0
    total_valid_obs = 0
    sampled_valid_obs = 0

    for tid in trip_ids:
        df = pd.read_parquet(
            cand_dir / f"{tid}.parquet",
            columns=FEATURE_COLUMNS + ["observation_id", "target", "is_valid_label"],
        )
        total_raw_rows += len(df)
        valid_mask = df["is_valid_label"].to_numpy(dtype=bool)
        excluded_invalid_rows += int(np.sum(~valid_mask))

        df_valid = df.loc[valid_mask]
        if len(df_valid) == 0:
            del df, df_valid
            continue

        obs_ids = df_valid["observation_id"].to_numpy()
        unique_obs = np.unique(obs_ids)
        total_valid_obs += len(unique_obs)

        if obs_fraction < 1.0 and len(unique_obs) > 1:
            n_keep = max(1, int(round(len(unique_obs) * obs_fraction)))
            chosen_obs = rng.choice(unique_obs, size=n_keep, replace=False)
            keep_mask = np.isin(obs_ids, chosen_obs)
            df_valid = df_valid.loc[keep_mask]
            sampled_valid_obs += n_keep
        else:
            sampled_valid_obs += len(unique_obs)

        x_chunks.append(df_valid[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
        y_chunks.append(df_valid["target"].to_numpy(dtype=np.int8))
        group_chunks.append(np.full(len(df_valid), tid, dtype=object))
        del df, df_valid

    X = np.vstack(x_chunks)
    y = np.concatenate(y_chunks)
    groups = np.concatenate(group_chunks)

    stats = {
        "n_trips": len(trip_ids),
        "obs_fraction": obs_fraction,
        "total_raw_candidate_rows": total_raw_rows,
        "excluded_invalid_rows": excluded_invalid_rows,
        "total_valid_gps_obs": total_valid_obs,
        "used_valid_gps_obs": sampled_valid_obs,
        "used_candidate_rows": int(len(y)),
        "positive_rows": int(np.sum(y == 1)),
        "negative_rows": int(np.sum(y == 0)),
        "positive_rate_pct": round(float(np.mean(y == 1) * 100.0), 3),
    }
    return X, y, groups, stats


def build_phase2_model(model_key: str, params: dict) -> Any:
    """Instantiate one of the Phase 2 candidate classifiers."""
    if model_key == "decision_tree":
        return DecisionTreeClassifier(**params)
    if model_key == "random_forest":
        return RandomForestClassifier(**params)
    if model_key == "extra_trees":
        return ExtraTreesClassifier(**params)
    if model_key == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(**params)
    raise ValueError(f"Unknown Phase 2 model key: {model_key}")


def save_model_artifact(model: Any, metadata: dict, artifact_path: Path) -> int:
    """Save trained model + metadata via joblib and return file size in bytes."""
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata}, artifact_path, compress=3)
    return int(artifact_path.stat().st_size)


def load_model_artifact(artifact_path: Path) -> tuple[Any, dict]:
    """Load trained Phase 1 or Phase 2 artifact seamlessly."""
    obj = joblib.load(artifact_path)
    if "pipeline" in obj:
        return obj["pipeline"], obj["metadata"]
    return obj["model"], obj["metadata"]

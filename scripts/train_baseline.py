"""Phase 1 Pipeline: Train preliminary CPU-friendly SGDClassifier baseline.

Fits StandardScaler + SGDClassifier(loss='log_loss', class_weight='balanced')
strictly on valid observations from the 64 train_fit trips.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.data.loaders import get_process_rss_mb
from backend.app.features.feature_engineering import FEATURE_COLUMNS
from backend.app.models.baseline import (
    SGDBaselineConfig,
    build_sgd_baseline_pipeline,
    save_baseline_artifact,
)


def load_valid_training_arrays(
    cand_dir: Path, trip_ids: list[str]
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load valid labeled candidate rows from train_fit Parquet partitions."""
    x_chunks = []
    y_chunks = []
    total_raw_rows = 0
    excluded_invalid_rows = 0

    for tid in trip_ids:
        df = pd.read_parquet(
            cand_dir / f"{tid}.parquet",
            columns=FEATURE_COLUMNS + ["target", "is_valid_label"],
        )
        total_raw_rows += len(df)
        valid_mask = df["is_valid_label"].to_numpy(dtype=bool)
        excluded_invalid_rows += int(np.sum(~valid_mask))

        df_valid = df.loc[valid_mask]
        if len(df_valid) > 0:
            x_chunks.append(df_valid[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
            y_chunks.append(df_valid["target"].to_numpy(dtype=np.int8))
        del df, df_valid

    X = np.vstack(x_chunks)
    y = np.concatenate(y_chunks)
    stats = {
        "train_fit_trips": len(trip_ids),
        "total_raw_candidate_rows": total_raw_rows,
        "excluded_invalid_rows": excluded_invalid_rows,
        "valid_training_rows": int(len(y)),
        "positive_rows": int(np.sum(y == 1)),
        "negative_rows": int(np.sum(y == 0)),
        "positive_rate_pct": round(float(np.mean(y == 1) * 100.0), 3),
    }
    return X, y, stats


def main() -> int:
    t0 = time.perf_counter()
    with open(PROJECT_ROOT / "configs" / "pipeline_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    with open(PROJECT_ROOT / "results" / "trip_splits.json", "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_fit_ids = splits["train_fit"]
    cand_train_dir = PROJECT_ROOT / "data" / "processed" / "candidates" / "train"

    print(f"Loading valid training candidates across {len(train_fit_ids)} train_fit trips...", flush=True)
    X_train, y_train, data_stats = load_valid_training_arrays(cand_train_dir, train_fit_ids)
    rss_after_load = get_process_rss_mb()
    print(
        f"Loaded X_train={X_train.shape} ({X_train.dtype}), "
        f"pos={data_stats['positive_rows']} ({data_stats['positive_rate_pct']}%), "
        f"RSS={rss_after_load:.1f} MB",
        flush=True,
    )

    sgd_params = {k: v for k, v in cfg["baseline_model"].items() if k != "model_type"}
    model_cfg = SGDBaselineConfig(**sgd_params)
    pipeline = build_sgd_baseline_pipeline(model_cfg)

    t_fit_start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_time_s = round(time.perf_counter() - t_fit_start, 3)
    peak_rss_mb = get_process_rss_mb()

    metadata = {
        "model_name": "preliminary_sgd_logistic_baseline",
        "config": cfg["baseline_model"],
        "feature_columns": FEATURE_COLUMNS,
        "train_fit_trip_ids": train_fit_ids,
        "training_data_stats": data_stats,
        "train_time_s": train_time_s,
        "peak_rss_mb": round(peak_rss_mb, 2),
        "total_script_time_s": round(time.perf_counter() - t0, 3),
    }

    artifact_path = PROJECT_ROOT / "artifacts" / "baseline_sgd_pipeline.joblib"
    save_baseline_artifact(pipeline, metadata, artifact_path)

    meta_path = PROJECT_ROOT / "artifacts" / "baseline_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    del X_train, y_train
    gc.collect()

    print("\n=== BASELINE TRAINING COMPLETE ===")
    print(f"Train fit time: {train_time_s}s | Peak RSS: {peak_rss_mb:.1f} MB")
    print(f"Saved artifact: {artifact_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

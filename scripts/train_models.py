"""Phase 2 Training Script: Train the mandatory five-model comparison suite.

Model 1 (SGD Logistic Regression) reuses the Phase 1 artifact.
Models 2-5 (Decision Tree, Random Forest, Extra Trees, HistGradientBoosting)
are trained strictly on train_fit (64 trips, valid observations only) with
per-model peak RSS and timing telemetry.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import psutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.features.feature_engineering import FEATURE_COLUMNS
from backend.app.models.model_comparison import (
    MODEL_DISPLAY_NAMES,
    MODEL_FAMILIES,
    PeakRSSMonitor,
    build_phase2_model,
    load_observation_grouped_dataset,
    save_model_artifact,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Phase 2 PathSnap models.")
    parser.add_argument(
        "--obs-fraction",
        type=float,
        default=1.0,
        help="Observation-level sampling fraction in (0, 1]. Default 1.0 (full train_fit).",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    with open(PROJECT_ROOT / "configs" / "pipeline_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    with open(PROJECT_ROOT / "results" / "trip_splits.json", "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_fit_ids = splits["train_fit"]
    cand_train_dir = PROJECT_ROOT / "data" / "processed" / "candidates" / "train"
    artifacts_dir = PROJECT_ROOT / "artifacts"
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Record Model 1 (existing SGD Logistic Regression baseline)
    sgd_artifact = artifacts_dir / "baseline_sgd_pipeline.joblib"
    sgd_meta_path = artifacts_dir / "baseline_metadata.json"
    if not sgd_artifact.is_file() or not sgd_meta_path.is_file():
        raise FileNotFoundError("Phase 1 SGD baseline artifact missing; run scripts/train_baseline.py first.")

    with open(sgd_meta_path, "r", encoding="utf-8") as f:
        sgd_meta = json.load(f)

    training_records: dict[str, dict] = {
        "sgd_logistic": {
            "model_key": "sgd_logistic",
            "display_name": MODEL_DISPLAY_NAMES["sgd_logistic"],
            "algorithm_family": MODEL_FAMILIES["sgd_logistic"],
            "artifact_path": str(sgd_artifact.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "artifact_size_bytes": int(sgd_artifact.stat().st_size),
            "artifact_size_mb": round(sgd_artifact.stat().st_size / (1024.0 * 1024.0), 4),
            "train_time_s": sgd_meta["train_time_s"],
            "peak_rss_mb": sgd_meta["peak_rss_mb"],
            "peak_vram_mb": 0.0,
            "data_stats": {
                **sgd_meta["training_data_stats"],
                "obs_fraction": 1.0,
            },
            "hyperparameters": sgd_meta["config"],
        }
    }

    # 2. Load train_fit data once
    avail_gb = psutil.virtual_memory().available / (1024.0 ** 3)
    print(f"System RAM available before load: {avail_gb:.2f} GB | obs_fraction={args.obs_fraction}", flush=True)

    X_train, y_train, _, data_stats = load_observation_grouped_dataset(
        cand_train_dir,
        train_fit_ids,
        obs_fraction=args.obs_fraction,
        random_state=42,
    )
    print(
        f"Loaded train_fit X={X_train.shape} ({X_train.dtype}), "
        f"valid_obs={data_stats['used_valid_gps_obs']}, "
        f"pos={data_stats['positive_rows']} ({data_stats['positive_rate_pct']}%)",
        flush=True,
    )

    model_order = [
        "decision_tree",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
    ]

    for model_key in model_order:
        params = cfg["phase2_models"][model_key]
        print(f"\nTraining [{model_key}] with params={params} ...", flush=True)
        model = build_phase2_model(model_key, params)

        with PeakRSSMonitor() as mem:
            t_fit = time.perf_counter()
            model.fit(X_train, y_train)
            fit_time_s = round(time.perf_counter() - t_fit, 3)

        meta = {
            "model_key": model_key,
            "display_name": MODEL_DISPLAY_NAMES[model_key],
            "algorithm_family": MODEL_FAMILIES[model_key],
            "feature_columns": FEATURE_COLUMNS,
            "hyperparameters": params,
            "data_stats": data_stats,
            "train_time_s": fit_time_s,
            "peak_rss_mb": round(mem.peak_rss_mb, 2),
            "peak_vram_mb": 0.0,
        }
        # Record actual iterations if HistGradientBoosting stopped early
        if hasattr(model, "n_iter_"):
            meta["n_iter_actual"] = int(model.n_iter_)

        art_path = artifacts_dir / f"{model_key}.joblib"
        art_bytes = save_model_artifact(model, meta, art_path)
        meta["artifact_path"] = str(art_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        meta["artifact_size_bytes"] = art_bytes
        meta["artifact_size_mb"] = round(art_bytes / (1024.0 * 1024.0), 4)

        training_records[model_key] = meta
        print(
            f"  Done [{model_key}]: train_time={fit_time_s}s | "
            f"peak_RSS={mem.peak_rss_mb:.1f} MB | artifact={meta['artifact_size_mb']:.2f} MB",
            flush=True,
        )
        del model
        gc.collect()

    del X_train, y_train
    gc.collect()

    summary_out = {
        "total_script_elapsed_s": round(time.perf_counter() - t0, 2),
        "models": training_records,
    }
    summary_path = results_dir / "model_training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2)

    print(f"\nSaved training summary: {summary_path} (total elapsed: {summary_out['total_script_elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

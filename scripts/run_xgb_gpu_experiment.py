"""Optional 6th Model Experiment: GPU-accelerated XGBoost on RTX 3050 6 GB.

Verifies CUDA runtime compatibility with xgboost before training on full
train_fit (1.94M rows, float32), logging Peak System RAM and Peak GPU VRAM.
"""

from __future__ import annotations

import gc
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.features.feature_engineering import FEATURE_COLUMNS
from backend.app.models.model_comparison import (
    PeakRSSMonitor,
    load_observation_grouped_dataset,
    save_model_artifact,
)


def query_gpu_vram_used_mb() -> float:
    """Return current GPU memory.used in MiB via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=2.0,
        ).strip()
        return float(out.splitlines()[0].strip())
    except Exception:
        return 0.0


class PeakVRAMMonitor:
    """Context manager polling GPU VRAM used (MiB) every 100 ms."""

    def __init__(self) -> None:
        self.baseline_vram_mb = query_gpu_vram_used_mb()
        self.peak_vram_mb = self.baseline_vram_mb
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll(self) -> None:
        while not self._stop.is_set():
            v = query_gpu_vram_used_mb()
            if v > self.peak_vram_mb:
                self.peak_vram_mb = v
            time.sleep(0.1)

    def __enter__(self) -> PeakVRAMMonitor:
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        v = query_gpu_vram_used_mb()
        if v > self.peak_vram_mb:
            self.peak_vram_mb = v


def main() -> int:
    try:
        import xgboost as xgb
    except ImportError as exc:
        print(f"SKIP: xgboost not installed ({exc})")
        return 0

    print(f"Detected xgboost version: {xgb.__version__}", flush=True)

    # 1. Verify actual CUDA GPU execution on a small synthetic probe
    try:
        probe = xgb.XGBClassifier(
            n_estimators=5,
            max_depth=3,
            tree_method="hist",
            device="cuda",
            random_state=42,
        )
        rng = np.random.default_rng(42)
        xp = rng.normal(size=(200, 10)).astype(np.float32)
        yp = rng.integers(0, 2, size=200, dtype=np.int8)
        probe.fit(xp, yp)
        _ = probe.predict_proba(xp)
        print("CUDA GPU execution verified cleanly on synthetic probe.", flush=True)
    except Exception as exc:
        print(f"SKIP: CUDA GPU execution failed verification: {exc}")
        return 0

    # 2. Train on full train_fit (1.94M x 26 float32)
    with open(PROJECT_ROOT / "results" / "trip_splits.json", "r", encoding="utf-8") as f:
        splits = json.load(f)

    cand_train_dir = PROJECT_ROOT / "data" / "processed" / "candidates" / "train"
    artifacts_dir = PROJECT_ROOT / "artifacts"
    results_dir = PROJECT_ROOT / "results"

    X_train, y_train, _, data_stats = load_observation_grouped_dataset(
        cand_train_dir, splits["train_fit"], obs_fraction=1.0, random_state=42
    )

    xgb_params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.08,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "device": "cuda",
        "eval_metric": "logloss",
        "random_state": 42,
    }
    model = xgb.XGBClassifier(**xgb_params)

    print(f"Training [xgboost_gpu] on X={X_train.shape} ({X_train.dtype})...", flush=True)
    with PeakRSSMonitor() as rss_mon, PeakVRAMMonitor() as vram_mon:
        t_fit = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time_s = round(time.perf_counter() - t_fit, 3)

    vram_delta_mb = round(max(0.0, vram_mon.peak_vram_mb - vram_mon.baseline_vram_mb), 1)
    meta = {
        "model_key": "xgboost_gpu",
        "display_name": "XGBoost (CUDA GPU)",
        "algorithm_family": "Gradient Boosting (GPU)",
        "xgboost_version": xgb.__version__,
        "feature_columns": FEATURE_COLUMNS,
        "hyperparameters": xgb_params,
        "data_stats": data_stats,
        "train_time_s": fit_time_s,
        "peak_rss_mb": round(rss_mon.peak_rss_mb, 2),
        "peak_vram_mb": round(vram_mon.peak_vram_mb, 1),
        "vram_delta_mb": vram_delta_mb,
    }

    art_path = artifacts_dir / "xgboost_gpu.joblib"
    art_bytes = save_model_artifact(model, meta, art_path)
    meta["artifact_path"] = str(art_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    meta["artifact_size_bytes"] = art_bytes
    meta["artifact_size_mb"] = round(art_bytes / (1024.0 * 1024.0), 4)

    summary_path = results_dir / "model_training_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        train_summary = json.load(f)
    train_summary["models"]["xgboost_gpu"] = meta
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(train_summary, f, indent=2)

    del X_train, y_train, model
    gc.collect()

    print(
        f"Done [xgboost_gpu]: train_time={fit_time_s}s | peak_RSS={rss_mon.peak_rss_mb:.1f} MB | "
        f"peak_VRAM={vram_mon.peak_vram_mb:.1f} MiB (+{vram_delta_mb} MiB) | artifact={meta['artifact_size_mb']:.2f} MB",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

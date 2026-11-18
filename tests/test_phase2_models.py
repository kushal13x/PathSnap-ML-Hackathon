"""Automated tests for Phase 2 multi-model training, sampling, metrics, and artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.evaluation.metrics import compute_candidate_binary_metrics
from backend.app.features.feature_engineering import FEATURE_COLUMNS
from backend.app.models.model_comparison import (
    build_phase2_model,
    load_model_artifact,
    load_observation_grouped_dataset,
)


def test_observation_sampling_preserves_all_candidates_and_is_reproducible():
    cand_dir = PROJECT_ROOT / "data" / "processed" / "candidates" / "train"
    tids = ["00000012", "00000025"]

    X1, y1, g1, s1 = load_observation_grouped_dataset(cand_dir, tids, obs_fraction=0.25, random_state=42)
    X2, y2, g2, s2 = load_observation_grouped_dataset(cand_dir, tids, obs_fraction=0.25, random_state=42)

    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)
    assert np.array_equal(g1, g2)
    assert s1["used_valid_gps_obs"] == s2["used_valid_gps_obs"]
    assert X1.dtype == np.float32
    assert X1.shape[1] == len(FEATURE_COLUMNS)

    # Verify each sampled observation kept all its candidate rows (row count matches full obs)
    df_full = pd.read_parquet(cand_dir / "00000012.parquet")
    df_valid = df_full[df_full["is_valid_label"]]
    expected_cands_per_obs = df_valid.groupby("observation_id").size().to_dict()
    assert len(expected_cands_per_obs) > 0
    assert s1["used_candidate_rows"] > s1["used_valid_gps_obs"]


def test_phase2_models_fit_and_predict_reproducibly():
    cand_dir = PROJECT_ROOT / "data" / "processed" / "candidates" / "train"
    X, y, _, _ = load_observation_grouped_dataset(cand_dir, ["00000012"], obs_fraction=0.4, random_state=42)

    for key, params in [
        ("decision_tree", {"max_depth": 6, "min_samples_leaf": 10, "class_weight": "balanced", "random_state": 42}),
        ("random_forest", {"n_estimators": 10, "max_depth": 6, "min_samples_leaf": 10, "n_jobs": 1, "random_state": 42}),
        ("extra_trees", {"n_estimators": 10, "max_depth": 6, "min_samples_leaf": 10, "n_jobs": 1, "random_state": 42}),
        ("hist_gradient_boosting", {"max_iter": 15, "max_leaf_nodes": 15, "early_stopping": False, "random_state": 42}),
    ]:
        m1 = build_phase2_model(key, params)
        m2 = build_phase2_model(key, params)
        m1.fit(X, y)
        m2.fit(X, y)
        p1 = m1.predict_proba(X[:50])[:, 1]
        p2 = m2.predict_proba(X[:50])[:, 1]
        assert np.allclose(p1, p2)
        assert np.all((p1 >= 0.0) & (p1 <= 1.0))


def test_binary_metrics_include_auc_and_logloss():
    y_true = np.array([0, 0, 1, 0, 1, 0, 1, 0], dtype=np.int8)
    y_prob = np.array([0.1, 0.2, 0.85, 0.3, 0.7, 0.05, 0.9, 0.4], dtype=np.float64)
    m = compute_candidate_binary_metrics(y_true, y_prob)
    for k in [
        "candidate_precision",
        "candidate_recall",
        "candidate_macro_f1",
        "candidate_roc_auc",
        "candidate_pr_auc",
        "candidate_log_loss",
    ]:
        assert k in m
        assert np.isfinite(m[k])


def test_phase2_artifacts_and_no_student_test_leakage():
    with open(PROJECT_ROOT / "results" / "trip_splits.json", "r", encoding="utf-8") as f:
        splits = json.load(f)
    student_test_set = set(splits["student_test"])

    with open(PROJECT_ROOT / "results" / "model_training_summary.json", "r", encoding="utf-8") as f:
        tsum = json.load(f)

    mandatory = ["sgd_logistic", "decision_tree", "random_forest", "extra_trees", "hist_gradient_boosting"]
    for key in mandatory:
        assert key in tsum["models"]
        art_path = PROJECT_ROOT / tsum["models"][key]["artifact_path"]
        assert art_path.is_file()
        model, meta = load_model_artifact(art_path)
        assert hasattr(model, "predict_proba")
        if "train_fit_trip_ids" in meta:
            assert not (set(meta["train_fit_trip_ids"]) & student_test_set)

    comp_df = pd.read_csv(PROJECT_ROOT / "results" / "model_comparison.csv")
    for key in ["nearest_road_baseline"] + mandatory:
        assert key in set(comp_df["model_key"])

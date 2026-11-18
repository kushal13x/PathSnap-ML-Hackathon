"""Basic automated tests for Phase 1 PathSnap data, candidate, label, and baseline pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.data.label_alignment import (
    LabelAlignmentConfig,
    align_track_to_route,
)
from backend.app.data.loaders import load_reference_route, load_trip_inputs
from backend.app.data.preprocessing import (
    LocalTangentProjection,
    compute_gps_kinematics,
    preprocess_road_network,
)
from backend.app.features.candidate_generation import (
    CandidateConfig,
    generate_candidates_for_trip,
    validate_spatial_index_vs_bruteforce,
)
from backend.app.features.feature_engineering import (
    FEATURE_COLUMNS,
    build_candidate_feature_table,
)


def test_spatial_index_matches_bruteforce_with_long_arcs():
    data_root = PROJECT_ROOT / "data" / "raw"
    ti = load_trip_inputs(data_root, "train", "00000019")
    proj = LocalTangentProjection(float(ti.lon.mean()), float(ti.lat.mean()))
    px, py = proj.to_xy(ti.lon, ti.lat)
    road_net = preprocess_road_network(ti.node_lon, ti.node_lat, ti.arcs, proj)
    cfg = CandidateConfig(k_candidates=20, radius_m=100.0, short_arc_threshold_m=300.0)

    sample_idx = np.linspace(0, len(px) - 1, 25, dtype=int)
    res = validate_spatial_index_vs_bruteforce(px, py, road_net, cfg, sample_idx)
    assert res["missed_top1_count"] == 0
    assert res["missed_topk_count"] == 0
    assert res["max_distance_diff_m"] < 1e-6


def test_candidate_generation_and_features_leak_free():
    data_root = PROJECT_ROOT / "data" / "raw"
    ti = load_trip_inputs(data_root, "student_test", "00000052")
    proj = LocalTangentProjection(float(ti.lon.mean()), float(ti.lat.mean()))
    px, py = proj.to_xy(ti.lon, ti.lat)
    kin = compute_gps_kinematics(px, py, ti.timestamps)
    road_net = preprocess_road_network(ti.node_lon, ti.node_lat, ti.arcs, proj)
    cands = generate_candidates_for_trip(px, py, road_net, CandidateConfig())

    df = build_candidate_feature_table("00000052", kin, road_net, cands, alignment=None)
    assert "target" not in df.columns
    assert "is_valid_label" not in df.columns
    forbidden = {"trip_id", "candidate_arc_id", "lon", "lat", "target", "is_valid_label"}
    assert not (forbidden & set(FEATURE_COLUMNS))
    assert not df[FEATURE_COLUMNS].isna().any().any()
    for col in FEATURE_COLUMNS:
        assert df[col].dtype == np.float32


def test_monotonic_alignment_and_pause_handling():
    data_root = PROJECT_ROOT / "data" / "raw"
    # Trip 00000090 has a 153,064s pause
    ti = load_trip_inputs(data_root, "train", "00000090")
    route = load_reference_route(data_root, "train", "00000090")
    proj = LocalTangentProjection(float(ti.lon.mean()), float(ti.lat.mean()))
    px, py = proj.to_xy(ti.lon, ti.lat)
    kin = compute_gps_kinematics(px, py, ti.timestamps)
    road_net = preprocess_road_network(ti.node_lon, ti.node_lat, ti.arcs, proj)

    align = align_track_to_route(kin, road_net, route, LabelAlignmentConfig())
    assert len(align.assigned_route_idx) == len(ti.lon)
    assert np.all(np.diff(align.assigned_route_idx) >= 0)
    assert np.all(align.is_valid_label)  # All points on trip 90 are within 50m


def test_invalid_label_flagging_on_off_route_trip():
    data_root = PROJECT_ROOT / "data" / "raw"
    ti = load_trip_inputs(data_root, "train", "00000054")
    route = load_reference_route(data_root, "train", "00000054")
    proj = LocalTangentProjection(float(ti.lon.mean()), float(ti.lat.mean()))
    px, py = proj.to_xy(ti.lon, ti.lat)
    kin = compute_gps_kinematics(px, py, ti.timestamps)
    road_net = preprocess_road_network(ti.node_lon, ti.node_lat, ti.arcs, proj)

    align = align_track_to_route(kin, road_net, route, LabelAlignmentConfig())
    assert np.mean(~align.is_valid_label) > 0.80


def test_grouped_splits_and_generated_outputs():
    with open(PROJECT_ROOT / "results" / "trip_splits.json", "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_fit = set(splits["train_fit"])
    train_val = set(splits["train_val"])
    student_test = set(splits["student_test"])

    assert len(train_fit) == 64
    assert len(train_val) == 16
    assert len(student_test) == 10
    assert len(train_fit & train_val) == 0
    assert len(train_fit & student_test) == 0
    assert len(train_val & student_test) == 0

    assert (PROJECT_ROOT / "results" / "candidate_dataset_summary.csv").is_file()
    assert (PROJECT_ROOT / "results" / "baseline_metrics.json").is_file()
    assert (PROJECT_ROOT / "results" / "baseline_per_trip_metrics.csv").is_file()
    assert (PROJECT_ROOT / "artifacts" / "baseline_sgd_pipeline.joblib").is_file()

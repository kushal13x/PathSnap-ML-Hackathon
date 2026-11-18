"""Phase 1 Pipeline: Build per-trip candidate-level Parquet dataset & validation split.

Generates leak-free candidates (K=20, R=100m), offline monotonic DP training labels,
and 26 float32 features per (observation, candidate_arc) pair, saving per-trip
Parquet partitions under data/processed/candidates/{train,student_test}/.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.data.label_alignment import (
    LabelAlignmentConfig,
    align_track_to_route,
)
from backend.app.data.loaders import (
    get_process_rss_mb,
    iter_trip_ids,
    load_reference_route,
    load_trip_inputs,
)
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


def main() -> int:
    t0 = time.perf_counter()
    cfg_path = PROJECT_ROOT / "configs" / "pipeline_config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cand_cfg = CandidateConfig(**cfg["candidate_generation"])
    align_cfg = LabelAlignmentConfig(**cfg["label_alignment"])
    val_cfg = cfg["validation"]

    data_root = PROJECT_ROOT / "data" / "raw"
    out_cand_root = PROJECT_ROOT / "data" / "processed" / "candidates"
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Validate spatial index against brute-force on representative trips (incl. long arcs > 300m)
    index_check_trips = [
        ("train", "00000001"),
        ("train", "00000019"),
        ("train", "00000054"),
        ("train", "00000090"),
        ("student_test", "00000056"),
        ("student_test", "00000076"),
    ]
    index_validations = []
    rng = np.random.default_rng(42)
    print("Validating SpatialRoadIndex against brute-force scan...", flush=True)
    for sp, tid in index_check_trips:
        ti = load_trip_inputs(data_root, sp, tid)
        proj = LocalTangentProjection(float(ti.lon.mean()), float(ti.lat.mean()))
        px, py = proj.to_xy(ti.lon, ti.lat)
        road_net = preprocess_road_network(ti.node_lon, ti.node_lat, ti.arcs, proj)
        sample_n = min(40, len(px))
        sample_idx = np.sort(rng.choice(len(px), size=sample_n, replace=False))
        val_res = validate_spatial_index_vs_bruteforce(px, py, road_net, cand_cfg, sample_idx)
        val_res["split"] = sp
        val_res["trip_id"] = tid
        val_res["n_long_arcs_gt_300m"] = int(np.sum(road_net.arc_lengths_m > cand_cfg.short_arc_threshold_m))
        index_validations.append(val_res)
        del ti, road_net
        gc.collect()

    with open(results_dir / "spatial_index_validation.json", "w", encoding="utf-8") as f:
        json.dump(index_validations, f, indent=2)

    total_missed = sum(v["missed_top1_count"] + v["missed_topk_count"] for v in index_validations)
    print(f"  Spatial index validation complete: total missed top-1/top-K = {total_missed}", flush=True)

    # 2. Build trip-grouped train / val split over the 80 train trips
    train_ids = list(iter_trip_ids(data_root, "train"))
    test_ids = list(iter_trip_ids(data_root, "student_test"))

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=val_cfg["val_size"],
        random_state=val_cfg["random_state"],
    )
    dummy_x = np.zeros(len(train_ids))
    train_idx, val_idx = next(splitter.split(dummy_x, groups=train_ids))
    train_fit_ids = sorted([train_ids[i] for i in train_idx])
    train_val_ids = sorted([train_ids[i] for i in val_idx])

    split_manifest = {
        "train_fit": train_fit_ids,
        "train_val": train_val_ids,
        "student_test": sorted(test_ids),
        "val_size": val_cfg["val_size"],
        "random_state": val_cfg["random_state"],
    }
    with open(results_dir / "trip_splits.json", "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2)

    train_fit_set = set(train_fit_ids)
    train_val_set = set(train_val_ids)

    # 3. Process all 90 trips iteratively and write Parquet partitions
    summary_rows = []
    peak_rss_mb = get_process_rss_mb()

    for split in ["train", "student_test"]:
        split_out = out_cand_root / split
        split_out.mkdir(parents=True, exist_ok=True)
        trip_ids = train_ids if split == "train" else test_ids

        for idx, tid in enumerate(trip_ids, start=1):
            t_trip = time.perf_counter()
            # Load inference inputs ONLY (never .route)
            ti = load_trip_inputs(data_root, split, tid)
            proj = LocalTangentProjection(float(ti.lon.mean()), float(ti.lat.mean()))
            px, py = proj.to_xy(ti.lon, ti.lat)
            kinematics = compute_gps_kinematics(px, py, ti.timestamps)
            road_net = preprocess_road_network(ti.node_lon, ti.node_lat, ti.arcs, proj)

            # Candidate generation (strictly .route-free)
            cands = generate_candidates_for_trip(px, py, road_net, cand_cfg)

            # Offline label alignment (uses .route)
            route = load_reference_route(data_root, split, tid)
            alignment = align_track_to_route(kinematics, road_net, route, align_cfg)

            # Build feature table
            df_trip = build_candidate_feature_table(tid, kinematics, road_net, cands, alignment)
            parquet_path = split_out / f"{tid}.parquet"
            df_trip.to_parquet(parquet_path, index=False, compression="snappy")

            # Compute trip candidate/label diagnostics
            n_gps = len(ti.lon)
            n_valid = int(np.sum(alignment.is_valid_label))
            n_invalid = n_gps - n_valid
            n_ambig_valid = int(np.sum(alignment.is_valid_label & alignment.is_ambiguous))
            n_cand_rows = len(df_trip)
            n_pos_rows = int(df_trip["target"].sum()) if n_cand_rows > 0 else 0

            if n_cand_rows > 0:
                valid_pos_obs = df_trip.loc[
                    (df_trip["is_valid_label"]) & (df_trip["target"] == 1),
                    "observation_id",
                ].nunique()
                all_pos_obs = df_trip.loc[df_trip["target"] == 1, "observation_id"].nunique()
            else:
                valid_pos_obs = 0
                all_pos_obs = 0

            cand_recall_valid_pct = (valid_pos_obs / max(1, n_valid)) * 100.0 if n_valid > 0 else 0.0
            cand_recall_all_pct = (all_pos_obs / max(1, n_gps)) * 100.0

            sub_split = (
                "train_fit"
                if tid in train_fit_set and split == "train"
                else ("train_val" if tid in train_val_set and split == "train" else "student_test")
            )

            summary_rows.append(
                {
                    "split": split,
                    "cv_fold": sub_split,
                    "trip_id": tid,
                    "n_gps": n_gps,
                    "route_len": len(route),
                    "n_cand_rows": n_cand_rows,
                    "mean_cands_per_obs": round(n_cand_rows / max(1, n_gps), 2),
                    "n_obs_no_candidates": cands.n_obs_no_candidates,
                    "n_valid_labels": n_valid,
                    "n_invalid_labels": n_invalid,
                    "invalid_label_pct": round(n_invalid / max(1, n_gps) * 100.0, 2),
                    "n_ambiguous_valid_obs": n_ambig_valid,
                    "n_positive_rows": n_pos_rows,
                    "n_valid_obs_with_true_cand": int(valid_pos_obs),
                    "cand_recall_valid_pct": round(cand_recall_valid_pct, 2),
                    "cand_recall_all_pct": round(cand_recall_all_pct, 2),
                    "route_coverage_pct": round(alignment.route_coverage_pct, 2),
                    "aligned_dist_median_m": round(float(np.median(alignment.assigned_dist_m)), 2),
                    "aligned_dist_max_m": round(float(alignment.assigned_dist_m.max()), 2),
                    "parquet_bytes": parquet_path.stat().st_size,
                    "elapsed_s": round(time.perf_counter() - t_trip, 3),
                }
            )

            peak_rss_mb = max(peak_rss_mb, get_process_rss_mb())
            del ti, road_net, cands, alignment, df_trip, route
            gc.collect()

            if idx % 20 == 0 or idx == len(trip_ids):
                print(
                    f"  [{split}] {idx}/{len(trip_ids)} trips processed | RSS={get_process_rss_mb():.1f} MB",
                    flush=True,
                )

    df_sum = pd.DataFrame(summary_rows)
    sum_path = results_dir / "candidate_dataset_summary.csv"
    df_sum.to_csv(sum_path, index=False)

    total_elapsed = round(time.perf_counter() - t0, 2)
    print("\n=== CANDIDATE DATASET BUILD COMPLETE ===")
    print(f"Total elapsed: {total_elapsed}s | Peak RSS: {peak_rss_mb:.1f} MB")
    print(f"Features ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")
    for grp in ["train_fit", "train_val", "student_test"]:
        sub = df_sum[df_sum["cv_fold"] == grp]
        tot_g = int(sub["n_gps"].sum())
        tot_v = int(sub["n_valid_labels"].sum())
        tot_r = int(sub["n_cand_rows"].sum())
        tot_hit_v = int(sub["n_valid_obs_with_true_cand"].sum())
        print(
            f"  {grp:12s}: {len(sub):2d} trips | {tot_g:6d} obs ({tot_v:6d} valid, "
            f"{(tot_g-tot_v)/tot_g*100:.2f}% invalid) | {tot_r:7d} cand rows | "
            f"Valid Cand Recall = {tot_hit_v / max(1, tot_v) * 100:.2f}%"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

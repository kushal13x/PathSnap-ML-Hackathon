"""Phase 1 Pipeline: Evaluate preliminary baseline vs nearest-road-distance baseline.

Evaluates on train_fit (64), held-out train_val (16), and student_test (10),
reporting candidate-level, observation-level, and route-continuity metrics.
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
from backend.app.evaluation.metrics import (
    aggregate_per_trip_metrics,
    compute_candidate_binary_metrics,
    evaluate_trip_ranking,
)
from backend.app.features.feature_engineering import FEATURE_COLUMNS
from backend.app.models.baseline import (
    NearestRoadDistanceBaseline,
    load_baseline_artifact,
)


def evaluate_split(
    split_name: str,
    raw_split: str,
    trip_ids: list[str],
    pipeline,
    summary_df: pd.DataFrame,
    cand_root: Path,
    raw_root: Path,
) -> tuple[dict, pd.DataFrame]:
    """Evaluate both Nearest-Road baseline and SGD baseline on a trip group."""
    all_y_true_valid = []
    all_sgd_prob_valid = []
    all_dist_prob_valid = []

    sgd_trip_rows = []
    dist_trip_rows = []

    sgd_infer_time_s = 0.0
    dist_infer_time_s = 0.0

    sum_lookup = summary_df.set_index(["split", "trip_id"], drop=False)

    for tid in trip_ids:
        df = pd.read_parquet(cand_root / raw_split / f"{tid}.parquet")
        arcs = pd.read_csv(
            raw_root / raw_split / tid / f"{tid}.arcs",
            sep=r"\s+",
            header=None,
            engine="c",
            dtype=np.int64,
        ).to_numpy()

        obs_sum = sum_lookup.loc[(raw_split, tid)]
        X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

        t1 = time.perf_counter()
        sgd_probs = pipeline.predict_proba(X)[:, 1] if len(X) > 0 else np.empty(0)
        sgd_infer_time_s += time.perf_counter() - t1

        t2 = time.perf_counter()
        dist_probs = (
            NearestRoadDistanceBaseline.predict_proba(df["dist_to_seg_m"].to_numpy())
            if len(X) > 0
            else np.empty(0)
        )
        dist_infer_time_s += time.perf_counter() - t2

        if len(df) > 0:
            valid_mask = df["is_valid_label"].to_numpy(dtype=bool)
            if np.any(valid_mask):
                all_y_true_valid.append(df["target"].to_numpy()[valid_mask])
                all_sgd_prob_valid.append(sgd_probs[valid_mask])
                all_dist_prob_valid.append(dist_probs[valid_mask])

        m_sgd, _ = evaluate_trip_ranking(df, sgd_probs, obs_sum, arcs=arcs)
        m_sgd["cv_fold"] = split_name
        m_sgd["model"] = "sgd_logistic_baseline"
        sgd_trip_rows.append(m_sgd)

        m_dist, _ = evaluate_trip_ranking(df, dist_probs, obs_sum, arcs=arcs)
        m_dist["cv_fold"] = split_name
        m_dist["model"] = "nearest_road_baseline"
        dist_trip_rows.append(m_dist)

        del df, arcs, X
        gc.collect()

    y_valid = np.concatenate(all_y_true_valid)
    p_sgd_valid = np.concatenate(all_sgd_prob_valid)
    p_dist_valid = np.concatenate(all_dist_prob_valid)

    df_sgd_trips = pd.DataFrame(sgd_trip_rows)
    df_dist_trips = pd.DataFrame(dist_trip_rows)

    res = {
        "sgd_logistic_baseline": {
            **aggregate_per_trip_metrics(df_sgd_trips),
            **compute_candidate_binary_metrics(y_valid, p_sgd_valid),
            "inference_time_s": round(sgd_infer_time_s, 4),
        },
        "nearest_road_baseline": {
            **aggregate_per_trip_metrics(df_dist_trips),
            **compute_candidate_binary_metrics(y_valid, p_dist_valid),
            "inference_time_s": round(dist_infer_time_s, 4),
        },
    }
    combined_trips = pd.concat([df_sgd_trips, df_dist_trips], ignore_index=True)
    return res, combined_trips


def main() -> int:
    t0 = time.perf_counter()
    results_dir = PROJECT_ROOT / "results"
    cand_root = PROJECT_ROOT / "data" / "processed" / "candidates"
    raw_root = PROJECT_ROOT / "data" / "raw"

    with open(results_dir / "trip_splits.json", "r", encoding="utf-8") as f:
        splits = json.load(f)
    summary_df = pd.read_csv(results_dir / "candidate_dataset_summary.csv", dtype={"trip_id": str})
    pipeline, train_meta = load_baseline_artifact(
        PROJECT_ROOT / "artifacts" / "baseline_sgd_pipeline.joblib"
    )

    eval_configs = [
        ("train_fit", "train", splits["train_fit"]),
        ("train_val", "train", splits["train_val"]),
        ("student_test", "student_test", splits["student_test"]),
    ]

    metrics_out = {
        "train_time_s": train_meta["train_time_s"],
        "training_data_stats": train_meta["training_data_stats"],
        "splits": {},
    }
    all_per_trip = []

    for split_name, raw_split, tids in eval_configs:
        print(f"Evaluating {split_name} ({len(tids)} trips)...", flush=True)
        res, trip_df = evaluate_split(
            split_name, raw_split, tids, pipeline, summary_df, cand_root, raw_root
        )
        metrics_out["splits"][split_name] = res
        all_per_trip.append(trip_df)

    metrics_out["eval_peak_rss_mb"] = round(get_process_rss_mb(), 2)
    metrics_out["eval_elapsed_s"] = round(time.perf_counter() - t0, 2)

    with open(results_dir / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)

    per_trip_df = pd.concat(all_per_trip, ignore_index=True)
    per_trip_df.to_csv(results_dir / "baseline_per_trip_metrics.csv", index=False)

    print("\n=== BASELINE EVALUATION SUMMARY ===")
    for sname in ["train_fit", "train_val", "student_test"]:
        s = metrics_out["splits"][sname]
        sgd = s["sgd_logistic_baseline"]
        dst = s["nearest_road_baseline"]
        print(f"\n[{sname}] ({sgd['num_trips']} trips, {sgd['total_gps_obs']} obs, {sgd['invalid_label_pct_micro']}% invalid labels)")
        print(
            f"  Nearest-Road Baseline : Top1(valid)={dst['top1_acc_valid_pct_micro']:.2f}% | "
            f"Top5(valid)={dst['top5_acc_valid_pct_micro']:.2f}% | "
            f"Top1(all)={dst['top1_acc_all_obs_pct_micro']:.2f}% | "
            f"StepCont={dst['route_step_continuity_pct_mean']:.2f}% | "
            f"MacroF1={dst['candidate_macro_f1']:.4f}"
        )
        print(
            f"  SGD Logistic Baseline : Top1(valid)={sgd['top1_acc_valid_pct_micro']:.2f}% | "
            f"Top5(valid)={sgd['top5_acc_valid_pct_micro']:.2f}% | "
            f"Top1(all)={sgd['top1_acc_all_obs_pct_micro']:.2f}% | "
            f"StepCont={sgd['route_step_continuity_pct_mean']:.2f}% | "
            f"MacroF1={sgd['candidate_macro_f1']:.4f} | "
            f"Prec={sgd['candidate_precision']:.4f} | Rec={sgd['candidate_recall']:.4f} | "
            f"LogLoss={sgd['candidate_log_loss']:.4f} | Infer={sgd['inference_time_s']:.3f}s"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

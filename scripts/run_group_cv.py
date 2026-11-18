"""Phase 2 3-Fold GroupKFold Cross-Validation across the 80 training trips.

Uses reproducible observation-level sampling (preserving all candidates per
selected GPS observation) to run all 5 mandatory models across 3 trip-grouped
folds safely within available RAM.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.data.loaders import iter_trip_ids
from backend.app.evaluation.metrics import (
    aggregate_per_trip_metrics,
    compute_candidate_binary_metrics,
    evaluate_trip_ranking,
)
from backend.app.features.feature_engineering import FEATURE_COLUMNS
from backend.app.models.baseline import SGDBaselineConfig, build_sgd_baseline_pipeline
from backend.app.models.model_comparison import (
    MODEL_DISPLAY_NAMES,
    PeakRSSMonitor,
    build_phase2_model,
    load_observation_grouped_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 3-fold GroupKFold CV for Phase 2 models.")
    parser.add_argument("--obs-fraction", type=float, default=0.35, help="Observation sampling fraction for CV fit.")
    args = parser.parse_args()

    t0 = time.perf_counter()
    with open(PROJECT_ROOT / "configs" / "pipeline_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    data_root = PROJECT_ROOT / "data" / "raw"
    cand_train_dir = PROJECT_ROOT / "data" / "processed" / "candidates" / "train"
    results_dir = PROJECT_ROOT / "results"
    summary_df = pd.read_csv(results_dir / "candidate_dataset_summary.csv", dtype={"trip_id": str})
    sum_lookup = summary_df[summary_df["split"] == "train"].set_index("trip_id", drop=False)

    train_trip_ids = np.array(sorted(iter_trip_ids(data_root, "train")))
    gkf = GroupKFold(n_splits=cfg["group_kfold"]["n_splits"])

    model_keys = [
        "sgd_logistic",
        "decision_tree",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
    ]
    sgd_params = {k: v for k, v in cfg["baseline_model"].items() if k != "model_type"}

    fold_rows = []
    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(train_trip_ids, groups=train_trip_ids), start=1):
        tr_trips = train_trip_ids[tr_idx].tolist()
        va_trips = train_trip_ids[va_idx].tolist()
        print(f"\n=== Fold {fold_idx}/3 (train={len(tr_trips)} trips, val={len(va_trips)} trips) ===", flush=True)

        X_tr, y_tr, _, tr_stats = load_observation_grouped_dataset(
            cand_train_dir,
            tr_trips,
            obs_fraction=args.obs_fraction,
            random_state=42 + fold_idx,
        )
        print(
            f"  Loaded fold train X={X_tr.shape} (obs_fraction={args.obs_fraction}, "
            f"valid_obs={tr_stats['used_valid_gps_obs']})",
            flush=True,
        )

        for mkey in model_keys:
            if mkey == "sgd_logistic":
                model = build_sgd_baseline_pipeline(SGDBaselineConfig(**sgd_params))
            else:
                model = build_phase2_model(mkey, cfg["phase2_models"][mkey])

            with PeakRSSMonitor() as mem:
                t_fit = time.perf_counter()
                model.fit(X_tr, y_tr)
                fit_s = round(time.perf_counter() - t_fit, 3)

            # Evaluate on full held-out fold trips (no subsampling on validation!)
            all_y_val, all_p_val, trip_metrics = [], [], []
            t_inf_total = 0.0
            for tid in va_trips:
                df = pd.read_parquet(cand_train_dir / f"{tid}.parquet")
                arcs = pd.read_csv(
                    data_root / "train" / tid / f"{tid}.arcs",
                    sep=r"\s+",
                    header=None,
                    engine="c",
                    dtype=np.int64,
                ).to_numpy()
                obs_sum = sum_lookup.loc[tid]
                if len(df) > 0:
                    Xv = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
                    t_inf = time.perf_counter()
                    probs = model.predict_proba(Xv)[:, 1]
                    t_inf_total += time.perf_counter() - t_inf
                    del Xv
                    vmask = df["is_valid_label"].to_numpy(dtype=bool)
                    if np.any(vmask):
                        all_y_val.append(df["target"].to_numpy()[vmask])
                        all_p_val.append(probs[vmask])
                else:
                    probs = np.empty(0)
                mt, _ = evaluate_trip_ranking(df, probs, obs_sum, arcs=arcs)
                trip_metrics.append(mt)
                del df, arcs

            agg = aggregate_per_trip_metrics(pd.DataFrame(trip_metrics))
            bin_m = compute_candidate_binary_metrics(
                np.concatenate(all_y_val), np.concatenate(all_p_val)
            )
            row = {
                "fold": fold_idx,
                "model_key": mkey,
                "display_name": MODEL_DISPLAY_NAMES[mkey],
                "train_obs_used": tr_stats["used_valid_gps_obs"],
                "train_rows_used": tr_stats["used_candidate_rows"],
                "train_time_s": fit_s,
                "infer_time_s": round(t_inf_total, 4),
                "peak_rss_mb": round(mem.peak_rss_mb, 2),
                "top1_acc_valid": agg["top1_acc_valid_pct_micro"],
                "top5_acc_valid": agg["top5_acc_valid_pct_micro"],
                "step_continuity": agg["route_step_continuity_pct_mean"],
                "macro_f1": bin_m["candidate_macro_f1"],
                "roc_auc": bin_m["candidate_roc_auc"],
                "pr_auc": bin_m["candidate_pr_auc"],
                "log_loss": bin_m["candidate_log_loss"],
            }
            fold_rows.append(row)
            print(
                f"    [{mkey:22s}] Top1={row['top1_acc_valid']:.2f}% | "
                f"MacroF1={row['macro_f1']:.4f} | ROC-AUC={row['roc_auc']:.4f} | "
                f"LogLoss={row['log_loss']:.4f} | fit={fit_s}s",
                flush=True,
            )
            del model
            gc.collect()

        del X_tr, y_tr
        gc.collect()

    df_folds = pd.DataFrame(fold_rows)
    df_folds.to_csv(results_dir / "model_cv_folds.csv", index=False)

    # Aggregate mean and std across folds
    agg_rows = []
    for mkey in model_keys:
        sub = df_folds[df_folds["model_key"] == mkey]
        agg_rows.append(
            {
                "model_key": mkey,
                "display_name": MODEL_DISPLAY_NAMES[mkey],
                "cv_top1_mean": round(float(sub["top1_acc_valid"].mean()), 2),
                "cv_top1_std": round(float(sub["top1_acc_valid"].std(ddof=1)), 2),
                "cv_top5_mean": round(float(sub["top5_acc_valid"].mean()), 2),
                "cv_step_cont_mean": round(float(sub["step_continuity"].mean()), 2),
                "cv_macro_f1_mean": round(float(sub["macro_f1"].mean()), 4),
                "cv_macro_f1_std": round(float(sub["macro_f1"].std(ddof=1)), 4),
                "cv_roc_auc_mean": round(float(sub["roc_auc"].mean()), 4),
                "cv_pr_auc_mean": round(float(sub["pr_auc"].mean()), 4),
                "cv_log_loss_mean": round(float(sub["log_loss"].mean()), 4),
                "cv_fit_time_mean_s": round(float(sub["train_time_s"].mean()), 2),
            }
        )
    df_cv = pd.DataFrame(agg_rows)
    df_cv.to_csv(results_dir / "model_cv_summary.csv", index=False)

    with open(results_dir / "model_cv_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_splits": 3,
                "obs_fraction": args.obs_fraction,
                "elapsed_s": round(time.perf_counter() - t0, 2),
                "summary": agg_rows,
                "folds": fold_rows,
            },
            f,
            indent=2,
        )

    print(f"\n=== 3-FOLD GROUPKFOLD COMPLETE in {time.perf_counter() - t0:.2f}s ===")
    print(df_cv.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

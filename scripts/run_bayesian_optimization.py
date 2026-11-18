"""Phase 3 Script: Run Optuna TPE Bayesian Optimization on HistGradientBoostingClassifier.

Optimizes hyperparameters strictly on train_fit (64 trips via 3-fold GroupKFold),
refits on full train_fit, compares Tuned vs Untuned on held-out train_val (16 trips)
and frozen student_test (10 trips), and saves artifacts/plots/JSON summaries.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.evaluation.metrics import (
    aggregate_per_trip_metrics,
    compute_candidate_binary_metrics,
    evaluate_trip_ranking,
)
from backend.app.features.feature_engineering import FEATURE_COLUMNS
from backend.app.models.bayesian_optimization import (
    compute_calibration_stats,
    prepare_inner_cv_folds,
    run_hgb_optuna_search,
)
from backend.app.models.model_comparison import (
    PeakRSSMonitor,
    load_model_artifact,
    load_observation_grouped_dataset,
    save_model_artifact,
)


def evaluate_model_full(
    model: HistGradientBoostingClassifier,
    split_name: str,
    raw_split: str,
    trip_ids: list[str],
    summary_df: pd.DataFrame,
    cand_root: Path,
    raw_root: Path,
) -> dict:
    """Evaluate a classifier on a full trip partition including calibration stats."""
    sum_lookup = summary_df.set_index(["split", "trip_id"], drop=False)
    all_y, all_p, trip_rows = [], [], []
    infer_s = 0.0

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

        if len(df) > 0:
            X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
            t1 = time.perf_counter()
            probs = model.predict_proba(X)[:, 1]
            infer_s += time.perf_counter() - t1
            del X
            vm = df["is_valid_label"].to_numpy(dtype=bool)
            if np.any(vm):
                all_y.append(df["target"].to_numpy()[vm])
                all_p.append(probs[vm])
        else:
            probs = np.empty(0)

        mt, _ = evaluate_trip_ranking(df, probs, obs_sum, arcs=arcs)
        trip_rows.append(mt)
        del df, arcs

    yv = np.concatenate(all_y)
    pv = np.concatenate(all_p)
    agg = aggregate_per_trip_metrics(pd.DataFrame(trip_rows))
    bin_m = compute_candidate_binary_metrics(yv, pv)
    cal_m = compute_calibration_stats(yv, pv)
    return {
        **agg,
        **bin_m,
        **cal_m,
        "inference_time_s": round(infer_s, 4),
    }


def plot_optimization_history(trials_df: pd.DataFrame, out_path: Path) -> None:
    """Plot Optuna convergence: trial Top-1 + running best, and LogLoss vs Macro F1."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    t_nums = trials_df["trial_number"].to_numpy()
    top1 = trials_df["cv_top1_pct"].to_numpy()
    best_so_far = np.maximum.accumulate(top1)

    ax = axes[0]
    ax.scatter(t_nums, top1, color="#1f77b4", s=55, zorder=3, label="Trial 3-Fold CV Top-1")
    ax.plot(t_nums, best_so_far, color="#d62728", linewidth=2.2, zorder=2, label="Best So Far")
    ax.set_xlabel("Optuna Trial Number")
    ax.set_ylabel("Inner 3-Fold CV Top-1 Accuracy (%)")
    ax.set_title("(a) Bayesian Optimisation Convergence (train_fit CV)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    sc = ax.scatter(
        trials_df["cv_log_loss"],
        trials_df["cv_macro_f1"],
        c=trials_df["cv_top1_pct"],
        cmap="viridis",
        s=75,
        edgecolor="black",
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("CV Top-1 (%)")
    best_idx = int(np.argmax(top1))
    ax.scatter(
        [trials_df.iloc[best_idx]["cv_log_loss"]],
        [trials_df.iloc[best_idx]["cv_macro_f1"]],
        color="red",
        marker="*",
        s=180,
        label=f"Best Trial #{int(trials_df.iloc[best_idx]['trial_number'])}",
        zorder=5,
    )
    ax.set_xlabel("Inner CV Log Loss (lower is better)")
    ax.set_ylabel("Inner CV Macro F1 (higher is better)")
    ax.set_title("(b) Trial Calibration (Log Loss) vs Macro F1")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("PathSnap Phase 3 — Optuna TPE Search on HistGradientBoosting", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 3 Optuna Bayesian Optimization.")
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--timeout-s", type=float, default=360.0)
    parser.add_argument("--obs-fraction", type=float, default=0.45)
    args = parser.parse_args()

    t0 = time.perf_counter()
    results_dir = PROJECT_ROOT / "results"
    artifacts_dir = PROJECT_ROOT / "artifacts"
    viz_dir = PROJECT_ROOT / "visualizations"
    cand_root = PROJECT_ROOT / "data" / "processed" / "candidates"
    raw_root = PROJECT_ROOT / "data" / "raw"

    with open(results_dir / "trip_splits.json", "r", encoding="utf-8") as f:
        splits = json.load(f)
    summary_df = pd.read_csv(results_dir / "candidate_dataset_summary.csv", dtype={"trip_id": str})

    print(
        f"Preparing 3-fold GroupKFold inside train_fit (64 trips, obs_fraction={args.obs_fraction})...",
        flush=True,
    )
    folds = prepare_inner_cv_folds(
        cand_root / "train",
        splits["train_fit"],
        n_splits=3,
        train_obs_fraction=args.obs_fraction,
        random_state=42,
    )
    print(f"Folds ready. Running Optuna TPE ({args.n_trials} trials, timeout={args.timeout_s}s)...", flush=True)

    t_opt = time.perf_counter()
    study, trials_df = run_hgb_optuna_search(
        folds, n_trials=args.n_trials, timeout_s=args.timeout_s, random_state=42
    )
    opt_elapsed_s = round(time.perf_counter() - t_opt, 2)

    trials_csv = results_dir / "bayesian_optimization_trials.csv"
    trials_df.to_csv(trials_csv, index=False)
    plot_optimization_history(trials_df, viz_dir / "bayesian_optimization_history.png")

    del folds
    gc.collect()

    best_params = {
        **study.best_params,
        "early_stopping": True,
        "n_iter_no_change": 15,
        "random_state": 42,
    }
    print(f"\nBest Trial #{study.best_trial.number}: Inner CV Top-1 = {study.best_value:.3f}%")
    print(f"Best Params: {best_params}", flush=True)

    # Refit tuned model on full train_fit (64 trips, 1.94M rows)
    X_full, y_full, _, data_stats = load_observation_grouped_dataset(
        cand_root / "train", splits["train_fit"], obs_fraction=1.0, random_state=42
    )
    tuned_model = HistGradientBoostingClassifier(**best_params)
    with PeakRSSMonitor() as mem:
        t_fit = time.perf_counter()
        tuned_model.fit(X_full, y_full)
        refit_time_s = round(time.perf_counter() - t_fit, 3)

    tuned_meta = {
        "model_key": "hist_gradient_boosting_tuned",
        "display_name": "HistGradientBoosting (Tuned)",
        "feature_columns": FEATURE_COLUMNS,
        "hyperparameters": best_params,
        "best_trial_number": study.best_trial.number,
        "best_inner_cv_top1_pct": round(float(study.best_value), 4),
        "data_stats": data_stats,
        "train_time_s": refit_time_s,
        "peak_rss_mb": round(mem.peak_rss_mb, 2),
        "n_iter_actual": int(tuned_model.n_iter_),
    }
    tuned_art_path = artifacts_dir / "hist_gradient_boosting_tuned.joblib"
    tuned_bytes = save_model_artifact(tuned_model, tuned_meta, tuned_art_path)
    tuned_meta["artifact_size_mb"] = round(tuned_bytes / (1024.0 * 1024.0), 4)

    del X_full, y_full
    gc.collect()

    # Load untuned Phase 2 HistGradientBoosting for direct side-by-side comparison
    untuned_model, untuned_meta = load_model_artifact(
        artifacts_dir / "hist_gradient_boosting.joblib"
    )

    comparison = {"untuned": {}, "tuned": {}}
    for label, model in [("untuned", untuned_model), ("tuned", tuned_model)]:
        print(f"\nEvaluating [{label}] model...", flush=True)
        for sp_name, raw_sp, tids in [
            ("train_fit", "train", splits["train_fit"]),
            ("train_val", "train", splits["train_val"]),
            ("student_test", "student_test", splits["student_test"]),
        ]:
            m = evaluate_model_full(model, sp_name, raw_sp, tids, summary_df, cand_root, raw_root)
            comparison[label][sp_name] = m
            print(
                f"  {sp_name:12s}: Top1={m['top1_acc_valid_pct_micro']:.2f}% | "
                f"Top5={m['top5_acc_valid_pct_micro']:.2f}% | "
                f"MacroF1={m['candidate_macro_f1']:.4f} | "
                f"LogLoss={m['candidate_log_loss']:.4f} | "
                f"Brier={m['brier_score']:.5f} | ECE={m['ece_10bin']:.5f} | "
                f"StepCont={m['route_step_continuity_pct_mean']:.2f}% | Infer={m['inference_time_s']:.3f}s",
                flush=True,
            )

    summary_out = {
        "sampler": "TPESampler(seed=42)",
        "n_trials_completed": len(trials_df),
        "inner_cv_folds": 3,
        "inner_cv_obs_fraction": args.obs_fraction,
        "search_elapsed_s": opt_elapsed_s,
        "total_script_elapsed_s": round(time.perf_counter() - t0, 2),
        "best_trial": {
            "trial_number": study.best_trial.number,
            "inner_cv_top1_pct": round(float(study.best_value), 4),
            "user_attrs": study.best_trial.user_attrs,
            "params": best_params,
        },
        "untuned_meta": {
            "hyperparameters": untuned_meta["hyperparameters"],
            "train_time_s": untuned_meta["train_time_s"],
            "artifact_size_mb": round((artifacts_dir / "hist_gradient_boosting.joblib").stat().st_size / (1024**2), 4),
        },
        "tuned_meta": {
            "hyperparameters": tuned_meta["hyperparameters"],
            "train_time_s": tuned_meta["train_time_s"],
            "peak_rss_mb": tuned_meta["peak_rss_mb"],
            "n_iter_actual": tuned_meta["n_iter_actual"],
            "artifact_size_mb": tuned_meta["artifact_size_mb"],
        },
        "comparison": comparison,
    }

    out_json = results_dir / "bayesian_optimization_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2)

    print(f"\n=== PHASE 3 BAYESIAN OPTIMIZATION COMPLETE ({summary_out['total_script_elapsed_s']}s) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Phase 2 Evaluation & Plotting Script: Compare all trained models + baselines.

Evaluates on train_fit (64 trips), held-out train_val (16 trips), and
student_test (10 trips), saving CSV/JSON summaries and comparison plots.
"""

from __future__ import annotations

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.evaluation.metrics import (
    aggregate_per_trip_metrics,
    compute_candidate_binary_metrics,
    evaluate_trip_ranking,
)
from backend.app.features.feature_engineering import FEATURE_COLUMNS
from backend.app.models.baseline import NearestRoadDistanceBaseline
from backend.app.models.model_comparison import (
    MODEL_DISPLAY_NAMES,
    MODEL_FAMILIES,
    PeakRSSMonitor,
    load_model_artifact,
)


def evaluate_model_on_split(
    model_key: str,
    model_obj,
    split_name: str,
    raw_split: str,
    trip_ids: list[str],
    summary_df: pd.DataFrame,
    cand_root: Path,
    raw_root: Path,
) -> tuple[dict, list[dict]]:
    """Evaluate a single model (or nearest-road heuristic) on one partition."""
    sum_lookup = summary_df.set_index(["split", "trip_id"], drop=False)

    all_y_valid = []
    all_p_valid = []
    trip_rows = []
    infer_time_s = 0.0

    with PeakRSSMonitor() as mem:
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
                if model_key == "nearest_road_baseline":
                    dists = df["dist_to_seg_m"].to_numpy()
                    t1 = time.perf_counter()
                    probs = NearestRoadDistanceBaseline.predict_proba(dists)
                    infer_time_s += time.perf_counter() - t1
                else:
                    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
                    t1 = time.perf_counter()
                    probs = model_obj.predict_proba(X)[:, 1]
                    infer_time_s += time.perf_counter() - t1
                    del X

                valid_mask = df["is_valid_label"].to_numpy(dtype=bool)
                if np.any(valid_mask):
                    all_y_valid.append(df["target"].to_numpy()[valid_mask])
                    all_p_valid.append(probs[valid_mask])
            else:
                probs = np.empty(0, dtype=np.float64)

            m_trip, _ = evaluate_trip_ranking(df, probs, obs_sum, arcs=arcs)
            m_trip["cv_fold"] = split_name
            m_trip["model_key"] = model_key
            trip_rows.append(m_trip)
            del df, arcs

    y_valid = np.concatenate(all_y_valid)
    p_valid = np.concatenate(all_p_valid)
    df_trips = pd.DataFrame(trip_rows)

    agg = {
        **aggregate_per_trip_metrics(df_trips),
        **compute_candidate_binary_metrics(y_valid, p_valid),
        "inference_time_s": round(infer_time_s, 4),
        "eval_peak_rss_mb": round(mem.peak_rss_mb, 2),
    }
    return agg, trip_rows


def generate_comparison_plots(comp_df: pd.DataFrame, viz_dir: Path) -> None:
    """Create Phase 2 model comparison plots from actual experimental metrics."""
    viz_dir.mkdir(parents=True, exist_ok=True)
    models = comp_df["model_key"].tolist()
    labels = comp_df["display_name"].tolist()
    x = np.arange(len(models))
    width = 0.25

    # Plot 1: Top-1 & Top-5 Observation-Level Accuracy Comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=150)
    ax = axes[0]
    ax.bar(x - width, comp_df["train_fit_top1_valid"], width, label="train_fit (64)", color="#9ecae1")
    ax.bar(x, comp_df["train_val_top1_valid"], width, label="train_val (16)", color="#1f77b4")
    ax.bar(x + width, comp_df["student_test_top1_valid"], width, label="student_test (10)", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=9)
    ax.set_ylabel("Top-1 Accuracy on Valid Labels (%)")
    ax.set_ylim(40, 102)
    ax.set_title("(a) Observation-Level Top-1 Accuracy Comparison")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(x - width / 2, comp_df["train_val_top5_valid"], width, label="train_val Top-5", color="#2ca02c")
    ax.bar(x + width / 2, comp_df["student_test_top5_valid"], width, label="student_test Top-5", color="#98df8a", edgecolor="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=9)
    ax.set_ylabel("Top-5 Accuracy on Valid Labels (%)")
    ax.set_ylim(90, 101)
    ax.set_title("(b) Observation-Level Top-5 Accuracy Comparison")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle("PathSnap Phase 2 — Top-1 & Top-5 Model Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(viz_dir / "phase2_top1_top5_comparison.png")
    plt.close(fig)

    # Plot 2: Candidate Macro F1, ROC-AUC, PR-AUC, Log Loss & Step Continuity
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), dpi=150)
    ax = axes[0]
    ax.bar(x - width / 2, comp_df["train_val_macro_f1"], width, label="train_val", color="#1f77b4")
    ax.bar(x + width / 2, comp_df["student_test_macro_f1"], width, label="student_test", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=9)
    ax.set_ylabel("Candidate Macro F1")
    ax.set_ylim(0.45, 1.0)
    ax.set_title("(a) Candidate Macro F1 Comparison")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(x - width / 2, comp_df["train_val_roc_auc"], width, label="train_val ROC-AUC", color="#9467bd")
    ax.bar(x + width / 2, comp_df["train_val_pr_auc"], width, label="train_val PR-AUC", color="#c5b0d5", edgecolor="#9467bd")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=9)
    ax.set_ylabel("AUC Score")
    ax.set_ylim(0.35, 1.02)
    ax.set_title("(b) ROC-AUC & PR-AUC on Held-Out train_val")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[2]
    ax.bar(x - width / 2, comp_df["train_val_step_continuity"], width, label="train_val", color="#2ca02c")
    ax.bar(x + width / 2, comp_df["student_test_step_continuity"], width, label="student_test", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=9)
    ax.set_ylabel("Step Continuity (%)")
    ax.set_ylim(70, 100)
    ax.set_title("(c) Route Step Continuity (Classifier-Only)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle("PathSnap Phase 2 — Classification Quality & Route Continuity", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(viz_dir / "phase2_macro_f1_auc_comparison.png")
    plt.close(fig)

    # Plot 3: Training Time, Inference Time, Peak RAM & Artifact Size
    trained_df = comp_df[comp_df["model_key"] != "nearest_road_baseline"].copy()
    tx = np.arange(len(trained_df))
    t_labels = trained_df["display_name"].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5), dpi=150)
    ax = axes[0]
    ax.bar(tx, trained_df["train_time_s"], color="#1f77b4", alpha=0.85)
    ax.set_xticks(tx)
    ax.set_xticklabels(t_labels, rotation=22, ha="right", fontsize=9)
    ax.set_ylabel("Training Time (seconds)")
    ax.set_title("(a) Training Time on 1.94M Rows")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(tx, trained_df["train_val_infer_s"] + trained_df["student_test_infer_s"], color="#ff7f0e", alpha=0.85)
    ax.set_xticks(tx)
    ax.set_xticklabels(t_labels, rotation=22, ha="right", fontsize=9)
    ax.set_ylabel("Inference Time: val + student_test (s)")
    ax.set_title("(b) Inference Time (26 Trips / 773k Cands)")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[2]
    ax.bar(tx, trained_df["train_peak_rss_mb"], color="#2ca02c", alpha=0.85, label="Train Peak RSS (MB)")
    ax.set_xticks(tx)
    ax.set_xticklabels(t_labels, rotation=22, ha="right", fontsize=9)
    ax.set_ylabel("Peak Process RAM (MB)")
    ax.set_title("(c) Peak System RAM During Training")
    ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle("PathSnap Phase 2 — Compute Efficiency & Memory Footprint", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(viz_dir / "phase2_runtime_memory_comparison.png")
    plt.close(fig)


def main() -> int:
    t0 = time.perf_counter()
    results_dir = PROJECT_ROOT / "results"
    artifacts_dir = PROJECT_ROOT / "artifacts"
    cand_root = PROJECT_ROOT / "data" / "processed" / "candidates"
    raw_root = PROJECT_ROOT / "data" / "raw"
    viz_dir = PROJECT_ROOT / "visualizations"

    with open(results_dir / "trip_splits.json", "r", encoding="utf-8") as f:
        splits = json.load(f)
    with open(results_dir / "model_training_summary.json", "r", encoding="utf-8") as f:
        train_summary = json.load(f)
    summary_df = pd.read_csv(results_dir / "candidate_dataset_summary.csv", dtype={"trip_id": str})

    # Evaluate heuristic baseline + all trained models in training summary
    eval_order = ["nearest_road_baseline"] + list(train_summary["models"].keys())

    all_metrics: dict[str, dict] = {}
    all_per_trip_rows = []
    comparison_rows = []

    split_defs = [
        ("train_fit", "train", splits["train_fit"]),
        ("train_val", "train", splits["train_val"]),
        ("student_test", "student_test", splits["student_test"]),
    ]

    for model_key in eval_order:
        if model_key == "nearest_road_baseline":
            model_obj = NearestRoadDistanceBaseline()
            train_info = {
                "train_time_s": 0.0,
                "peak_rss_mb": 0.0,
                "peak_vram_mb": 0.0,
                "artifact_size_mb": 0.0,
            }
        else:
            train_info = train_summary["models"][model_key]
            art_path = PROJECT_ROOT / train_info["artifact_path"]
            model_obj, _ = load_model_artifact(art_path)

        print(f"\nEvaluating [{model_key}]...", flush=True)
        model_splits_res = {}
        for split_name, raw_split, tids in split_defs:
            agg, trip_rows = evaluate_model_on_split(
                model_key, model_obj, split_name, raw_split, tids, summary_df, cand_root, raw_root
            )
            model_splits_res[split_name] = agg
            all_per_trip_rows.extend(trip_rows)
            print(
                f"  {split_name:12s}: Top1(val)={agg['top1_acc_valid_pct_micro']:.2f}% | "
                f"Top5={agg['top5_acc_valid_pct_micro']:.2f}% | "
                f"MacroF1={agg['candidate_macro_f1']:.4f} | "
                f"ROC-AUC={agg['candidate_roc_auc']:.4f} | "
                f"PR-AUC={agg['candidate_pr_auc']:.4f} | "
                f"LogLoss={agg['candidate_log_loss']:.4f} | "
                f"StepCont={agg['route_step_continuity_pct_mean']:.2f}% | "
                f"Infer={agg['inference_time_s']:.3f}s",
                flush=True,
            )

        all_metrics[model_key] = {
            "display_name": MODEL_DISPLAY_NAMES.get(model_key, model_key),
            "algorithm_family": MODEL_FAMILIES.get(model_key, "Unknown"),
            "train_time_s": train_info["train_time_s"],
            "train_peak_rss_mb": train_info["peak_rss_mb"],
            "peak_vram_mb": train_info.get("peak_vram_mb", 0.0),
            "artifact_size_mb": train_info["artifact_size_mb"],
            "splits": model_splits_res,
        }

        tf = model_splits_res["train_fit"]
        tv = model_splits_res["train_val"]
        st = model_splits_res["student_test"]
        comparison_rows.append(
            {
                "model_key": model_key,
                "display_name": MODEL_DISPLAY_NAMES.get(model_key, model_key),
                "algorithm_family": MODEL_FAMILIES.get(model_key, "Unknown"),
                "train_time_s": train_info["train_time_s"],
                "train_peak_rss_mb": train_info["peak_rss_mb"],
                "peak_vram_mb": train_info.get("peak_vram_mb", 0.0),
                "artifact_size_mb": train_info["artifact_size_mb"],
                "train_fit_top1_valid": tf["top1_acc_valid_pct_micro"],
                "train_fit_macro_f1": tf["candidate_macro_f1"],
                "train_val_top1_valid": tv["top1_acc_valid_pct_micro"],
                "train_val_top5_valid": tv["top5_acc_valid_pct_micro"],
                "train_val_top1_all": tv["top1_acc_all_obs_pct_micro"],
                "train_val_step_continuity": tv["route_step_continuity_pct_mean"],
                "train_val_cand_miss_rate": tv["candidate_miss_rate_valid_pct_micro"],
                "train_val_precision": tv["candidate_precision"],
                "train_val_recall": tv["candidate_recall"],
                "train_val_pos_f1": tv["candidate_positive_f1"],
                "train_val_macro_f1": tv["candidate_macro_f1"],
                "train_val_roc_auc": tv["candidate_roc_auc"],
                "train_val_pr_auc": tv["candidate_pr_auc"],
                "train_val_log_loss": tv["candidate_log_loss"],
                "train_val_infer_s": tv["inference_time_s"],
                "student_test_top1_valid": st["top1_acc_valid_pct_micro"],
                "student_test_top5_valid": st["top5_acc_valid_pct_micro"],
                "student_test_top1_all": st["top1_acc_all_obs_pct_micro"],
                "student_test_step_continuity": st["route_step_continuity_pct_mean"],
                "student_test_cand_miss_rate": st["candidate_miss_rate_valid_pct_micro"],
                "student_test_precision": st["candidate_precision"],
                "student_test_recall": st["candidate_recall"],
                "student_test_pos_f1": st["candidate_positive_f1"],
                "student_test_macro_f1": st["candidate_macro_f1"],
                "student_test_roc_auc": st["candidate_roc_auc"],
                "student_test_pr_auc": st["candidate_pr_auc"],
                "student_test_log_loss": st["candidate_log_loss"],
                "student_test_infer_s": st["inference_time_s"],
            }
        )

        del model_obj
        gc.collect()

    comp_df = pd.DataFrame(comparison_rows)
    comp_csv = results_dir / "model_comparison.csv"
    comp_df.to_csv(comp_csv, index=False)

    per_trip_df = pd.DataFrame(all_per_trip_rows)
    per_trip_csv = results_dir / "model_per_trip_metrics.csv"
    per_trip_df.to_csv(per_trip_csv, index=False)

    metrics_json = results_dir / "model_metrics.json"
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "eval_elapsed_s": round(time.perf_counter() - t0, 2),
                "models": all_metrics,
            },
            f,
            indent=2,
        )

    print("\nGenerating Phase 2 comparison plots...", flush=True)
    generate_comparison_plots(comp_df, viz_dir)

    print("\n=== EVALUATION COMPLETE ===")
    print(f"Saved: {comp_csv}")
    print(f"Saved: {metrics_json}")
    print(f"Saved: {per_trip_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

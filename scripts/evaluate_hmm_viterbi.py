"""Phase 4 Script: Calibrate HMM transitions on train_fit and evaluate Classifier vs HMM/Viterbi.

Outputs:
  - configs/hmm_config.json
  - results/hmm_validation_metrics.json
  - results/hmm_student_test_metrics.json
  - results/hmm_difficult_trips_metrics.json
  - results/hmm_per_trip_metrics.csv
  - results/classifier_vs_hmm_comparison.csv
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

from backend.app.data.loaders import load_reference_route, load_trip_inputs
from backend.app.data.preprocessing import (
    LocalTangentProjection,
    compute_gps_kinematics,
    preprocess_road_network,
)
from backend.app.decoder.hmm import (
    EmissionConfig,
    build_hmm_observation_steps,
    compute_classifier_probabilities,
)
from backend.app.decoder.route_reconstruction import (
    evaluate_route_against_reference,
    reconstruct_route_from_decoded,
)
from backend.app.decoder.transition_model import DirectedRoadGraph, TransitionConfig
from backend.app.decoder.viterbi import decode_viterbi_sequence
from backend.app.evaluation.metrics import (
    aggregate_per_trip_metrics,
    evaluate_trip_ranking,
)
from backend.app.features.feature_engineering import FEATURE_COLUMNS
from backend.app.models.model_comparison import load_model_artifact


def evaluate_trip_both(
    raw_split: str,
    trip_id: str,
    classifier,
    obs_summary_row: pd.Series,
    cand_root: Path,
    raw_root: Path,
    emit_cfg: EmissionConfig,
    trans_cfg: TransitionConfig,
) -> tuple[dict, dict, dict]:
    """Run Classifier-Only and HMM/Viterbi on one trip without exposing .route to decoders."""
    ti = load_trip_inputs(raw_root, raw_split, trip_id)
    proj = LocalTangentProjection(float(ti.lon.mean()), float(ti.lat.mean()))
    px, py = proj.to_xy(ti.lon, ti.lat)
    kin = compute_gps_kinematics(px, py, ti.timestamps)
    road_net = preprocess_road_network(ti.node_lon, ti.node_lat, ti.arcs, proj)
    graph = DirectedRoadGraph(road_net)

    df_cands = pd.read_parquet(cand_root / raw_split / f"{trip_id}.parquet")
    t_clf = time.perf_counter()
    cand_probs = compute_classifier_probabilities(classifier, df_cands, FEATURE_COLUMNS)
    clf_infer_s = time.perf_counter() - t_clf

    # A. Classifier-Only Evaluation
    clf_obs_m, clf_pred_seq = evaluate_trip_ranking(
        df_cands, cand_probs, obs_summary_row, arcs=ti.arcs
    )
    # Count invalid transitions in classifier-only sequence
    clf_invalid_trans = 0
    for k in range(len(clf_pred_seq) - 1):
        a0, a1 = int(clf_pred_seq[k]), int(clf_pred_seq[k + 1])
        if a0 >= 0 and a1 >= 0 and a0 != a1 and int(ti.arcs[a0, 1]) != int(ti.arcs[a1, 0]):
            clf_invalid_trans += 1

    # Collapsed classifier sequence (no graph bridge) for sequence comparison against .route
    clf_collapsed = []
    for a in clf_pred_seq:
        ai = int(a)
        if ai >= 0 and (not clf_collapsed or clf_collapsed[-1] != ai):
            clf_collapsed.append(ai)
    n_c_trans = max(0, len(clf_collapsed) - 1)
    clf_route_cont = (
        sum(1 for k in range(n_c_trans) if int(ti.arcs[clf_collapsed[k], 1]) == int(ti.arcs[clf_collapsed[k + 1], 0]))
        / n_c_trans
        * 100.0
        if n_c_trans > 0
        else 100.0
    )

    # B. HMM + Viterbi + Route Reconstruction (strictly .route-free!)
    t_hmm = time.perf_counter()
    steps = build_hmm_observation_steps(
        len(ti.lon), ti.timestamps, kin, df_cands, cand_probs, emit_cfg
    )
    viterbi_res = decode_viterbi_sequence(steps, road_net, graph, trans_cfg)
    reconstructed = reconstruct_route_from_decoded(
        viterbi_res.decoded_obs_arcs,
        viterbi_res.segment_ids,
        road_net,
        graph,
        connector_budget_m=trans_cfg.max_route_search_m,
    )
    hmm_decode_s = time.perf_counter() - t_hmm

    # Evaluate HMM observation-level accuracy by giving Viterbi-selected arcs score 1.0
    hmm_scores = np.zeros(len(df_cands), dtype=np.float64)
    if len(df_cands) > 0:
        obs_ids = df_cands["observation_id"].to_numpy(dtype=np.int32)
        cand_arcs = df_cands["candidate_arc_id"].to_numpy(dtype=np.int64)
        selected_for_row = viterbi_res.decoded_obs_arcs[obs_ids]
        # Give chosen arc score=2.0 and fall back to classifier prob for Top-5 ordering
        hmm_scores = np.where(cand_arcs == selected_for_row, 2.0 + cand_probs, cand_probs)

    hmm_obs_m, hmm_pred_seq = evaluate_trip_ranking(
        df_cands, hmm_scores, obs_summary_row, arcs=ti.arcs
    )
    hmm_invalid_trans = 0
    for k in range(len(hmm_pred_seq) - 1):
        a0, a1 = int(hmm_pred_seq[k]), int(hmm_pred_seq[k + 1])
        if a0 >= 0 and a1 >= 0 and a0 != a1 and int(ti.arcs[a0, 1]) != int(ti.arcs[a1, 0]):
            hmm_invalid_trans += 1

    # C. Offline Reference Route Sequence Similarity (evaluation ONLY!)
    ref_route = load_reference_route(raw_root, raw_split, trip_id)
    clf_seq_m = evaluate_route_against_reference(clf_collapsed, ref_route, road_net)
    hmm_seq_m = evaluate_route_against_reference(
        reconstructed.reconstructed_route_arcs, ref_route, road_net
    )

    clf_row = {
        **clf_obs_m,
        "method": "classifier_only",
        "invalid_transitions": clf_invalid_trans,
        "reconstructed_route_continuity_pct": round(clf_route_cont, 2),
        "n_inserted_connectors": 0,
        "n_gap_splits": 0,
        "runtime_s": round(clf_infer_s, 4),
        **clf_seq_m,
    }
    hmm_row = {
        **hmm_obs_m,
        "method": "hmm_viterbi",
        "invalid_transitions": hmm_invalid_trans,
        "reconstructed_route_continuity_pct": reconstructed.route_continuity_pct,
        "n_inserted_connectors": reconstructed.n_inserted_connectors,
        "n_unconnected_gaps": reconstructed.n_unconnected_gaps,
        "n_gap_splits": viterbi_res.n_gap_splits,
        "n_viterbi_disc_steps": viterbi_res.n_disconnected_viterbi_steps,
        "runtime_s": round(clf_infer_s + hmm_decode_s, 4),
        "hmm_decode_only_s": round(hmm_decode_s, 4),
        **hmm_seq_m,
    }

    del ti, road_net, graph, df_cands, ref_route
    return clf_row, hmm_row, {"reconstructed_len": len(reconstructed.reconstructed_route_arcs)}


def calibrate_hmm_on_train_fit(
    classifier,
    train_fit_ids: list[str],
    summary_df: pd.DataFrame,
    cand_root: Path,
    raw_root: Path,
) -> tuple[EmissionConfig, TransitionConfig, list[dict]]:
    """Tune transition parameters strictly on a representative subset of train_fit trips."""
    # 6 diverse train_fit trips covering urban, highway, multi-gap, and pause trips
    calib_trips = [
        tid for tid in ["00000001", "00000012", "00000019", "00000025", "00000044", "00000090"]
        if tid in set(train_fit_ids)
    ]
    sum_lookup = summary_df.set_index(["split", "trip_id"], drop=False)

    grid = [
        (1.0, 15.0, 4.0),
        (1.0, 25.0, 4.0),
        (1.5, 20.0, 4.0),
        (1.5, 25.0, 6.0),
        (2.0, 20.0, 5.0),
        (2.0, 30.0, 6.0),
    ]

    records = []
    best_score = -1.0
    best_emit = EmissionConfig()
    best_trans = TransitionConfig()

    print(f"Calibrating HMM parameters strictly on {len(calib_trips)} train_fit trips...", flush=True)
    for ew, beta, uturn in grid:
        ec = EmissionConfig(prob_eps=1e-6, emit_weight=ew)
        tc = TransitionConfig(gap_threshold_s=30.0, beta_m=beta, uturn_penalty=uturn)
        hmm_rows = []
        for tid in calib_trips:
            _, hrow, _ = evaluate_trip_both(
                "train", tid, classifier, sum_lookup.loc[("train", tid)], cand_root, raw_root, ec, tc
            )
            hmm_rows.append(hrow)
        df_h = pd.DataFrame(hmm_rows)
        agg = aggregate_per_trip_metrics(df_h)
        mean_f1_len = float(df_h["length_weighted_f1_pct"].mean())
        mean_rcont = float(df_h["reconstructed_route_continuity_pct"].mean())
        # Combined objective balancing observation Top-1 and route length F1
        score = 0.5 * agg["top1_acc_valid_pct_micro"] + 0.5 * mean_f1_len
        rec = {
            "emit_weight": ew,
            "beta_m": beta,
            "uturn_penalty": uturn,
            "top1_valid_pct": agg["top1_acc_valid_pct_micro"],
            "length_weighted_f1_pct": round(mean_f1_len, 2),
            "route_continuity_pct": round(mean_rcont, 2),
            "combined_score": round(score, 3),
        }
        records.append(rec)
        print(
            f"  ew={ew}, beta={beta}, uturn={uturn} -> Top1={rec['top1_valid_pct']:.2f}%, "
            f"LenF1={rec['length_weighted_f1_pct']:.2f}%, RouteCont={rec['route_continuity_pct']:.2f}%",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_emit = ec
            best_trans = tc

    return best_emit, best_trans, records


def summarize_method_rows(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    agg = aggregate_per_trip_metrics(df)
    return {
        **agg,
        "total_invalid_transitions": int(df["invalid_transitions"].sum()),
        "mean_reconstructed_route_continuity_pct": round(float(df["reconstructed_route_continuity_pct"].mean()), 2),
        "mean_arc_set_f1_pct": round(float(df["arc_set_f1_pct"].mean()), 2),
        "mean_length_weighted_f1_pct": round(float(df["length_weighted_f1_pct"].mean()), 2),
        "mean_sequence_lcs_similarity_pct": round(float(df["sequence_lcs_similarity_pct"].mean()), 2),
        "total_inserted_connectors": int(df["n_inserted_connectors"].sum()),
        "total_runtime_s": round(float(df["runtime_s"].sum()), 3),
    }


def main() -> int:
    t0 = time.perf_counter()
    results_dir = PROJECT_ROOT / "results"
    cand_root = PROJECT_ROOT / "data" / "processed" / "candidates"
    raw_root = PROJECT_ROOT / "data" / "raw"
    configs_dir = PROJECT_ROOT / "configs"

    with open(results_dir / "trip_splits.json", "r", encoding="utf-8") as f:
        splits = json.load(f)
    summary_df = pd.read_csv(results_dir / "candidate_dataset_summary.csv", dtype={"trip_id": str})
    sum_lookup = summary_df.set_index(["split", "trip_id"], drop=False)

    tuned_path = PROJECT_ROOT / "artifacts" / "hist_gradient_boosting_tuned.joblib"
    if not tuned_path.is_file():
        raise FileNotFoundError(f"Missing tuned model artifact: {tuned_path}. Run Phase 3 first.")
    classifier, _ = load_model_artifact(tuned_path)

    # 1. Calibrate HMM parameters strictly on train_fit subset
    emit_cfg, trans_cfg, calib_records = calibrate_hmm_on_train_fit(
        classifier, splits["train_fit"], summary_df, cand_root, raw_root
    )
    hmm_cfg_out = {
        "emission": {"prob_eps": emit_cfg.prob_eps, "emit_weight": emit_cfg.emit_weight},
        "transition": {
            "gap_threshold_s": trans_cfg.gap_threshold_s,
            "beta_m": trans_cfg.beta_m,
            "uturn_penalty": trans_cfg.uturn_penalty,
            "backward_jitter_penalty": trans_cfg.backward_jitter_penalty,
            "disconnected_penalty": trans_cfg.disconnected_penalty,
            "max_route_search_m": trans_cfg.max_route_search_m,
        },
        "calibration_trials_on_train_fit": calib_records,
    }
    with open(configs_dir / "hmm_config.json", "w", encoding="utf-8") as f:
        json.dump(hmm_cfg_out, f, indent=2)

    all_trip_rows = []
    comparison_rows = []

    # 2. Evaluate on train_val (16) and student_test (10)
    for split_name, raw_split, tids, out_json in [
        ("train_val", "train", splits["train_val"], "hmm_validation_metrics.json"),
        ("student_test", "student_test", splits["student_test"], "hmm_student_test_metrics.json"),
    ]:
        print(f"\nEvaluating Classifier-Only vs HMM/Viterbi on {split_name} ({len(tids)} trips)...", flush=True)
        clf_rows, hmm_rows = [], []
        for tid in tids:
            crow, hrow, _ = evaluate_trip_both(
                raw_split, tid, classifier, sum_lookup.loc[(raw_split, tid)], cand_root, raw_root, emit_cfg, trans_cfg
            )
            crow["split_group"] = split_name
            hrow["split_group"] = split_name
            clf_rows.append(crow)
            hmm_rows.append(hrow)
            all_trip_rows.extend([crow, hrow])
            gc.collect()

        clf_sum = summarize_method_rows(clf_rows)
        hmm_sum = summarize_method_rows(hmm_rows)

        with open(results_dir / out_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "split": split_name,
                    "classifier_only": clf_sum,
                    "hmm_viterbi": hmm_sum,
                },
                f,
                indent=2,
            )

        for method_key, msum in [("classifier_only", clf_sum), ("hmm_viterbi", hmm_sum)]:
            comparison_rows.append({"split": split_name, "method": method_key, **msum})

        print(
            f"  [{split_name}] Classifier-Only: Top1(val)={clf_sum['top1_acc_valid_pct_micro']:.2f}% | "
            f"StepCont={clf_sum['route_step_continuity_pct_mean']:.2f}% | "
            f"InvalidTrans={clf_sum['total_invalid_transitions']} | "
            f"RouteCont={clf_sum['mean_reconstructed_route_continuity_pct']:.2f}% | "
            f"LenF1={clf_sum['mean_length_weighted_f1_pct']:.2f}% | "
            f"LCS={clf_sum['mean_sequence_lcs_similarity_pct']:.2f}%",
            flush=True,
        )
        print(
            f"  [{split_name}] HMM + Viterbi  : Top1(val)={hmm_sum['top1_acc_valid_pct_micro']:.2f}% | "
            f"StepCont={hmm_sum['route_step_continuity_pct_mean']:.2f}% | "
            f"InvalidTrans={hmm_sum['total_invalid_transitions']} | "
            f"RouteCont={hmm_sum['mean_reconstructed_route_continuity_pct']:.2f}% | "
            f"LenF1={hmm_sum['mean_length_weighted_f1_pct']:.2f}% | "
            f"LCS={hmm_sum['mean_sequence_lcs_similarity_pct']:.2f}% | "
            f"Connectors={hmm_sum['total_inserted_connectors']}",
            flush=True,
        )

    # 3. Evaluate difficult diagnostic trips specifically
    diff_trips = ["00000054", "00000033", "00000019", "00000090"]
    diff_results = []
    print("\nEvaluating difficult diagnostic trips...", flush=True)
    for tid in diff_trips:
        crow, hrow, _ = evaluate_trip_both(
            "train", tid, classifier, sum_lookup.loc[("train", tid)], cand_root, raw_root, emit_cfg, trans_cfg
        )
        crow["split_group"] = "difficult_diagnostic"
        hrow["split_group"] = "difficult_diagnostic"
        diff_results.extend([crow, hrow])
        all_trip_rows.extend([crow, hrow])
        print(
            f"  [train/{tid}] Clf Top1={crow['top1_acc_valid_pct']:.2f}% (InvTrans={crow['invalid_transitions']}, LenF1={crow['length_weighted_f1_pct']:.2f}%) "
            f"-> HMM Top1={hrow['top1_acc_valid_pct']:.2f}% (InvTrans={hrow['invalid_transitions']}, LenF1={hrow['length_weighted_f1_pct']:.2f}%, GapSplits={hrow['n_gap_splits']})",
            flush=True,
        )

    with open(results_dir / "hmm_difficult_trips_metrics.json", "w", encoding="utf-8") as f:
        json.dump(diff_results, f, indent=2)

    pd.DataFrame(all_trip_rows).to_csv(results_dir / "hmm_per_trip_metrics.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(results_dir / "classifier_vs_hmm_comparison.csv", index=False)

    print(f"\n=== PHASE 4 HMM/VITERBI EVALUATION COMPLETE ({time.perf_counter() - t0:.2f}s) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

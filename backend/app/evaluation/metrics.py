"""Candidate-level, observation-level, and sequence-level evaluation metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


def compute_candidate_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute binary candidate classification metrics on valid observations."""
    y_pred = (y_prob >= threshold).astype(np.int8)
    clipped_prob = np.clip(y_prob, 1e-7, 1.0 - 1e-7)
    return {
        "candidate_accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "candidate_precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "candidate_recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "candidate_macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "candidate_positive_f1": round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "candidate_log_loss": round(float(log_loss(y_true, clipped_prob, labels=[0, 1])), 4),
    }


def evaluate_trip_ranking(
    df_cands: pd.DataFrame,
    scores: np.ndarray,
    obs_summary_row: pd.Series,
    arcs: np.ndarray | None = None,
) -> tuple[dict, np.ndarray]:
    """Convert candidate scores into predicted road-arc sequence for one trip and evaluate.

    Prediction rule: for each observation t, select argmax_{c in C_t} score(t, c).
    """
    n_obs_total = int(obs_summary_row["n_gps"])
    n_obs_valid = int(obs_summary_row["n_valid_labels"])
    n_obs_no_cand = int(obs_summary_row["n_obs_no_candidates"])
    cand_hit_valid = int(obs_summary_row["n_valid_obs_with_true_cand"])

    pred_seq = np.full(n_obs_total, -1, dtype=np.int64)
    if len(df_cands) == 0:
        metrics = {
            "trip_id": str(obs_summary_row["trip_id"]),
            "n_gps": n_obs_total,
            "n_valid_labels": n_obs_valid,
            "invalid_label_pct": round((n_obs_total - n_obs_valid) / max(1, n_obs_total) * 100.0, 2),
            "no_candidate_pct": round(n_obs_no_cand / max(1, n_obs_total) * 100.0, 2),
            "candidate_recall_valid_pct": 0.0,
            "candidate_miss_rate_valid_pct": 100.0,
            "top1_acc_valid_pct": 0.0,
            "top5_acc_valid_pct": 0.0,
            "top1_acc_all_obs_pct": 0.0,
            "route_step_continuity_pct": 0.0,
            "route_transition_continuity_pct": 0.0,
        }
        return metrics, pred_seq

    obs_ids = df_cands["observation_id"].to_numpy()
    cand_arcs = df_cands["candidate_arc_id"].to_numpy()
    targets = df_cands["target"].to_numpy()
    is_valid = df_cands["is_valid_label"].to_numpy(dtype=bool)

    breaks = np.flatnonzero(np.diff(obs_ids) != 0) + 1
    starts = np.concatenate([[0], breaks])
    ends = np.concatenate([breaks, [len(obs_ids)]])

    top1_hits_valid = 0
    top5_hits_valid = 0
    top1_hits_all = 0
    valid_obs_with_cands = 0

    for s, e in zip(starts, ends):
        oid = int(obs_ids[s])
        sub_scores = scores[s:e]
        sub_arcs = cand_arcs[s:e]
        sub_targets = targets[s:e]
        obs_valid = bool(is_valid[s])

        # Descending score order; deterministic tie-break by appearance (distance rank)
        order = np.argsort(-sub_scores, kind="mergesort")
        pred_seq[oid] = int(sub_arcs[order[0]])

        is_top1 = bool(sub_targets[order[0]] == 1)
        is_top5 = bool(np.any(sub_targets[order[:5]] == 1))

        if is_top1:
            top1_hits_all += 1
        if obs_valid:
            valid_obs_with_cands += 1
            if is_top1:
                top1_hits_valid += 1
            if is_top5:
                top5_hits_valid += 1

    # Compute predicted route continuity if arcs array is available
    step_cont_pct = np.nan
    trans_cont_pct = np.nan
    if arcs is not None and n_obs_total > 1:
        a_prev = pred_seq[:-1]
        a_curr = pred_seq[1:]
        both_matched = (a_prev >= 0) & (a_curr >= 0)
        same_arc = both_matched & (a_prev == a_curr)
        connected = np.zeros(n_obs_total - 1, dtype=bool)
        if np.any(both_matched):
            u_curr = arcs[a_curr[both_matched], 0]
            v_prev = arcs[a_prev[both_matched], 1]
            conn_vals = v_prev == u_curr
            connected[both_matched] = conn_vals

        step_cont_pct = round(float(np.mean(same_arc | connected) * 100.0), 2)
        changed = both_matched & (a_prev != a_curr)
        n_changed = int(np.sum(changed))
        if n_changed > 0:
            trans_cont_pct = round(float(np.sum(connected & changed) / n_changed * 100.0), 2)
        else:
            trans_cont_pct = 100.0

    cand_recall = cand_hit_valid / max(1, n_obs_valid) * 100.0
    metrics = {
        "trip_id": str(obs_summary_row["trip_id"]),
        "n_gps": n_obs_total,
        "n_valid_labels": n_obs_valid,
        "invalid_label_pct": round((n_obs_total - n_obs_valid) / max(1, n_obs_total) * 100.0, 2),
        "no_candidate_pct": round(n_obs_no_cand / max(1, n_obs_total) * 100.0, 2),
        "candidate_recall_valid_pct": round(cand_recall, 2),
        "candidate_miss_rate_valid_pct": round(100.0 - cand_recall, 2),
        "top1_acc_valid_pct": round(top1_hits_valid / max(1, n_obs_valid) * 100.0, 2),
        "top5_acc_valid_pct": round(top5_hits_valid / max(1, n_obs_valid) * 100.0, 2),
        "top1_acc_all_obs_pct": round(top1_hits_all / max(1, n_obs_total) * 100.0, 2),
        "route_step_continuity_pct": step_cont_pct,
        "route_transition_continuity_pct": trans_cont_pct,
    }
    return metrics, pred_seq


def aggregate_per_trip_metrics(per_trip_df: pd.DataFrame) -> dict[str, float]:
    """Compute micro-weighted and macro-averaged observation-level metrics across trips."""
    total_gps = int(per_trip_df["n_gps"].sum())
    total_valid = int(per_trip_df["n_valid_labels"].sum())

    def weighted_avg(col: str, weights_col: str) -> float:
        w = per_trip_df[weights_col].to_numpy(dtype=np.float64)
        v = per_trip_df[col].to_numpy(dtype=np.float64)
        return round(float(np.sum(w * v) / max(1.0, np.sum(w))), 2)

    return {
        "num_trips": int(len(per_trip_df)),
        "total_gps_obs": total_gps,
        "total_valid_obs": total_valid,
        "invalid_label_pct_micro": round((total_gps - total_valid) / max(1, total_gps) * 100.0, 2),
        "no_candidate_pct_micro": weighted_avg("no_candidate_pct", "n_gps"),
        "candidate_recall_valid_pct_micro": weighted_avg("candidate_recall_valid_pct", "n_valid_labels"),
        "candidate_miss_rate_valid_pct_micro": weighted_avg("candidate_miss_rate_valid_pct", "n_valid_labels"),
        "top1_acc_valid_pct_micro": weighted_avg("top1_acc_valid_pct", "n_valid_labels"),
        "top5_acc_valid_pct_micro": weighted_avg("top5_acc_valid_pct", "n_valid_labels"),
        "top1_acc_valid_pct_macro": round(float(per_trip_df["top1_acc_valid_pct"].mean()), 2),
        "top1_acc_all_obs_pct_micro": weighted_avg("top1_acc_all_obs_pct", "n_gps"),
        "route_step_continuity_pct_mean": round(float(per_trip_df["route_step_continuity_pct"].mean()), 2),
        "route_transition_continuity_pct_mean": round(float(per_trip_df["route_transition_continuity_pct"].mean()), 2),
    }

"""HMM emission probability computation from trained candidate classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from backend.app.data.preprocessing import GPSKinematics
from backend.app.features.feature_engineering import FEATURE_COLUMNS


@dataclass(frozen=True)
class EmissionConfig:
    """Configuration for converting classifier probabilities into log emissions."""

    prob_eps: float = 1e-6
    emit_weight: float = 1.0


@dataclass(frozen=True)
class HMMObservationStep:
    """Hidden states (candidate arc IDs) and log-emissions for one GPS observation."""

    obs_id: int
    timestamp_s: float
    px: float
    py: float
    cand_arcs: np.ndarray       # shape (K,), int64
    proj_t: np.ndarray          # shape (K,), float32
    probs: np.ndarray           # shape (K,), float64
    emit_log_probs: np.ndarray  # shape (K,), float64


def compute_classifier_probabilities(
    classifier: Any,
    df_cands: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> np.ndarray:
    """Score candidate rows using a trained classifier without touching .route."""
    if len(df_cands) == 0:
        return np.empty(0, dtype=np.float64)
    cols = feature_cols or FEATURE_COLUMNS
    X = df_cands[cols].to_numpy(dtype=np.float32)
    probs = classifier.predict_proba(X)[:, 1].astype(np.float64)
    return probs


def build_hmm_observation_steps(
    n_obs: int,
    timestamps: np.ndarray,
    kinematics: GPSKinematics,
    df_cands: pd.DataFrame,
    cand_probs: np.ndarray,
    cfg: EmissionConfig,
) -> list[HMMObservationStep]:
    """Assemble chronological HMMObservationStep list for all T GPS points."""
    steps_by_obs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    if len(df_cands) > 0:
        obs_ids = df_cands["observation_id"].to_numpy(dtype=np.int32)
        arcs = df_cands["candidate_arc_id"].to_numpy(dtype=np.int64)
        proj_t = df_cands["proj_t"].to_numpy(dtype=np.float32)

        breaks = np.flatnonzero(np.diff(obs_ids) != 0) + 1
        starts = np.concatenate([[0], breaks])
        ends = np.concatenate([breaks, [len(obs_ids)]])

        for s, e in zip(starts, ends):
            oid = int(obs_ids[s])
            p_slice = np.nan_to_num(cand_probs[s:e], nan=cfg.prob_eps, posinf=1.0, neginf=0.0)
            p_clip = np.clip(p_slice, cfg.prob_eps, 1.0 - cfg.prob_eps)
            steps_by_obs[oid] = (arcs[s:e], proj_t[s:e], p_clip)

    empty_i64 = np.empty(0, dtype=np.int64)
    empty_f32 = np.empty(0, dtype=np.float32)
    empty_f64 = np.empty(0, dtype=np.float64)

    steps: list[HMMObservationStep] = []
    for t in range(n_obs):
        if t in steps_by_obs:
            c_arcs, c_t, c_probs = steps_by_obs[t]
            emit_lp = cfg.emit_weight * np.log(c_probs)
        else:
            c_arcs, c_t, c_probs, emit_lp = empty_i64, empty_f32, empty_f64, empty_f64

        steps.append(
            HMMObservationStep(
                obs_id=t,
                timestamp_s=float(timestamps[t]),
                px=float(kinematics.px[t]),
                py=float(kinematics.py[t]),
                cand_arcs=c_arcs,
                proj_t=c_t,
                probs=c_probs,
                emit_log_probs=emit_lp,
            )
        )
    return steps

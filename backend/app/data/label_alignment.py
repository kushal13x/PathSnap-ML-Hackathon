"""Offline monotonic constrained DP label alignment for PathSnap.

Aligns chronological GPS observations (.track) to the chronological reference
route sequence (.route) using spatial distance, speed-gated heading consistency,
and monotonic forward progression. Used ONLY offline on training and
development evaluation data (never during candidate generation or live inference).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.app.data.preprocessing import (
    GPSKinematics,
    RoadNetworkMetric,
    angular_diff_deg,
    point_to_segments_distance,
)


@dataclass(frozen=True)
class LabelAlignmentConfig:
    """Configuration for offline monotonic DP route-to-track alignment."""

    sigma_dist_m: float = 15.0
    sigma_heading_deg: float = 45.0
    moving_speed_threshold_mps: float = 2.0
    step_penalty: float = 0.05
    valid_label_max_dist_m: float = 50.0
    ambiguous_margin_cost: float = 0.25


@dataclass(frozen=True)
class TripLabelAlignment:
    """Observation-level aligned reference labels and quality flags for one trip."""

    assigned_route_idx: np.ndarray  # shape (T,), int32 index into .route
    assigned_arc_id: np.ndarray     # shape (T,), int64 arc ID
    assigned_dist_m: np.ndarray     # shape (T,), float32 distance to aligned arc
    is_valid_label: np.ndarray      # shape (T,), bool (dist <= valid_label_max_dist_m)
    is_ambiguous: np.ndarray        # shape (T,), bool (distinct arc within margin)
    route_coverage_pct: float       # % of .route entries matched by >=1 valid obs


def align_track_to_route(
    kinematics: GPSKinematics,
    road_net: RoadNetworkMetric,
    route_arc_ids: np.ndarray,
    config: LabelAlignmentConfig,
) -> TripLabelAlignment:
    """Align T GPS observations monotonically to M reference route positions [0..M-1].

    Uses vectorized O(T * M) prefix-minimum recurrence via np.minimum.accumulate so
    even 19k-point trajectories align in milliseconds on an i5 CPU.
    """
    t_len = len(kinematics.px)
    m_len = len(route_arc_ids)

    if t_len == 0 or m_len == 0:
        return TripLabelAlignment(
            assigned_route_idx=np.empty(0, dtype=np.int32),
            assigned_arc_id=np.empty(0, dtype=np.int64),
            assigned_dist_m=np.empty(0, dtype=np.float32),
            is_valid_label=np.empty(0, dtype=bool),
            is_ambiguous=np.empty(0, dtype=bool),
            route_coverage_pct=0.0,
        )

    rax = road_net.ax[route_arc_ids]
    ray = road_net.ay[route_arc_ids]
    rbx = road_net.bx[route_arc_ids]
    rby = road_net.by[route_arc_ids]
    r_bearings = road_net.arc_bearings_deg[route_arc_ids]

    # Point-to-segment distances (T, M)
    d_mat, _ = point_to_segments_distance(
        kinematics.px[:, None],
        kinematics.py[:, None],
        rax[None, :],
        ray[None, :],
        rbx[None, :],
        rby[None, :],
    )

    # Emission cost: normalized squared distance + speed-gated heading difference
    emit = (d_mat / config.sigma_dist_m) ** 2
    moving = kinematics.speed_smooth_mps > config.moving_speed_threshold_mps
    if np.any(moving):
        ang_diff = angular_diff_deg(
            kinematics.heading_deg[moving, None], r_bearings[None, :]
        )
        emit[moving] += (ang_diff / config.sigma_heading_deg) ** 2

    # Vectorized monotonic prefix-minimum DP:
    # best_prev[k] = min_{j <= k} (dp[t-1, j] + step_penalty * (k - j))
    #              = step_penalty * k + min_{j <= k} (dp[t-1, j] - step_penalty * j)
    k_idx = np.arange(m_len, dtype=np.float64)
    k_int = np.arange(m_len, dtype=np.int32)
    penalty_ramp = config.step_penalty * k_idx

    dp = np.empty((t_len, m_len), dtype=np.float64)
    back = np.zeros((t_len, m_len), dtype=np.int32)

    # Mild start offset allows starting cleanly even if track begins slightly mid-route
    dp[0] = emit[0] + 0.5 * k_idx

    for t in range(1, t_len):
        shifted = dp[t - 1] - penalty_ramp
        cum_min = np.minimum.accumulate(shifted)

        is_new_min = np.empty(m_len, dtype=bool)
        is_new_min[0] = True
        if m_len > 1:
            is_new_min[1:] = shifted[1:] <= cum_min[:-1]

        best_prev_val = cum_min + penalty_ramp
        best_prev_idx = np.maximum.accumulate(np.where(is_new_min, k_int, 0))

        dp[t] = emit[t] + best_prev_val
        back[t] = best_prev_idx

    # Mild end offset prevents skipping straight to M-1 prematurely
    end_cost = dp[-1] + 0.5 * (m_len - 1 - k_idx)
    k_seq = np.empty(t_len, dtype=np.int32)
    k_seq[-1] = int(np.argmin(end_cost))
    for t in range(t_len - 1, 0, -1):
        k_seq[t - 1] = back[t, k_seq[t]]

    assigned_arc_id = route_arc_ids[k_seq].astype(np.int64)
    assigned_dist_m = d_mat[np.arange(t_len), k_seq].astype(np.float32)
    is_valid_label = assigned_dist_m <= config.valid_label_max_dist_m

    # Ambiguity check: emission margin against the best DIFFERENT arc ID in .route
    assigned_emit = emit[np.arange(t_len), k_seq]
    same_arc_mask = route_arc_ids[None, :] == assigned_arc_id[:, None]
    other_emit = np.where(same_arc_mask, np.inf, emit)
    second_best_emit = other_emit.min(axis=1)
    is_ambiguous = (second_best_emit - assigned_emit) < config.ambiguous_margin_cost

    if np.any(is_valid_label):
        matched_unique_k = len(np.unique(k_seq[is_valid_label]))
        route_coverage_pct = float(matched_unique_k / m_len * 100.0)
    else:
        route_coverage_pct = 0.0

    return TripLabelAlignment(
        assigned_route_idx=k_seq,
        assigned_arc_id=assigned_arc_id,
        assigned_dist_m=assigned_dist_m,
        is_valid_label=is_valid_label,
        is_ambiguous=is_ambiguous,
        route_coverage_pct=route_coverage_pct,
    )

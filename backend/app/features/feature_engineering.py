"""Leak-free feature engineering for (observation, candidate_arc) pairs.

Strictly excludes trip_id, raw absolute coordinates (lon/lat), candidate arc IDs,
future ground truth, and .route information from predictor features.
Uses float32 for continuous features to minimize RAM and Parquet disk footprint.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.data.label_alignment import TripLabelAlignment
from backend.app.data.preprocessing import (
    GPSKinematics,
    RoadNetworkMetric,
    angular_diff_deg,
)
from backend.app.features.candidate_generation import ObservationCandidates

FEATURE_COLUMNS: list[str] = [
    # Spatial & geometric proximity
    "dist_to_seg_m",
    "dist_to_start_m",
    "dist_to_end_m",
    "proj_t",
    "proj_dist_from_endpoint_m",
    "arc_length_m",
    "log_arc_length_m",
    # Kinematics & temporal context
    "gps_speed_smooth_mps",
    "gps_speed_prev_mps",
    "dt_prev_s",
    "log_dt_prev_s",
    "step_dist_prev_m",
    "is_moving",
    "heading_diff_deg",
    "cos_heading_diff",
    "heading_turn_deg",
    # Candidate-relative & temporal-continuity features
    "candidate_rank",
    "dist_diff_from_best_m",
    "dist_ratio_to_best",
    "n_cands_in_radius",
    "heading_rank_at_obs",
    "prev_obs_dist_m",
    "dist_change_from_prev_m",
    # Road-network local topology
    "from_node_deg",
    "to_node_deg",
    "is_bidirectional",
]


def build_candidate_feature_table(
    trip_id: str,
    kinematics: GPSKinematics,
    road_net: RoadNetworkMetric,
    candidates: ObservationCandidates,
    alignment: TripLabelAlignment | None = None,
) -> pd.DataFrame:
    """Construct candidate-level DataFrame for a single trip.

    If `alignment` is provided (offline train / student_test), attaches target
    and label-quality columns; otherwise (live inference) omits label columns.
    """
    n_cands = len(candidates.observation_id)
    if n_cands == 0:
        cols = (
            ["trip_id", "observation_id", "candidate_arc_id", "candidate_rank"]
            + [c for c in FEATURE_COLUMNS if c != "candidate_rank"]
        )
        if alignment is not None:
            cols += ["target", "is_valid_label", "is_ambiguous", "aligned_dist_m"]
        return pd.DataFrame(columns=cols)

    obs_id = candidates.observation_id
    arc_id = candidates.candidate_arc_id

    # Segment geometry
    arc_len = road_net.arc_lengths_m[arc_id].astype(np.float32)
    log_arc_len = np.log1p(arc_len).astype(np.float32)
    proj_t = candidates.proj_t
    proj_dist_end = (np.minimum(proj_t, 1.0 - proj_t) * arc_len).astype(np.float32)

    # GPS kinematics mapped to candidate rows
    spd_smooth = kinematics.speed_smooth_mps[obs_id].astype(np.float32)
    spd_prev = kinematics.speed_prev_mps[obs_id].astype(np.float32)
    dt_prev = kinematics.dt_prev_s[obs_id].astype(np.float32)
    log_dt_prev = np.log1p(dt_prev).astype(np.float32)
    step_prev = kinematics.step_dist_prev_m[obs_id].astype(np.float32)
    is_moving_bool = kinematics.is_moving[obs_id]
    is_moving = is_moving_bool.astype(np.float32)
    heading_turn = kinematics.heading_turn_deg[obs_id].astype(np.float32)

    # Heading difference (neutral 90 deg / cos=0 when vehicle is stationary)
    raw_hdiff = angular_diff_deg(
        kinematics.heading_deg[obs_id], road_net.arc_bearings_deg[arc_id]
    )
    heading_diff = np.where(is_moving_bool, raw_hdiff, 90.0).astype(np.float32)
    cos_hdiff = np.where(
        is_moving_bool, np.cos(np.radians(raw_hdiff)), 0.0
    ).astype(np.float32)

    # Candidate-relative features within each observation slice
    dist_m = candidates.dist_to_seg_m
    dist_diff_best = (dist_m - candidates.obs_min_dist_m).astype(np.float32)
    dist_ratio_best = ((dist_m + 1.0) / (candidates.obs_min_dist_m + 1.0)).astype(np.float32)
    dist_change_prev = (dist_m - candidates.prev_obs_dist_m).astype(np.float32)

    # Compute heading rank within each contiguous observation block
    heading_rank = np.ones(n_cands, dtype=np.float32)
    if is_moving_bool.any():
        breaks = np.flatnonzero(np.diff(obs_id) != 0) + 1
        starts = np.concatenate([[0], breaks])
        ends = np.concatenate([breaks, [n_cands]])
        for s, e in zip(starts, ends):
            if e - s > 1:
                sub_h = heading_diff[s:e]
                order = np.argsort(sub_h, kind="mergesort")
                ranks = np.empty(e - s, dtype=np.float32)
                ranks[order] = np.arange(1, e - s + 1, dtype=np.float32)
                heading_rank[s:e] = ranks

    # Road-network topology features
    from_deg = road_net.node_tot_deg[road_net.u[arc_id]].astype(np.float32)
    to_deg = road_net.node_tot_deg[road_net.v[arc_id]].astype(np.float32)
    is_bidir = road_net.is_bidirectional[arc_id].astype(np.float32)

    data: dict[str, np.ndarray | list[str]] = {
        "trip_id": np.full(n_cands, trip_id, dtype=object),
        "observation_id": obs_id,
        "candidate_arc_id": arc_id,
        "dist_to_seg_m": dist_m,
        "dist_to_start_m": candidates.dist_to_start_m,
        "dist_to_end_m": candidates.dist_to_end_m,
        "proj_t": proj_t,
        "proj_dist_from_endpoint_m": proj_dist_end,
        "arc_length_m": arc_len,
        "log_arc_length_m": log_arc_len,
        "gps_speed_smooth_mps": spd_smooth,
        "gps_speed_prev_mps": spd_prev,
        "dt_prev_s": dt_prev,
        "log_dt_prev_s": log_dt_prev,
        "step_dist_prev_m": step_prev,
        "is_moving": is_moving,
        "heading_diff_deg": heading_diff,
        "cos_heading_diff": cos_hdiff,
        "heading_turn_deg": heading_turn,
        "candidate_rank": candidates.candidate_rank.astype(np.float32),
        "dist_diff_from_best_m": dist_diff_best,
        "dist_ratio_to_best": dist_ratio_best,
        "n_cands_in_radius": candidates.n_cands_in_radius.astype(np.float32),
        "heading_rank_at_obs": heading_rank,
        "prev_obs_dist_m": candidates.prev_obs_dist_m,
        "dist_change_from_prev_m": dist_change_prev,
        "from_node_deg": from_deg,
        "to_node_deg": to_deg,
        "is_bidirectional": is_bidir,
    }

    if alignment is not None:
        true_arc = alignment.assigned_arc_id[obs_id]
        data["target"] = (arc_id == true_arc).astype(np.int8)
        data["is_valid_label"] = alignment.is_valid_label[obs_id]
        data["is_ambiguous"] = alignment.is_ambiguous[obs_id]
        data["aligned_dist_m"] = alignment.assigned_dist_m[obs_id]

    return pd.DataFrame(data)

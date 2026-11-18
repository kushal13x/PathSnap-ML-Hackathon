"""Leak-free spatial candidate generation for PathSnap.

Generates up to K nearest road-segment candidates within radius R (meters) for
every GPS observation using ONLY .track, .nodes, and .arcs (never .route).
Handles long road segments (> 300 m) without midpoint-only blind spots.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from backend.app.data.preprocessing import (
    RoadNetworkMetric,
    point_to_segments_distance,
)


@dataclass(frozen=True)
class CandidateConfig:
    """Configuration for spatial candidate retrieval."""

    k_candidates: int = 20
    radius_m: float = 100.0
    short_arc_threshold_m: float = 300.0


@dataclass(frozen=True)
class ObservationCandidates:
    """Flattened candidate-level records for a single trip (before feature matrix expansion)."""

    observation_id: np.ndarray     # shape (C,), int32
    candidate_arc_id: np.ndarray   # shape (C,), int64 (original 0-based .arcs row index)
    candidate_rank: np.ndarray     # shape (C,), int16 (1..K ordered by dist_to_seg_m)
    dist_to_seg_m: np.ndarray      # shape (C,), float32
    proj_t: np.ndarray             # shape (C,), float32
    dist_to_start_m: np.ndarray    # shape (C,), float32
    dist_to_end_m: np.ndarray      # shape (C,), float32
    prev_obs_dist_m: np.ndarray    # shape (C,), float32
    n_cands_in_radius: np.ndarray  # shape (C,), int16
    obs_min_dist_m: np.ndarray     # shape (C,), float32
    obs_cand_counts: np.ndarray    # shape (T,), int16 (total candidates in radius per obs)
    n_obs_total: int
    n_obs_no_candidates: int


class SpatialRoadIndex:
    """Hybrid spatial index: cKDTree on short-arc midpoints + bbox-exact long-arc check."""

    def __init__(
        self,
        road_net: RoadNetworkMetric,
        short_arc_threshold_m: float = 300.0,
    ) -> None:
        self.road_net = road_net
        self.short_thresh = float(short_arc_threshold_m)

        short_mask = road_net.arc_lengths_m <= self.short_thresh
        self.short_idx = np.flatnonzero(short_mask)
        self.long_idx = np.flatnonzero(~short_mask)

        if len(self.short_idx) > 0:
            mx = 0.5 * (road_net.ax[self.short_idx] + road_net.bx[self.short_idx])
            my = 0.5 * (road_net.ay[self.short_idx] + road_net.by[self.short_idx])
            self.short_tree: cKDTree | None = cKDTree(np.column_stack([mx, my]))
        else:
            self.short_tree = None

        if len(self.long_idx) > 0:
            lax, lbx = road_net.ax[self.long_idx], road_net.bx[self.long_idx]
            lay, lby = road_net.ay[self.long_idx], road_net.by[self.long_idx]
            self.long_xmin = np.minimum(lax, lbx)
            self.long_xmax = np.maximum(lax, lbx)
            self.long_ymin = np.minimum(lay, lby)
            self.long_ymax = np.maximum(lay, lby)
        else:
            self.long_xmin = np.empty(0, dtype=np.float64)
            self.long_xmax = np.empty(0, dtype=np.float64)
            self.long_ymin = np.empty(0, dtype=np.float64)
            self.long_ymax = np.empty(0, dtype=np.float64)

    def query_observation_candidates(
        self, px: float, py: float, radius_m: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (arc_ids, dists_m, proj_t) for all arcs within radius_m of (px, py)."""
        parts = []
        if self.short_tree is not None:
            ball_r = radius_m + 0.5 * self.short_thresh
            nb = self.short_tree.query_ball_point([px, py], r=ball_r)
            if nb:
                parts.append(self.short_idx[np.asarray(nb, dtype=np.int64)])

        if len(self.long_idx) > 0:
            lmask = (
                (self.long_xmax >= px - radius_m)
                & (self.long_xmin <= px + radius_m)
                & (self.long_ymax >= py - radius_m)
                & (self.long_ymin <= py + radius_m)
            )
            if np.any(lmask):
                parts.append(self.long_idx[lmask])

        if not parts:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
            )

        cand_arcs = parts[0] if len(parts) == 1 else np.concatenate(parts)
        d, t = point_to_segments_distance(
            px,
            py,
            self.road_net.ax[cand_arcs],
            self.road_net.ay[cand_arcs],
            self.road_net.bx[cand_arcs],
            self.road_net.by[cand_arcs],
        )
        within = d <= radius_m
        if not np.any(within):
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
            )
        return cand_arcs[within], d[within], t[within]


def generate_candidates_for_trip(
    px: np.ndarray,
    py: np.ndarray,
    road_net: RoadNetworkMetric,
    config: CandidateConfig,
) -> ObservationCandidates:
    """Generate top-K candidates within radius R for every GPS point in a trip."""
    n_obs = len(px)
    index = SpatialRoadIndex(road_net, short_arc_threshold_m=config.short_arc_threshold_m)

    # Batch cKDTree query for short arcs (much faster than per-point Python calls)
    if index.short_tree is not None and n_obs > 0:
        ball_r = config.radius_m + 0.5 * index.short_thresh
        short_neighbors = index.short_tree.query_ball_point(
            np.column_stack([px, py]), r=ball_r
        )
    else:
        short_neighbors = [[] for _ in range(n_obs)]

    # Pre-filter long arcs once against the entire track bbox + radius_m
    if len(index.long_idx) > 0 and n_obs > 0:
        txmin, txmax = float(px.min()) - config.radius_m, float(px.max()) + config.radius_m
        tymin, tymax = float(py.min()) - config.radius_m, float(py.max()) + config.radius_m
        track_lmask = (
            (index.long_xmax >= txmin)
            & (index.long_xmin <= txmax)
            & (index.long_ymax >= tymin)
            & (index.long_ymin <= tymax)
        )
        track_long_idx = index.long_idx[track_lmask]
        track_lxmin = index.long_xmin[track_lmask]
        track_lxmax = index.long_xmax[track_lmask]
        track_lymin = index.long_ymin[track_lmask]
        track_lymax = index.long_ymax[track_lmask]
    else:
        track_long_idx = np.empty(0, dtype=np.int64)
        track_lxmin = track_lxmax = track_lymin = track_lymax = np.empty(0)

    obs_ids_list = []
    cand_arcs_list = []
    ranks_list = []
    dists_list = []
    proj_t_list = []
    d_start_list = []
    d_end_list = []
    prev_dist_list = []
    n_in_rad_list = []
    min_dist_list = []

    obs_cand_counts = np.zeros(n_obs, dtype=np.int16)
    n_obs_no_candidates = 0

    for t in range(n_obs):
        nb = short_neighbors[t]
        short_c = index.short_idx[np.asarray(nb, dtype=np.int64)] if nb else np.empty(0, dtype=np.int64)
        if len(track_long_idx) > 0:
            lmask = (
                (track_lxmax >= px[t] - config.radius_m)
                & (track_lxmin <= px[t] + config.radius_m)
                & (track_lymax >= py[t] - config.radius_m)
                & (track_lymin <= py[t] + config.radius_m)
            )
            long_c = track_long_idx[lmask]
        else:
            long_c = np.empty(0, dtype=np.int64)

        if len(short_c) > 0 and len(long_c) > 0:
            raw_cands = np.concatenate([short_c, long_c])
        elif len(short_c) > 0:
            raw_cands = short_c
        elif len(long_c) > 0:
            raw_cands = long_c
        else:
            n_obs_no_candidates += 1
            continue

        d, proj_t = point_to_segments_distance(
            px[t],
            py[t],
            road_net.ax[raw_cands],
            road_net.ay[raw_cands],
            road_net.bx[raw_cands],
            road_net.by[raw_cands],
        )
        within = d <= config.radius_m
        if not np.any(within):
            n_obs_no_candidates += 1
            continue

        valid_arcs = raw_cands[within]
        valid_d = d[within]
        valid_t = proj_t[within]
        total_in_r = len(valid_arcs)
        obs_cand_counts[t] = min(total_in_r, 32767)

        # Deterministic sort by distance ascending, tie-breaking by arc_id
        order = np.lexsort((valid_arcs, valid_d))[: config.k_candidates]
        top_arcs = valid_arcs[order]
        top_d = valid_d[order]
        top_t = valid_t[order]
        k_len = len(top_arcs)

        d_start = np.hypot(px[t] - road_net.ax[top_arcs], py[t] - road_net.ay[top_arcs])
        d_end = np.hypot(px[t] - road_net.bx[top_arcs], py[t] - road_net.by[top_arcs])

        if t > 0:
            prev_d, _ = point_to_segments_distance(
                px[t - 1],
                py[t - 1],
                road_net.ax[top_arcs],
                road_net.ay[top_arcs],
                road_net.bx[top_arcs],
                road_net.by[top_arcs],
            )
        else:
            prev_d = top_d

        obs_ids_list.append(np.full(k_len, t, dtype=np.int32))
        cand_arcs_list.append(top_arcs.astype(np.int64))
        ranks_list.append(np.arange(1, k_len + 1, dtype=np.int16))
        dists_list.append(top_d.astype(np.float32))
        proj_t_list.append(top_t.astype(np.float32))
        d_start_list.append(d_start.astype(np.float32))
        d_end_list.append(d_end.astype(np.float32))
        prev_dist_list.append(prev_d.astype(np.float32))
        n_in_rad_list.append(np.full(k_len, total_in_r, dtype=np.int16))
        min_dist_list.append(np.full(k_len, float(top_d[0]), dtype=np.float32))

    if not obs_ids_list:
        empty_f32 = np.empty(0, dtype=np.float32)
        return ObservationCandidates(
            observation_id=np.empty(0, dtype=np.int32),
            candidate_arc_id=np.empty(0, dtype=np.int64),
            candidate_rank=np.empty(0, dtype=np.int16),
            dist_to_seg_m=empty_f32,
            proj_t=empty_f32,
            dist_to_start_m=empty_f32,
            dist_to_end_m=empty_f32,
            prev_obs_dist_m=empty_f32,
            n_cands_in_radius=np.empty(0, dtype=np.int16),
            obs_min_dist_m=empty_f32,
            obs_cand_counts=obs_cand_counts,
            n_obs_total=n_obs,
            n_obs_no_candidates=n_obs_no_candidates,
        )

    return ObservationCandidates(
        observation_id=np.concatenate(obs_ids_list),
        candidate_arc_id=np.concatenate(cand_arcs_list),
        candidate_rank=np.concatenate(ranks_list),
        dist_to_seg_m=np.concatenate(dists_list),
        proj_t=np.concatenate(proj_t_list),
        dist_to_start_m=np.concatenate(d_start_list),
        dist_to_end_m=np.concatenate(d_end_list),
        prev_obs_dist_m=np.concatenate(prev_dist_list),
        n_cands_in_radius=np.concatenate(n_in_rad_list),
        obs_min_dist_m=np.concatenate(min_dist_list),
        obs_cand_counts=obs_cand_counts,
        n_obs_total=n_obs,
        n_obs_no_candidates=n_obs_no_candidates,
    )


def validate_spatial_index_vs_bruteforce(
    px: np.ndarray,
    py: np.ndarray,
    road_net: RoadNetworkMetric,
    config: CandidateConfig,
    sample_indices: np.ndarray,
) -> dict[str, int | float]:
    """Validate SpatialRoadIndex exactness against full brute-force scan on sampled points."""
    index = SpatialRoadIndex(road_net, short_arc_threshold_m=config.short_arc_threshold_m)
    all_arcs = np.arange(len(road_net.ax), dtype=np.int64)

    missed_top1 = 0
    missed_topk = 0
    max_dist_diff = 0.0

    for t in sample_indices:
        # Brute force over ALL arcs in the network
        d_bf, _ = point_to_segments_distance(
            px[t], py[t], road_net.ax, road_net.ay, road_net.bx, road_net.by
        )
        within_bf = d_bf <= config.radius_m
        bf_arcs = all_arcs[within_bf]
        bf_d = d_bf[within_bf]
        bf_order = np.lexsort((bf_arcs, bf_d))[: config.k_candidates]
        bf_topk = bf_arcs[bf_order]
        bf_topk_d = bf_d[bf_order]

        # Indexed retrieval
        idx_arcs, idx_d, _ = index.query_observation_candidates(
            float(px[t]), float(py[t]), config.radius_m
        )
        idx_order = np.lexsort((idx_arcs, idx_d))[: config.k_candidates]
        idx_topk = idx_arcs[idx_order]
        idx_topk_d = idx_d[idx_order]

        if len(bf_topk) > 0:
            if len(idx_topk) == 0 or idx_topk[0] != bf_topk[0]:
                missed_top1 += 1
            if not np.array_equal(bf_topk, idx_topk):
                missed_topk += 1
            if len(idx_topk_d) == len(bf_topk_d):
                max_dist_diff = max(max_dist_diff, float(np.max(np.abs(bf_topk_d - idx_topk_d))))

    return {
        "sampled_observations": int(len(sample_indices)),
        "missed_top1_count": missed_top1,
        "missed_topk_count": missed_topk,
        "max_distance_diff_m": max_dist_diff,
    }

"""Phase 0 Prototype & Audit of Reference-Route Label Association Methodology.

Tests monotonic DP sequence alignment (spatial proximity + heading consistency +
chronological progression along .route) vs naive nearest-route-arc matching on the
80 training trips, and measures candidate-set recall (radius + top-K) without leakage.

Outputs:
  - results/label_association_audit.csv
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from geo_utils import (
    LocalTangentProjection,
    angular_diff_deg,
    point_to_segments_distance,
    segment_bearings_deg,
)


def compute_gps_headings(
    px: np.ndarray, py: np.ndarray, ts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate instantaneous vehicle bearing (deg) and speed (m/s) using centered differences."""
    n = len(px)
    bearings = np.zeros(n, dtype=np.float64)
    speeds = np.zeros(n, dtype=np.float64)
    if n < 2:
        return bearings, speeds

    dx = np.empty(n, dtype=np.float64)
    dy = np.empty(n, dtype=np.float64)
    dt = np.empty(n, dtype=np.float64)

    dx[0], dy[0], dt[0] = px[1] - px[0], py[1] - py[0], max(ts[1] - ts[0], 1e-3)
    dx[-1], dy[-1], dt[-1] = px[-1] - px[-2], py[-1] - py[-2], max(ts[-1] - ts[-2], 1e-3)
    if n > 2:
        dx[1:-1] = px[2:] - px[:-2]
        dy[1:-1] = py[2:] - py[:-2]
        dt[1:-1] = np.maximum(ts[2:] - ts[:-2], 1e-3)

    step_m = np.hypot(dx, dy)
    speeds = step_m / dt
    bearings = np.mod(np.degrees(np.arctan2(dx, dy)), 360.0)
    return bearings, speeds


def monotonic_dp_align(
    d_mat: np.ndarray,
    gps_bearings: np.ndarray,
    gps_speeds: np.ndarray,
    route_bearings: np.ndarray,
    sigma_d: float = 15.0,
    sigma_theta: float = 45.0,
    max_jump: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    """Align T GPS observations monotonically to M route indices k in [0..M-1].

    Uses banded monotonic DP: k(t) >= k(t-1), allowing jumps up to max_jump
    (with fallback to full prefix minimum if needed for large GPS gaps).
    """
    t_len, m_len = d_mat.shape
    # Emission cost: squared normalized distance + gated heading mismatch
    emit = (d_mat / sigma_d) ** 2
    moving = gps_speeds > 2.0  # > 7.2 km/h
    if np.any(moving):
        ang_diff = angular_diff_deg(gps_bearings[moving, None], route_bearings[None, :])
        emit[moving] += (ang_diff / sigma_theta) ** 2

    # Transition penalty per skipped route arc (mild penalty so progression follows track)
    step_penalty = 0.05

    dp = np.full((t_len, m_len), np.inf, dtype=np.float64)
    back = np.zeros((t_len, m_len), dtype=np.int32)

    # Allow starting near beginning, or anywhere with small prefix penalty if track starts mid-route
    dp[0, :] = emit[0, :] + 0.5 * np.arange(m_len, dtype=np.float64)

    for t in range(1, t_len):
        prev = dp[t - 1]
        # Compute running prefix minimum with step penalty:
        # best_prev[k] = min_{j <= k} (prev[j] + step_penalty * (k - j))
        # Recurrence: best[0] = prev[0]; best[k] = min(prev[k], best[k-1] + step_penalty)
        best_val = np.empty(m_len, dtype=np.float64)
        best_idx = np.empty(m_len, dtype=np.int32)
        best_val[0] = prev[0]
        best_idx[0] = 0
        for k in range(1, m_len):
            shifted = best_val[k - 1] + step_penalty
            if prev[k] <= shifted:
                best_val[k] = prev[k]
                best_idx[k] = k
            else:
                best_val[k] = shifted
                best_idx[k] = best_idx[k - 1]
        dp[t] = emit[t] + best_val
        back[t] = best_idx

    # Backtrack from best ending state (small penalty for ending before M-1)
    end_cost = dp[-1] + 0.5 * np.arange(m_len - 1, -1, -1, dtype=np.float64)
    k_seq = np.empty(t_len, dtype=np.int32)
    k_seq[-1] = int(np.argmin(end_cost))
    for t in range(t_len - 1, 0, -1):
        k_seq[t - 1] = back[t, k_seq[t]]

    assigned_dists = d_mat[np.arange(t_len), k_seq]
    return k_seq, assigned_dists


def get_topk_candidates_recall(
    px: np.ndarray,
    py: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    arc_lengths_m: np.ndarray,
    assigned_arc_ids: np.ndarray,
    valid_mask: np.ndarray,
    k_values: tuple[int, ...] = (5, 10, 15, 20),
    r_max: float = 100.0,
) -> dict[str, float]:
    """Measure candidate-set recall (without using .route) at various top-K and radius limits."""
    short_thresh = 300.0
    short_idx = np.flatnonzero(arc_lengths_m <= short_thresh)
    long_idx = np.flatnonzero(arc_lengths_m > short_thresh)

    xmin, xmax = float(px.min()) - r_max, float(px.max()) + r_max
    ymin, ymax = float(py.min()) - r_max, float(py.max()) + r_max
    if len(long_idx) > 0:
        lax, lbx = ax[long_idx], bx[long_idx]
        lay, lby = ay[long_idx], by[long_idx]
        lmask = (
            (np.maximum(lax, lbx) >= xmin)
            & (np.minimum(lax, lbx) <= xmax)
            & (np.maximum(lay, lby) >= ymin)
            & (np.minimum(lay, lby) <= ymax)
        )
        local_long_idx = long_idx[lmask]
    else:
        local_long_idx = np.empty(0, dtype=int)

    mx = 0.5 * (ax[short_idx] + bx[short_idx])
    my = 0.5 * (ay[short_idx] + by[short_idx])
    tree = cKDTree(np.column_stack([mx, my]))
    ball_r = r_max + 0.5 * short_thresh
    neighbors_list = tree.query_ball_point(np.column_stack([px, py]), r=ball_r)

    hits_r50 = 0
    hits_r100 = 0
    hits_k = {k: 0 for k in k_values}
    total_valid = int(np.sum(valid_mask))

    for i in np.flatnonzero(valid_mask):
        nb_short = short_idx[neighbors_list[i]] if neighbors_list[i] else np.empty(0, dtype=int)
        cand_idx = (
            np.concatenate([nb_short, local_long_idx])
            if len(local_long_idx) > 0
            else nb_short
        )
        if len(cand_idx) == 0:
            continue
        d_i, _ = point_to_segments_distance(
            px[i], py[i], ax[cand_idx], ay[cand_idx], bx[cand_idx], by[cand_idx]
        )
        true_arc = assigned_arc_ids[i]
        in_100 = cand_idx[d_i <= 100.0]
        in_50 = cand_idx[d_i <= 50.0]
        if true_arc in in_50:
            hits_r50 += 1
        if true_arc in in_100:
            hits_r100 += 1

        # Top-K sorted by exact point-to-segment distance within r_max=100m
        order = np.argsort(d_i)
        sorted_cands = cand_idx[order]
        sorted_dists = d_i[order]
        in_radius = sorted_cands[sorted_dists <= r_max]
        for k in k_values:
            if true_arc in in_radius[:k]:
                hits_k[k] += 1

    res = {
        "recall_r50_pct": round(hits_r50 / max(1, total_valid) * 100.0, 2),
        "recall_r100_pct": round(hits_r100 / max(1, total_valid) * 100.0, 2),
    }
    for k in k_values:
        res[f"recall_top{k}_r100_pct"] = round(hits_k[k] / max(1, total_valid) * 100.0, 2)
    return res


def main() -> int:
    t0 = time.perf_counter()
    data_root = PROJECT_ROOT / "data" / "raw"
    results_dir = PROJECT_ROOT / "results"

    rows = []
    for split in ["train", "student_test"]:
        split_dir = data_root / split
        for td in sorted([d for d in split_dir.iterdir() if d.is_dir()]):
            tid = td.name
            track = pd.read_csv(td / f"{tid}.track", sep=r"\s+", header=None).to_numpy()
            nodes = pd.read_csv(td / f"{tid}.nodes", sep=r"\s+", header=None).to_numpy()
            arcs = pd.read_csv(td / f"{tid}.arcs", sep=r"\s+", header=None).to_numpy(dtype=int)
            route = pd.read_csv(td / f"{tid}.route", sep=r"\s+", header=None).to_numpy(dtype=int).ravel()

            proj = LocalTangentProjection(track[:, 0].mean(), track[:, 1].mean())
            px, py = proj.to_xy(track[:, 0], track[:, 1])
            nx, ny = proj.to_xy(nodes[:, 0], nodes[:, 1])

            ax, ay = nx[arcs[:, 0]], ny[arcs[:, 0]]
            bx, by = nx[arcs[:, 1]], ny[arcs[:, 1]]
            arc_lengths_m = np.hypot(bx - ax, by - ay)

            ru, rv = arcs[route, 0], arcs[route, 1]
            rax, ray, rbx, rby = nx[ru], ny[ru], nx[rv], ny[rv]

            d_mat, _ = point_to_segments_distance(
                px[:, None], py[:, None], rax[None, :], ray[None, :], rbx[None, :], rby[None, :]
            )

            # 1. Naive nearest route arc
            naive_k = np.argmin(d_mat, axis=1)
            naive_dists = d_mat[np.arange(len(track)), naive_k]

            # 2. Monotonic DP alignment (distance + heading + order)
            gps_bearings, gps_speeds = compute_gps_headings(px, py, track[:, 2])
            route_bearings = segment_bearings_deg(rax, ray, rbx, rby)
            dp_k, dp_dists = monotonic_dp_align(
                d_mat, gps_bearings, gps_speeds, route_bearings
            )
            dp_arc_ids = route[dp_k]

            # Agreement and continuity of DP assigned sequence
            pct_agree_naive_dp = float(np.mean(route[naive_k] == dp_arc_ids) * 100.0)
            pct_dp_within_25m = float(np.mean(dp_dists <= 25.0) * 100.0)
            pct_dp_within_50m = float(np.mean(dp_dists <= 50.0) * 100.0)
            pct_dp_within_100m = float(np.mean(dp_dists <= 100.0) * 100.0)
            dp_route_coverage_pct = float(len(np.unique(dp_k)) / len(route) * 100.0)

            # Candidate set recall evaluated on observations within 50m (well-matched points)
            # AND across all observations
            recall_all = get_topk_candidates_recall(
                px, py, ax, ay, bx, by, arc_lengths_m, dp_arc_ids, np.ones(len(track), dtype=bool)
            )
            within50 = dp_dists <= 50.0
            recall_clean = (
                get_topk_candidates_recall(
                    px, py, ax, ay, bx, by, arc_lengths_m, dp_arc_ids, within50
                )
                if np.any(within50)
                else {k: 0.0 for k in recall_all}
            )

            rows.append(
                {
                    "split": split,
                    "trip_id": tid,
                    "n_gps": len(track),
                    "route_len": len(route),
                    "naive_dist_median_m": round(float(np.median(naive_dists)), 2),
                    "dp_dist_median_m": round(float(np.median(dp_dists)), 2),
                    "dp_dist_p95_m": round(float(np.percentile(dp_dists, 95)), 2),
                    "dp_dist_max_m": round(float(dp_dists.max()), 2),
                    "pct_agree_naive_dp": round(pct_agree_naive_dp, 2),
                    "pct_dp_within_25m": round(pct_dp_within_25m, 2),
                    "pct_dp_within_50m": round(pct_dp_within_50m, 2),
                    "pct_dp_within_100m": round(pct_dp_within_100m, 2),
                    "dp_route_coverage_pct": round(dp_route_coverage_pct, 2),
                    "all_recall_r50_pct": recall_all["recall_r50_pct"],
                    "all_recall_r100_pct": recall_all["recall_r100_pct"],
                    "all_recall_top10_pct": recall_all["recall_top10_r100_pct"],
                    "all_recall_top20_pct": recall_all["recall_top20_r100_pct"],
                    "clean50_recall_top5_pct": recall_clean["recall_top5_r100_pct"],
                    "clean50_recall_top10_pct": recall_clean["recall_top10_r100_pct"],
                    "clean50_recall_top15_pct": recall_clean["recall_top15_r100_pct"],
                    "clean50_recall_top20_pct": recall_clean["recall_top20_r100_pct"],
                }
            )
            del track, nodes, arcs, route, d_mat
            gc.collect()

    df = pd.DataFrame(rows)
    out_path = results_dir / "label_association_audit.csv"
    df.to_csv(out_path, index=False)

    print(f"Completed label association audit in {time.perf_counter() - t0:.2f}s")
    for s in ["train", "student_test"]:
        sub = df[df["split"] == s]
        print(f"\n--- {s} ({len(sub)} trips) ---")
        print(f"  Median DP dist (median across trips): {sub['dp_dist_median_m'].median():.2f} m")
        print(f"  Mean % obs within 50m of DP arc:      {sub['pct_dp_within_50m'].mean():.2f}%")
        print(f"  Mean % obs within 100m of DP arc:     {sub['pct_dp_within_100m'].mean():.2f}%")
        print(f"  Naive vs DP arc agreement (mean):     {sub['pct_agree_naive_dp'].mean():.2f}%")
        print(f"  Mean route arcs hit by >=1 GPS point: {sub['dp_route_coverage_pct'].mean():.2f}%")
        print(f"  Clean(<=50m) Top-5 / Top-10 / Top-20 candidate recall: "
              f"{sub['clean50_recall_top5_pct'].mean():.2f}% / "
              f"{sub['clean50_recall_top10_pct'].mean():.2f}% / "
              f"{sub['clean50_recall_top20_pct'].mean():.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Phase 0 Comprehensive Dataset Audit & EDA Script for PathSnap.

Iteratively inspects every trip in data/raw/train (80 trips) and
data/raw/student_test (10 trips) using CPU-friendly, low-memory operations.

Outputs:
  - results/dataset_inventory.csv
  - results/timestamp_gap_analysis.csv
  - results/network_connectivity_summary.csv
  - results/reference_route_validity.csv
  - results/data_quality_summary.json
"""

from __future__ import annotations

import csv
import gc
import json
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from geo_utils import (
    LocalTangentProjection,
    angular_diff_deg,
    haversine_meters,
    point_to_segments_distance,
    segment_bearings_deg,
)


def iter_trips(data_root: Path) -> Iterator[tuple[str, str, Path]]:
    """Yield (split, trip_id, trip_dir) sorted deterministically."""
    for split in ["train", "student_test"]:
        split_dir = data_root / split
        if not split_dir.is_dir():
            continue
        for trip_dir in sorted([d for d in split_dir.iterdir() if d.is_dir()]):
            yield split, trip_dir.name, trip_dir


def load_numeric_array(
    filepath: Path, expected_cols: int, dtype: type
) -> tuple[np.ndarray, int, int]:
    """Load whitespace-separated numeric file quickly while verifying malformed/NaN rows."""
    try:
        np_dtype = np.int64 if dtype is int else np.float64
        df = pd.read_csv(
            filepath,
            sep=r"\s+",
            header=None,
            engine="c",
            dtype=np_dtype,
        )
        if df.shape[1] == expected_cols:
            arr = df.to_numpy(dtype=np_dtype)
            if np.all(np.isfinite(arr)):
                return arr, 0, 0
    except Exception:
        pass

    rows = []
    malformed_rows = 0
    nan_inf_count = 0

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) != expected_cols:
                malformed_rows += 1
                continue
            try:
                if dtype is int:
                    vals = [int(x) for x in parts]
                else:
                    vals = [float(x) for x in parts]
                    if any(not np.isfinite(v) for v in vals):
                        nan_inf_count += 1
                        continue
                rows.append(vals)
            except ValueError:
                malformed_rows += 1

    if not rows:
        return np.empty((0, expected_cols), dtype=dtype), malformed_rows, nan_inf_count
    return np.asarray(rows, dtype=dtype), malformed_rows, nan_inf_count


def compute_exact_nearest_road_distances(
    px: np.ndarray,
    py: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    arc_lengths_m: np.ndarray,
    search_r: float = 150.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact distance (m) from each GPS point to the nearest road arc,
    plus count of candidate arcs within 50m and 100m.

    Uses cKDTree on short-arc midpoints (<= 300m) + direct check on bbox-filtered long arcs,
    guaranteeing exact distances within search_r=150m, with fallback for points > 150m away.
    """
    n_gps = len(px)
    if n_gps == 0 or len(ax) == 0:
        return (
            np.full(n_gps, np.nan),
            np.zeros(n_gps, dtype=int),
            np.zeros(n_gps, dtype=int),
        )

    short_len_thresh = 300.0
    short_mask = arc_lengths_m <= short_len_thresh
    short_idx = np.flatnonzero(short_mask)
    long_idx = np.flatnonzero(~short_mask)

    # Filter long arcs by track bounding box + search_r
    xmin, xmax = float(px.min()) - search_r, float(px.max()) + search_r
    ymin, ymax = float(py.min()) - search_r, float(py.max()) + search_r
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

    min_dists = np.full(n_gps, np.inf, dtype=np.float64)
    cand_50m = np.zeros(n_gps, dtype=int)
    cand_100m = np.zeros(n_gps, dtype=int)

    # 1. Check local long arcs against all GPS points (usually very few long arcs)
    if len(local_long_idx) > 0:
        d_long, _ = point_to_segments_distance(
            px[:, None],
            py[:, None],
            ax[local_long_idx][None, :],
            ay[local_long_idx][None, :],
            bx[local_long_idx][None, :],
            by[local_long_idx][None, :],
        )
        min_dists = np.minimum(min_dists, d_long.min(axis=1))
        cand_50m += (d_long <= 50.0).sum(axis=1)
        cand_100m += (d_long <= 100.0).sum(axis=1)

    # 2. Query cKDTree on short-arc midpoints with radius = search_r + short_len_thresh/2
    if len(short_idx) > 0:
        mx = 0.5 * (ax[short_idx] + bx[short_idx])
        my = 0.5 * (ay[short_idx] + by[short_idx])
        tree = cKDTree(np.column_stack([mx, my]))
        ball_r = search_r + 0.5 * short_len_thresh
        gps_pts = np.column_stack([px, py])
        neighbors_list = tree.query_ball_point(gps_pts, r=ball_r)

        for i, nb in enumerate(neighbors_list):
            if not nb:
                continue
            a_idx = short_idx[nb]
            d_i, _ = point_to_segments_distance(
                px[i], py[i], ax[a_idx], ay[a_idx], bx[a_idx], by[a_idx]
            )
            d_min = float(d_i.min())
            if d_min < min_dists[i]:
                min_dists[i] = d_min
            cand_50m[i] += int(np.sum(d_i <= 50.0))
            cand_100m[i] += int(np.sum(d_i <= 100.0))

    # 3. Fallback for any GPS point > search_r from nearest found arc
    far_gps = np.flatnonzero(min_dists > search_r)
    if len(far_gps) > 0:
        # Check nearest 64 arc midpoints across ALL arcs + any long arcs
        all_mx = 0.5 * (ax + bx)
        all_my = 0.5 * (ay + by)
        full_tree = cKDTree(np.column_stack([all_mx, all_my]))
        k_query = min(64, len(ax))
        _, nn_idx = full_tree.query(np.column_stack([px[far_gps], py[far_gps]]), k=k_query)
        if k_query == 1:
            nn_idx = nn_idx[:, None]
        for row_i, g_i in enumerate(far_gps):
            a_idx = nn_idx[row_i]
            d_i, _ = point_to_segments_distance(
                px[g_i], py[g_i], ax[a_idx], ay[a_idx], bx[a_idx], by[a_idx]
            )
            min_dists[g_i] = min(min_dists[g_i], float(d_i.min()))

    return min_dists, cand_50m, cand_100m


def inspect_trip(split: str, trip_id: str, trip_dir: Path) -> tuple[dict, list[dict]]:
    """Audit a single trip and return (summary_row, timestamp_gap_events)."""
    track_path = trip_dir / f"{trip_id}.track"
    nodes_path = trip_dir / f"{trip_id}.nodes"
    arcs_path = trip_dir / f"{trip_id}.arcs"
    route_path = trip_dir / f"{trip_id}.route"

    # File sizes
    track_bytes = track_path.stat().st_size
    nodes_bytes = nodes_path.stat().st_size
    arcs_bytes = arcs_path.stat().st_size
    route_bytes = route_path.stat().st_size

    # Load files
    track, track_bad, track_nan = load_numeric_array(track_path, 3, float)
    nodes, nodes_bad, nodes_nan = load_numeric_array(nodes_path, 2, float)
    arcs, arcs_bad, arcs_nan = load_numeric_array(arcs_path, 2, int)
    route_arr, route_bad, route_nan = load_numeric_array(route_path, 1, int)
    route = route_arr.ravel() if len(route_arr) > 0 else np.empty(0, dtype=int)

    n_gps = len(track)
    n_nodes = len(nodes)
    n_arcs = len(arcs)
    route_len = len(route)

    # Track coordinates & timestamps
    lon = track[:, 0]
    lat = track[:, 1]
    ts = track[:, 2]

    coord_out_of_bounds = int(
        np.sum((lon < -180.0) | (lon > 180.0) | (lat < -90.0) | (lat > 90.0))
    )
    lon_min, lon_max = float(lon.min()), float(lon.max())
    lat_min, lat_max = float(lat.min()), float(lat.max())
    centroid_lon = float(lon.mean())
    centroid_lat = float(lat.mean())

    bbox_width_km = float(
        haversine_meters(lon_min, centroid_lat, lon_max, centroid_lat) / 1000.0
    )
    bbox_height_km = float(
        haversine_meters(centroid_lon, lat_min, centroid_lon, lat_max) / 1000.0
    )

    # Timestamps & gaps
    t_start = float(ts[0])
    t_end = float(ts[-1])
    duration_s = float(t_end - t_start)
    dt = np.diff(ts)
    step_dists_m = (
        haversine_meters(lon[:-1], lat[:-1], lon[1:], lat[1:])
        if n_gps > 1
        else np.empty(0)
    )
    track_length_km = float(step_dists_m.sum() / 1000.0)
    straight_line_km = (
        float(haversine_meters(lon[0], lat[0], lon[-1], lat[-1]) / 1000.0)
        if n_gps > 1
        else 0.0
    )

    n_dt_total = len(dt)
    n_dt_negative = int(np.sum(dt < 0))
    n_dt_zero = int(np.sum(dt == 0))
    n_dt_non_monotonic = n_dt_negative + n_dt_zero
    n_dt_1s = int(np.sum(np.isclose(dt, 1.0, atol=1e-3)))
    pct_dt_1s = float(n_dt_1s / n_dt_total * 100.0) if n_dt_total > 0 else 0.0
    n_dt_lt_1s = int(np.sum((dt > 0) & (dt < 1.0 - 1e-3)))
    n_dt_gt_1s = int(np.sum(dt > 1.0 + 1e-3))
    n_dt_gt_5s = int(np.sum(dt > 5.0))
    n_dt_gt_10s = int(np.sum(dt > 10.0))
    n_dt_gt_30s = int(np.sum(dt > 30.0))
    n_dt_gt_60s = int(np.sum(dt > 60.0))

    dt_min = float(dt.min()) if n_dt_total > 0 else np.nan
    dt_median = float(np.median(dt)) if n_dt_total > 0 else np.nan
    dt_max = float(dt.max()) if n_dt_total > 0 else np.nan

    valid_speed_mask = dt > 0
    speeds_kmh = (
        (step_dists_m[valid_speed_mask] / dt[valid_speed_mask]) * 3.6
        if np.any(valid_speed_mask)
        else np.empty(0)
    )
    speed_median_kmh = float(np.median(speeds_kmh)) if len(speeds_kmh) > 0 else np.nan
    speed_p95_kmh = (
        float(np.percentile(speeds_kmh, 95)) if len(speeds_kmh) > 0 else np.nan
    )
    speed_max_kmh = float(speeds_kmh.max()) if len(speeds_kmh) > 0 else np.nan
    n_speed_gt_150 = int(np.sum(speeds_kmh > 150.0))
    n_speed_gt_200 = int(np.sum(speeds_kmh > 200.0))
    max_step_dist_m = float(step_dists_m.max()) if len(step_dists_m) > 0 else 0.0

    gap_events = []
    irregular_idx = np.flatnonzero(~np.isclose(dt, 1.0, atol=1e-3))
    for idx in irregular_idx:
        dt_val = float(dt[idx])
        dist_val = float(step_dists_m[idx])
        spd_val = float((dist_val / dt_val) * 3.6) if dt_val > 0 else np.nan
        gap_events.append(
            {
                "split": split,
                "trip_id": trip_id,
                "obs_idx": int(idx),
                "t_prev": float(ts[idx]),
                "t_curr": float(ts[idx + 1]),
                "dt_s": dt_val,
                "step_dist_m": round(dist_val, 3),
                "implied_speed_kmh": round(spd_val, 3) if np.isfinite(spd_val) else np.nan,
            }
        )

    # Road network checks
    nodes_lon_min, nodes_lon_max = float(nodes[:, 0].min()), float(nodes[:, 0].max())
    nodes_lat_min, nodes_lat_max = float(nodes[:, 1].min()), float(nodes[:, 1].max())
    nodes_out_of_bounds = int(
        np.sum(
            (nodes[:, 0] < -180.0)
            | (nodes[:, 0] > 180.0)
            | (nodes[:, 1] < -90.0)
            | (nodes[:, 1] > 90.0)
        )
    )

    u = arcs[:, 0]
    v = arcs[:, 1]
    invalid_arc_endpoints = int(
        np.sum((u < 0) | (u >= n_nodes) | (v < 0) | (v >= n_nodes))
    )
    valid_arc_mask = (u >= 0) & (u < n_nodes) & (v >= 0) & (v < n_nodes)
    u_valid = u[valid_arc_mask]
    v_valid = v[valid_arc_mask]

    self_loops = int(np.sum(u_valid == v_valid))

    # Duplicate directed arcs and bidirectional check using 64-bit packed keys
    edge_keys = u_valid.astype(np.int64) * np.int64(n_nodes) + v_valid.astype(np.int64)
    rev_keys = v_valid.astype(np.int64) * np.int64(n_nodes) + u_valid.astype(np.int64)
    unique_edge_keys = np.unique(edge_keys)
    duplicate_directed_arcs = int(len(edge_keys) - len(unique_edge_keys))
    has_reverse = np.isin(edge_keys, unique_edge_keys)
    # Check reverse existence properly:
    rev_exists = np.isin(rev_keys, unique_edge_keys)
    bidirectional_arc_pct = (
        float(np.mean(rev_exists) * 100.0) if len(edge_keys) > 0 else 0.0
    )

    in_deg = np.bincount(v_valid, minlength=n_nodes)
    out_deg = np.bincount(u_valid, minlength=n_nodes)
    tot_deg = in_deg + out_deg
    isolated_nodes = int(np.sum(tot_deg == 0))
    dead_end_out0 = int(np.sum((out_deg == 0) & (in_deg > 0)))
    dead_end_in0 = int(np.sum((in_deg == 0) & (out_deg > 0)))
    mean_out_deg = float(out_deg.mean())
    max_out_deg = int(out_deg.max())

    # Connected components via scipy.sparse.csgraph
    data_ones = np.ones(len(u_valid), dtype=np.int8)
    adj = csr_matrix((data_ones, (u_valid, v_valid)), shape=(n_nodes, n_nodes))
    n_weak_comp, weak_labels = connected_components(
        adj, directed=False, return_labels=True
    )
    weak_sizes = np.bincount(weak_labels)
    largest_weak_comp_nodes = int(weak_sizes.max()) if len(weak_sizes) > 0 else 0
    largest_weak_comp_pct = (
        float(largest_weak_comp_nodes / n_nodes * 100.0) if n_nodes > 0 else 0.0
    )

    n_strong_comp, strong_labels = connected_components(
        adj, directed=True, connection="strong", return_labels=True
    )
    strong_sizes = np.bincount(strong_labels)
    largest_strong_comp_nodes = int(strong_sizes.max()) if len(strong_sizes) > 0 else 0
    largest_strong_comp_pct = (
        float(largest_strong_comp_nodes / n_nodes * 100.0) if n_nodes > 0 else 0.0
    )

    # Project track and road network to local tangent plane in meters
    proj = LocalTangentProjection(centroid_lon, centroid_lat)
    px, py = proj.to_xy(lon, lat)
    nx, ny = proj.to_xy(nodes[:, 0], nodes[:, 1])

    ax = nx[u_valid]
    ay = ny[u_valid]
    bx = nx[v_valid]
    by = ny[v_valid]
    arc_lengths_m = np.hypot(bx - ax, by - ay)
    arc_len_min_m = float(arc_lengths_m.min()) if len(arc_lengths_m) > 0 else np.nan
    arc_len_median_m = (
        float(np.median(arc_lengths_m)) if len(arc_lengths_m) > 0 else np.nan
    )
    arc_len_mean_m = float(arc_lengths_m.mean()) if len(arc_lengths_m) > 0 else np.nan
    arc_len_max_m = float(arc_lengths_m.max()) if len(arc_lengths_m) > 0 else np.nan
    n_zero_len_arcs = int(np.sum(arc_lengths_m < 0.1))

    # GPS-to-nearest-road distances across entire road network
    gps_to_road_m, cand_50m, cand_100m = compute_exact_nearest_road_distances(
        px, py, ax, ay, bx, by, arc_lengths_m, search_r=150.0
    )
    gps_to_road_median_m = float(np.median(gps_to_road_m))
    gps_to_road_p95_m = float(np.percentile(gps_to_road_m, 95))
    gps_to_road_max_m = float(gps_to_road_m.max())
    cand_50m_mean = float(cand_50m.mean())
    cand_100m_mean = float(cand_100m.mean())
    pct_gps_no_cand_50m = float(np.mean(cand_50m == 0) * 100.0)
    pct_gps_no_cand_100m = float(np.mean(cand_100m == 0) * 100.0)

    # Reference route checks
    invalid_route_arcs = int(np.sum((route < 0) | (route >= n_arcs)))
    valid_route = route[(route >= 0) & (route < n_arcs)]
    route_unique_arcs = int(len(np.unique(valid_route)))
    route_repeated_arcs = int(len(valid_route) - route_unique_arcs)

    ru = u[valid_route]
    rv = v[valid_route]
    rax, ray = nx[ru], ny[ru]
    rbx, rby = nx[rv], ny[rv]
    r_lens_m = np.hypot(rbx - rax, rby - ray)
    route_length_km = float(r_lens_m.sum() / 1000.0)

    # Continuity across consecutive route arcs
    n_transitions = max(0, len(valid_route) - 1)
    if n_transitions > 0:
        strict_conn = rv[:-1] == ru[1:]
        same_arc_rep = valid_route[:-1] == valid_route[1:]
        u_turn = strict_conn & (ru[:-1] == rv[1:])
        disconnected_mask = (~strict_conn) & (~same_arc_rep)

        n_strict_connected = int(np.sum(strict_conn))
        n_same_arc_repeat = int(np.sum(same_arc_rep))
        n_u_turn = int(np.sum(u_turn))
        n_disconnected = int(np.sum(disconnected_mask))
        route_continuity_pct = float(n_strict_connected / n_transitions * 100.0)

        if n_disconnected > 0:
            disc_gaps_m = np.hypot(
                nx[ru[1:][disconnected_mask]] - nx[rv[:-1][disconnected_mask]],
                ny[ru[1:][disconnected_mask]] - ny[rv[:-1][disconnected_mask]],
            )
            max_route_gap_m = float(disc_gaps_m.max())
        else:
            max_route_gap_m = 0.0
    else:
        n_strict_connected = 0
        n_same_arc_repeat = 0
        n_u_turn = 0
        n_disconnected = 0
        route_continuity_pct = 100.0 if len(valid_route) == 1 else 0.0
        max_route_gap_m = 0.0

    # Check whether route nodes belong to largest weakly/strongly connected component
    largest_weak_label = int(np.argmax(weak_sizes)) if len(weak_sizes) > 0 else -1
    largest_strong_label = int(np.argmax(strong_sizes)) if len(strong_sizes) > 0 else -1
    route_nodes = np.unique(np.concatenate([ru, rv])) if len(valid_route) > 0 else np.empty(0, dtype=int)
    route_in_largest_weak = bool(np.all(weak_labels[route_nodes] == largest_weak_label)) if len(route_nodes) > 0 else False
    route_in_largest_strong = bool(np.all(strong_labels[route_nodes] == largest_strong_label)) if len(route_nodes) > 0 else False

    # GPS-to-Route spatial alignment
    if len(valid_route) > 0 and n_gps > 0:
        d_gps_route, _ = point_to_segments_distance(
            px[:, None], py[:, None], rax[None, :], ray[None, :], rbx[None, :], rby[None, :]
        )
        gps_to_route_min = d_gps_route.min(axis=1)
        route_arc_to_gps_min = d_gps_route.min(axis=0)

        gps_to_route_median_m = float(np.median(gps_to_route_min))
        gps_to_route_p95_m = float(np.percentile(gps_to_route_min, 95))
        gps_to_route_max_m = float(gps_to_route_min.max())
        pct_gps_within_10m_route = float(np.mean(gps_to_route_min <= 10.0) * 100.0)
        pct_gps_within_25m_route = float(np.mean(gps_to_route_min <= 25.0) * 100.0)
        pct_gps_within_50m_route = float(np.mean(gps_to_route_min <= 50.0) * 100.0)
        pct_gps_gt_50m_route = float(np.mean(gps_to_route_min > 50.0) * 100.0)

        route_arc_to_gps_median_m = float(np.median(route_arc_to_gps_min))
        route_arc_to_gps_max_m = float(route_arc_to_gps_min.max())

        # Start & end alignment (distance from GPS start to first route arc, GPS end to last route arc)
        route_start_to_gps_start_m = float(d_gps_route[0, 0])
        route_end_to_gps_end_m = float(d_gps_route[-1, -1])

        # Candidate recall check: is a route arc within 50m / 100m of GPS point?
        route_recall_50m_pct = float(np.mean(gps_to_route_min <= 50.0) * 100.0)
        route_recall_100m_pct = float(np.mean(gps_to_route_min <= 100.0) * 100.0)

        # Heading consistency check along GPS track vs nearest route arc
        if n_gps >= 3:
            gps_dx = px[2:] - px[:-2]
            gps_dy = py[2:] - py[:-2]
            gps_move = np.hypot(gps_dx, gps_dy)
            gps_bearings = np.mod(np.degrees(np.arctan2(gps_dx, gps_dy)), 360.0)
            nearest_route_col = np.argmin(d_gps_route[1:-1], axis=1)
            route_bearings = segment_bearings_deg(rax, ray, rbx, rby)
            matched_route_bearings = route_bearings[nearest_route_col]
            moving_mask = gps_move > 2.0
            if np.any(moving_mask):
                hdg_diffs = angular_diff_deg(
                    gps_bearings[moving_mask], matched_route_bearings[moving_mask]
                )
                route_heading_diff_median_deg = float(np.median(hdg_diffs))
                pct_opposite_heading = float(np.mean(hdg_diffs > 120.0) * 100.0)
            else:
                route_heading_diff_median_deg = np.nan
                pct_opposite_heading = np.nan
        else:
            route_heading_diff_median_deg = np.nan
            pct_opposite_heading = np.nan
    else:
        gps_to_route_median_m = np.nan
        gps_to_route_p95_m = np.nan
        gps_to_route_max_m = np.nan
        pct_gps_within_10m_route = np.nan
        pct_gps_within_25m_route = np.nan
        pct_gps_within_50m_route = np.nan
        pct_gps_gt_50m_route = np.nan
        route_arc_to_gps_median_m = np.nan
        route_arc_to_gps_max_m = np.nan
        route_start_to_gps_start_m = np.nan
        route_end_to_gps_end_m = np.nan
        route_recall_50m_pct = np.nan
        route_recall_100m_pct = np.nan
        route_heading_diff_median_deg = np.nan
        pct_opposite_heading = np.nan

    obs_per_route_arc = float(n_gps / route_len) if route_len > 0 else np.nan

    row = {
        "split": split,
        "trip_id": trip_id,
        # File sizes
        "track_bytes": track_bytes,
        "nodes_bytes": nodes_bytes,
        "arcs_bytes": arcs_bytes,
        "route_bytes": route_bytes,
        "total_bytes": track_bytes + nodes_bytes + arcs_bytes + route_bytes,
        # Counts
        "n_gps": n_gps,
        "n_nodes": n_nodes,
        "n_arcs": n_arcs,
        "route_len": route_len,
        "obs_per_route_arc": round(obs_per_route_arc, 3),
        # Data quality
        "track_malformed": track_bad,
        "track_nan_inf": track_nan,
        "nodes_malformed": nodes_bad,
        "nodes_nan_inf": nodes_nan,
        "arcs_malformed": arcs_bad,
        "arcs_nan_inf": arcs_nan,
        "route_malformed": route_bad,
        "route_nan_inf": route_nan,
        "coord_out_of_bounds": coord_out_of_bounds,
        "nodes_out_of_bounds": nodes_out_of_bounds,
        "invalid_arc_endpoints": invalid_arc_endpoints,
        "invalid_route_arcs": invalid_route_arcs,
        # Spatial extent
        "lon_min": round(lon_min, 6),
        "lon_max": round(lon_max, 6),
        "lat_min": round(lat_min, 6),
        "lat_max": round(lat_max, 6),
        "centroid_lon": round(centroid_lon, 6),
        "centroid_lat": round(centroid_lat, 6),
        "bbox_width_km": round(bbox_width_km, 3),
        "bbox_height_km": round(bbox_height_km, 3),
        "nodes_lon_min": round(nodes_lon_min, 6),
        "nodes_lon_max": round(nodes_lon_max, 6),
        "nodes_lat_min": round(nodes_lat_min, 6),
        "nodes_lat_max": round(nodes_lat_max, 6),
        # Temporal & speed
        "t_start": round(t_start, 3),
        "t_end": round(t_end, 3),
        "duration_s": round(duration_s, 3),
        "n_dt_non_monotonic": n_dt_non_monotonic,
        "n_dt_negative": n_dt_negative,
        "n_dt_zero": n_dt_zero,
        "n_dt_1s": n_dt_1s,
        "pct_dt_1s": round(pct_dt_1s, 3),
        "n_dt_lt_1s": n_dt_lt_1s,
        "n_dt_gt_1s": n_dt_gt_1s,
        "n_dt_gt_5s": n_dt_gt_5s,
        "n_dt_gt_10s": n_dt_gt_10s,
        "n_dt_gt_30s": n_dt_gt_30s,
        "n_dt_gt_60s": n_dt_gt_60s,
        "dt_min": round(dt_min, 3),
        "dt_median": round(dt_median, 3),
        "dt_max": round(dt_max, 3),
        "track_length_km": round(track_length_km, 4),
        "straight_line_km": round(straight_line_km, 4),
        "speed_median_kmh": round(speed_median_kmh, 2),
        "speed_p95_kmh": round(speed_p95_kmh, 2),
        "speed_max_kmh": round(speed_max_kmh, 2),
        "n_speed_gt_150": n_speed_gt_150,
        "n_speed_gt_200": n_speed_gt_200,
        "max_step_dist_m": round(max_step_dist_m, 2),
        # Road network connectivity
        "self_loops": self_loops,
        "duplicate_directed_arcs": duplicate_directed_arcs,
        "bidirectional_arc_pct": round(bidirectional_arc_pct, 2),
        "isolated_nodes": isolated_nodes,
        "dead_end_out0": dead_end_out0,
        "dead_end_in0": dead_end_in0,
        "mean_out_deg": round(mean_out_deg, 3),
        "max_out_deg": max_out_deg,
        "n_weak_comp": n_weak_comp,
        "largest_weak_comp_nodes": largest_weak_comp_nodes,
        "largest_weak_comp_pct": round(largest_weak_comp_pct, 3),
        "n_strong_comp": n_strong_comp,
        "largest_strong_comp_nodes": largest_strong_comp_nodes,
        "largest_strong_comp_pct": round(largest_strong_comp_pct, 3),
        "arc_len_min_m": round(arc_len_min_m, 3),
        "arc_len_median_m": round(arc_len_median_m, 2),
        "arc_len_mean_m": round(arc_len_mean_m, 2),
        "arc_len_max_m": round(arc_len_max_m, 2),
        "n_zero_len_arcs": n_zero_len_arcs,
        # GPS-to-road spatial stats
        "gps_to_road_median_m": round(gps_to_road_median_m, 2),
        "gps_to_road_p95_m": round(gps_to_road_p95_m, 2),
        "gps_to_road_max_m": round(gps_to_road_max_m, 2),
        "cand_50m_mean": round(cand_50m_mean, 2),
        "cand_100m_mean": round(cand_100m_mean, 2),
        "pct_gps_no_cand_50m": round(pct_gps_no_cand_50m, 2),
        "pct_gps_no_cand_100m": round(pct_gps_no_cand_100m, 2),
        # Reference route validity
        "route_unique_arcs": route_unique_arcs,
        "route_repeated_arcs": route_repeated_arcs,
        "route_length_km": round(route_length_km, 4),
        "n_strict_connected": n_strict_connected,
        "n_same_arc_repeat": n_same_arc_repeat,
        "n_u_turn": n_u_turn,
        "n_disconnected": n_disconnected,
        "route_continuity_pct": round(route_continuity_pct, 3),
        "max_route_gap_m": round(max_route_gap_m, 2),
        "route_in_largest_weak": route_in_largest_weak,
        "route_in_largest_strong": route_in_largest_strong,
        "gps_to_route_median_m": round(gps_to_route_median_m, 2),
        "gps_to_route_p95_m": round(gps_to_route_p95_m, 2),
        "gps_to_route_max_m": round(gps_to_route_max_m, 2),
        "pct_gps_within_10m_route": round(pct_gps_within_10m_route, 2),
        "pct_gps_within_25m_route": round(pct_gps_within_25m_route, 2),
        "pct_gps_within_50m_route": round(pct_gps_within_50m_route, 2),
        "pct_gps_gt_50m_route": round(pct_gps_gt_50m_route, 2),
        "route_arc_to_gps_median_m": round(route_arc_to_gps_median_m, 2),
        "route_arc_to_gps_max_m": round(route_arc_to_gps_max_m, 2),
        "route_start_to_gps_start_m": round(route_start_to_gps_start_m, 2),
        "route_end_to_gps_end_m": round(route_end_to_gps_end_m, 2),
        "route_recall_50m_pct": round(route_recall_50m_pct, 2),
        "route_recall_100m_pct": round(route_recall_100m_pct, 2),
        "route_heading_diff_median_deg": round(route_heading_diff_median_deg, 2)
        if np.isfinite(route_heading_diff_median_deg)
        else np.nan,
        "pct_opposite_heading": round(pct_opposite_heading, 2)
        if np.isfinite(pct_opposite_heading)
        else np.nan,
    }

    # Explicit cleanup
    del track, nodes, arcs, route, adj
    return row, gap_events


def main() -> int:
    t0 = time.perf_counter()
    data_root = PROJECT_ROOT / "data" / "raw"
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    inventory_rows = []
    all_gap_events = []

    trips = list(iter_trips(data_root))
    print(f"Found {len(trips)} trips to audit.")

    for idx, (split, trip_id, trip_dir) in enumerate(trips, start=1):
        row, gaps = inspect_trip(split, trip_id, trip_dir)
        inventory_rows.append(row)
        all_gap_events.extend(gaps)
        if idx % 10 == 0 or idx == len(trips):
            print(f"  Processed {idx}/{len(trips)} trips ({split}/{trip_id})", flush=True)
        gc.collect()

    df_inv = pd.DataFrame(inventory_rows)
    df_gaps = pd.DataFrame(all_gap_events)

    # Save 1: Dataset inventory
    inv_path = results_dir / "dataset_inventory.csv"
    df_inv.to_csv(inv_path, index=False)

    # Save 2: Timestamp gap analysis
    gaps_path = results_dir / "timestamp_gap_analysis.csv"
    df_gaps.to_csv(gaps_path, index=False)

    # Save 3: Network connectivity summary
    net_cols = [
        "split",
        "trip_id",
        "n_nodes",
        "n_arcs",
        "invalid_arc_endpoints",
        "self_loops",
        "duplicate_directed_arcs",
        "bidirectional_arc_pct",
        "isolated_nodes",
        "dead_end_out0",
        "dead_end_in0",
        "mean_out_deg",
        "max_out_deg",
        "n_weak_comp",
        "largest_weak_comp_nodes",
        "largest_weak_comp_pct",
        "n_strong_comp",
        "largest_strong_comp_nodes",
        "largest_strong_comp_pct",
        "arc_len_min_m",
        "arc_len_median_m",
        "arc_len_mean_m",
        "arc_len_max_m",
        "n_zero_len_arcs",
    ]
    net_path = results_dir / "network_connectivity_summary.csv"
    df_inv[net_cols].to_csv(net_path, index=False)

    # Save 4: Reference route validity
    route_cols = [
        "split",
        "trip_id",
        "n_gps",
        "route_len",
        "obs_per_route_arc",
        "invalid_route_arcs",
        "route_unique_arcs",
        "route_repeated_arcs",
        "route_length_km",
        "track_length_km",
        "n_strict_connected",
        "n_same_arc_repeat",
        "n_u_turn",
        "n_disconnected",
        "route_continuity_pct",
        "max_route_gap_m",
        "route_in_largest_weak",
        "route_in_largest_strong",
        "gps_to_route_median_m",
        "gps_to_route_p95_m",
        "gps_to_route_max_m",
        "pct_gps_within_10m_route",
        "pct_gps_within_25m_route",
        "pct_gps_within_50m_route",
        "pct_gps_gt_50m_route",
        "route_arc_to_gps_median_m",
        "route_arc_to_gps_max_m",
        "route_start_to_gps_start_m",
        "route_end_to_gps_end_m",
        "route_recall_50m_pct",
        "route_recall_100m_pct",
        "route_heading_diff_median_deg",
        "pct_opposite_heading",
    ]
    route_path = results_dir / "reference_route_validity.csv"
    df_inv[route_cols].to_csv(route_path, index=False)

    # Save 5: Data quality summary JSON
    def summarize_split(sub: pd.DataFrame) -> dict:
        return {
            "num_trips": int(len(sub)),
            "total_gps_observations": int(sub["n_gps"].sum()),
            "gps_per_trip": {
                "min": int(sub["n_gps"].min()),
                "median": float(sub["n_gps"].median()),
                "mean": round(float(sub["n_gps"].mean()), 2),
                "max": int(sub["n_gps"].max()),
            },
            "nodes_per_trip": {
                "min": int(sub["n_nodes"].min()),
                "median": float(sub["n_nodes"].median()),
                "max": int(sub["n_nodes"].max()),
                "unique_counts": int(sub["n_nodes"].nunique()),
            },
            "arcs_per_trip": {
                "min": int(sub["n_arcs"].min()),
                "median": float(sub["n_arcs"].median()),
                "max": int(sub["n_arcs"].max()),
                "unique_counts": int(sub["n_arcs"].nunique()),
            },
            "route_len_per_trip": {
                "min": int(sub["route_len"].min()),
                "median": float(sub["route_len"].median()),
                "mean": round(float(sub["route_len"].mean()), 2),
                "max": int(sub["route_len"].max()),
            },
            "quality_anomalies": {
                "total_track_malformed": int(sub["track_malformed"].sum()),
                "total_track_nan_inf": int(sub["track_nan_inf"].sum()),
                "total_nodes_malformed": int(sub["nodes_malformed"].sum()),
                "total_nodes_nan_inf": int(sub["nodes_nan_inf"].sum()),
                "total_arcs_malformed": int(sub["arcs_malformed"].sum()),
                "total_route_malformed": int(sub["route_malformed"].sum()),
                "total_coord_out_of_bounds": int(sub["coord_out_of_bounds"].sum()),
                "total_invalid_arc_endpoints": int(sub["invalid_arc_endpoints"].sum()),
                "total_invalid_route_arcs": int(sub["invalid_route_arcs"].sum()),
                "total_non_monotonic_dt": int(sub["n_dt_non_monotonic"].sum()),
                "trips_with_non_monotonic_dt": int((sub["n_dt_non_monotonic"] > 0).sum()),
            },
            "temporal": {
                "total_intervals": int(sub["n_gps"].sum() - len(sub)),
                "total_dt_1s": int(sub["n_dt_1s"].sum()),
                "overall_pct_dt_1s": round(
                    float(sub["n_dt_1s"].sum() / max(1, sub["n_gps"].sum() - len(sub)) * 100.0),
                    3,
                ),
                "total_dt_gt_1s": int(sub["n_dt_gt_1s"].sum()),
                "total_dt_gt_5s": int(sub["n_dt_gt_5s"].sum()),
                "total_dt_gt_10s": int(sub["n_dt_gt_10s"].sum()),
                "total_dt_gt_30s": int(sub["n_dt_gt_30s"].sum()),
                "max_dt_s": float(sub["dt_max"].max()),
                "trips_with_gaps_gt_5s": int((sub["n_dt_gt_5s"] > 0).sum()),
            },
            "spatial_and_noise": {
                "lon_range": [float(sub["lon_min"].min()), float(sub["lon_max"].max())],
                "lat_range": [float(sub["lat_min"].min()), float(sub["lat_max"].max())],
                "track_length_km_median": round(float(sub["track_length_km"].median()), 3),
                "track_length_km_max": round(float(sub["track_length_km"].max()), 3),
                "gps_to_road_median_m_across_trips": round(float(sub["gps_to_road_median_m"].median()), 2),
                "gps_to_road_max_m": round(float(sub["gps_to_road_max_m"].max()), 2),
                "gps_to_route_median_m_across_trips": round(float(sub["gps_to_route_median_m"].median()), 2),
                "gps_to_route_max_m": round(float(sub["gps_to_route_max_m"].max()), 2),
                "trips_with_speed_gt_150_kmh": int((sub["n_speed_gt_150"] > 0).sum()),
            },
            "route_validity": {
                "trips_with_100pct_continuity": int((sub["n_disconnected"] == 0).sum()),
                "trips_with_disconnected_route": int((sub["n_disconnected"] > 0).sum()),
                "total_disconnected_transitions": int(sub["n_disconnected"].sum()),
                "total_same_arc_repeats": int(sub["n_same_arc_repeat"].sum()),
                "total_u_turns": int(sub["n_u_turn"].sum()),
                "max_route_gap_m": float(sub["max_route_gap_m"].max()),
                "min_route_recall_50m_pct": float(sub["route_recall_50m_pct"].min()),
                "min_route_recall_100m_pct": float(sub["route_recall_100m_pct"].min()),
            },
        }

    summary = {
        "overall": summarize_split(df_inv),
        "train": summarize_split(df_inv[df_inv["split"] == "train"]),
        "student_test": summarize_split(df_inv[df_inv["split"] == "student_test"]),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    }

    summary_path = results_dir / "data_quality_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== AUDIT COMPLETE ===")
    print(f"Elapsed: {summary['elapsed_seconds']}s")
    print(f"Saved: {inv_path}")
    print(f"Saved: {gaps_path} ({len(df_gaps)} irregular interval events)")
    print(f"Saved: {net_path}")
    print(f"Saved: {route_path}")
    print(f"Saved: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

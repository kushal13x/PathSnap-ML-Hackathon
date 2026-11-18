"""Metric projection, GPS kinematics, and road-network preprocessing for PathSnap.

All distances and bearings are computed in physical meters via a per-trip
WGS84 local tangent-plane projection (never Euclidean degree space).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

WGS84_A = 6378137.0
WGS84_E2 = 0.0066943799901413165
EARTH_MEAN_RADIUS_M = 6371008.8


def haversine_meters(
    lon1: np.ndarray | float,
    lat1: np.ndarray | float,
    lon2: np.ndarray | float,
    lat2: np.ndarray | float,
) -> np.ndarray | float:
    """Compute great-circle distance in meters using the Haversine formula."""
    lon1_r = np.radians(lon1)
    lat1_r = np.radians(lat1)
    lon2_r = np.radians(lon2)
    lat2_r = np.radians(lat2)

    dlon = lon2_r - lon1_r
    dlat = lat2_r - lat1_r

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    return EARTH_MEAN_RADIUS_M * 2.0 * np.arcsin(np.sqrt(a))


class LocalTangentProjection:
    """Local WGS84 tangent-plane projection in meters around (ref_lon, ref_lat)."""

    def __init__(self, ref_lon: float, ref_lat: float) -> None:
        self.ref_lon = float(ref_lon)
        self.ref_lat = float(ref_lat)
        lat_rad = math.radians(self.ref_lat)
        sin_lat = math.sin(lat_rad)
        cos_lat = math.cos(lat_rad)

        denom = 1.0 - WGS84_E2 * sin_lat * sin_lat
        r_m = WGS84_A * (1.0 - WGS84_E2) / (denom ** 1.5)
        r_n = WGS84_A / math.sqrt(denom)

        self.meters_per_deg_lat = math.radians(1.0) * r_m
        self.meters_per_deg_lon = math.radians(1.0) * r_n * cos_lat

    def to_xy(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project (lon, lat) arrays in degrees to local (x_east, y_north) in meters."""
        x = (np.asarray(lon, dtype=np.float64) - self.ref_lon) * self.meters_per_deg_lon
        y = (np.asarray(lat, dtype=np.float64) - self.ref_lat) * self.meters_per_deg_lat
        return x, y


def point_to_segments_distance(
    px: np.ndarray | float,
    py: np.ndarray | float,
    ax: np.ndarray,
    ay: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute physical distance (m) and projection ratio t in [0, 1] from P to segments AB."""
    abx = bx - ax
    aby = by - ay
    ab_len2 = abx * abx + aby * aby

    apx = px - ax
    apy = py - ay

    safe_len2 = np.where(ab_len2 > 1e-12, ab_len2, 1.0)
    t = np.where(ab_len2 > 1e-12, (apx * abx + apy * aby) / safe_len2, 0.0)
    t = np.clip(t, 0.0, 1.0)

    proj_x = ax + t * abx
    proj_y = ay + t * aby
    dist = np.hypot(px - proj_x, py - proj_y)
    return dist, t


def segment_bearings_deg(
    ax: np.ndarray,
    ay: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
) -> np.ndarray:
    """Clockwise bearing from North in [0, 360) degrees for segments A->B."""
    angles = np.degrees(np.arctan2(bx - ax, by - ay))
    return np.mod(angles, 360.0)


def angular_diff_deg(
    a: np.ndarray | float, b: np.ndarray | float
) -> np.ndarray | float:
    """Smallest absolute angular difference in [0, 180] degrees."""
    diff = np.abs(a - b) % 360.0
    return np.where(diff > 180.0, 360.0 - diff, diff)


@dataclass(frozen=True)
class GPSKinematics:
    """Gap-aware GPS kinematics computed strictly from .track coordinates & timestamps."""

    px: np.ndarray
    py: np.ndarray
    dt_prev_s: np.ndarray
    step_dist_prev_m: np.ndarray
    speed_prev_mps: np.ndarray
    speed_smooth_mps: np.ndarray
    heading_deg: np.ndarray
    is_moving: np.ndarray
    heading_turn_deg: np.ndarray


def compute_gps_kinematics(
    px: np.ndarray,
    py: np.ndarray,
    ts: np.ndarray,
    gap_threshold_s: float = 10.0,
    moving_speed_mps: float = 1.5,
) -> GPSKinematics:
    """Compute gap-aware speed, heading, and local movement consistency from .track."""
    n = len(px)
    dt_prev = np.ones(n, dtype=np.float64)
    step_dist_prev = np.zeros(n, dtype=np.float64)
    speed_prev = np.zeros(n, dtype=np.float64)
    speed_smooth = np.zeros(n, dtype=np.float64)
    heading = np.zeros(n, dtype=np.float64)
    heading_turn = np.zeros(n, dtype=np.float64)

    if n >= 2:
        dt_raw = np.maximum(np.diff(ts), 1e-3)
        dist_raw = np.hypot(np.diff(px), np.diff(py))
        dt_prev[1:] = dt_raw
        step_dist_prev[1:] = dist_raw
        speed_prev[1:] = dist_raw / dt_raw
        speed_prev[0] = speed_prev[1]

        # Gap-aware instantaneous heading & speed:
        # Use centered difference only when neither neighbor is across a long gap.
        for t in range(n):
            has_prev = t > 0 and dt_prev[t] <= gap_threshold_s
            has_next = t < n - 1 and dt_prev[t + 1] <= gap_threshold_s

            if has_prev and has_next:
                dx = px[t + 1] - px[t - 1]
                dy = py[t + 1] - py[t - 1]
                dt_span = dt_prev[t] + dt_prev[t + 1]
            elif has_next:
                dx = px[t + 1] - px[t]
                dy = py[t + 1] - py[t]
                dt_span = dt_prev[t + 1]
            elif has_prev:
                dx = px[t] - px[t - 1]
                dy = py[t] - py[t - 1]
                dt_span = dt_prev[t]
            else:
                # Isolated point surrounded by long gaps: fallback to backward/forward step
                if t > 0:
                    dx, dy, dt_span = px[t] - px[t - 1], py[t] - py[t - 1], dt_prev[t]
                else:
                    dx, dy, dt_span = px[t + 1] - px[t], py[t + 1] - py[t], dt_prev[t + 1]

            disp = math.hypot(dx, dy)
            speed_smooth[t] = disp / max(dt_span, 1e-3)
            heading[t] = math.degrees(math.atan2(dx, dy)) % 360.0

        is_moving = speed_smooth > moving_speed_mps
        for t in range(1, n):
            if dt_prev[t] <= gap_threshold_s and is_moving[t] and is_moving[t - 1]:
                heading_turn[t] = float(angular_diff_deg(heading[t], heading[t - 1]))
            else:
                heading_turn[t] = 0.0
    else:
        is_moving = np.zeros(n, dtype=bool)

    return GPSKinematics(
        px=px,
        py=py,
        dt_prev_s=dt_prev,
        step_dist_prev_m=step_dist_prev,
        speed_prev_mps=speed_prev,
        speed_smooth_mps=speed_smooth,
        heading_deg=heading,
        is_moving=is_moving,
        heading_turn_deg=heading_turn,
    )


@dataclass(frozen=True)
class RoadNetworkMetric:
    """Projected road network arrays and topological degree features."""

    nx: np.ndarray
    ny: np.ndarray
    u: np.ndarray
    v: np.ndarray
    ax: np.ndarray
    ay: np.ndarray
    bx: np.ndarray
    by: np.ndarray
    arc_lengths_m: np.ndarray
    arc_bearings_deg: np.ndarray
    node_tot_deg: np.ndarray
    is_bidirectional: np.ndarray


def preprocess_road_network(
    node_lon: np.ndarray,
    node_lat: np.ndarray,
    arcs: np.ndarray,
    proj: LocalTangentProjection,
) -> RoadNetworkMetric:
    """Project road network nodes and precompute segment geometry & topological features."""
    nx, ny = proj.to_xy(node_lon, node_lat)
    u = arcs[:, 0]
    v = arcs[:, 1]
    ax, ay = nx[u], ny[u]
    bx, by = nx[v], ny[v]

    arc_lengths_m = np.hypot(bx - ax, by - ay)
    arc_bearings = segment_bearings_deg(ax, ay, bx, by)

    n_nodes = len(nx)
    in_deg = np.bincount(v, minlength=n_nodes)
    out_deg = np.bincount(u, minlength=n_nodes)
    node_tot_deg = in_deg + out_deg

    # Bidirectional check via 64-bit packed (u, v) keys
    edge_keys = u.astype(np.int64) * np.int64(n_nodes) + v.astype(np.int64)
    rev_keys = v.astype(np.int64) * np.int64(n_nodes) + u.astype(np.int64)
    unique_keys = np.unique(edge_keys)
    is_bidirectional = np.isin(rev_keys, unique_keys).astype(np.float32)

    return RoadNetworkMetric(
        nx=nx,
        ny=ny,
        u=u,
        v=v,
        ax=ax,
        ay=ay,
        bx=bx,
        by=by,
        arc_lengths_m=arc_lengths_m,
        arc_bearings_deg=arc_bearings,
        node_tot_deg=node_tot_deg,
        is_bidirectional=is_bidirectional,
    )

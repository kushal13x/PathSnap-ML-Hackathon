"""Geographical distance and local metric projection utilities for PathSnap.

Avoids Euclidean distance in angular degrees by providing:
1. Vectorized WGS84 Haversine distance in meters.
2. Local WGS84 tangent-plane projection (meters) centered at a trip reference point.
3. Exact vectorized point-to-segment distances and segment bearings in meters.
"""

from __future__ import annotations

import math
import numpy as np

# WGS84 ellipsoid constants (meters)
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
    c = 2.0 * np.arcsin(np.sqrt(a))
    return EARTH_MEAN_RADIUS_M * c


class LocalTangentProjection:
    """Local WGS84 tangent-plane projection in meters around (ref_lon, ref_lat)."""

    def __init__(self, ref_lon: float, ref_lat: float) -> None:
        self.ref_lon = float(ref_lon)
        self.ref_lat = float(ref_lat)
        lat_rad = math.radians(self.ref_lat)
        sin_lat = math.sin(lat_rad)
        cos_lat = math.cos(lat_rad)

        denom = 1.0 - WGS84_E2 * sin_lat * sin_lat
        # Meridional (north-south) and prime-vertical (east-west) radii of curvature
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
    px: np.ndarray,
    py: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute distance in meters from P=(px,py) to line segments A=(ax,ay)->B=(bx,by).

    Broadcasts cleanly:
      - If P is shape (N, 1) and A, B are shape (1, M), returns (N, M) distances and (N, M) t in [0, 1].
      - If P, A, B are 1D of length M, returns 1D arrays of length M.
    """
    abx = bx - ax
    aby = by - ay
    ab_len2 = abx * abx + aby * aby

    apx = px - ax
    apy = py - ay

    # Safe division when segment has zero length
    t = np.where(ab_len2 > 1e-12, (apx * abx + apy * aby) / np.where(ab_len2 > 1e-12, ab_len2, 1.0), 0.0)
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
    """Compute clockwise bearing from North in [0, 360) degrees for segments A->B."""
    dx = bx - ax
    dy = by - ay
    angles = np.degrees(np.arctan2(dx, dy))
    return np.mod(angles, 360.0)


def angular_diff_deg(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray | float:
    """Smallest absolute angular difference in [0, 180] degrees."""
    diff = np.abs(a - b) % 360.0
    return np.where(diff > 180.0, 360.0 - diff, diff)

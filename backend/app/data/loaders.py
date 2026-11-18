"""Data loading utilities for PathSnap.

Separates inference inputs (.track, .nodes, .arcs) from offline reference
routes (.route) to prevent target leakage during candidate generation and inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import psutil


@dataclass(frozen=True)
class TripInputs:
    """Inference-time trip inputs (explicitly excludes .route)."""

    split: str
    trip_id: str
    lon: np.ndarray        # shape (T,), float64
    lat: np.ndarray        # shape (T,), float64
    timestamps: np.ndarray # shape (T,), float64
    node_lon: np.ndarray   # shape (N,), float64
    node_lat: np.ndarray   # shape (N,), float64
    arcs: np.ndarray       # shape (E, 2), int64 [from_node_id, to_node_id]


def iter_trip_ids(data_root: Path, split: str) -> Iterator[str]:
    """Yield sorted trip IDs for a dataset split ('train' or 'student_test')."""
    split_dir = data_root / split
    if not split_dir.is_dir():
        return
    for trip_dir in sorted([d for d in split_dir.iterdir() if d.is_dir()]):
        yield trip_dir.name


def load_trip_inputs(data_root: Path, split: str, trip_id: str) -> TripInputs:
    """Load .track, .nodes, and .arcs for a single trip (never touches .route)."""
    td = data_root / split / trip_id
    track = pd.read_csv(td / f"{trip_id}.track", sep=r"\s+", header=None, engine="c", dtype=np.float64).to_numpy()
    nodes = pd.read_csv(td / f"{trip_id}.nodes", sep=r"\s+", header=None, engine="c", dtype=np.float64).to_numpy()
    arcs = pd.read_csv(td / f"{trip_id}.arcs", sep=r"\s+", header=None, engine="c", dtype=np.int64).to_numpy()

    return TripInputs(
        split=split,
        trip_id=trip_id,
        lon=track[:, 0],
        lat=track[:, 1],
        timestamps=track[:, 2],
        node_lon=nodes[:, 0],
        node_lat=nodes[:, 1],
        arcs=arcs,
    )


def load_reference_route(data_root: Path, split: str, trip_id: str) -> np.ndarray:
    """Load offline reference .route arc ID sequence (for training alignment & evaluation only)."""
    route_path = data_root / split / trip_id / f"{trip_id}.route"
    route = pd.read_csv(route_path, sep=r"\s+", header=None, engine="c", dtype=np.int64).to_numpy().ravel()
    return route


def get_process_rss_mb() -> float:
    """Return current process Resident Set Size (RSS) in megabytes."""
    return psutil.Process().memory_info().rss / (1024.0 * 1024.0)

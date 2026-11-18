"""Pydantic v2 API contract schemas for PathSnap FastAPI backend.

All geographical coordinates returned by the API follow WGS84 [latitude, longitude]
ordering ('coordinate_order': 'lat_lon') for direct Folium rendering.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Standard error payload."""

    detail: str
    error_code: str


class HealthResponse(BaseModel):
    """Response schema for GET /health."""

    status: str = Field(..., description="'ok' when model & config are loaded, else 'degraded'")
    service: str
    version: str
    model_loaded: bool
    model_name: str
    model_path: str
    hmm_config_loaded: bool
    inference_dataset_name: str
    available_trip_count: int


class TripSummaryItem(BaseModel):
    """Metadata for a single inference trip (strictly excludes .route)."""

    trip_id: str
    n_gps_points: int
    duration_s: float
    has_required_files: bool


class TripListResponse(BaseModel):
    """Response schema for GET /trips."""

    inference_dataset_name: str
    trip_count: int
    trips: list[TripSummaryItem]


class TripMapDataResponse(BaseModel):
    """Response schema for GET /trips/{trip_id}/map-data (no ground-truth .route)."""

    trip_id: str
    coordinate_order: str = Field(default="lat_lon", description="All coordinates are [latitude, longitude]")
    center: list[float]
    bounds: list[list[float]]
    n_gps_observations: int
    gps_coordinates: list[list[float]]
    road_network_segments: list[list[list[float]]]
    total_network_arcs: int
    displayed_network_arcs: int


class PredictionDiagnostics(BaseModel):
    """Sequence-level HMM/Viterbi reconstruction diagnostics (non-ground-truth)."""

    route_continuity_pct: float
    connector_count: int
    unconnected_gaps: int
    temporal_gap_splits: int
    empty_candidate_obs_count: int
    viterbi_disconnected_steps: int
    viterbi_total_log_score: float
    warnings: list[str]


class PredictionResponse(BaseModel):
    """Response schema for POST /predict/{trip_id}."""

    trip_id: str
    status: str
    coordinate_order: str = Field(default="lat_lon", description="All coordinates are [latitude, longitude]")
    model_used: str
    predicted_obs_arc_ids: list[int]
    reconstructed_route_arc_ids: list[int]
    is_connector_arc: list[bool]
    gps_coordinates: list[list[float]]
    reconstructed_route_segments: list[list[list[float]]]
    start_coordinate: list[float]
    end_coordinate: list[float]
    bounds: list[list[float]]
    observation_count: int
    candidate_count: int
    mean_candidates_per_obs: float
    processing_time_s: float
    diagnostics: PredictionDiagnostics

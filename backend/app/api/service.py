"""Inference service executing the real PathSnap Phase 1-4 pipeline.

CRITICAL DATA SEPARATION:
This module NEVER imports, loads, reads, or exposes .route ground-truth files.
Predictions rely strictly on .track, .nodes, and .arcs from inference_data_dir.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.app.api.config import APISettings
from backend.app.api.schemas import (
    HealthResponse,
    PredictionDiagnostics,
    PredictionResponse,
    TripListResponse,
    TripMapDataResponse,
    TripSummaryItem,
)
from backend.app.data.loaders import TripInputs
from backend.app.data.preprocessing import (
    LocalTangentProjection,
    compute_gps_kinematics,
    preprocess_road_network,
)
from backend.app.decoder.hmm import (
    EmissionConfig,
    build_hmm_observation_steps,
    compute_classifier_probabilities,
)
from backend.app.decoder.route_reconstruction import reconstruct_route_from_decoded
from backend.app.decoder.transition_model import DirectedRoadGraph, TransitionConfig
from backend.app.decoder.viterbi import decode_viterbi_sequence
from backend.app.features.candidate_generation import (
    CandidateConfig,
    generate_candidates_for_trip,
)
from backend.app.features.feature_engineering import (
    FEATURE_COLUMNS,
    build_candidate_feature_table,
)
from backend.app.models.model_comparison import load_model_artifact

logger = logging.getLogger("pathsnap.api")

_VALID_TRIP_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class InvalidTripIdError(ValueError):
    """Raised when trip_id contains invalid or unsafe characters."""


class TripNotFoundError(FileNotFoundError):
    """Raised when trip directory or required inference files are missing."""


class InferenceService:
    """Stateful inference engine holding startup-loaded model and decoder configs."""

    def __init__(self, settings: APISettings) -> None:
        self.settings = settings
        self.model: Any = None
        self.model_meta: dict[str, Any] = {}
        self.model_loaded: bool = False
        self.hmm_config_loaded: bool = False
        self.cand_cfg = CandidateConfig()
        self.emit_cfg = EmissionConfig()
        self.trans_cfg = TransitionConfig()

    def startup(self) -> None:
        """Load trained classifier and configs once at FastAPI startup."""
        for label, p in [
            ("Model artifact", self.settings.model_path),
            ("Pipeline config", self.settings.pipeline_config_path),
            ("HMM config", self.settings.hmm_config_path),
        ]:
            if not p.is_file():
                raise FileNotFoundError(f"{label} not found: {p}")

        logger.info("Loading model artifact from %s", self.settings.model_path)
        self.model, self.model_meta = load_model_artifact(self.settings.model_path)
        self.model_loaded = True

        with open(self.settings.pipeline_config_path, "r", encoding="utf-8") as f:
            pipe_cfg = json.load(f)
        self.cand_cfg = CandidateConfig(**pipe_cfg["candidate_generation"])

        with open(self.settings.hmm_config_path, "r", encoding="utf-8") as f:
            hmm_cfg = json.load(f)
        self.emit_cfg = EmissionConfig(**hmm_cfg["emission"])
        self.trans_cfg = TransitionConfig(**hmm_cfg["transition"])
        self.hmm_config_loaded = True

        logger.info(
            "Startup complete: model=%s | inference_dir=%s | K=%d, R=%.1fm",
            self.model_meta.get("model_key", "hist_gradient_boosting_tuned"),
            self.settings.inference_data_dir,
            self.cand_cfg.k_candidates,
            self.cand_cfg.radius_m,
        )

    def _validate_and_resolve_trip_dir(self, trip_id: str) -> Path:
        if not _VALID_TRIP_ID_RE.fullmatch(trip_id):
            raise InvalidTripIdError(f"Invalid trip_id format: {trip_id!r}")

        trip_dir = (self.settings.inference_data_dir / trip_id).resolve()
        if not trip_dir.is_dir():
            raise TripNotFoundError(f"Trip '{trip_id}' not found in inference directory.")

        for ext in (".track", ".nodes", ".arcs"):
            fpath = trip_dir / f"{trip_id}{ext}"
            if not fpath.is_file():
                raise TripNotFoundError(f"Required inference file missing: {fpath.name}")
        return trip_dir

    def _load_inference_trip_inputs(self, trip_id: str) -> TripInputs:
        """Load strictly .track, .nodes, and .arcs (never reads .route)."""
        trip_dir = self._validate_and_resolve_trip_dir(trip_id)
        track = pd.read_csv(
            trip_dir / f"{trip_id}.track",
            sep=r"\s+",
            header=None,
            engine="c",
            dtype=np.float64,
        ).to_numpy()
        nodes = pd.read_csv(
            trip_dir / f"{trip_id}.nodes",
            sep=r"\s+",
            header=None,
            engine="c",
            dtype=np.float64,
        ).to_numpy()
        arcs = pd.read_csv(
            trip_dir / f"{trip_id}.arcs",
            sep=r"\s+",
            header=None,
            engine="c",
            dtype=np.int64,
        ).to_numpy()

        if len(track) == 0 or len(nodes) == 0 or len(arcs) == 0:
            raise TripNotFoundError(f"Trip '{trip_id}' has empty inference files.")

        return TripInputs(
            split=self.settings.inference_data_dir.name,
            trip_id=trip_id,
            lon=track[:, 0],
            lat=track[:, 1],
            timestamps=track[:, 2],
            node_lon=nodes[:, 0],
            node_lat=nodes[:, 1],
            arcs=arcs,
        )

    def _list_trip_dirs(self) -> list[Path]:
        if not self.settings.inference_data_dir.is_dir():
            return []
        return sorted(
            [d for d in self.settings.inference_data_dir.iterdir() if d.is_dir()]
        )

    def get_health(self) -> HealthResponse:
        trip_dirs = self._list_trip_dirs()
        ok = self.model_loaded and self.hmm_config_loaded
        return HealthResponse(
            status="ok" if ok else "degraded",
            service="PathSnap Distributed Map-Matching API",
            version="1.0.0",
            model_loaded=self.model_loaded,
            model_name=str(
                self.model_meta.get("display_name", "HistGradientBoosting (Tuned)")
            ),
            model_path=self.settings.model_path.name,
            hmm_config_loaded=self.hmm_config_loaded,
            inference_dataset_name=self.settings.inference_data_dir.name,
            available_trip_count=len(trip_dirs),
        )

    def list_trips(self) -> TripListResponse:
        items: list[TripSummaryItem] = []
        for td in self._list_trip_dirs():
            tid = td.name
            has_req = all((td / f"{tid}{ext}").is_file() for ext in (".track", ".nodes", ".arcs"))
            n_pts = 0
            dur_s = 0.0
            if (td / f"{tid}.track").is_file():
                try:
                    track = pd.read_csv(
                        td / f"{tid}.track",
                        sep=r"\s+",
                        header=None,
                        engine="c",
                        dtype=np.float64,
                    ).to_numpy()
                    n_pts = int(len(track))
                    if n_pts >= 2:
                        dur_s = round(float(track[-1, 2] - track[0, 2]), 2)
                except Exception as exc:
                    logger.warning("Failed reading metadata for trip '%s': %s", tid, exc)
                    has_req = False
            items.append(
                TripSummaryItem(
                    trip_id=tid,
                    n_gps_points=n_pts,
                    duration_s=dur_s,
                    has_required_files=has_req,
                )
            )
        return TripListResponse(
            inference_dataset_name=self.settings.inference_data_dir.name,
            trip_count=len(items),
            trips=items,
        )

    def get_trip_map_data(self, trip_id: str) -> TripMapDataResponse:
        ti = self._load_inference_trip_inputs(trip_id)
        lon, lat = ti.lon, ti.lat

        # Clip local road network around trajectory bounding box (~350m margin)
        lon_pad, lat_pad = 0.004, 0.003
        lmin, lmax = float(lon.min()) - lon_pad, float(lon.max()) + lon_pad
        bmin, bmax = float(lat.min()) - lat_pad, float(lat.max()) + lat_pad

        u_lon, u_lat = ti.node_lon[ti.arcs[:, 0]], ti.node_lat[ti.arcs[:, 0]]
        v_lon, v_lat = ti.node_lon[ti.arcs[:, 1]], ti.node_lat[ti.arcs[:, 1]]
        in_box = (
            (np.maximum(u_lon, v_lon) >= lmin)
            & (np.minimum(u_lon, v_lon) <= lmax)
            & (np.maximum(u_lat, v_lat) >= bmin)
            & (np.minimum(u_lat, v_lat) <= bmax)
        )
        local_idx = np.flatnonzero(in_box)
        if len(local_idx) > self.settings.max_map_road_segments:
            step = max(1, len(local_idx) // self.settings.max_map_road_segments)
            local_idx = local_idx[::step][: self.settings.max_map_road_segments]

        road_segments = [
            [
                [round(float(u_lat[i]), 6), round(float(u_lon[i]), 6)],
                [round(float(v_lat[i]), 6), round(float(v_lon[i]), 6)],
            ]
            for i in local_idx
        ]
        gps_coords = [
            [round(float(lat[i]), 6), round(float(lon[i]), 6)]
            for i in range(len(lat))
        ]

        return TripMapDataResponse(
            trip_id=trip_id,
            coordinate_order="lat_lon",
            center=[round(float(lat.mean()), 6), round(float(lon.mean()), 6)],
            bounds=[
                [round(float(lat.min()), 6), round(float(lon.min()), 6)],
                [round(float(lat.max()), 6), round(float(lon.max()), 6)],
            ],
            n_gps_observations=len(gps_coords),
            gps_coordinates=gps_coords,
            road_network_segments=road_segments,
            total_network_arcs=int(len(ti.arcs)),
            displayed_network_arcs=len(road_segments),
        )

    def predict_trip(self, trip_id: str) -> PredictionResponse:
        if not self.model_loaded or not self.hmm_config_loaded:
            raise RuntimeError("Inference model or HMM config is not loaded.")

        t0 = time.perf_counter()
        ti = self._load_inference_trip_inputs(trip_id)

        # 1. Local metric projection, kinematics & road network
        proj = LocalTangentProjection(float(ti.lon.mean()), float(ti.lat.mean()))
        px, py = proj.to_xy(ti.lon, ti.lat)
        kin = compute_gps_kinematics(px, py, ti.timestamps)
        road_net = preprocess_road_network(ti.node_lon, ti.node_lat, ti.arcs, proj)

        # 2. Candidate generation & 26-feature extraction (strictly .route-free)
        cands = generate_candidates_for_trip(px, py, road_net, self.cand_cfg)
        df_cands = build_candidate_feature_table(
            trip_id, kin, road_net, cands, alignment=None
        )

        # 3. Classifier positive probabilities & HMM observation steps
        cand_probs = compute_classifier_probabilities(
            self.model, df_cands, FEATURE_COLUMNS
        )
        steps = build_hmm_observation_steps(
            len(ti.lon), ti.timestamps, kin, df_cands, cand_probs, self.emit_cfg
        )

        # 4. Graph routing, Viterbi decoding & auditable route reconstruction
        graph = DirectedRoadGraph(road_net)
        viterbi_res = decode_viterbi_sequence(steps, road_net, graph, self.trans_cfg)
        reconstructed = reconstruct_route_from_decoded(
            viterbi_res.decoded_obs_arcs,
            viterbi_res.segment_ids,
            road_net,
            graph,
            connector_budget_m=self.trans_cfg.max_route_search_m,
        )

        elapsed_s = round(time.perf_counter() - t0, 4)

        # Build compact coordinate responses ([lat, lon])
        gps_coords = [
            [round(float(ti.lat[i]), 6), round(float(ti.lon[i]), 6)]
            for i in range(len(ti.lat))
        ]
        route_segments = []
        for aid in reconstructed.reconstructed_route_arcs:
            u = int(ti.arcs[aid, 0])
            v = int(ti.arcs[aid, 1])
            route_segments.append(
                [
                    [round(float(ti.node_lat[u]), 6), round(float(ti.node_lon[u]), 6)],
                    [round(float(ti.node_lat[v]), 6), round(float(ti.node_lon[v]), 6)],
                ]
            )

        warnings: list[str] = []
        if viterbi_res.n_gap_splits > 0:
            warnings.append(
                f"{viterbi_res.n_gap_splits} temporal gap(s) > {self.trans_cfg.gap_threshold_s:.0f}s "
                "triggered independent segment decoding."
            )
        if cands.n_obs_no_candidates > 0:
            warnings.append(
                f"{cands.n_obs_no_candidates} GPS observation(s) had 0 road candidates "
                f"within {self.cand_cfg.radius_m:.0f}m and were marked unmatched (-1)."
            )
        if reconstructed.n_unconnected_gaps > 0:
            warnings.append(
                f"{reconstructed.n_unconnected_gaps} discontinuity/segment break(s) remain "
                "in the reconstructed route (no synthetic connectors invented)."
            )

        logger.info(
            "POST /predict/%s | obs=%d | cands=%d | route_arcs=%d (connectors=%d) | "
            "cont=%.2f%% | time=%.3fs",
            trip_id,
            len(ti.lon),
            len(df_cands),
            len(reconstructed.reconstructed_route_arcs),
            reconstructed.n_inserted_connectors,
            reconstructed.route_continuity_pct,
            elapsed_s,
        )

        return PredictionResponse(
            trip_id=trip_id,
            status="success",
            coordinate_order="lat_lon",
            model_used=str(self.model_meta.get("model_key", "hist_gradient_boosting_tuned")),
            predicted_obs_arc_ids=[int(x) for x in viterbi_res.decoded_obs_arcs],
            reconstructed_route_arc_ids=reconstructed.reconstructed_route_arcs,
            is_connector_arc=reconstructed.is_connector_mask,
            gps_coordinates=gps_coords,
            reconstructed_route_segments=route_segments,
            start_coordinate=gps_coords[0],
            end_coordinate=gps_coords[-1],
            bounds=[
                [round(float(ti.lat.min()), 6), round(float(ti.lon.min()), 6)],
                [round(float(ti.lat.max()), 6), round(float(ti.lon.max()), 6)],
            ],
            observation_count=int(len(ti.lon)),
            candidate_count=int(len(df_cands)),
            mean_candidates_per_obs=round(len(df_cands) / max(1, len(ti.lon)), 2),
            processing_time_s=elapsed_s,
            diagnostics=PredictionDiagnostics(
                route_continuity_pct=reconstructed.route_continuity_pct,
                connector_count=reconstructed.n_inserted_connectors,
                unconnected_gaps=reconstructed.n_unconnected_gaps,
                temporal_gap_splits=viterbi_res.n_gap_splits,
                empty_candidate_obs_count=int(cands.n_obs_no_candidates),
                viterbi_disconnected_steps=viterbi_res.n_disconnected_viterbi_steps,
                viterbi_total_log_score=round(float(viterbi_res.total_log_score), 3),
                warnings=warnings,
            ),
        )

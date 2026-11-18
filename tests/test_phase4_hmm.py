"""Automated tests for Phase 3 Bayesian Optimization and Phase 4 HMM/Viterbi."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.data.preprocessing import (
    GPSKinematics,
    LocalTangentProjection,
    RoadNetworkMetric,
)
from backend.app.decoder.hmm import (
    EmissionConfig,
    HMMObservationStep,
    build_hmm_observation_steps,
)
from backend.app.decoder.route_reconstruction import reconstruct_route_from_decoded
from backend.app.decoder.transition_model import (
    DirectedRoadGraph,
    TransitionConfig,
    compute_transition_log_probs,
)
from backend.app.decoder.viterbi import decode_viterbi_sequence


def _make_synthetic_network():
    """Create a small directed graph with 2 parallel paths and 1 disconnected arc.

    Nodes:
      0:(0,0) -> 1:(100,0) -> 2:(200,0) -> 3:(300,0)   [connected main road: arcs 0,1,2]
      4:(0,40) -> 5:(100,40)                            [parallel disconnected street: arc 3]
      6:(100,-50) -> 7:(200,-50)                        [another disconnected street: arc 4]
    """
    nx = np.array([0.0, 100.0, 200.0, 300.0, 0.0, 100.0, 100.0, 200.0])
    ny = np.array([0.0, 0.0, 0.0, 0.0, 40.0, 40.0, -50.0, -50.0])
    u = np.array([0, 1, 2, 4, 6], dtype=np.int64)
    v = np.array([1, 2, 3, 5, 7], dtype=np.int64)
    ax, ay = nx[u], ny[u]
    bx, by = nx[v], ny[v]
    lens = np.hypot(bx - ax, by - ay)
    bearings = np.full(len(u), 90.0)
    deg = np.ones(len(nx), dtype=np.int64)
    bidir = np.zeros(len(u), dtype=np.float32)

    road_net = RoadNetworkMetric(
        nx=nx, ny=ny, u=u, v=v, ax=ax, ay=ay, bx=bx, by=by,
        arc_lengths_m=lens, arc_bearings_deg=bearings,
        node_tot_deg=deg, is_bidirectional=bidir,
    )
    return road_net, DirectedRoadGraph(road_net)


def test_viterbi_recovers_connected_path_over_noisy_emission():
    road_net, graph = _make_synthetic_network()
    trans_cfg = TransitionConfig(gap_threshold_s=30.0, beta_m=20.0, disconnected_penalty=15.0)

    # At t=1, disconnected arc 4 has slightly higher emission (0.6 vs 0.4),
    # but arcs 0 -> 1 -> 2 form a connected path matching d_gps=100m!
    steps = [
        HMMObservationStep(0, 1.0, 50.0, 0.0, np.array([0, 3], dtype=np.int64), np.array([0.5, 0.5], dtype=np.float32), np.array([0.85, 0.15]), np.log([0.85, 0.15])),
        HMMObservationStep(1, 2.0, 150.0, 0.0, np.array([1, 4], dtype=np.int64), np.array([0.5, 0.5], dtype=np.float32), np.array([0.40, 0.60]), np.log([0.40, 0.60])),
        HMMObservationStep(2, 3.0, 250.0, 0.0, np.array([2], dtype=np.int64), np.array([0.5], dtype=np.float32), np.array([0.90]), np.log([0.90])),
    ]
    res = decode_viterbi_sequence(steps, road_net, graph, trans_cfg)
    assert np.array_equal(res.decoded_obs_arcs, np.array([0, 1, 2], dtype=np.int64))
    assert res.n_disconnected_viterbi_steps == 0
    assert res.n_segments == 1


def test_viterbi_splits_on_long_temporal_gap_and_empty_candidates():
    road_net, graph = _make_synthetic_network()
    trans_cfg = TransitionConfig(gap_threshold_s=30.0, beta_m=20.0, disconnected_penalty=15.0)

    steps = [
        HMMObservationStep(0, 1.0, 50.0, 0.0, np.array([0], dtype=np.int64), np.array([0.5], dtype=np.float32), np.array([0.9]), np.log([0.9])),
        # Empty candidate at obs 1
        HMMObservationStep(1, 2.0, 150.0, 200.0, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32), np.empty(0), np.empty(0)),
        # Connected again at obs 2, followed by 120s gap before obs 3
        HMMObservationStep(2, 3.0, 150.0, 0.0, np.array([1], dtype=np.int64), np.array([0.5], dtype=np.float32), np.array([0.9]), np.log([0.9])),
        HMMObservationStep(3, 123.0, 50.0, 40.0, np.array([3], dtype=np.int64), np.array([0.5], dtype=np.float32), np.array([0.9]), np.log([0.9])),
    ]
    res = decode_viterbi_sequence(steps, road_net, graph, trans_cfg)
    assert res.decoded_obs_arcs[1] == -1
    assert res.n_empty_splits == 1
    assert res.n_gap_splits == 1
    assert res.n_segments == 3
    assert res.decoded_obs_arcs[0] == 0
    assert res.decoded_obs_arcs[2] == 1
    assert res.decoded_obs_arcs[3] == 3


def test_route_reconstruction_bridges_multi_arc_without_inventing_disconnected():
    road_net, graph = _make_synthetic_network()
    # Decoded obs 0 -> arc 0, obs 1 -> arc 2 (bridge should insert arc 1!), obs 2 -> disconnected arc 3
    decoded = np.array([0, 2, 3], dtype=np.int64)
    seg_ids = np.array([0, 0, 0], dtype=np.int32)
    rec = reconstruct_route_from_decoded(decoded, seg_ids, road_net, graph, connector_budget_m=1000.0)

    assert rec.collapsed_decoded_arcs == [0, 2, 3]
    assert rec.reconstructed_route_arcs == [0, 1, 2, 3]
    assert rec.is_connector_mask == [False, True, False, False]
    assert rec.n_inserted_connectors == 1
    assert rec.n_unconnected_gaps == 1


def test_transition_connectivity_and_disconnected_handling():
    road_net, graph = _make_synthetic_network()
    trans_cfg = TransitionConfig(beta_m=20.0, disconnected_penalty=15.0)
    lp, conn = compute_transition_log_probs(
        np.array([0], dtype=np.int64),
        np.array([0.5], dtype=np.float32),
        np.array([1, 3], dtype=np.int64),
        np.array([0.5, 0.5], dtype=np.float32),
        100.0,
        road_net,
        graph,
        trans_cfg,
    )
    assert conn[0, 0] and not conn[0, 1]
    assert np.isfinite(lp).all()
    assert lp[0, 0] > lp[0, 1]


def test_decoder_never_accepts_route_and_is_reproducible():
    for fn in [build_hmm_observation_steps, decode_viterbi_sequence, reconstruct_route_from_decoded]:
        sig = inspect.signature(fn)
        assert "route" not in sig.parameters
        assert "ref_route" not in sig.parameters

    # Reproducibility check on synthetic sequence
    road_net, graph = _make_synthetic_network()
    trans_cfg = TransitionConfig(gap_threshold_s=30.0, beta_m=20.0)
    steps = [
        HMMObservationStep(0, 1.0, 50.0, 0.0, np.array([0, 3], dtype=np.int64), np.array([0.5, 0.5], dtype=np.float32), np.array([0.85, 0.15]), np.log([0.85, 0.15])),
        HMMObservationStep(1, 2.0, 150.0, 0.0, np.array([1, 4], dtype=np.int64), np.array([0.5, 0.5], dtype=np.float32), np.array([0.40, 0.60]), np.log([0.40, 0.60])),
    ]
    r1 = decode_viterbi_sequence(steps, road_net, graph, trans_cfg)
    r2 = decode_viterbi_sequence(steps, road_net, graph, trans_cfg)
    assert np.array_equal(r1.decoded_obs_arcs, r2.decoded_obs_arcs)


def test_phase3_and_phase4_outputs_exist():
    for rel_path in [
        "artifacts/hist_gradient_boosting_tuned.joblib",
        "configs/hmm_config.json",
        "results/bayesian_optimization_summary.json",
        "results/bayesian_optimization_trials.csv",
        "results/hmm_validation_metrics.json",
        "results/hmm_student_test_metrics.json",
        "results/classifier_vs_hmm_comparison.csv",
        "visualizations/bayesian_optimization_history.png",
        "visualizations/phase4_classifier_vs_hmm.png",
        "visualizations/phase4_folium_student_test_00000076.html",
    ]:
        assert (PROJECT_ROOT / rel_path).is_file()


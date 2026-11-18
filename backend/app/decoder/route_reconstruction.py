"""Route reconstruction and sequence-similarity evaluation against .route."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.app.data.preprocessing import RoadNetworkMetric
from backend.app.decoder.transition_model import DirectedRoadGraph


@dataclass(frozen=True)
class ReconstructedRoute:
    """Auditable route reconstruction separating decoded states from graph connectors."""

    decoded_obs_arcs: np.ndarray          # length T (-1 for unmatched obs)
    collapsed_decoded_arcs: list[int]     # consecutive duplicates collapsed
    reconstructed_route_arcs: list[int]   # with valid graph connectors inserted
    is_connector_mask: list[bool]         # True for inserted road connectors
    n_inserted_connectors: int
    n_unconnected_gaps: int
    route_continuity_pct: float


def reconstruct_route_from_decoded(
    decoded_obs_arcs: np.ndarray,
    segment_ids: np.ndarray,
    road_net: RoadNetworkMetric,
    graph: DirectedRoadGraph,
    connector_budget_m: float = 2500.0,
) -> ReconstructedRoute:
    """Collapse consecutive duplicate arcs and bridge multi-arc transitions via valid road paths.

    Never invents synthetic connectors; retains is_connector_mask for auditability.
    """
    collapsed: list[int] = []
    collapsed_segs: list[int] = []

    for arc_id, seg_id in zip(decoded_obs_arcs, segment_ids):
        a = int(arc_id)
        s = int(seg_id)
        if a < 0:
            continue
        if not collapsed or collapsed[-1] != a or collapsed_segs[-1] != s:
            collapsed.append(a)
            collapsed_segs.append(s)

    if not collapsed:
        return ReconstructedRoute(
            decoded_obs_arcs=decoded_obs_arcs,
            collapsed_decoded_arcs=[],
            reconstructed_route_arcs=[],
            is_connector_mask=[],
            n_inserted_connectors=0,
            n_unconnected_gaps=0,
            route_continuity_pct=0.0,
        )

    reconstructed: list[int] = [collapsed[0]]
    is_connector: list[bool] = [False]
    n_inserted = 0
    n_unconnected = 0

    for idx in range(1, len(collapsed)):
        prev_arc = collapsed[idx - 1]
        curr_arc = collapsed[idx]
        same_seg = collapsed_segs[idx] == collapsed_segs[idx - 1]

        v_prev = int(road_net.v[prev_arc])
        u_curr = int(road_net.u[curr_arc])

        if v_prev == u_curr:
            reconstructed.append(curr_arc)
            is_connector.append(False)
        elif same_seg:
            # Try bridging multi-arc gap using ONLY valid road graph paths
            bridge = graph.shortest_path_arcs(v_prev, u_curr, budget_m=connector_budget_m)
            if bridge is not None:
                for b_arc in bridge:
                    reconstructed.append(b_arc)
                    is_connector.append(True)
                    n_inserted += 1
            else:
                n_unconnected += 1
            reconstructed.append(curr_arc)
            is_connector.append(False)
        else:
            # Split across a long temporal gap or missing-candidate section
            n_unconnected += 1
            reconstructed.append(curr_arc)
            is_connector.append(False)

    # Compute directed topological continuity along reconstructed route
    n_trans = max(0, len(reconstructed) - 1)
    if n_trans > 0:
        conn_count = sum(
            1
            for k in range(n_trans)
            if int(road_net.v[reconstructed[k]]) == int(road_net.u[reconstructed[k + 1]])
        )
        cont_pct = float(conn_count / n_trans * 100.0)
    else:
        cont_pct = 100.0

    return ReconstructedRoute(
        decoded_obs_arcs=decoded_obs_arcs,
        collapsed_decoded_arcs=collapsed,
        reconstructed_route_arcs=reconstructed,
        is_connector_mask=is_connector,
        n_inserted_connectors=n_inserted,
        n_unconnected_gaps=n_unconnected,
        route_continuity_pct=round(cont_pct, 2),
    )


def _lcs_length(seq_a: list[int], seq_b: list[int]) -> int:
    """Compute Longest Common Subsequence length in O(min(N,M)) space."""
    if not seq_a or not seq_b:
        return 0
    if len(seq_a) < len(seq_b):
        seq_a, seq_b = seq_b, seq_a
    prev = [0] * (len(seq_b) + 1)
    for a in seq_a:
        curr = [0] * (len(seq_b) + 1)
        for j, b in enumerate(seq_b, start=1):
            if a == b:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = prev[j] if prev[j] >= curr[j - 1] else curr[j - 1]
        prev = curr
    return prev[-1]


def evaluate_route_against_reference(
    pred_route_arcs: list[int],
    ref_route_arcs: np.ndarray,
    road_net: RoadNetworkMetric,
) -> dict[str, float]:
    """Compare reconstructed arc sequence against reference .route without 1-to-1 row assumption."""
    ref_list = [int(x) for x in ref_route_arcs]
    pred_set = set(pred_route_arcs)
    ref_set = set(ref_list)

    inter = pred_set & ref_set
    union = pred_set | ref_set

    arc_prec = len(inter) / max(1, len(pred_set)) * 100.0
    arc_rec = len(inter) / max(1, len(ref_set)) * 100.0
    arc_f1 = (2 * arc_prec * arc_rec / (arc_prec + arc_rec)) if (arc_prec + arc_rec) > 0 else 0.0
    arc_jaccard = len(inter) / max(1, len(union)) * 100.0

    # Length-weighted precision/recall/F1 (meters along unique arcs)
    pred_len_m = sum(float(road_net.arc_lengths_m[a]) for a in pred_set)
    ref_len_m = sum(float(road_net.arc_lengths_m[a]) for a in ref_set)
    inter_len_m = sum(float(road_net.arc_lengths_m[a]) for a in inter)
    len_prec = inter_len_m / max(1e-6, pred_len_m) * 100.0
    len_rec = inter_len_m / max(1e-6, ref_len_m) * 100.0
    len_f1 = (2 * len_prec * len_rec / (len_prec + len_rec)) if (len_prec + len_rec) > 0 else 0.0

    # Order-preserving LCS similarity
    lcs_len = _lcs_length(pred_route_arcs, ref_list)
    seq_lcs_sim = (2.0 * lcs_len / max(1, len(pred_route_arcs) + len(ref_list))) * 100.0

    return {
        "pred_route_len": len(pred_route_arcs),
        "ref_route_len": len(ref_list),
        "arc_set_precision_pct": round(arc_prec, 2),
        "arc_set_recall_pct": round(arc_rec, 2),
        "arc_set_f1_pct": round(arc_f1, 2),
        "arc_jaccard_pct": round(arc_jaccard, 2),
        "length_weighted_f1_pct": round(len_f1, 2),
        "sequence_lcs_similarity_pct": round(seq_lcs_sim, 2),
    }

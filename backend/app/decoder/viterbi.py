"""Numerically stable log-space Viterbi decoder with temporal gap segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.app.data.preprocessing import RoadNetworkMetric
from backend.app.decoder.hmm import HMMObservationStep
from backend.app.decoder.transition_model import (
    DirectedRoadGraph,
    TransitionConfig,
    compute_transition_log_probs,
)


@dataclass(frozen=True)
class ViterbiDecodeResult:
    """Decoded observation-level road arc sequence and segment diagnostics."""

    decoded_obs_arcs: np.ndarray        # shape (T,), int64 (-1 if 0 candidates)
    segment_ids: np.ndarray             # shape (T,), int32 (-1 if 0 candidates)
    n_segments: int
    n_gap_splits: int
    n_empty_splits: int
    n_disconnected_viterbi_steps: int
    total_log_score: float


def _decode_contiguous_segment(
    seg_steps: list[HMMObservationStep],
    road_net: RoadNetworkMetric,
    graph: DirectedRoadGraph,
    trans_cfg: TransitionConfig,
) -> tuple[np.ndarray, int, float]:
    """Run log-space Viterbi over a contiguous non-empty candidate segment."""
    seg_len = len(seg_steps)
    if seg_len == 0:
        return np.empty(0, dtype=np.int64), 0, 0.0

    if seg_len == 1:
        best_idx = int(np.argmax(seg_steps[0].emit_log_probs))
        return (
            np.array([int(seg_steps[0].cand_arcs[best_idx])], dtype=np.int64),
            0,
            float(seg_steps[0].emit_log_probs[best_idx]),
        )

    v_scores: list[np.ndarray] = [seg_steps[0].emit_log_probs.copy()]
    backpointers: list[np.ndarray] = []
    conn_matrices: list[np.ndarray] = []

    for t in range(1, seg_len):
        prev_step = seg_steps[t - 1]
        curr_step = seg_steps[t]
        d_gps = float(np.hypot(curr_step.px - prev_step.px, curr_step.py - prev_step.py))

        trans_lp, is_conn = compute_transition_log_probs(
            prev_step.cand_arcs,
            prev_step.proj_t,
            curr_step.cand_arcs,
            curr_step.proj_t,
            d_gps,
            road_net,
            graph,
            trans_cfg,
        )

        # scores[i, j] = v_prev[i] + trans_lp[i, j]
        scores = v_scores[-1][:, None] + trans_lp
        best_prev = np.argmax(scores, axis=0).astype(np.int32)
        best_prev_scores = scores[best_prev, np.arange(len(curr_step.cand_arcs))]
        v_curr = best_prev_scores + curr_step.emit_log_probs

        v_scores.append(v_curr)
        backpointers.append(best_prev)
        conn_matrices.append(is_conn)

    # Backtrack most likely state path
    state_idx = np.empty(seg_len, dtype=np.int32)
    state_idx[-1] = int(np.argmax(v_scores[-1]))
    seg_score = float(v_scores[-1][state_idx[-1]])

    for t in range(seg_len - 1, 0, -1):
        state_idx[t - 1] = int(backpointers[t - 1][state_idx[t]])

    decoded_arcs = np.array(
        [int(seg_steps[t].cand_arcs[state_idx[t]]) for t in range(seg_len)],
        dtype=np.int64,
    )
    n_disc = sum(
        0 if conn_matrices[t - 1][state_idx[t - 1], state_idx[t]] else 1
        for t in range(1, seg_len)
    )
    return decoded_arcs, n_disc, seg_score


def decode_viterbi_sequence(
    steps: list[HMMObservationStep],
    road_net: RoadNetworkMetric,
    graph: DirectedRoadGraph,
    trans_cfg: TransitionConfig,
) -> ViterbiDecodeResult:
    """Decode full trip sequence, splitting independently across long GPS gaps and empty candidates."""
    n_obs = len(steps)
    decoded = np.full(n_obs, -1, dtype=np.int64)
    segment_ids = np.full(n_obs, -1, dtype=np.int32)

    if n_obs == 0:
        return ViterbiDecodeResult(decoded, segment_ids, 0, 0, 0, 0, 0.0)

    # Partition observations into contiguous valid-candidate segments without gaps > gap_threshold_s
    segments: list[list[HMMObservationStep]] = []
    current_seg: list[HMMObservationStep] = []
    n_gap_splits = 0
    n_empty_splits = 0

    for t, step in enumerate(steps):
        if len(step.cand_arcs) == 0:
            if current_seg:
                segments.append(current_seg)
                current_seg = []
                n_empty_splits += 1
            continue

        if current_seg:
            dt = step.timestamp_s - current_seg[-1].timestamp_s
            if dt > trans_cfg.gap_threshold_s:
                segments.append(current_seg)
                current_seg = [step]
                n_gap_splits += 1
                continue

        current_seg.append(step)

    if current_seg:
        segments.append(current_seg)

    total_score = 0.0
    total_disc = 0
    for seg_id, seg in enumerate(segments):
        seg_arcs, seg_disc, seg_score = _decode_contiguous_segment(
            seg, road_net, graph, trans_cfg
        )
        total_disc += seg_disc
        total_score += seg_score
        for local_i, step in enumerate(seg):
            decoded[step.obs_id] = seg_arcs[local_i]
            segment_ids[step.obs_id] = seg_id

    return ViterbiDecodeResult(
        decoded_obs_arcs=decoded,
        segment_ids=segment_ids,
        n_segments=len(segments),
        n_gap_splits=n_gap_splits,
        n_empty_splits=n_empty_splits,
        n_disconnected_viterbi_steps=total_disc,
        total_log_score=total_score,
    )

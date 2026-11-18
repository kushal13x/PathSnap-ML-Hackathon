"""Phase 4 Visualization Script: Classifier-Only vs HMM/Viterbi plots & Folium map."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.data.loaders import load_reference_route, load_trip_inputs
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
from backend.app.evaluation.metrics import evaluate_trip_ranking
from backend.app.features.feature_engineering import FEATURE_COLUMNS
from backend.app.models.model_comparison import load_model_artifact


def plot_classifier_vs_hmm(comp_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=150)

    splits = ["train_val", "student_test"]
    clf_sub = comp_df[comp_df["method"] == "classifier_only"].set_index("split").loc[splits]
    hmm_sub = comp_df[comp_df["method"] == "hmm_viterbi"].set_index("split").loc[splits]

    x = np.arange(len(splits))
    w = 0.34

    # (a) Top-1 & Step Continuity
    ax = axes[0]
    ax.bar(x - w / 2, clf_sub["top1_acc_valid_pct_micro"], w, label="Classifier-Only Top-1", color="#9ecae1")
    ax.bar(x + w / 2, hmm_sub["top1_acc_valid_pct_micro"], w, label="HMM + Viterbi Top-1", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(["train_val (16)", "student_test (10)"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(75, 100)
    ax.set_title("(a) Observation-Level Top-1 Accuracy")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # (b) Route Continuity & Invalid Transitions
    ax = axes[1]
    ax.bar(x - w / 2, clf_sub["mean_reconstructed_route_continuity_pct"], w, label="Classifier-Only Route Cont.", color="#fdae6b")
    ax.bar(x + w / 2, hmm_sub["mean_reconstructed_route_continuity_pct"], w, label="HMM Reconstructed Cont.", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(["train_val (16)", "student_test (10)"])
    ax.set_ylabel("Route Continuity (%)")
    ax.set_ylim(50, 102)
    ax.set_title("(b) Reconstructed Route Continuity")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # (c) Length-Weighted Route F1 & LCS Similarity
    ax = axes[2]
    ax.bar(x - w / 2, clf_sub["mean_length_weighted_f1_pct"], w, label="Classifier-Only Len-F1", color="#c5b0d5")
    ax.bar(x + w / 2, hmm_sub["mean_length_weighted_f1_pct"], w, label="HMM + Viterbi Len-F1", color="#9467bd")
    ax.set_xticks(x)
    ax.set_xticklabels(["train_val (16)", "student_test (10)"])
    ax.set_ylabel("Length-Weighted Route F1 (%)")
    ax.set_ylim(70, 102)
    ax.set_title("(c) Reference Route Length-Weighted F1")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle("PathSnap Phase 4 — Tuned Classifier vs. HMM/Viterbi Route Reconstruction", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path)
    plt.close(fig)


def generate_reconstructed_route_folium(
    raw_split: str, trip_id: str, out_path: Path
) -> None:
    """Create interactive Folium map comparing GPS, Reference Route, Classifier-Only & HMM Route."""
    raw_root = PROJECT_ROOT / "data" / "raw"
    cand_root = PROJECT_ROOT / "data" / "processed" / "candidates"
    with open(PROJECT_ROOT / "configs" / "hmm_config.json", "r", encoding="utf-8") as f:
        hcfg = json.load(f)
    summary_df = pd.read_csv(PROJECT_ROOT / "results" / "candidate_dataset_summary.csv", dtype={"trip_id": str})
    obs_sum = summary_df.set_index(["split", "trip_id"], drop=False).loc[(raw_split, trip_id)]

    classifier, _ = load_model_artifact(PROJECT_ROOT / "artifacts" / "hist_gradient_boosting_tuned.joblib")
    ti = load_trip_inputs(raw_root, raw_split, trip_id)
    ref_route = load_reference_route(raw_root, raw_split, trip_id)

    proj = LocalTangentProjection(float(ti.lon.mean()), float(ti.lat.mean()))
    px, py = proj.to_xy(ti.lon, ti.lat)
    kin = compute_gps_kinematics(px, py, ti.timestamps)
    road_net = preprocess_road_network(ti.node_lon, ti.node_lat, ti.arcs, proj)
    graph = DirectedRoadGraph(road_net)

    df_cands = pd.read_parquet(cand_root / raw_split / f"{trip_id}.parquet")
    cand_probs = compute_classifier_probabilities(classifier, df_cands, FEATURE_COLUMNS)
    _, clf_seq = evaluate_trip_ranking(df_cands, cand_probs, obs_sum, arcs=ti.arcs)

    emit_cfg = EmissionConfig(**hcfg["emission"])
    trans_cfg = TransitionConfig(**{k: v for k, v in hcfg["transition"].items()})
    steps = build_hmm_observation_steps(len(ti.lon), ti.timestamps, kin, df_cands, cand_probs, emit_cfg)
    viterbi_res = decode_viterbi_sequence(steps, road_net, graph, trans_cfg)
    reconstructed = reconstruct_route_from_decoded(
        viterbi_res.decoded_obs_arcs, viterbi_res.segment_ids, road_net, graph
    )

    m = folium.Map(location=[float(ti.lat.mean()), float(ti.lon.mean())], zoom_start=14, tiles="OpenStreetMap")

    def arcs_to_segments(arc_ids):
        segs = []
        for aid in arc_ids:
            if aid < 0:
                continue
            u, v = int(ti.arcs[aid, 0]), int(ti.arcs[aid, 1])
            segs.append([[float(ti.node_lat[u]), float(ti.node_lon[u])], [float(ti.node_lat[v]), float(ti.node_lon[v])]])
        return segs

    # Layer 1: Reference route
    ref_fg = folium.FeatureGroup(name="Reference Route (.route)", show=True)
    folium.PolyLine(arcs_to_segments(ref_route), color="#2ca02c", weight=7, opacity=0.55, tooltip="Reference Route").add_to(ref_fg)
    ref_fg.add_to(m)

    # Layer 2: Classifier-only unique predicted arcs
    clf_unique = list(dict.fromkeys([int(a) for a in clf_seq if a >= 0]))
    clf_fg = folium.FeatureGroup(name="Classifier-Only Predicted Arcs", show=False)
    folium.PolyLine(arcs_to_segments(clf_unique), color="#ff7f0e", weight=4, opacity=0.85, tooltip="Classifier-Only").add_to(clf_fg)
    clf_fg.add_to(m)

    # Layer 3: HMM Reconstructed Route
    hmm_fg = folium.FeatureGroup(name="HMM/Viterbi Reconstructed Route", show=True)
    direct_arcs = [a for a, is_c in zip(reconstructed.reconstructed_route_arcs, reconstructed.is_connector_mask) if not is_c]
    conn_arcs = [a for a, is_c in zip(reconstructed.reconstructed_route_arcs, reconstructed.is_connector_mask) if is_c]
    folium.PolyLine(arcs_to_segments(direct_arcs), color="#1f77b4", weight=4, opacity=0.9, tooltip="HMM Decoded Arcs").add_to(hmm_fg)
    if conn_arcs:
        folium.PolyLine(arcs_to_segments(conn_arcs), color="#9467bd", weight=4, opacity=0.9, dash_array="5,5", tooltip="HMM Graph Connectors").add_to(hmm_fg)
    hmm_fg.add_to(m)

    # Layer 4: Raw GPS
    gps_fg = folium.FeatureGroup(name="Raw GPS Points (.track)", show=True)
    step = max(1, len(ti.lat) // 1000)
    for i in range(0, len(ti.lat), step):
        folium.CircleMarker([float(ti.lat[i]), float(ti.lon[i])], radius=2.2, color="#d62728", fill=True, fill_opacity=0.8).add_to(gps_fg)
    gps_fg.add_to(m)

    legend_html = f"""
    <div style="position: fixed; bottom: 25px; left: 25px; z-index: 9999;
                background-color: white; border: 2px solid #555; border-radius: 6px;
                padding: 10px 14px; font-size: 12px; box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
      <b>PathSnap Phase 4: {raw_split}/{trip_id}</b><br>
      <span style="color:#2ca02c;">&#9473;</span> Reference Route ({len(ref_route)} arcs)<br>
      <span style="color:#1f77b4;">&#9473;</span> HMM Decoded Route ({len(reconstructed.reconstructed_route_arcs)} arcs)<br>
      <span style="color:#9467bd;">- -</span> Graph Connectors ({reconstructed.n_inserted_connectors})<br>
      <span style="color:#ff7f0e;">&#9473;</span> Classifier-Only (toggleable)<br>
      <span style="color:#d62728;">&#9679;</span> Raw GPS Track
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[float(ti.lat.min()), float(ti.lon.min())], [float(ti.lat.max()), float(ti.lon.max())]])
    m.save(str(out_path))


def main() -> int:
    results_dir = PROJECT_ROOT / "results"
    viz_dir = PROJECT_ROOT / "visualizations"
    comp_df = pd.read_csv(results_dir / "classifier_vs_hmm_comparison.csv")
    plot_classifier_vs_hmm(comp_df, viz_dir / "phase4_classifier_vs_hmm.png")
    generate_reconstructed_route_folium(
        "student_test", "00000076", viz_dir / "phase4_folium_student_test_00000076.html"
    )
    print("Generated Phase 4 comparison plot and interactive Folium route map.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

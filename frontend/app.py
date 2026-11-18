"""PathSnap Evaluator Dashboard (Streamlit + Folium).

Communicates strictly with the FastAPI backend over HTTP using PATHSNAP_API_URL.
Never imports internal backend ML modules or reads ground-truth .route files.
"""

from __future__ import annotations

import os
from typing import Any

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

DEFAULT_API_URL = os.environ.get("PATHSNAP_API_URL", "").strip().rstrip("/")


def fetch_health(api_url: str) -> tuple[bool, dict[str, Any] | str]:
    try:
        resp = requests.get(f"{api_url}/health", timeout=5)
        resp.raise_for_status()
        return True, resp.json()
    except Exception as exc:
        return False, str(exc)


def fetch_trips(api_url: str) -> tuple[bool, dict[str, Any] | str]:
    try:
        resp = requests.get(f"{api_url}/trips", timeout=10)
        resp.raise_for_status()
        return True, resp.json()
    except Exception as exc:
        return False, str(exc)


def fetch_map_data(api_url: str, trip_id: str) -> tuple[bool, dict[str, Any] | str]:
    try:
        resp = requests.get(f"{api_url}/trips/{trip_id}/map-data", timeout=20)
        resp.raise_for_status()
        return True, resp.json()
    except Exception as exc:
        return False, str(exc)


def request_prediction(api_url: str, trip_id: str) -> tuple[bool, dict[str, Any] | str]:
    try:
        resp = requests.post(f"{api_url}/predict/{trip_id}", timeout=120)
        if resp.status_code != 200:
            try:
                err = resp.json()
                return False, f"HTTP {resp.status_code}: {err.get('detail', resp.text)}"
            except Exception:
                return False, f"HTTP {resp.status_code}: {resp.text}"
        return True, resp.json()
    except Exception as exc:
        return False, str(exc)


def build_prediction_map(
    map_data: dict[str, Any] | None,
    prediction: dict[str, Any],
) -> folium.Map:
    """Build interactive Folium map with GPS track, road network, and predicted route."""
    gps_coords = prediction["gps_coordinates"]
    center_lat = (prediction["bounds"][0][0] + prediction["bounds"][1][0]) / 2.0
    center_lon = (prediction["bounds"][0][1] + prediction["bounds"][1][1]) / 2.0

    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="OpenStreetMap")

    # 1. Local road-network context
    if map_data and map_data.get("road_network_segments"):
        road_fg = folium.FeatureGroup(name="Local Road Network", show=True)
        folium.PolyLine(
            locations=map_data["road_network_segments"],
            color="#7f7f7f",
            weight=1.5,
            opacity=0.55,
            tooltip=f"Road network ({map_data['displayed_network_arcs']} arcs shown)",
        ).add_to(road_fg)
        road_fg.add_to(m)

    # 2. Reconstructed predicted route (separating direct HMM arcs vs road connectors)
    route_fg = folium.FeatureGroup(name="Predicted Route (HMM + Viterbi)", show=True)
    route_segs = prediction["reconstructed_route_segments"]
    is_conn = prediction["is_connector_arc"]

    direct_segs = [seg for seg, c in zip(route_segs, is_conn) if not c]
    conn_segs = [seg for seg, c in zip(route_segs, is_conn) if c]

    if direct_segs:
        folium.PolyLine(
            locations=direct_segs,
            color="#1f77b4",
            weight=5,
            opacity=0.9,
            tooltip=f"Decoded Road Arcs ({len(direct_segs)})",
        ).add_to(route_fg)
    if conn_segs:
        folium.PolyLine(
            locations=conn_segs,
            color="#9467bd",
            weight=4.5,
            opacity=0.9,
            dash_array="6, 6",
            tooltip=f"Graph Connector Arcs ({len(conn_segs)})",
        ).add_to(route_fg)
    route_fg.add_to(m)

    # 3. Raw GPS trajectory & start/end markers
    gps_fg = folium.FeatureGroup(name="Raw GPS Observations", show=True)
    folium.PolyLine(
        locations=gps_coords,
        color="#d62728",
        weight=2,
        opacity=0.65,
        tooltip="Raw GPS sequence",
    ).add_to(gps_fg)

    marker_step = max(1, len(gps_coords) // 500)
    for idx in range(0, len(gps_coords), marker_step):
        folium.CircleMarker(
            location=gps_coords[idx],
            radius=2.5,
            color="#d62728",
            fill=True,
            fill_opacity=0.8,
            tooltip=f"GPS obs #{idx}",
        ).add_to(gps_fg)

    folium.Marker(
        location=prediction["start_coordinate"],
        icon=folium.Icon(color="green", icon="play"),
        tooltip="Trip Start",
    ).add_to(gps_fg)
    folium.Marker(
        location=prediction["end_coordinate"],
        icon=folium.Icon(color="red", icon="stop"),
        tooltip="Trip End",
    ).add_to(gps_fg)
    gps_fg.add_to(m)

    legend_html = f"""
    <div style="position: fixed; bottom: 20px; left: 20px; z-index: 9999;
                background-color: white; border: 2px solid #555; border-radius: 6px;
                padding: 10px 14px; font-size: 12px; box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
      <b>PathSnap Prediction: Trip {prediction['trip_id']}</b><br>
      <span style="color:#1f77b4;">&#9473;</span> HMM Decoded Arcs ({len(direct_segs)})<br>
      <span style="color:#9467bd;">- -</span> Graph Connectors ({len(conn_segs)})<br>
      <span style="color:#d62728;">&#9679;</span> Raw GPS Track ({len(gps_coords)} pts)<br>
      <span style="color:#7f7f7f;">&#9473;</span> Local Road Network
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds(prediction["bounds"])
    return m


def main() -> None:
    st.set_page_config(
        page_title="PathSnap — Distributed Map-Matching",
        page_icon="🗺️",
        layout="wide",
    )

    # A. Project Overview & Backend Connection Status
    st.title("PathSnap — Distributed Map-Matching & Route Reconstruction")
    st.markdown(
        "Reconstructs topologically coherent vehicle routes from noisy GPS traces using "
        "spatial candidate retrieval, tuned `HistGradientBoosting` candidate scoring, and "
        "HMM/Viterbi sequence decoding over the directed road network."
    )

    with st.sidebar:
        st.header("Backend Connection")
        api_url = st.text_input(
            "FastAPI Endpoint URL (`PATHSNAP_API_URL`)",
            value=DEFAULT_API_URL,
            placeholder="http://<LAPTOP_1_LAN_IP>:8000",
            help="Read from PATHSNAP_API_URL environment variable for two-laptop deployment.",
        ).strip().rstrip("/")

        if not api_url:
            st.warning(
                "`PATHSNAP_API_URL` is not set. Enter Laptop 1's FastAPI URL above "
                "(e.g. `http://<LAPTOP_1_IP>:8000`) or launch via `scripts/start_frontend.ps1`."
            )
            st.stop()

        healthy, health_info = fetch_health(api_url)
        if healthy and isinstance(health_info, dict):
            st.success(f"Connected ({health_info['status'].upper()})")
            st.caption(f"Service: {health_info['service']}")
            st.caption(f"Model: {health_info['model_name']}")
            st.caption(f"Model Loaded: {health_info['model_loaded']} | HMM Config: {health_info['hmm_config_loaded']}")
            st.caption(f"Dataset Partition: `{health_info['inference_dataset_name']}` ({health_info['available_trip_count']} trips)")
        else:
            st.error("Backend Offline / Unreachable")
            st.caption(str(health_info))
            st.stop()

    # B. Trip Selection
    trips_ok, trips_payload = fetch_trips(api_url)
    if not trips_ok or not isinstance(trips_payload, dict):
        st.error(f"Failed to fetch trip list from backend: {trips_payload}")
        st.stop()

    trip_items = trips_payload.get("trips", [])
    if not trip_items:
        st.warning("No inference trips found in configured backend directory.")
        st.stop()

    trip_labels = {
        t["trip_id"]: f"{t['trip_id']}  ({t['n_gps_points']} GPS pts, {t['duration_s']:.0f}s)"
        for t in trip_items
    }
    trip_ids = [t["trip_id"] for t in trip_items]

    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        selected_trip = st.selectbox(
            f"Select Evaluation Trip (`{trips_payload['inference_dataset_name']}`)",
            options=trip_ids,
            format_func=lambda tid: trip_labels[tid],
        )
    with col_btn:
        st.write("")
        st.write("")
        predict_clicked = st.button("Predict Route", type="primary", use_container_width=True)

    if predict_clicked:
        with st.spinner(f"Running HMM + Viterbi inference for trip `{selected_trip}`..."):
            pred_ok, pred_res = request_prediction(api_url, selected_trip)
            map_ok, map_res = fetch_map_data(api_url, selected_trip)

        if not pred_ok or not isinstance(pred_res, dict):
            st.error(f"Prediction failed for trip `{selected_trip}`: {pred_res}")
            st.stop()

        st.session_state["last_prediction"] = pred_res
        st.session_state["last_map_data"] = map_res if map_ok and isinstance(map_res, dict) else None

    # C, D, E: Display prediction results if available for selected trip
    prediction = st.session_state.get("last_prediction")
    map_data = st.session_state.get("last_map_data")

    if prediction and prediction.get("trip_id") == selected_trip:
        st.success(
            f"Prediction succeeded for trip `{prediction['trip_id']}` using `{prediction['model_used']}` "
            f"in {prediction['processing_time_s']:.3f}s."
        )

        # C. Key Prediction Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("GPS Observations", f"{prediction['observation_count']:,}")
        c2.metric("Candidates Evaluated", f"{prediction['candidate_count']:,}")
        c3.metric("Reconstructed Route Arcs", f"{len(prediction['reconstructed_route_arc_ids']):,}")
        c4.metric("Processing Time", f"{prediction['processing_time_s']:.3f} s")

        # E. Prediction Diagnostics
        diag = prediction["diagnostics"]
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Route Continuity", f"{diag['route_continuity_pct']:.2f}%")
        d2.metric("Inserted Graph Connectors", f"{diag['connector_count']}")
        d3.metric("Temporal Gap Splits (>30s)", f"{diag['temporal_gap_splits']}")
        d4.metric("Unmatched Obs (0 cands)", f"{diag['empty_candidate_obs_count']}")

        if diag["warnings"]:
            for w in diag["warnings"]:
                st.warning(w)

        # D. Interactive Folium Map
        st.subheader("Interactive Geographical Route Visualisation")
        folium_map = build_prediction_map(map_data, prediction)
        st_folium(folium_map, width=None, height=560, returned_objects=[])

        # Expandable Technical Details
        with st.expander("Predicted Road-Arc Sequence & Technical Diagnostics"):
            st.write(f"**Coordinate Order:** `{prediction['coordinate_order']}`")
            st.write(
                f"**Mean Candidates / Obs:** `{prediction['mean_candidates_per_obs']}` | "
                f"**Viterbi Log Score:** `{diag['viterbi_total_log_score']}` | "
                f"**Unconnected Route Gaps:** `{diag['unconnected_gaps']}` | "
                f"**Viterbi Disconnected Steps:** `{diag['viterbi_disconnected_steps']}`"
            )
            st.text_area(
                "Reconstructed Road Arc IDs (chronological)",
                value=", ".join(str(a) for a in prediction["reconstructed_route_arc_ids"]),
                height=120,
            )
            st.text_area(
                "Per-Observation Decoded Arc IDs (-1 = unmatched)",
                value=", ".join(str(a) for a in prediction["predicted_obs_arc_ids"]),
                height=120,
            )
    else:
        st.info("Select a trip and click **Predict Route** to run HMM/Viterbi map-matching.")


if __name__ == "__main__":
    main()

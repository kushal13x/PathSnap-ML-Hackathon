"""Phase 0 EDA Static Plots and Interactive Folium Map Generator for PathSnap.

Generates:
  Static PNGs in visualizations/:
    - eda_dataset_overview.png
    - eda_temporal_gaps_and_speeds.png
    - eda_network_and_route_quality.png
    - eda_sample_trajectories.png

  Interactive Folium HTML maps in visualizations/:
    - folium_global_trips_overview.html
    - folium_train_typical_00000012.html
    - folium_train_anomaly_00000054.html
    - folium_train_anomaly_00000033.html
    - folium_student_test_00000076.html
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from geo_utils import LocalTangentProjection, haversine_meters


def load_trip_raw(split: str, trip_id: str):
    """Load raw .track, .nodes, .arcs, .route for a single trip."""
    td = PROJECT_ROOT / "data" / "raw" / split / trip_id
    track = pd.read_csv(td / f"{trip_id}.track", sep=r"\s+", header=None).to_numpy()
    nodes = pd.read_csv(td / f"{trip_id}.nodes", sep=r"\s+", header=None).to_numpy()
    arcs = pd.read_csv(td / f"{trip_id}.arcs", sep=r"\s+", header=None).to_numpy(dtype=int)
    route = pd.read_csv(td / f"{trip_id}.route", sep=r"\s+", header=None).to_numpy(dtype=int).ravel()
    return track, nodes, arcs, route


def plot_dataset_overview(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "student_test"]

    # (a) GPS observation count distribution
    ax = axes[0, 0]
    bins = np.logspace(np.log10(200), np.log10(25000), 22)
    ax.hist(train_df["n_gps"], bins=bins, alpha=0.75, label="Train (n=80)", color="#1f77b4", edgecolor="white")
    ax.hist(test_df["n_gps"], bins=bins, alpha=0.85, label="Student Test (n=10)", color="#ff7f0e", edgecolor="black")
    ax.set_xscale("log")
    ax.set_xlabel("GPS Observations per Trip (log scale)")
    ax.set_ylabel("Trip Count")
    ax.set_title("(a) GPS Observation Count Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) Road Network Nodes vs Arcs
    ax = axes[0, 1]
    ax.scatter(train_df["n_nodes"], train_df["n_arcs"], alpha=0.75, s=45, label="Train", color="#1f77b4")
    ax.scatter(test_df["n_nodes"], test_df["n_arcs"], alpha=0.95, s=65, marker="^", edgecolor="black", label="Student Test", color="#ff7f0e")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Road Network Nodes (log scale)")
    ax.set_ylabel("Road Network Directed Arcs (log scale)")
    ax.set_title("(b) Road Network Scale per Trip (Ratio ~2.04 arcs/node)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (c) Track Length vs Reference Route Length
    ax = axes[1, 0]
    ax.scatter(train_df["track_length_km"], train_df["route_length_km"], alpha=0.75, s=45, label="Train", color="#1f77b4")
    ax.scatter(test_df["track_length_km"], test_df["route_length_km"], alpha=0.95, s=65, marker="^", edgecolor="black", label="Student Test", color="#ff7f0e")
    lim = max(df["track_length_km"].max(), df["route_length_km"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5, linewidth=1, label="1:1 Reference")
    offsets = {"00000054": (-55, 8), "00000002": (6, 6), "00000033": (-15, -14)}
    for tid, off in offsets.items():
        r = df[df["trip_id"] == int(tid)]
        if len(r) == 0:
            r = df[df["trip_id"].astype(str) == tid]
        if len(r) > 0:
            rr = r.iloc[0]
            ax.annotate(str(rr["trip_id"]).zfill(8), (rr["track_length_km"], rr["route_length_km"]), fontsize=8, xytext=off, textcoords="offset points", color="red")
    ax.set_xlabel("GPS Track Length (km)")
    ax.set_ylabel("Reference Route Length (km)")
    ax.set_title("(c) GPS Trajectory Length vs Reference Route Length")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (d) Global Geographical Coverage
    ax = axes[1, 1]
    ax.scatter(train_df["centroid_lon"], train_df["centroid_lat"], alpha=0.75, s=50, label="Train (80)", color="#1f77b4")
    ax.scatter(test_df["centroid_lon"], test_df["centroid_lat"], alpha=0.95, s=75, marker="^", edgecolor="black", label="Student Test (10)", color="#ff7f0e")
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitude (degrees)")
    ax.set_title("(d) Global Spatial Distribution of All 90 Trips")
    ax.set_xlim(-130, 160)
    ax.set_ylim(-50, 75)
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("PathSnap Phase 0 — Dataset Scale & Spatial Overview (90 Trips)", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_dir / "eda_dataset_overview.png")
    plt.close(fig)


def plot_temporal_and_speeds(df: pd.DataFrame, gaps_df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150)

    # (a) Irregular time gaps histogram
    ax = axes[0]
    pos_gaps = gaps_df[gaps_df["dt_s"] > 1.0]["dt_s"]
    bins = np.logspace(np.log10(1.5), np.log10(max(pos_gaps.max(), 100.0) * 1.1), 28)
    ax.hist(pos_gaps, bins=bins, color="#d62728", alpha=0.8, edgecolor="white")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Temporal Gap dt (seconds, log scale)")
    ax.set_ylabel("Count of Irregular Intervals (log)")
    ax.set_title(f"(a) Distribution of dt > 1s Gaps (n={len(pos_gaps)})")
    ax.axvline(5.0, color="black", linestyle="--", linewidth=1, label="5s threshold")
    ax.axvline(30.0, color="navy", linestyle=":", linewidth=1.2, label="30s threshold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Speed distribution across trips
    ax = axes[1]
    ax.boxplot(
        [df["speed_median_kmh"], df["speed_p95_kmh"], df["speed_max_kmh"]],
        tick_labels=["Median Speed", "95th Pct Speed", "Max Step Speed"],
        patch_artist=True,
        boxprops=dict(facecolor="#a6cee3"),
    )
    ax.axhline(150, color="red", linestyle="--", linewidth=1, label="150 km/h reference")
    ax.set_ylabel("Implied GPS Step Speed (km/h)")
    ax.set_title("(b) Implied Speed Summary Across 90 Trips")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) Sampling regularity vs max gap
    ax = axes[2]
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "student_test"]
    ax.scatter(train_df["pct_dt_1s"], np.maximum(train_df["dt_max"], 1.0), alpha=0.75, s=45, label="Train", color="#1f77b4")
    ax.scatter(test_df["pct_dt_1s"], np.maximum(test_df["dt_max"], 1.0), alpha=0.95, s=65, marker="^", edgecolor="black", label="Student Test", color="#ff7f0e")
    ax.set_yscale("log")
    ax.set_xlabel("% Intervals with dt == 1.0 s")
    ax.set_ylabel("Maximum Temporal Gap dt_max (s, log)")
    ax.set_title("(c) Sampling Regularity vs Max Dropout Gap")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle("PathSnap Phase 0 — Temporal Sampling & Kinematics Audit", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_dir / "eda_temporal_gaps_and_speeds.png")
    plt.close(fig)


def plot_network_and_route_quality(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)

    # (a) Weak vs Strong largest component %
    ax = axes[0, 0]
    ax.scatter(df["largest_weak_comp_pct"], df["largest_strong_comp_pct"], c=np.log10(df["n_nodes"]), cmap="viridis", s=50, alpha=0.85)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("log10(n_nodes)")
    ax.plot([60, 101], [60, 101], "k--", alpha=0.4)
    ax.set_xlabel("Largest Weakly Connected Component (% of nodes)")
    ax.set_ylabel("Largest Strongly Connected Component (% of nodes)")
    ax.set_title("(a) Road Network Connectivity Across 90 Subgraphs")
    ax.grid(True, alpha=0.3)

    # (b) Bidirectional arc % vs median arc length
    ax = axes[0, 1]
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "student_test"]
    ax.scatter(train_df["arc_len_median_m"], train_df["bidirectional_arc_pct"], alpha=0.75, s=45, label="Train", color="#1f77b4")
    ax.scatter(test_df["arc_len_median_m"], test_df["bidirectional_arc_pct"], alpha=0.95, s=65, marker="^", edgecolor="black", label="Student Test", color="#ff7f0e")
    ax.set_xlabel("Median Road Arc Length (m)")
    ax.set_ylabel("Bidirectional Arcs (%)")
    ax.set_title("(b) Road Segment Granularity vs Bidirectionality")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (c) GPS-to-Road vs GPS-to-Route P95 distances
    ax = axes[1, 0]
    ax.scatter(train_df["gps_to_road_p95_m"], train_df["gps_to_route_p95_m"], alpha=0.75, s=45, label="Train", color="#1f77b4")
    ax.scatter(test_df["gps_to_road_p95_m"], test_df["gps_to_route_p95_m"], alpha=0.95, s=65, marker="^", edgecolor="black", label="Student Test", color="#ff7f0e")
    ax.axvline(50.0, color="gray", linestyle="--", linewidth=1)
    ax.axhline(50.0, color="red", linestyle="--", linewidth=1, label="50m noise/anomaly line")
    for _, rr in df[df["gps_to_route_p95_m"] > 60.0].iterrows():
        ax.annotate(str(rr["trip_id"]).zfill(8), (rr["gps_to_road_p95_m"], rr["gps_to_route_p95_m"]), fontsize=8, xytext=(4, 4), textcoords="offset points", color="red")
    ax.set_xlabel("95th Pct GPS-to-Nearest-Road Distance (m)")
    ax.set_ylabel("95th Pct GPS-to-Reference-Route Distance (m)")
    ax.set_title("(c) GPS Noise vs Reference Route Alignment")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) Route Continuity & Disconnected Transitions
    ax = axes[1, 1]
    disc_df = df[df["n_disconnected"] > 0].sort_values("max_route_gap_m", ascending=True)
    y_labels = [f"{r['split'][:2]}:{str(r['trip_id']).zfill(8)} ({int(r['n_disconnected'])}x)" for _, r in disc_df.iterrows()]
    colors = ["#ff7f0e" if r["split"] == "student_test" else "#d62728" for _, r in disc_df.iterrows()]
    ax.barh(y_labels, disc_df["max_route_gap_m"], color=colors, alpha=0.85)
    ax.set_xlabel("Max Discontinuity Gap in .route (m)")
    ax.set_title(f"(d) Trips with Disconnected .route Transitions ({len(disc_df)}/90)")
    ax.grid(True, axis="x", alpha=0.3)

    plt.suptitle("PathSnap Phase 0 — Road Network Connectivity & Reference Route Validity", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_dir / "eda_network_and_route_quality.png")
    plt.close(fig)


def plot_sample_trajectories(out_dir: Path) -> None:
    cases = [
        ("student_test", "00000007", "Typical Urban/Suburban (student_test/00000007)"),
        ("train", "00000001", "High-Speed Highway (train/00000001)"),
        ("train", "00000054", "Severe Off-Route Anomaly (train/00000054)"),
        ("train", "00000033", "Partial Route / Detour Mismatch (train/00000033)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=150)
    for ax, (split, tid, title) in zip(axes.ravel(), cases):
        track, nodes, arcs, route = load_trip_raw(split, tid)
        proj = LocalTangentProjection(track[:, 0].mean(), track[:, 1].mean())
        px, py = proj.to_xy(track[:, 0], track[:, 1])
        nx, ny = proj.to_xy(nodes[:, 0], nodes[:, 1])

        # Local road network clipping for clean plot
        pad = 350.0
        xmin, xmax = px.min() - pad, px.max() + pad
        ymin, ymax = py.min() - pad, py.max() + pad
        # Ensure narrow corridors have readable aspect width
        x_span = xmax - xmin
        y_span = ymax - ymin
        if x_span < 0.35 * y_span:
            x_mid = 0.5 * (xmin + xmax)
            half_w = 0.5 * 0.35 * y_span
            xmin, xmax = x_mid - half_w, x_mid + half_w

        ax0, ay0 = nx[arcs[:, 0]], ny[arcs[:, 0]]
        bx0, by0 = nx[arcs[:, 1]], ny[arcs[:, 1]]
        mask = (
            (np.maximum(ax0, bx0) >= xmin)
            & (np.minimum(ax0, bx0) <= xmax)
            & (np.maximum(ay0, by0) >= ymin)
            & (np.minimum(ay0, by0) <= ymax)
        )
        local_arcs = np.flatnonzero(mask)
        # Draw up to 8000 local segments cleanly
        if len(local_arcs) > 8000:
            local_arcs = local_arcs[:8000]
        xs = np.vstack([ax0[local_arcs], bx0[local_arcs], np.full(len(local_arcs), np.nan)])
        ys = np.vstack([ay0[local_arcs], by0[local_arcs], np.full(len(local_arcs), np.nan)])
        ax.plot(xs.ravel(order="F"), ys.ravel(order="F"), color="#cccccc", linewidth=0.6, label="Road Network", zorder=1)

        # Reference route
        ru, rv = arcs[route, 0], arcs[route, 1]
        rxs = np.vstack([nx[ru], nx[rv], np.full(len(route), np.nan)])
        rys = np.vstack([ny[ru], ny[rv], np.full(len(route), np.nan)])
        ax.plot(rxs.ravel(order="F"), rys.ravel(order="F"), color="#ff7f0e", linewidth=3.5, alpha=0.85, label="Reference Route (.route)", zorder=2)

        # GPS track
        ax.scatter(px, py, s=2.2, color="#1f77b4", alpha=0.85, label="Raw GPS (.track)", zorder=3)
        ax.scatter([px[0]], [py[0]], s=50, color="lime", edgecolor="black", zorder=4, label="GPS Start")
        ax.scatter([px[-1]], [py[-1]], s=50, color="red", edgecolor="black", zorder=4, label="GPS End")

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Local Easting (m)")
        ax.set_ylabel("Local Northing (m)")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.25)

        del track, nodes, arcs, route
        gc.collect()

    plt.suptitle("PathSnap Phase 0 — Local Metric Trajectory & Road Network Inspection", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_dir / "eda_sample_trajectories.png")
    plt.close(fig)


def build_global_folium_map(df: pd.DataFrame, out_path: Path) -> None:
    m = folium.Map(location=[35.0, 10.0], zoom_start=2, tiles="OpenStreetMap")

    train_fg = folium.FeatureGroup(name="Train Trips (80)")
    test_fg = folium.FeatureGroup(name="Student Test Trips (10)")
    anom_fg = folium.FeatureGroup(name="Flagged Route Anomalies")

    for _, row in df.iterrows():
        tid = str(row["trip_id"]).zfill(8)
        popup_html = (
            f"<b>{row['split']} / {tid}</b><br>"
            f"GPS obs: {int(row['n_gps'])} | Route arcs: {int(row['route_len'])}<br>"
            f"Nodes/Arcs: {int(row['n_nodes'])} / {int(row['n_arcs'])}<br>"
            f"Track len: {row['track_length_km']:.2f} km | Route len: {row['route_length_km']:.2f} km<br>"
            f"GPS->Route median: {row['gps_to_route_median_m']:.1f}m | max: {row['gps_to_route_max_m']:.1f}m<br>"
            f"Max dt: {row['dt_max']:.1f}s | Disconnected: {int(row['n_disconnected'])}"
        )
        is_anom = (row["gps_to_route_p95_m"] > 60.0) or (row["n_disconnected"] > 0)
        if row["split"] == "student_test":
            folium.CircleMarker(
                location=[row["centroid_lat"], row["centroid_lon"]],
                radius=6,
                color="black",
                weight=1.5,
                fill=True,
                fill_color="#ff7f0e",
                fill_opacity=0.9,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"student_test/{tid}",
            ).add_to(test_fg)
        else:
            folium.CircleMarker(
                location=[row["centroid_lat"], row["centroid_lon"]],
                radius=5,
                color="#1f77b4",
                fill=True,
                fill_color="#1f77b4",
                fill_opacity=0.75,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"train/{tid}",
            ).add_to(train_fg)

        if is_anom:
            folium.CircleMarker(
                location=[row["centroid_lat"], row["centroid_lon"]],
                radius=9,
                color="#d62728",
                weight=2,
                fill=False,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"ANOMALY {row['split']}/{tid}",
            ).add_to(anom_fg)

    train_fg.add_to(m)
    test_fg.add_to(m)
    anom_fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(out_path))


def build_trip_folium_map(
    split: str,
    trip_id: str,
    out_path: Path,
    subtitle: str,
    max_road_arcs: int = 3500,
) -> None:
    track, nodes, arcs, route = load_trip_raw(split, trip_id)
    lon, lat, ts = track[:, 0], track[:, 1], track[:, 2]
    center_lat, center_lon = float(lat.mean()), float(lon.mean())

    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="OpenStreetMap")

    # 1. Local road network layer (clipped to track bbox + ~0.004 deg margin)
    road_fg = folium.FeatureGroup(name="Local Road Network (.nodes/.arcs)", show=True)
    lon_pad, lat_pad = 0.004, 0.003
    lmin, lmax = float(lon.min()) - lon_pad, float(lon.max()) + lon_pad
    bmin, bmax = float(lat.min()) - lat_pad, float(lat.max()) + lat_pad

    u_lon, u_lat = nodes[arcs[:, 0], 0], nodes[arcs[:, 0], 1]
    v_lon, v_lat = nodes[arcs[:, 1], 0], nodes[arcs[:, 1], 1]
    in_box = (
        (np.maximum(u_lon, v_lon) >= lmin)
        & (np.minimum(u_lon, v_lon) <= lmax)
        & (np.maximum(u_lat, v_lat) >= bmin)
        & (np.minimum(u_lat, v_lat) <= bmax)
    )
    local_arc_idx = np.flatnonzero(in_box)
    if len(local_arc_idx) > max_road_arcs:
        step = max(1, len(local_arc_idx) // max_road_arcs)
        local_arc_idx = local_arc_idx[::step]

    road_segments = [
        [[float(u_lat[i]), float(u_lon[i])], [float(v_lat[i]), float(v_lon[i])]]
        for i in local_arc_idx
    ]
    if road_segments:
        folium.PolyLine(
            locations=road_segments,
            color="#7f7f7f",
            weight=1.5,
            opacity=0.55,
            tooltip=f"Road network ({len(road_segments)} displayed arcs)",
        ).add_to(road_fg)
    road_fg.add_to(m)

    # 2. Reference route layer (.route)
    route_fg = folium.FeatureGroup(name="Reference Route (.route)", show=True)
    ru, rv = arcs[route, 0], arcs[route, 1]
    route_segments = [
        [
            [float(nodes[ru[k], 1]), float(nodes[ru[k], 0])],
            [float(nodes[rv[k], 1]), float(nodes[rv[k], 0])],
        ]
        for k in range(len(route))
    ]
    if route_segments:
        folium.PolyLine(
            locations=route_segments,
            color="#2ca02c",
            weight=5,
            opacity=0.85,
            tooltip=f"Reference route ({len(route)} arcs)",
        ).add_to(route_fg)

        # Highlight any disconnected transitions in .route
        for k in range(len(route) - 1):
            if rv[k] != ru[k + 1]:
                folium.PolyLine(
                    locations=[
                        [float(nodes[rv[k], 1]), float(nodes[rv[k], 0])],
                        [float(nodes[ru[k + 1], 1]), float(nodes[ru[k + 1], 0])],
                    ],
                    color="#ff00ff",
                    weight=4,
                    dash_array="6, 6",
                    tooltip=f"Route discontinuity step {k}->{k+1}",
                ).add_to(route_fg)
    route_fg.add_to(m)

    # 3. Raw GPS observations layer (.track)
    gps_fg = folium.FeatureGroup(name="Raw GPS Observations (.track)", show=True)
    # Draw connecting thin polyline for sequential clarity
    step_pts = max(1, len(track) // 3000)
    gps_coords = [[float(lat[i]), float(lon[i])] for i in range(0, len(track), step_pts)]
    folium.PolyLine(
        locations=gps_coords,
        color="#1f77b4",
        weight=2,
        opacity=0.65,
        tooltip="GPS trajectory sequence",
    ).add_to(gps_fg)

    # Individual GPS markers (downsampled only if huge > 2500 pts so HTML remains responsive)
    dt = np.diff(ts, prepend=ts[0] - 1.0)
    marker_step = max(1, len(track) // 1200)
    for i in range(0, len(track), marker_step):
        is_gap = dt[i] > 5.0
        folium.CircleMarker(
            location=[float(lat[i]), float(lon[i])],
            radius=4.5 if is_gap else 2.5,
            color="#d62728" if is_gap else "#1f77b4",
            fill=True,
            fill_opacity=0.85,
            tooltip=f"idx={i}, t={ts[i]:.1f}s, dt={dt[i]:.1f}s",
        ).add_to(gps_fg)

    # Always mark gap endpoints > 5s and Start/End
    for i in np.flatnonzero(dt > 5.0):
        folium.CircleMarker(
            location=[float(lat[i]), float(lon[i])],
            radius=6,
            color="#d62728",
            fill=True,
            fill_color="#d62728",
            fill_opacity=0.9,
            tooltip=f"GPS Gap arrival idx={i}, dt={dt[i]:.1f}s",
        ).add_to(gps_fg)

    folium.Marker(
        location=[float(lat[0]), float(lon[0])],
        icon=folium.Icon(color="green", icon="play"),
        tooltip="GPS Start",
    ).add_to(gps_fg)
    folium.Marker(
        location=[float(lat[-1]), float(lon[-1])],
        icon=folium.Icon(color="red", icon="stop"),
        tooltip="GPS End",
    ).add_to(gps_fg)
    gps_fg.add_to(m)

    # Legend overlay
    legend_html = f"""
    <div style="position: fixed; bottom: 25px; left: 25px; z-index: 9999;
                background-color: white; border: 2px solid #555; border-radius: 6px;
                padding: 10px 14px; font-size: 12px; box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
      <b>PathSnap Phase 0: {split}/{trip_id}</b><br>
      <span style="color:#444;">{subtitle}</span><br><br>
      <span style="color:#1f77b4;">&#9679;</span> Raw GPS Track ({len(track)} pts)<br>
      <span style="color:#d62728;">&#9679;</span> GPS Gap &gt; 5s<br>
      <span style="color:#2ca02c;">&#9473;</span> Reference Route ({len(route)} arcs)<br>
      <span style="color:#ff00ff;">- -</span> Route Discontinuity<br>
      <span style="color:#7f7f7f;">&#9473;</span> Local Road Network
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[float(lat.min()), float(lon.min())], [float(lat.max()), float(lon.max())]])
    m.save(str(out_path))

    del track, nodes, arcs, route
    gc.collect()


def main() -> int:
    results_dir = PROJECT_ROOT / "results"
    viz_dir = PROJECT_ROOT / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_dir / "dataset_inventory.csv")
    gaps_df = pd.read_csv(results_dir / "timestamp_gap_analysis.csv")

    print("Generating static EDA plots...", flush=True)
    plot_dataset_overview(df, viz_dir)
    plot_temporal_and_speeds(df, gaps_df, viz_dir)
    plot_network_and_route_quality(df, viz_dir)
    plot_sample_trajectories(viz_dir)

    print("Generating interactive Folium maps...", flush=True)
    build_global_folium_map(df, viz_dir / "folium_global_trips_overview.html")
    build_trip_folium_map(
        "train",
        "00000012",
        viz_dir / "folium_train_typical_00000012.html",
        "Typical clean training trip",
    )
    build_trip_folium_map(
        "train",
        "00000054",
        viz_dir / "folium_train_anomaly_00000054.html",
        "Severe off-route anomaly (87.2% obs > 50m from .route)",
    )
    build_trip_folium_map(
        "train",
        "00000033",
        viz_dir / "folium_train_anomaly_00000033.html",
        "Partial route / detour mismatch (49% recall at 100m)",
    )
    build_trip_folium_map(
        "student_test",
        "00000076",
        viz_dir / "folium_student_test_00000076.html",
        "Student test trip with 32s GPS gap & 1 route discontinuity",
    )

    print("All visualizations generated successfully in visualizations/!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

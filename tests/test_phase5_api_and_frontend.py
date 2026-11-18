"""Automated tests for Phase 5 FastAPI backend and Streamlit frontend integration."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.api.main import create_app
from frontend import app as frontend_app


def test_health_and_model_loaded():
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert data["hmm_config_loaded"] is True
        assert data["available_trip_count"] == 10
        assert data["inference_dataset_name"] == "student_test"


def test_list_trips_and_error_handling():
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/trips")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["trip_count"] == 10
        trip_ids = {t["trip_id"] for t in payload["trips"]}
        assert "00000052" in trip_ids
        assert "00000076" in trip_ids

        # Missing trip ID -> 404
        r404 = client.post("/predict/99999999")
        assert r404.status_code == 404
        assert r404.json()["error_code"] == "TRIP_NOT_FOUND"

        # Invalid characters -> 400
        r400 = client.post("/predict/bad$trip")
        assert r400.status_code == 400
        assert r400.json()["error_code"] == "INVALID_TRIP_ID"


def test_end_to_end_prediction_and_map_data():
    app = create_app()
    with TestClient(app) as client:
        m_resp = client.get("/trips/00000052/map-data")
        assert m_resp.status_code == 200
        m_data = m_resp.json()
        assert m_data["coordinate_order"] == "lat_lon"
        assert m_data["n_gps_observations"] == 262
        assert len(m_data["gps_coordinates"]) == 262
        assert m_data["displayed_network_arcs"] > 0

        p_resp = client.post("/predict/00000052")
        assert p_resp.status_code == 200
        pred = p_resp.json()
        assert pred["status"] == "success"
        assert pred["trip_id"] == "00000052"
        assert pred["observation_count"] == 262
        assert pred["candidate_count"] > 262
        assert len(pred["predicted_obs_arc_ids"]) == 262
        assert len(pred["reconstructed_route_arc_ids"]) > 0
        assert len(pred["is_connector_arc"]) == len(pred["reconstructed_route_arc_ids"])
        assert len(pred["reconstructed_route_segments"]) == len(pred["reconstructed_route_arc_ids"])
        assert pred["diagnostics"]["route_continuity_pct"] == 100.0
        assert pred["processing_time_s"] > 0.0

        # Also test trip 00000076 which has a 32s temporal gap (> 30s split)
        p76 = client.post("/predict/00000076")
        assert p76.status_code == 200
        d76 = p76.json()["diagnostics"]
        assert d76["temporal_gap_splits"] >= 1
        assert len(d76["warnings"]) >= 1


def test_ground_truth_route_isolation(tmp_path: Path, monkeypatch):
    # 1. Copy 00000052 WITHOUT its .route file to verify API works without .route
    src_trip = PROJECT_ROOT / "data" / "raw" / "student_test" / "00000052"
    dst_root = tmp_path / "eval_no_route"
    dst_trip = dst_root / "00000052"
    dst_trip.mkdir(parents=True)
    for ext in (".track", ".nodes", ".arcs"):
        shutil.copy2(src_trip / f"00000052{ext}", dst_trip / f"00000052{ext}")

    monkeypatch.setenv("PATHSNAP_INFERENCE_DATA_DIR", str(dst_root))
    app = create_app()
    with TestClient(app) as client:
        assert client.get("/trips").json()["trip_count"] == 1
        assert client.get("/trips/00000052/map-data").status_code == 200
        pred = client.post("/predict/00000052")
        assert pred.status_code == 200
        assert pred.json()["status"] == "success"

    # 2. Static source check: API and Frontend never reference load_reference_route or .route reads
    for pyfile in (PROJECT_ROOT / "backend" / "app" / "api").glob("*.py"):
        src = pyfile.read_text(encoding="utf-8")
        assert "load_reference_route" not in src
        assert '".route"' not in src and "'.route'" not in src

    import ast
    fe_src = (PROJECT_ROOT / "frontend" / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(fe_src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("backend")
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("backend")
    assert "load_reference_route" not in fe_src


def test_frontend_folium_builder_with_api_payload():
    app = create_app()
    with TestClient(app) as client:
        m_data = client.get("/trips/00000052/map-data").json()
        pred = client.post("/predict/00000052").json()
        fmap = frontend_app.build_prediction_map(m_data, pred)
        html = fmap._repr_html_()
        assert "Predicted Route" in html
        assert "Raw GPS Observations" in html

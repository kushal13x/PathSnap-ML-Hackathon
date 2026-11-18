"""Phase 6 hardening tests: multi-trip E2E, unreachable backend handling, env config."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.api.main import create_app
from frontend import app as frontend_app


def test_frontend_respects_env_var_and_handles_unreachable_backend(monkeypatch):
    # Ensure frontend does not hardcode localhost when PATHSNAP_API_URL is unset
    monkeypatch.delenv("PATHSNAP_API_URL", raising=False)
    reloaded = importlib.reload(frontend_app)
    assert reloaded.DEFAULT_API_URL == ""

    # Verify unreachable backend returns clean (False, message) without crashing
    ok_h, msg_h = reloaded.fetch_health("http://127.0.0.1:59999")
    ok_t, msg_t = reloaded.fetch_trips("http://127.0.0.1:59999")
    ok_p, msg_p = reloaded.request_prediction("http://127.0.0.1:59999", "00000052")
    assert ok_h is False and isinstance(msg_h, str) and len(msg_h) > 0
    assert ok_t is False and isinstance(msg_t, str) and len(msg_t) > 0
    assert ok_p is False and isinstance(msg_p, str) and len(msg_p) > 0

    # Restore module cleanly
    monkeypatch.setenv("PATHSNAP_API_URL", "http://127.0.0.1:8000")
    importlib.reload(frontend_app)


def test_multiple_trips_e2e_and_evaluator_dir_override(tmp_path: Path, monkeypatch):
    app = create_app()
    with TestClient(app) as client:
        for tid in ("00000000", "00000076"):
            r = client.post(f"/predict/{tid}")
            assert r.status_code == 200
            data = r.json()
            assert data["trip_id"] == tid
            assert data["model_used"] == "hist_gradient_boosting_tuned"
            assert data["diagnostics"]["route_continuity_pct"] >= 99.0
            assert len(data["reconstructed_route_segments"]) == len(data["reconstructed_route_arc_ids"])

            fmap = frontend_app.build_prediction_map(
                client.get(f"/trips/{tid}/map-data").json(), data
            )
            assert "folium" in fmap._repr_html_().lower()

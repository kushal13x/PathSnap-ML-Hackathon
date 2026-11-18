# PathSnap — System Architecture

## 1. End-to-End Inference Pipeline

```text
Raw Trip (.track, .nodes, .arcs)   [strictly NO .route access at inference time]
        │
        ▼
[1] Local WGS84 Tangent-Plane Projection & Gap-Aware Kinematics
        │  (backend/app/data/preprocessing.py)
        ▼
[2] Spatial Candidate Retrieval (K = 20, R = 100m, short <= 300m cKDTree + long bbox)
        │  (backend/app/features/candidate_generation.py)
        ▼
[3] Leak-Free 26-Feature Engineering (float32)
        │  (backend/app/features/feature_engineering.py)
        ▼
[4] Tuned HistGradientBoostingClassifier (CPU, loaded once at startup)
        │  (artifacts/hist_gradient_boosting_tuned.joblib)
        ▼
[5] HMM Emission Log-Probabilities + Cached Bounded Dijkstra Transitions
        │  (backend/app/decoder/hmm.py, transition_model.py, configs/hmm_config.json)
        ▼
[6] Log-Space Viterbi Decoding (splits on dt > 30s & empty candidates)
        │  (backend/app/decoder/viterbi.py)
        ▼
[7] Auditable Road-Route Reconstruction (collapsing + valid graph connectors)
        │  (backend/app/decoder/route_reconstruction.py)
        ▼
FastAPI JSON Response ([lat, lon] coordinates) ──HTTP──► Streamlit + Folium UI
```

## 2. Two-Laptop Distributed Deployment Topology

| Node | Role | Host / Port | Responsibilities |
| :--- | :--- | :--- | :--- |
| **Laptop 1** | **Backend & ML Server** | `0.0.0.0:8000` | Hosts FastAPI (`uvicorn`), startup-loaded `hist_gradient_boosting_tuned.joblib`, HMM/Viterbi decoder, road-network data, inference dataset (`PATHSNAP_INFERENCE_DATA_DIR`), and server-side API logs. |
| **Laptop 2** | **Evaluator UI Client** | `0.0.0.0:8501` | Hosts Streamlit + Folium (`frontend/app.py`). Communicates strictly over HTTP via `PATHSNAP_API_URL=http://<LAPTOP_1_LAN_IP>:8000`. Never imports backend code or accesses raw `.route` files. |

## 3. Data Separation & Anti-Leakage Guarantees

1. **Inference Isolation:** `backend/app/api/service.py` loads strictly `.track`, `.nodes`, and `.arcs` and never imports `load_reference_route`.
2. **Dynamic Trip Discovery:** `GET /trips` scans `PATHSNAP_INFERENCE_DATA_DIR` dynamically (default: `data/raw/student_test`; switched to reserved evaluator dataset on Laptop 1 during final evaluation without code changes).
3. **API Contract:** Documented in [`docs/api_contract.md`](file:///c:/Users/Kushal/Desktop/ML%20Hackathon/docs/api_contract.md) with Swagger UI at `/docs`.

# PathSnap Inference API Contract (v1.0.0)

## 1. Base URL & Two-Laptop Configuration

- **Local Development Default:** `http://127.0.0.1:8000`
- **Frontend Environment Variable:** `PATHSNAP_API_URL` (e.g., `http://<LAPTOP_1_LAN_IP>:8000`)
- **Backend Environment Variables:**
  - `PATHSNAP_INFERENCE_DATA_DIR` (default: `data/raw/student_test`; switchable to reserved evaluator dataset)
  - `PATHSNAP_MODEL_PATH` (default: `artifacts/hist_gradient_boosting_tuned.joblib`)
  - `PATHSNAP_HMM_CONFIG_PATH` (default: `configs/hmm_config.json`)
  - `PATHSNAP_CORS_ORIGINS` (default: `*`)
- **Interactive Swagger Docs:** `GET /docs`

## 2. Coordinate Standard & Data Separation

- **Coordinate Ordering:** All coordinate pairs returned by every endpoint use WGS84 **`[latitude, longitude]`** (`"coordinate_order": "lat_lon"`), matching Folium / Leaflet conventions directly.
- **Ground-Truth Isolation:** The API never reads, loads, or exposes `.route` files. Inference uses `.track`, `.nodes`, and `.arcs` only.

## 3. Endpoints

| Method | Path | Request | Response Model | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/health` | None | `HealthResponse` | Service health, startup model load status, HMM config status, trip count |
| `GET` | `/trips` | None | `TripListResponse` | Lists available trip IDs, GPS point counts, and duration (no `.route`) |
| `GET` | `/trips/{trip_id}/map-data` | Path `trip_id` | `TripMapDataResponse` | Returns `gps_coordinates`, bbox-clipped `road_network_segments`, `bounds`, `center` |
| `POST` | `/predict/{trip_id}` | Path `trip_id` | `PredictionResponse` | Executes full Phase 1–4 pipeline and returns predicted arcs, route geometry, and diagnostics |

## 4. Error Response Format

```json
{
  "detail": "Trip '99999999' not found in inference directory.",
  "error_code": "TRIP_NOT_FOUND"
}
```

- `400 Bad Request` (`INVALID_TRIP_ID`): Invalid/unsafe `trip_id` characters.
- `404 Not Found` (`TRIP_NOT_FOUND`): Missing trip folder or missing `.track`/`.nodes`/`.arcs`.
- `503 Service Unavailable`: Service model/config not initialized.

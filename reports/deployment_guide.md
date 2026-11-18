# PathSnap — Deployment & Operations Guide

## 1. Environment Variables

| Variable | Component | Default | Description |
| :--- | :--- | :--- | :--- |
| `PATHSNAP_INFERENCE_DATA_DIR` | Backend (Laptop 1) | `data/raw/student_test` | Directory containing trip folders (`<trip_id>/{trip_id}.{track,nodes,arcs}`) |
| `PATHSNAP_MODEL_PATH` | Backend (Laptop 1) | `artifacts/hist_gradient_boosting_tuned.joblib` | CPU-compatible tuned classifier artifact |
| `PATHSNAP_HMM_CONFIG_PATH` | Backend (Laptop 1) | `configs/hmm_config.json` | Calibrated HMM emission/transition parameters |
| `PATHSNAP_CORS_ORIGINS` | Backend (Laptop 1) | `*` | Allowed CORS origins |
| `PATHSNAP_API_URL` | Frontend (Laptop 2) | Required (set by `start_frontend.ps1`) | Base URL of Laptop 1 FastAPI server (e.g., `http://<LAPTOP_1_LAN_IP>:8000`) |

---

## 2. Single-Laptop Local Execution (Two PowerShell Terminals)

**Terminal 1 — Start FastAPI Backend:**
```powershell
cd "C:\Users\Kushal\Desktop\ML Hackathon"
.\scripts\start_backend.ps1
```

**Terminal 2 — Start Streamlit Frontend:**
```powershell
cd "C:\Users\Kushal\Desktop\ML Hackathon"
.\scripts\start_frontend.ps1 -ApiUrl "http://127.0.0.1:8000"
```
Open `http://127.0.0.1:8501` in a browser (Swagger API docs at `http://127.0.0.1:8000/docs`).

---

## 3. Two-Laptop LAN Deployment & Integration Checklist

### On Laptop 1 (Backend Server)

1. **Find Laptop 1's LAN IPv4 address:**
   ```powershell
   ipconfig
   ```
   Look under `Wireless LAN adapter Wi-Fi` (or active Ethernet adapter) for `IPv4 Address` (e.g., `10.x.x.x` or `192.168.x.x`).

2. **Start FastAPI pointing to the evaluation dataset (or `student_test` for rehearsal):**
   ```powershell
   cd "C:\path\to\ML Hackathon"
   .\scripts\start_backend.ps1 -DataDir "C:\path\to\evaluator_test"
   ```
   *(Direct equivalent: `$env:PATHSNAP_INFERENCE_DATA_DIR = "C:\path\to\evaluator_test"; python -m uvicorn backend.app.api.main:app --host 0.0.0.0 --port 8000`)*

### On Laptop 2 (Evaluator UI Client)

3. **Verify network reachability and `/health` from Laptop 2 PowerShell:**
   ```powershell
   Test-NetConnection -ComputerName <LAPTOP_1_LAN_IP> -Port 8000
   Invoke-RestMethod -Uri "http://<LAPTOP_1_LAN_IP>:8000/health"
   ```

4. **Start Streamlit UI pointing to Laptop 1:**
   ```powershell
   cd "C:\path\to\ML Hackathon"
   .\scripts\start_frontend.ps1 -ApiUrl "http://<LAPTOP_1_LAN_IP>:8000"
   ```
   Open `http://127.0.0.1:8501` on Laptop 2, select any trip, click **Predict Route**, and verify live API logs appear in Laptop 1's PowerShell window.

---

## 4. Windows Firewall & Connection Troubleshooting

- **`Test-NetConnection` fails (`TcpTestSucceeded : False`) / Timeout from Laptop 2:**
  1. Ensure both laptops are on the **same Wi-Fi/LAN subnet** (avoid guest Wi-Fi client isolation).
  2. Set the Wi-Fi network profile on Laptop 1 to **Private** (Windows Settings → Network & Internet → Wi-Fi).
  3. If Windows Defender Firewall blocks inbound port `8000`, allow Python/port `8000` on the Private profile (requires Administrator PowerShell, run manually only if needed):
     ```powershell
     New-NetFirewallRule -DisplayName "PathSnap FastAPI 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Private
     ```
- **API returns `HTTP 404 TRIP_NOT_FOUND`:** Verify `PATHSNAP_INFERENCE_DATA_DIR` points to the parent directory containing `<trip_id>/<trip_id>.{track,nodes,arcs}`.
- **Frontend shows `PATHSNAP_API_URL is not set`:** Pass `-ApiUrl "http://<LAPTOP_1_LAN_IP>:8000"` to `scripts/start_frontend.ps1` or enter it in the sidebar.

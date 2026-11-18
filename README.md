# PathSnap: Distributed Map-Matching & Route Reconstruction

**PES University — UE24CS352A Machine Learning Hackathon**  
End-to-end, CPU-compatible machine learning system for snapping noisy GPS trajectories onto directed road networks using spatial candidate retrieval, gradient-boosted candidate scoring, HMM/Viterbi sequence decoding, and a distributed FastAPI + Streamlit/Folium web application.

---

## 1. Objective & Problem Statement

GPS observations recorded in real urban and highway environments suffer from multipath noise, signal dropouts, irregular temporal gaps, and speed spikes. Raw GPS trajectories can drift onto parallel streets or off-network areas, while `.track` observations ($T$ points) and reference `.route` sequences ($M$ arc IDs) are not 1-to-1 ($T \neq M$).

PathSnap reconstructs topologically coherent road routes through:
1. Per-trip local WGS84 tangent-plane metric projection and gap-aware kinematics.
2. Spatial candidate road-segment retrieval ($K = 20, R = 100\text{ m}$).
3. Leak-free 26-feature scoring with a Bayesian-tuned `HistGradientBoostingClassifier`.
4. Log-space HMM/Viterbi decoding with cached bounded Dijkstra transitions and `dt > 30 s` gap segmentation.
5. Auditable route reconstruction served via a two-laptop FastAPI + Streamlit/Folium architecture.

---

## 2. Implementation Status (Phases 0–6 Complete)

- **Phase 0 — Dataset Audit & EDA:** Audited `80` training and `10` student-testing trips (`227,577` GPS observations); analyzed timestamp gaps (`96.682%` at `1.0 s`), network connectivity, and reference-route anomalies; generated static plots and Folium maps.
- **Phase 1 — Candidate Generation, Label Alignment & Feature Engineering:** Built `cKDTree` + long-arc spatial retrieval (`0` missed candidates vs. brute-force), vectorized Monotonic Constrained DP label alignment, 26 `float32` leak-free features, `64 / 16` trip-grouped split (`GroupShuffleSplit`), and preliminary `SGDClassifier` baseline.
- **Phase 2 — Five-Model Experimentation:** Trained and evaluated `SGDClassifier`, `DecisionTreeClassifier`, `RandomForestClassifier`, `ExtraTreesClassifier`, and `HistGradientBoostingClassifier` (plus optional RTX 3050 CUDA `XGBoost`) with 3-fold `GroupKFold` CV.
- **Phase 3 — Bayesian Hyperparameter Optimization:** Ran `12` Optuna TPE trials strictly inside `train_fit` (`82.563%` inner CV Top-1) and saved `artifacts/hist_gradient_boosting_tuned.joblib`.
- **Phase 4 — HMM & Viterbi Route Reconstruction:** Implemented emission log-probs, cached bounded Dijkstra transitions, temporal-gap splitting (`dt > 30 s`), log-space Viterbi, and auditable connector reconstruction.
- **Phase 5 & Phase 6 — FastAPI Backend, Streamlit UI & Two-Laptop Hardening:** Built REST API (`backend/app/api/`), interactive Streamlit + Folium dashboard (`frontend/app.py`), Windows PowerShell startup scripts (`scripts/start_backend.ps1`, `scripts/start_frontend.ps1`), and `22` automated tests.

---

## 3. Repository Structure

```text
PathSnap-ML-Hackathon/
├── backend/app/
│   ├── api/                 # FastAPI app, schemas, config & .route-free inference service
│   ├── data/                # Trip loaders, metric preprocessing & offline DP label alignment
│   ├── features/            # Hybrid cKDTree+long-arc candidate retrieval & 26 features
│   ├── models/              # Baselines, 5-model comparison & Optuna Bayesian search
│   ├── decoder/             # HMM emissions, bounded Dijkstra transitions, Viterbi & reconstruction
│   └── evaluation/          # Candidate, ranking & sequence-similarity metrics
├── frontend/
│   └── app.py               # Streamlit + Folium evaluator UI (HTTP-only via PATHSNAP_API_URL)
├── artifacts/
│   └── hist_gradient_boosting_tuned.joblib  # Primary CPU inference artifact (plus Phase 2 models)
├── configs/
│   ├── pipeline_config.json # Candidate, alignment, split & model hyperparameters
│   └── hmm_config.json      # Calibrated HMM emission/transition parameters
├── data/raw/student_test/   # 10 default inference trips (.track, .nodes, .arcs only; NO .route)
├── docs/
│   ├── api_contract.md      # REST endpoints, coordinate ordering [lat, lon] & error format
│   └── reference/           # Official hackathon guideline PDFs
├── reports/                 # Phase 0-6 reports, system_architecture.md & deployment_guide.md
├── results/                 # Audit CSVs, model comparison tables, HPO & HMM evaluation JSONs
├── scripts/                 # Pipeline scripts + start_backend.ps1 & start_frontend.ps1
├── tests/                   # 22 automated tests across Phases 1-6
├── visualizations/          # Static PNG plots & interactive Folium HTML maps
├── requirements.txt         # Full Phase 0-6 dependencies
└── .gitignore               # Strictly excludes .route, train data, Parquet partitions & secrets
```

---

## 4. Key Experimental Results

### Five-Model (+ Optional GPU XGBoost) Comparison (`train_val`, 16 Held-Out Trips)

| Model | Top-1 (Valid) | Top-5 (Valid) | Macro F1 | ROC-AUC | Log Loss | Step Continuity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Nearest-Road Distance | 58.34% | 96.88% | 0.5611 | 0.9231 | 0.1941 | 78.69% |
| SGD Logistic Regression | 80.08% | 98.06% | 0.8021 | 0.9804 | 0.1848 | 91.90% |
| Decision Tree | 81.61% | 97.44% | 0.8357 | 0.9793 | 0.1534 | 91.03% |
| Random Forest | 81.61% | 98.30% | 0.8691 | 0.9882 | 0.1128 | 92.21% |
| Extra Trees | 80.80% | 98.16% | 0.8430 | 0.9865 | 0.1422 | 91.33% |
| **HistGradientBoosting (Untuned)** | **82.27%** | 98.25% | **0.9000** | **0.9887** | **0.0687** | 92.38% |
| XGBoost (CUDA GPU, optional) | 82.10% | **98.35%** | 0.8995 | 0.9886 | 0.0691 | **92.52%** |

### Tuned Classifier-Only vs. HMM/Viterbi Route Reconstruction

| Partition | Method | Obs Top-1 (Valid) | Step Cont. | Invalid Obs Trans. | Reconstructed Route Cont. | Length-Weighted Route F1 | LCS Seq Sim. |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`train_val` (16)** | Classifier-Only (Tuned) | 82.33% | 92.56% | 2,231 | 77.08% | 88.79% | 72.77% |
| **`train_val` (16)** | **HMM + Viterbi** | **87.49%** | **97.16%** | **747** | **99.32%** | **94.72%** | **87.99%** |
| **`student_test` (10)** | Classifier-Only (Tuned) | 83.88% | 93.27% | 1,284 | 77.25% | 91.49% | 72.39% |
| **`student_test` (10)** | **HMM + Viterbi** | **88.73%** | **97.35%** | **471** | **99.92%** | **97.52%** | **91.42%** |

---

## 5. Quick Start (Clone & Run Out of the Box)

The repository includes `artifacts/hist_gradient_boosting_tuned.joblib`, `configs/hmm_config.json`, and the 10 inference-only `data/raw/student_test/` trips (`.track`, `.nodes`, `.arcs` without `.route`), so you can run the full application immediately after cloning:

```powershell
git clone https://github.com/kushal13x/PathSnap-ML-Hackathon.git
cd PathSnap-ML-Hackathon

python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run on One Laptop (Two PowerShell Terminals)

**Terminal 1 — Start FastAPI Backend (`0.0.0.0:8000`):**
```powershell
.\scripts\start_backend.ps1
```

**Terminal 2 — Start Streamlit Evaluator UI (`0.0.0.0:8501`):**
```powershell
.\scripts\start_frontend.ps1 -ApiUrl "http://127.0.0.1:8000"
```
Open `http://127.0.0.1:8501` in your browser (Swagger API docs at `http://127.0.0.1:8000/docs`).

### Run Across Two Laptops (LAN Deployment)

See [`reports/deployment_guide.md`](file:///c:/Users/Kushal/Desktop/ML%20Hackathon/reports/deployment_guide.md) and [`docs/api_contract.md`](file:///c:/Users/Kushal/Desktop/ML%20Hackathon/docs/api_contract.md) for full details.

1. **Laptop 1 (Backend):** Find LAN IP via `ipconfig` and start the API (optionally pointing `-DataDir` to the evaluator dataset):
   ```powershell
   .\scripts\start_backend.ps1
   # Or with reserved evaluator directory:
   .\scripts\start_backend.ps1 -DataDir "C:\path\to\evaluator_test"
   ```
2. **Laptop 2 (Frontend):** Test connectivity (`Invoke-RestMethod -Uri "http://<LAPTOP_1_LAN_IP>:8000/health"`), then launch Streamlit:
   ```powershell
   .\scripts\start_frontend.ps1 -ApiUrl "http://<LAPTOP_1_LAN_IP>:8000"
   ```

---

## 6. Automated Tests & Excluded Data

Run the test suite (note: offline training/candidate-parquet tests require local training data, while API, HMM/Viterbi, and `student_test` inference tests run directly):
```powershell
python -m pytest tests/ -v
```

**Deliberately Excluded from Git:**
- Ground-truth `*.route` files everywhere (including inside `data/raw/student_test/`).
- Training dataset (`data/raw/train/`), processed Parquet partitions (`data/processed/`), and original ZIP archives (`source_zips/`).
- Virtual environments, caches (`__pycache__/`, `.pytest_cache/`), logs, and credentials.

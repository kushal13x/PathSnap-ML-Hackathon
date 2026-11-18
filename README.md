# PathSnap: Distributed Map-Matching & Route Reconstruction

**PES University — UE24CS352A Machine Learning Hackathon**  
A modular, CPU-friendly machine learning pipeline for snapping noisy vehicle GPS trajectories onto directed road networks.

---

## 1. Objective

PathSnap aims to reconstruct accurate, topologically coherent vehicle routes from noisy real-world GPS traces by combining spatial candidate retrieval, supervised road-segment candidate scoring, and sequential route reconstruction.

## 2. Problem Statement

Raw GPS tracking observations suffer from multipath noise, urban canyon offsets, irregular sampling gaps, and speed spikes, causing raw coordinates to drift onto parallel streets or off-network areas. Furthermore, supplied reference `.route` sequences list chronological road arc IDs ($M$ arcs) rather than observation-aligned labels ($T$ GPS points, where $T \neq M$), and may contain route discontinuities or omitted detours.

---

## 3. Technology Stack (Currently Used in Phases 0–1)

- **Language:** Python 3.13
- **Numerical & Spatial Processing:** NumPy, Pandas, SciPy (`cKDTree`, `csgraph`), PyArrow (Snappy-compressed Parquet partitions)
- **Machine Learning & Serialization:** scikit-learn (`GroupShuffleSplit`, `StandardScaler`, `SGDClassifier`), Joblib
- **Visualization & Mapping:** Matplotlib, Folium (`OpenStreetMap` interactive maps)
- **Testing & Telemetry:** pytest, psutil

---

## 4. Project Architecture

```text
PathSnap-ML-Hackathon/
├── backend/
│   ├── app/
│   │   ├── data/
│   │   │   ├── loaders.py              # Strictly separates .track/.nodes/.arcs from .route
│   │   │   ├── preprocessing.py        # Per-trip WGS84 local tangent projection & kinematics
│   │   │   └── label_alignment.py      # Offline monotonic constrained DP alignment
│   │   ├── features/
│   │   │   ├── candidate_generation.py # Hybrid cKDTree + long-arc spatial retrieval (K=20, R=100m)
│   │   │   └── feature_engineering.py  # 26 leak-free float32 candidate features
│   │   ├── models/
│   │   │   └── baseline.py             # Nearest-road & SGDClassifier(log_loss) baselines
│   │   └── evaluation/
│   │       └── metrics.py              # Candidate, observation-ranking, and continuity metrics
│   └── requirements.txt
├── configs/
│   └── pipeline_config.json            # Central Phase 1 hyperparameters & split seed
├── scripts/
│   ├── verify_workspace.py             # Independent workspace & dataset integrity audit
│   ├── geo_utils.py                    # Phase 0 metric projection & geometry helpers
│   ├── inspect_dataset.py              # Phase 0 full audit across all 90 trips
│   ├── analyze_label_association.py    # Phase 0 DP alignment vs naive matching audit
│   ├── generate_eda_visualizations.py  # Static EDA plots & interactive Folium maps
│   ├── build_candidate_dataset.py      # Phase 1 candidate & Parquet dataset builder
│   ├── train_baseline.py               # Phase 1 SGD baseline trainer
│   └── evaluate_baseline.py            # Phase 1 split-wise & per-trip evaluator
├── tests/
│   └── test_phase1_pipeline.py         # Automated verification tests (5 tests)
├── artifacts/                          # Fitted baseline pipeline (.joblib) & metadata
├── results/                            # Audit CSVs, split manifest & baseline metrics JSON
├── visualizations/                     # EDA plots (.png) & interactive Folium maps (.html)
├── reports/
│   ├── dataset_audit.md                # Phase 0 audit report
│   └── phase1_report.md                # Phase 1 pipeline & baseline report
├── docs/reference/                     # Official hackathon guideline PDFs
├── frontend/                           # Reserved for future evaluator UI
├── requirements.txt                    # Root dependencies for Phase 0 & Phase 1
└── .gitignore                          # Excludes raw/processed data, ZIPs, caches & secrets
```

---

## 5. Current Implementation Status

### Completed
- **Phase 0 — Dataset Audit & Exploratory Data Analysis:**
  - Independent workspace verification (`80` train trips, `10` student-test trips, `227,577` total GPS observations).
  - Global metric spatial audit, timestamp-gap analysis (`96.682%` at `1.0 s`, `7,548` gaps `> 1 s` up to `153,064 s`), road-network connectivity analysis, and reference-route validity audit.
  - Generation of 4 multi-panel static plots and 5 interactive Folium maps.
- **Phase 1 — Candidate Generation, Label Alignment, Feature Engineering & Preliminary Baseline:**
  - Leak-free spatial candidate generation (`K = 20`, `R = 100 m`) validated with `0` missed candidates against exact brute-force search.
  - Offline monotonic constrained dynamic programming label association with `> 50 m` off-route invalidation.
  - 26 leak-free `float32` engineered features exported to per-trip Parquet partitions (stored locally, excluded from Git).
  - Trip-grouped `64 / 16` train/validation split (`GroupShuffleSplit`) plus separate `10`-trip `student_test` evaluation.
  - Preliminary `SGDClassifier (loss="log_loss")` baseline compared against a nearest-road-distance baseline, backed by 5 passing `pytest` unit tests.

### Planned / Not Yet Implemented
The following components are **not yet implemented** and are reserved for upcoming phases and team branches:
- Five-model cross-family experimentation and comparison.
- Bayesian hyperparameter optimization.
- HMM and Viterbi sequential route reconstruction.
- FastAPI backend inference and evaluation endpoints.
- Streamlit evaluator-facing frontend UI.
- Two-laptop distributed deployment configuration.

---

## 6. Dataset Format and Local Setup

> **Note:** Organiser-provided raw datasets (`data/raw/`), original archives (`source_zips/`), and the generated `143.46 MB` Parquet candidate partitions (`data/processed/`) are deliberately excluded from this Git repository. Obtain the official dataset archives separately before running pipeline scripts.

Each trip folder (`data/raw/train/<trip_id>/` or `data/raw/student_test/<trip_id>/`) contains four headerless whitespace-separated files:
- `<trip_id>.track`: `longitude`, `latitude`, `timestamp_s`
- `<trip_id>.nodes`: `longitude`, `latitude` (0-based `node_id`)
- `<trip_id>.arcs`: `from_node_id`, `to_node_id` (0-based `arc_id`)
- `<trip_id>.route`: chronological sequence of reference `arc_id` values

Place extracted trip folders under `data/raw/train/` (`80` trips) and `data/raw/student_test/` (`10` trips).

---

## 7. Candidate-Generation Methodology

Candidate generation (`backend/app/features/candidate_generation.py`) uses **only** `.track`, `.nodes`, and `.arcs` (never `.route`):
1. Coordinates are projected per trip into physical meters via a **WGS84 Local Tangent-Plane projection** (`LocalTangentProjection`) centered at the trajectory centroid, avoiding invalid degree-space Euclidean distances across global trip locations.
2. Road arcs are partitioned into **short arcs (`<= 300 m`)** indexed with a `scipy.spatial.cKDTree` over arc midpoints (`query_ball_point` with `r = R + 150 m = 250 m`) and **long arcs (`> 300 m`)** retrieved via exact segment bounding-box intersection.
3. Exact point-to-segment distances filter candidates to `R = 100.0 m` and retain the top `K = 20` closest arcs (`0` missed top-1/top-K candidates across `240` brute-force validation checks).
4. Valid candidate recall: **`99.64%`** (`train_fit`), **`99.85%`** (`train_val`), **`99.98%`** (`student_test`).

---

## 8. Label-Association Methodology

Because `.track` observations ($T$) and `.route` arcs ($M$) are not 1-to-1, offline label alignment (`backend/app/data/label_alignment.py`) uses **vectorized Monotonic Constrained Dynamic Programming**:
- Aligns observation $t \in [0..T-1]$ to route index $k(t) \in [0..M-1]$ enforcing $k(t) \ge k(t-1)$ with a mild forward-skip penalty (`step_penalty = 0.05`) so stationary GPS pauses remain anchored without forced transitions.
- Emission cost combines normalized point-to-segment distance ($\sigma_d = 15\text{ m}$) and speed-gated heading difference ($\sigma_\theta = 45^\circ$ when $v > 2.0\text{ m/s}$).
- Observations whose aligned arc distance exceeds `50.0 m` (e.g., off-route anomalies in `train/00000054` and `train/00000033`) are flagged `is_valid_label = False` and excluded from supervised classifier fitting without fabricating labels.

---

## 9. The 26 Engineered Features

All features in `backend/app/features/feature_engineering.py` are stored as `float32` and strictly exclude `trip_id`, `candidate_arc_id`, raw absolute `lon`/`lat`, future labels, and `.route` data:

| Category | Features |
| :--- | :--- |
| **Spatial & Geometric (7)** | `dist_to_seg_m`, `dist_to_start_m`, `dist_to_end_m`, `proj_t`, `proj_dist_from_endpoint_m`, `arc_length_m`, `log_arc_length_m` |
| **Kinematics & Temporal (9)** | `gps_speed_smooth_mps`, `gps_speed_prev_mps`, `dt_prev_s`, `log_dt_prev_s`, `step_dist_prev_m`, `is_moving`, `heading_diff_deg`, `cos_heading_diff`, `heading_turn_deg` |
| **Candidate-Relative & Continuity (7)** | `candidate_rank`, `dist_diff_from_best_m`, `dist_ratio_to_best`, `n_cands_in_radius`, `heading_rank_at_obs`, `prev_obs_dist_m`, `dist_change_from_prev_m` |
| **Road-Network Topology (3)** | `from_node_deg`, `to_node_deg`, `is_bidirectional` |

---

## 10. Trip-Grouped Validation Strategy

- `GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)` partitions the `80` training trips into:
  - **`train_fit` (`64` trips):** `1,939,133` valid candidate rows (`145,973` positives, `7.528%` positive rate) used for `StandardScaler` and classifier fitting.
  - **`train_val` (`16` trips):** `39,460` observations (`38,652` valid) held out with zero trip overlap.
  - **`student_test` (`10` trips):** `19,160` observations (`19,133` valid) kept strictly separate for development evaluation.

---

## 11. Preliminary Baseline Results

Candidate probability scores are converted into a predicted road-arc sequence via $\hat{a}_t = \arg\max_{c \in \mathcal{C}_t} P(y=1 \mid x_{t,c})$:

| Split | Model | Top-1 (Valid) | Top-5 (Valid) | Top-1 (All Obs) | Step Continuity | Cand Macro F1 | Log Loss | Infer Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`train_fit` (64)** | Nearest-Road | 51.69% | 97.42% | 45.08% | 77.46% | 0.5676 | 0.1867 | 0.015 s |
| **`train_fit` (64)** | **SGD Logistic** | **81.07%** | **98.42%** | **70.50%** | **90.42%** | **0.8039** | 0.1789 | 0.217 s |
| **`train_val` (16)** | Nearest-Road | 58.34% | 96.88% | 57.75% | 78.69% | 0.5611 | 0.1941 | 0.005 s |
| **`train_val` (16)** | **SGD Logistic** | **80.08%** | **98.06%** | **79.14%** | **91.90%** | **0.8021** | 0.1848 | 0.052 s |
| **`student_test` (10)** | Nearest-Road | 51.34% | 97.45% | 51.26% | 80.21% | 0.5591 | 0.1785 | 0.002 s |
| **`student_test` (10)** | **SGD Logistic** | **81.78%** | **98.61%** | **81.66%** | **92.15%** | **0.7996** | 0.1691 | 0.030 s |

---

## 12. Installation Instructions

```bash
git clone https://github.com/kushal13x/PathSnap-ML-Hackathon.git
cd PathSnap-ML-Hackathon

python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt
```

---

## 13. Reproduction Commands

Once raw trips are placed in `data/raw/{train,student_test}/`:

```bash
# Phase 0: Workspace check, dataset audit, label-association audit & EDA visuals
python scripts/verify_workspace.py
python -u scripts/inspect_dataset.py
python -u scripts/analyze_label_association.py
python -u scripts/generate_eda_visualizations.py

# Phase 1: Candidate Parquet generation, baseline training & evaluation
python -u scripts/build_candidate_dataset.py
python -u scripts/train_baseline.py
python -u scripts/evaluate_baseline.py
```

---

## 14. Automated Test Instructions

```bash
python -m pytest tests/test_phase1_pipeline.py -v
```

---

## 15. Known Limitations

- **Linear Model Capacity & Calibration:** `SGDClassifier` with `class_weight="balanced"` learns linear decision boundaries and shifts probabilities upward (`Precision ≈ 0.49`, `Recall ≈ 0.95` at `0.5`), though ranking via `argmax` remains strong (`80.08%` Top-1 on `train_val`).
- **Pointwise Discontinuities:** Independent per-observation classification yields `91.90%` step continuity (`75.38%` transition continuity on `train_val`); sequential HMM/Viterbi decoding is required to enforce network connectivity.
- **Imperfect Reference Routes:** Training trips such as `00000054` (`89.44%` invalid labels) and `00000033` (`59.05%` invalid labels) exhibit off-network/unmapped deviations relative to `.route`.

---

## 16. Planned Future Development

1. **Multi-Model Experimentation:** Train and compare at least 5 ML algorithms across diverse families (trees, boosting, ensembles, kernel/neural models) using trip-grouped cross-validation.
2. **Bayesian Hyperparameter Optimization:** Tune the top candidate model on training folds.
3. **HMM & Viterbi Route Reconstruction:** Combine classifier emission probabilities with road-network shortest-path/connectivity transition modeling and gap handling.
4. **FastAPI Backend & Streamlit Evaluator UI:** Build distributed two-laptop inference and interactive Folium route visualization.

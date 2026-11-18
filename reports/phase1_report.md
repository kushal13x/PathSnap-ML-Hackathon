# PathSnap — Phase 1 Report: Candidate Generation, Label Alignment & Baseline

## 1. Pipeline Architecture & Spatial Index Validation

- **Candidate Generation (`K = 20`, `R = 100.0 m`):** Implemented in [`backend/app/features/candidate_generation.py`](file:///c:/Users/Kushal/Desktop/ML%20Hackathon/backend/app/features/candidate_generation.py) using ONLY `.track`, `.nodes`, and `.arcs` (`.route` is never accessed).
- **Long-Segment Handling:** Partitioned road networks into short arcs (`<= 300 m`, indexed via `cKDTree` over midpoints with `ball_r = R + 150 m = 250 m`) and long arcs (`> 300 m`, exact segment bounding-box intersection).
- **Brute-Force Index Verification ([`results/spatial_index_validation.json`](file:///c:/Users/Kushal/Desktop/ML%20Hackathon/results/spatial_index_validation.json)):** Audited across 240 sampled observations spanning 6 trips (including `train/00000001` with `1,351` arcs > 300 m and `train/00000019` with `1,890` arcs > 300 m): **`0` missed Top-1, `0` missed Top-K, `0.0 m` max distance discrepancy**.

## 2. Offline Label Association & Difficult Trip Findings

Implemented via vectorized Monotonic Constrained DP in [`backend/app/data/label_alignment.py`](file:///c:/Users/Kushal/Desktop/ML%20Hackathon/backend/app/data/label_alignment.py). Observations with aligned distance `> 50.0 m` are flagged `is_valid_label = False` and excluded from supervised classifier fitting:

- **`train/00000054`:** `89.44%` invalid labels (`2,039 / 19,313` valid), `1,104` observations with 0 road candidates inside 100 m; valid candidate recall `92.64%`.
- **`train/00000033`:** `59.05%` invalid labels (`2,393 / 5,844` valid) due to off-route loops; valid candidate recall `87.76%`.
- **`train/00000019`:** `1.66%` invalid labels (`2,315 / 2,354` valid); `100.00%` valid candidate recall despite 3 route discontinuities.
- **`train/00000090`:** `0.00%` invalid labels (`536 / 536` valid); `100.00%` valid candidate recall; `153,064 s` stationary pause handled cleanly with `k(t) = k(t-1)` without forced transitions.
- **`train/00000002`:** `11.98%` invalid labels (`6,375 / 7,243` valid), `271` no-candidate observations; `100.00%` valid candidate recall.

## 3. Candidate Dataset & Trip-Grouped Validation Split

Generated per-trip Parquet partitions under `data/processed/candidates/` (`143.46 MB` total across `2,881,345` candidate rows, `12.66` candidates/obs average, `1,415` no-candidate observations).

| Partition | Trips | Total GPS Obs | Valid Obs (`<= 50m`) | Invalid % | Candidate Rows | Valid Cand Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`train_fit`** | 64 | 168,957 | 146,494 | 13.30% | 2,107,934 | **99.64%** |
| **`train_val`** | 16 | 39,460 | 38,652 | 2.05% | 497,979 | **99.85%** |
| **`student_test`** | 10 | 19,160 | 19,133 | 0.14% | 275,432 | **99.98%** |

**Engineered Features (26 `float32` leak-free features):** `dist_to_seg_m`, `dist_to_start_m`, `dist_to_end_m`, `proj_t`, `proj_dist_from_endpoint_m`, `arc_length_m`, `log_arc_length_m`, `gps_speed_smooth_mps`, `gps_speed_prev_mps`, `dt_prev_s`, `log_dt_prev_s`, `step_dist_prev_m`, `is_moving`, `heading_diff_deg`, `cos_heading_diff`, `heading_turn_deg`, `candidate_rank`, `dist_diff_from_best_m`, `dist_ratio_to_best`, `n_cands_in_radius`, `heading_rank_at_obs`, `prev_obs_dist_m`, `dist_change_from_prev_m`, `from_node_deg`, `to_node_deg`, `is_bidirectional`.

## 4. Preliminary Baseline Results

Trained `StandardScaler + SGDClassifier(loss='log_loss', class_weight='balanced')` strictly on `1,939,133` valid `train_fit` candidate rows (`145,973` positives, `7.528%` positive rate) in **`8.384 s`** (`628.0 MB` Peak RSS). Predictions are converted to a predicted road-arc sequence via $\hat{a}_t = \arg\max_{c \in \mathcal{C}_t} P(y=1 \mid x_{t,c})$:

| Split | Model | Top-1 (Valid) | Top-5 (Valid) | Top-1 (All Obs) | Step Continuity | Cand Macro F1 | Prec / Rec | Log Loss | Infer Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`train_fit`** (64) | Nearest-Road | 51.69% | 97.42% | 45.08% | 77.46% | 0.5676 | 0.4763 / 0.1073 | 0.1867 | 0.015 s |
| **`train_fit`** (64) | **SGD Logistic** | **81.07%** | **98.42%** | **70.50%** | **90.42%** | **0.8039** | 0.4922 / 0.9628 | 0.1789 | 0.217 s |
| **`train_val`** (16) | Nearest-Road | 58.34% | 96.88% | 57.75% | 78.69% | 0.5611 | 0.5215 / 0.0966 | 0.1941 | 0.005 s |
| **`train_val`** (16) | **SGD Logistic** | **80.08%** | **98.06%** | **79.14%** | **91.90%** | **0.8021** | 0.4937 / 0.9495 | 0.1848 | 0.052 s |
| **`student_test`** (10) | Nearest-Road | 51.34% | 97.45% | 51.26% | 80.21% | 0.5591 | 0.4602 / 0.0932 | 0.1785 | 0.002 s |
| **`student_test`** (10) | **SGD Logistic** | **81.78%** | **98.61%** | **81.66%** | **92.15%** | **0.7996** | 0.4802 / 0.9641 | 0.1691 | 0.030 s |

## 5. Automated Tests & Resource Usage

- `pytest tests/test_phase1_pipeline.py -v`: **5/5 passed** in `2.38 s`.
- Build runtime: `90.66 s` (`270.7 MB` Peak RSS).
- Training runtime: `8.384 s` (`628.0 MB` Peak RSS).
- Evaluation runtime: `9.71 s` (`369.65 MB` Peak RSS).

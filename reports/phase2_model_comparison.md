# PathSnap — Phase 2 Report: Five-Model Comparison & Evaluation

**Hardware Environment:** Intel Core i5-13450HX (16 logical CPUs), 15.69 GB usable RAM, NVIDIA GeForce RTX 3050 6 GB Laptop GPU (Driver 610.60), Windows 11 64-bit.

---

## 1. Experimental Setup & Leakage Prevention

- **Splits:** Strictly preserved `train_fit` (`64` trips, `1,939,133` valid candidate rows across `146,494` valid GPS observations), `train_val` (`16` trips, `38,652` valid observations), and `student_test` (`10` trips, `19,133` valid observations).
- **Features:** Same `26` `float32` leak-free features across all models.
- **Models Evaluated:**
  - *Heuristic Baseline:* Nearest-Road Distance (`1 / (1 + dist_to_seg_m)`).
  - *Mandatory 5 Models:* `SGDClassifier` (Model 1), `DecisionTreeClassifier` (Model 2), `RandomForestClassifier` (Model 3), `ExtraTreesClassifier` (Model 4), `HistGradientBoostingClassifier` (Model 5).
  - *Optional 6th Model (GPU):* `XGBClassifier(tree_method="hist", device="cuda")` (`xgboost 3.4.1`) verified on RTX 3050.

---

## 2. Held-Out Validation (`train_val`, 16 Trips) & Student-Testing (`student_test`, 10 Trips)

| Model | Train Time | Peak RAM / VRAM | Artifact Size | `train_val` Top-1 | `train_val` Top-5 | `train_val` Macro F1 | `train_val` ROC-AUC / PR-AUC | `train_val` LogLoss | `train_val` Step Cont. | `student_test` Top-1 | `student_test` Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nearest-Road Distance** | — | — | — | 58.34% | 96.88% | 0.5611 | 0.9231 / 0.4325 | 0.1941 | 78.69% | 51.34% | 0.5591 |
| **1. SGD Logistic** | 8.38 s | 628.0 MB / 0 | 0.002 MB | 80.08% | 98.06% | 0.8021 | 0.9804 / 0.8104 | 0.1848 | 91.90% | 81.78% | 0.7996 |
| **2. Decision Tree** | 62.81 s | 883.9 MB / 0 | 0.12 MB | 81.61% | 97.44% | 0.8357 | 0.9793 / 0.8839 | 0.1534 | 91.03% | 82.48% | 0.8414 |
| **3. Random Forest** | 286.30 s | 2,298.9 MB / 0 | 25.72 MB | 81.61% | 98.30% | 0.8691 | 0.9882 / 0.9046 | 0.1128 | 92.21% | 83.24% | 0.8776 |
| **4. Extra Trees** | 106.20 s | 2,318.4 MB / 0 | 26.56 MB | 80.80% | 98.16% | 0.8430 | 0.9865 / 0.8930 | 0.1422 | 91.33% | 82.04% | 0.8492 |
| **5. HistGradientBoosting** | 36.55 s | 1,753.9 MB / 0 | **0.50 MB** | **82.27%** | 98.25% | **0.9000** | **0.9887 / 0.9081** | **0.0687** | 92.38% | 83.97% | 0.9106 |
| **6. XGBoost (CUDA GPU)** | **5.25 s** | 1,154.4 MB / **277 MiB** | **0.28 MB** | 82.10% | **98.35%** | 0.8995 | 0.9886 / 0.9062 | 0.0691 | **92.52%** | **84.15%** | **0.9118** |

---

## 3. 3-Fold GroupKFold Cross-Validation (80 Training Trips, `obs_fraction=0.35`)

Using reproducible observation-level sampling (preserving every candidate per selected GPS observation; `~32k–50k` valid GPS observations per fold fit, evaluated on full held-out fold trips):

| Model | CV Top-1 (%) | CV Top-5 (%) | CV Step Cont. (%) | CV Macro F1 | CV ROC-AUC | CV PR-AUC | CV LogLoss | Mean Fit Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SGD Logistic** | 81.49 ± 3.23 | 98.17 | 90.60 | 0.8013 ± 0.0149 | 0.9824 | 0.8136 | 0.1823 | 3.19 s |
| **Decision Tree** | 81.79 ± 3.44 | 96.86 | 89.67 | 0.8402 ± 0.0202 | 0.9739 | 0.8816 | 0.1537 | 15.53 s |
| **Random Forest** | 82.25 ± 3.21 | 98.32 | 91.29 | 0.8787 ± 0.0156 | 0.9893 | 0.9110 | 0.1003 | 62.51 s |
| **Extra Trees** | 81.43 ± 3.07 | 98.30 | 90.44 | 0.8504 ± 0.0158 | 0.9881 | 0.9004 | 0.1324 | 19.54 s |
| **HistGradientBoosting** | **82.81 ± 3.42** | **98.37** | **91.60** | **0.9045 ± 0.0186** | **0.9897** | **0.9136** | **0.0628** | 12.83 s |

---

## 4. Technical Comparison & Best-Model Selection

1. **Best Overall Model (`HistGradientBoosting` among mandatory CPU models; `XGBoost GPU` virtually tied):**
   - `HistGradientBoosting` leads all 5 mandatory models on `train_val` Top-1 (`82.27%`), Macro F1 (`0.9000`), ROC-AUC (`0.9887`), PR-AUC (`0.9081`), Log Loss (`0.0687`), and 3-fold GroupKFold CV (`82.81%` Top-1 / `0.9045` Macro F1), with a compact `0.50 MB` artifact and zero GPU runtime dependency for two-laptop deployment.
   - `XGBoost (CUDA GPU)` matches `HistGradientBoosting` within ~`0.17%` on `train_val` (`82.10%` Top-1 / `0.8995` Macro F1), leads on `student_test` (`84.15%` Top-1 / `0.9118` Macro F1), and trains **7× faster (`5.25 s`)** with **5–6× faster inference** (`0.174 s` on `student_test`).
2. **Random Forest vs. Extra Trees:** Random Forest (`81.61%` Top-1, `0.8691` Macro F1, `0.1128` Log Loss) outperforms Extra Trees (`80.80%` Top-1, `0.8430` Macro F1, `0.1422` Log Loss) because exact optimal splits capture sharp distance/heading thresholds better than random split thresholds, though Extra Trees trains `2.7×` faster (`106.2 s` vs `286.3 s`).
3. **Overfitting Observation (Decision Tree vs. Ensembles):** Single Decision Tree drops from `83.34%` Top-1 / `0.1181` LogLoss on `train_fit` to `81.61%` / `0.1534` on `train_val` and `97.44%` Top-5 (worst Top-5 among ML models), confirming higher leaf probability variance compared to bagging/boosting ensembles.
4. **Why Pointwise Classifiers Plateau Near ~82–84% Top-1:** Classifier-only step continuity tops out at `92.38–93.26%`; remaining ambiguities (parallel ramps, tight urban junctions, GPS jitter) require sequential **HMM/Viterbi** decoding over well-calibrated emission probabilities (`LogLoss ≈ 0.0687` from `HistGradientBoosting` / `XGBoost`), which will be addressed after Phase 3 Bayesian optimization.

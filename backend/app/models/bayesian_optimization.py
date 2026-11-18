"""Phase 3: Bayesian Hyperparameter Optimization for HistGradientBoostingClassifier.

Uses Optuna TPE sampler strictly on train_fit (64 trips) via 3-fold GroupKFold
so that train_val (16 trips) and student_test (10 trips) remain completely unseen
during hyperparameter selection.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from backend.app.features.feature_engineering import FEATURE_COLUMNS


def prepare_inner_cv_folds(
    cand_train_dir: Path,
    train_fit_ids: list[str],
    n_splits: int = 3,
    train_obs_fraction: float = 0.45,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """Preload train_fit trips into memory and build 3-fold GroupKFold splits.

    Training folds use deterministic observation-level sampling (preserving all
    candidates per selected observation); inner validation folds use 100% of
    valid observations in the held-out fold trips for exact ranking evaluation.
    """
    rng = np.random.default_rng(random_state)
    trip_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for tid in train_fit_ids:
        df = pd.read_parquet(
            cand_train_dir / f"{tid}.parquet",
            columns=FEATURE_COLUMNS + ["observation_id", "target", "is_valid_label"],
        )
        df_v = df.loc[df["is_valid_label"].to_numpy(dtype=bool)]
        if len(df_v) > 0:
            trip_cache[tid] = (
                df_v[FEATURE_COLUMNS].to_numpy(dtype=np.float32),
                df_v["target"].to_numpy(dtype=np.int8),
                df_v["observation_id"].to_numpy(dtype=np.int32),
            )
        del df, df_v

    tids_arr = np.array(sorted(trip_cache.keys()))
    gkf = GroupKFold(n_splits=n_splits)
    folds = []

    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(tids_arr, groups=tids_arr), start=1):
        tr_tids = tids_arr[tr_idx]
        va_tids = tids_arr[va_idx]

        x_tr_list, y_tr_list = [], []
        for tid in tr_tids:
            X_t, y_t, obs_t = trip_cache[tid]
            if train_obs_fraction < 1.0:
                uniq_obs = np.unique(obs_t)
                n_keep = max(1, int(round(len(uniq_obs) * train_obs_fraction)))
                chosen = rng.choice(uniq_obs, size=n_keep, replace=False)
                mask = np.isin(obs_t, chosen)
                x_tr_list.append(X_t[mask])
                y_tr_list.append(y_t[mask])
            else:
                x_tr_list.append(X_t)
                y_tr_list.append(y_t)

        x_va_list, y_va_list = [], []
        obs_slices: list[tuple[int, int]] = []
        offset = 0
        for tid in va_tids:
            X_t, y_t, obs_t = trip_cache[tid]
            x_va_list.append(X_t)
            y_va_list.append(y_t)
            breaks = np.flatnonzero(np.diff(obs_t) != 0) + 1
            starts = np.concatenate([[0], breaks]) + offset
            ends = np.concatenate([breaks, [len(obs_t)]]) + offset
            for s, e in zip(starts, ends):
                obs_slices.append((int(s), int(e)))
            offset += len(obs_t)

        folds.append(
            {
                "fold": fold_idx,
                "X_tr": np.vstack(x_tr_list),
                "y_tr": np.concatenate(y_tr_list),
                "X_va": np.vstack(x_va_list),
                "y_va": np.concatenate(y_va_list),
                "va_obs_slices": np.asarray(obs_slices, dtype=np.int32),
            }
        )

    del trip_cache
    gc.collect()
    return folds


def compute_fast_obs_top1_and_top5(
    y_true: np.ndarray,
    probs: np.ndarray,
    obs_slices: np.ndarray,
) -> tuple[float, float]:
    """Compute exact observation-level Top-1 and Top-5 accuracy from precomputed slices."""
    top1 = 0
    top5 = 0
    n_obs = len(obs_slices)
    for s, e in obs_slices:
        sub_p = probs[s:e]
        sub_y = y_true[s:e]
        # Fast argmax for Top-1
        best_idx = int(np.argmax(sub_p))
        if sub_y[best_idx] == 1:
            top1 += 1
            top5 += 1
        elif e - s > 1:
            k = min(5, e - s)
            top_k_idx = np.argpartition(-sub_p, k - 1)[:k]
            if np.any(sub_y[top_k_idx] == 1):
                top5 += 1
    return (top1 / max(1, n_obs) * 100.0), (top5 / max(1, n_obs) * 100.0)


def run_hgb_optuna_search(
    folds: list[dict[str, Any]],
    n_trials: int = 12,
    timeout_s: float = 360.0,
    random_state: int = 42,
) -> tuple[optuna.Study, pd.DataFrame]:
    """Run Optuna TPE Bayesian optimization over HistGradientBoostingClassifier."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    trial_records: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        t_start = time.perf_counter()
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.20, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 275, step=25),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 31, 127),
            "max_depth": trial.suggest_categorical("max_depth", [None, 8, 12, 16]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 15, 80),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-2, 10.0, log=True),
            "validation_fraction": trial.suggest_float("validation_fraction", 0.1, 0.2),
            "early_stopping": True,
            "n_iter_no_change": 15,
            "random_state": random_state,
        }

        fold_top1, fold_top5, fold_f1, fold_ll, fold_auc = [], [], [], [], []
        for f in folds:
            clf = HistGradientBoostingClassifier(**params)
            clf.fit(f["X_tr"], f["y_tr"])
            probs = clf.predict_proba(f["X_va"])[:, 1]

            t1, t5 = compute_fast_obs_top1_and_top5(f["y_va"], probs, f["va_obs_slices"])
            y_pred = (probs >= 0.5).astype(np.int8)
            mf1 = float(f1_score(f["y_va"], y_pred, average="macro", zero_division=0))
            ll = float(log_loss(f["y_va"], np.clip(probs, 1e-7, 1.0 - 1e-7), labels=[0, 1]))
            auc = float(roc_auc_score(f["y_va"], probs))

            fold_top1.append(t1)
            fold_top5.append(t5)
            fold_f1.append(mf1)
            fold_ll.append(ll)
            fold_auc.append(auc)
            del clf

        mean_top1 = float(np.mean(fold_top1))
        mean_top5 = float(np.mean(fold_top5))
        mean_f1 = float(np.mean(fold_f1))
        mean_ll = float(np.mean(fold_ll))
        mean_auc = float(np.mean(fold_auc))
        elapsed = round(time.perf_counter() - t_start, 2)

        trial.set_user_attr("cv_top5", round(mean_top5, 4))
        trial.set_user_attr("cv_macro_f1", round(mean_f1, 4))
        trial.set_user_attr("cv_log_loss", round(mean_ll, 4))
        trial.set_user_attr("cv_roc_auc", round(mean_auc, 4))

        trial_records.append(
            {
                "trial_number": trial.number,
                **{k: v for k, v in params.items() if k not in {"early_stopping", "n_iter_no_change", "random_state"}},
                "cv_top1_pct": round(mean_top1, 4),
                "cv_top1_std": round(float(np.std(fold_top1, ddof=1)), 4),
                "cv_top5_pct": round(mean_top5, 4),
                "cv_macro_f1": round(mean_f1, 4),
                "cv_log_loss": round(mean_ll, 4),
                "cv_roc_auc": round(mean_auc, 4),
                "elapsed_s": elapsed,
            }
        )
        print(
            f"  [Trial {trial.number:02d}] Top1={mean_top1:.3f}% | Top5={mean_top5:.2f}% | "
            f"MacroF1={mean_f1:.4f} | LogLoss={mean_ll:.4f} | time={elapsed}s | "
            f"lr={params['learning_rate']:.4f}, leaves={params['max_leaf_nodes']}, iter={params['max_iter']}",
            flush=True,
        )
        return mean_top1

    study.optimize(objective, n_trials=n_trials, timeout=timeout_s)
    return study, pd.DataFrame(trial_records)


def compute_calibration_stats(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    """Compute Brier score and Expected Calibration Error (ECE)."""
    brier = float(brier_score_loss(y_true, probs))
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for b0, b1 in zip(bins[:-1], bins[1:]):
        mask = (probs >= b0) & (probs < b1 if b1 < 1.0 else probs <= b1)
        if np.any(mask):
            acc = float(np.mean(y_true[mask]))
            conf = float(np.mean(probs[mask]))
            ece += (np.sum(mask) / n) * abs(acc - conf)
    return {"brier_score": round(brier, 5), "ece_10bin": round(float(ece), 5)}

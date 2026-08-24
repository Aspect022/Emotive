"""
src/training/splits.py
Three-tier train/val/test splitting strategy for CogProfile-Net.

Tier 1 (Leaky / Window-level)   - compare only to existing baseline
Tier 2 (Clean / Trial-level)    - preferred for all new model results
Tier 3 (Strict / Subject-level) - leave-N-subjects-out generalization test
"""

from __future__ import annotations
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def tier2_trial_split(
    trial_ids: np.ndarray,
    y_trial: np.ndarray,
    window_to_trial: np.ndarray,
    y_window: np.ndarray,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    seed: int = 42,
) -> dict:
    """
    Tier 2 — split at TRIAL level, preventing adjacent sub-windows from
    leaking across train/test.

    Args:
        trial_ids:       unique trial IDs (N_trials,)
        y_trial:         class label per trial (N_trials,)
        window_to_trial: which trial each window belongs to (N_windows,)
        y_window:        class label per window (N_windows,)

    Returns dict with train_idx / val_idx / test_idx into the WINDOW array.
    """
    rng = np.random.default_rng(seed)
    unique_trials = np.unique(trial_ids)

    # Stratified shuffle split at trial level
    from sklearn.model_selection import StratifiedShuffleSplit
    y_per_trial = np.array([y_trial[np.where(trial_ids == t)[0][0]] for t in unique_trials])

    test_frac = 1.0 - train_frac - val_frac
    sss_outer = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    train_val_idx, test_trial_idx = next(sss_outer.split(unique_trials, y_per_trial))
    train_val_trials = unique_trials[train_val_idx]
    test_trials      = unique_trials[test_trial_idx]

    y_tv = y_per_trial[train_val_idx]
    val_share = val_frac / (train_frac + val_frac)
    sss_inner = StratifiedShuffleSplit(n_splits=1, test_size=val_share, random_state=seed)
    train_idx_tv, val_idx_tv = next(sss_inner.split(train_val_trials, y_tv))
    train_trials = train_val_trials[train_idx_tv]
    val_trials   = train_val_trials[val_idx_tv]

    def trial_to_window_idx(t_ids):
        mask = np.isin(window_to_trial, t_ids)
        return np.where(mask)[0]

    return {
        "tier": "trial_split",
        "train_idx": trial_to_window_idx(train_trials),
        "val_idx":   trial_to_window_idx(val_trials),
        "test_idx":  trial_to_window_idx(test_trials),
        "n_train_trials": len(train_trials),
        "n_val_trials":   len(val_trials),
        "n_test_trials":  len(test_trials),
    }


def tier3_subject_split(
    participant_ids: np.ndarray,
    y_window: np.ndarray,
    n_test_subjects: int = 3,
    n_val_subjects: int = 2,
    seed: int = 42,
) -> dict:
    """
    Tier 3 — leave-N-subjects-out. Strictly disjoint subject sets.
    No subject appears in more than one split.
    """
    rng = np.random.default_rng(seed)
    subjects = np.unique(participant_ids)
    rng.shuffle(subjects)

    test_subjs = subjects[:n_test_subjects]
    val_subjs  = subjects[n_test_subjects: n_test_subjects + n_val_subjects]
    train_subjs = subjects[n_test_subjects + n_val_subjects:]

    def subj_mask(s_list):
        return np.where(np.isin(participant_ids, s_list))[0]

    return {
        "tier": "subject_split",
        "train_idx":     subj_mask(train_subjs),
        "val_idx":       subj_mask(val_subjs),
        "test_idx":      subj_mask(test_subjs),
        "train_subjects": train_subjs.tolist(),
        "val_subjects":   val_subjs.tolist(),
        "test_subjects":  test_subjs.tolist(),
    }


def tier1_window_split(
    y_window: np.ndarray,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    seed: int = 42,
) -> dict:
    """
    Tier 1 — window-level random split. LEAKY but matches original baseline.
    Use ONLY to compare against the 73.34% LightGBM baseline.
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    N = len(y_window)
    idx = np.arange(N)
    test_frac = 1.0 - train_frac - val_frac

    sss_outer = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    tv_idx, test_idx = next(sss_outer.split(idx, y_window))

    y_tv = y_window[tv_idx]
    val_share = val_frac / (train_frac + val_frac)
    sss_inner = StratifiedShuffleSplit(n_splits=1, test_size=val_share, random_state=seed)
    train_sub, val_sub = next(sss_inner.split(tv_idx, y_tv))

    return {
        "tier": "window_split",
        "train_idx": tv_idx[train_sub],
        "val_idx":   tv_idx[val_sub],
        "test_idx":  test_idx,
    }

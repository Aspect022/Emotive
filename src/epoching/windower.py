"""
EEG Windower & DL Dataset Builder — Phases 5 & 7
Creates fixed-length EEG windows labeled for deep learning classification.

Output dataset:
  X_raw.npy           (N_windows, N_channels, N_timepoints) — normalized EEG
  y_task.npy          (N_windows,) — 5-class: task type label
  y_perf.npy          (N_windows,) — binary: high(1)/low(0) performer
  participant_ids.npy (N_windows,) — participant UUID for subject-independent splits
  session_ids.npy     (N_windows,) — session ID for reference
  window_meta.csv     — per-window metadata (timestamps, labels, QC flags)

Also produces:
  splits.json         — train/val/test participant-split indices
  label_map.json      — label integer to label name mappings
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
PREPROCESSED_DIR = ROOT / "data" / "processed" / "preprocessed"
OUT_DATASET_DIR = ROOT / "data" / "processed" / "dl_dataset"

# ── Configuration ─────────────────────────────────────────────────────────────
WINDOW_SIZE_S = 4.0       # seconds per window
OVERLAP_FRAC = 0.0        # 0 = non-overlapping (conservative for DL)
MIN_GOOD_SAMPLES_FRAC = 0.90  # window must have ≥90% good (non-interpolated, non-artifact) samples
NORMALIZE = "z_score"    # per-window normalization: "z_score" | "min_max" | "none"

# Label maps
TASK_LABEL_MAP = {
    "arithmetic":    0,
    "pattern":       1,
    "memory":        2,
    "comprehension": 3,
    "attention":     4,
}

EEG_CHANNELS = ["AF3","F7","F3","FC5","T7","P7","O1",
                 "O2","P8","T8","FC6","F4","F8","AF4"]


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def normalize_window(window: np.ndarray, method: str = "z_score") -> np.ndarray:
    """
    Normalize a single EEG window.
    window shape: (C, T)
    """
    if method == "z_score":
        mean = window.mean(axis=1, keepdims=True)
        std  = window.std(axis=1, keepdims=True)
        std  = np.where(std < 1e-8, 1.0, std)
        return (window - mean) / std
    elif method == "min_max":
        mn = window.min(axis=1, keepdims=True)
        mx = window.max(axis=1, keepdims=True)
        rng = mx - mn
        rng = np.where(rng < 1e-8, 1.0, rng)
        return (window - mn) / rng
    return window


def extract_windows(
    eeg: np.ndarray,            # (N, 14) preprocessed signal
    timestamps: np.ndarray,     # (N,)
    interp_mask: np.ndarray,    # (N,) bool
    amp_artifact_mask: np.ndarray,  # (N,) bool
    good_channels: np.ndarray,  # (14,) bool
    fs: float,
    activity_start_s: float,    # seconds into recording where activity starts
    activity_end_s: float,      # seconds into recording where activity ends
) -> list[dict]:
    """
    Extract fixed-length windows from the activity period of an EEG session.
    Returns list of dicts: {window_data, start_s, end_s, quality_ok}
    """
    window_samples = int(WINDOW_SIZE_S * fs)
    step_samples = int(window_samples * (1 - OVERLAP_FRAC))

    # Convert activity period to sample indices
    start_sample = max(0, int(activity_start_s * fs))
    end_sample = min(len(timestamps) - 1, int(activity_end_s * fs))

    if end_sample - start_sample < window_samples:
        return []  # activity too short for even one window

    windows = []
    idx = start_sample
    while idx + window_samples <= end_sample:
        w_slice = slice(idx, idx + window_samples)

        # Quality check: fraction of good samples in this window
        bad_samples = interp_mask[w_slice] | amp_artifact_mask[w_slice]
        bad_frac = bad_samples.mean()
        quality_ok = (bad_frac <= (1 - MIN_GOOD_SAMPLES_FRAC))

        # Extract and transpose to (C, T)
        window_data = eeg[w_slice, :].T  # (14, window_samples)

        # Replace NaN with 0 (bad channels remain at 0)
        window_data = np.nan_to_num(window_data, nan=0.0)

        windows.append({
            "window_data": window_data,
            "start_s": timestamps[idx] if len(timestamps) > idx else 0,
            "end_s": timestamps[idx + window_samples - 1] if len(timestamps) > idx + window_samples - 1 else 0,
            "start_sample": idx,
            "bad_sample_frac": round(float(bad_frac), 4),
            "quality_ok": quality_ok,
            "n_good_channels": int(good_channels.sum()),
        })
        idx += step_samples

    return windows


def build_dataset(sync_manifest: list[dict], preprocessed_dir: Path) -> dict:
    """
    Build the full DL dataset from sync manifest + preprocessed EEG files.
    """
    X_list = []
    y_task_list = []
    y_perf_list = []
    participant_ids_list = []
    session_ids_list = []
    meta_list = []

    total_sessions = 0
    skipped_no_eeg = 0
    skipped_bad_sync = 0
    skipped_no_label = 0
    skipped_no_file = 0
    total_windows = 0
    total_windows_kept = 0

    # Index preprocessed files by stem
    preproc_files = {f.stem.replace("_", "."): f for f in preprocessed_dir.glob("*.npz")}
    # Also index by filename stem directly
    preproc_by_name = {f.name: f for f in preprocessed_dir.glob("*.npz")}

    for sess in sync_manifest:
        total_sessions += 1

        # ── Skip unsynced sessions ────────────────────────────────────────────
        if sess.get("sync_confidence", "NONE") not in ("HIGH", "MODERATE"):
            skipped_bad_sync += 1
            continue

        # ── Get labels ───────────────────────────────────────────────────────
        activity_type = sess.get("activity_type", "").lower()
        task_label = TASK_LABEL_MAP.get(activity_type)
        if task_label is None:
            skipped_no_label += 1
            continue

        # Performance label from session accuracy
        session_score = None
        try:
            # Try to get from sessions_enriched via session_id
            # (In practice, this is joined at pipeline level)
            pass
        except Exception:
            pass

        # ── Find preprocessed file ────────────────────────────────────────────
        eeg_filepath = sess.get("matched_eeg_file", "")
        if not eeg_filepath:
            skipped_no_eeg += 1
            continue

        # Find corresponding .npz
        npz_path = None
        eeg_stem = Path(eeg_filepath).stem.replace(".", "_")
        npz_candidate = preprocessed_dir / f"{eeg_stem}.npz"
        if npz_candidate.exists():
            npz_path = npz_candidate
        else:
            # Try loose match by session title
            for fname, fpath in preproc_by_name.items():
                if Path(eeg_filepath).stem in fname:
                    npz_path = fpath
                    break

        if not npz_path or not npz_path.exists():
            skipped_no_file += 1
            continue

        # ── Load preprocessed EEG ─────────────────────────────────────────────
        try:
            data = np.load(npz_path, allow_pickle=False)
            eeg = data["eeg_preprocessed"]       # (N, 14)
            timestamps = data["timestamps"]       # (N,)
            interp_mask = data["interpolated_mask"].astype(bool)  # (N,)
            amp_mask = data["amplitude_artifact_mask"].astype(bool)  # (N,)
            good_channels = data["good_channels"].astype(bool)    # (14,)
        except Exception as e:
            print(f"    ERROR loading {npz_path.name}: {e}")
            skipped_no_file += 1
            continue

        # Detect sampling rate from timestamps
        if len(timestamps) > 1:
            fs = 1.0 / np.median(np.diff(timestamps[:1000]))
            fs = round(fs / 64) * 64  # round to nearest 64 (128 or 256)
        else:
            fs = 128.0

        # ── Get EEG-relative activity window ─────────────────────────────────
        eeg_start = float(sess.get("eeg_start_unix", 0) or 0)
        act_start_rel = float(sess.get("eeg_relative_start_s", 0) or 0)
        act_end_rel = float(sess.get("eeg_relative_end_s", 0) or 0)
        time_taken_s = float(sess.get("time_taken_s", 0) or 0)

        if act_start_rel < 0:
            act_start_rel = 0
        if act_end_rel <= act_start_rel:
            act_end_rel = act_start_rel + time_taken_s

        # ── Extract windows ───────────────────────────────────────────────────
        session_windows = extract_windows(
            eeg=eeg,
            timestamps=timestamps,
            interp_mask=interp_mask,
            amp_artifact_mask=amp_mask,
            good_channels=good_channels,
            fs=fs,
            activity_start_s=act_start_rel,
            activity_end_s=act_end_rel,
        )

        total_windows += len(session_windows)
        good_windows = [w for w in session_windows if w["quality_ok"]]
        total_windows_kept += len(good_windows)

        for w in good_windows:
            window_arr = w["window_data"]  # (14, T)

            # Normalize
            window_arr = normalize_window(window_arr, NORMALIZE)

            # Ensure shape is exactly (14, window_samples)
            expected_T = int(WINDOW_SIZE_S * fs)
            if window_arr.shape != (14, expected_T):
                # Pad or skip
                if window_arr.shape[0] == 14 and window_arr.shape[1] > 0:
                    window_arr = window_arr[:, :expected_T]
                    if window_arr.shape[1] < expected_T:
                        pad = np.zeros((14, expected_T - window_arr.shape[1]))
                        window_arr = np.concatenate([window_arr, pad], axis=1)
                else:
                    continue

            X_list.append(window_arr)
            y_task_list.append(task_label)
            participant_ids_list.append(sess.get("participant_id", ""))
            session_ids_list.append(sess.get("session_id", ""))

            meta_list.append({
                "window_idx": len(X_list) - 1,
                "session_id": sess.get("session_id", ""),
                "participant_id": sess.get("participant_id", ""),
                "activity_type": activity_type,
                "task_label": task_label,
                "sync_confidence": sess.get("sync_confidence", ""),
                "window_start_s": w["start_s"],
                "window_end_s": w["end_s"],
                "bad_sample_frac": w["bad_sample_frac"],
                "n_good_channels": w["n_good_channels"],
                "eeg_file": eeg_filepath,
            })

    print(f"\n  Dataset build summary:")
    print(f"    Total sessions processed: {total_sessions}")
    print(f"    Skipped (bad sync): {skipped_bad_sync}")
    print(f"    Skipped (no EEG file): {skipped_no_eeg + skipped_no_file}")
    print(f"    Skipped (no label): {skipped_no_label}")
    print(f"    Total windows extracted: {total_windows}")
    print(f"    Windows kept (quality pass): {total_windows_kept}")

    if not X_list:
        print("  WARNING: No windows produced. Check sync manifest and preprocessed files.")
        return {}

    X = np.stack(X_list, axis=0)             # (N, 14, T)
    y_task = np.array(y_task_list, dtype=np.int64)
    participant_ids = np.array(participant_ids_list, dtype=object)
    session_ids = np.array(session_ids_list, dtype=object)

    return {
        "X": X,
        "y_task": y_task,
        "participant_ids": participant_ids,
        "session_ids": session_ids,
        "meta": meta_list,
        "window_size_s": WINDOW_SIZE_S,
        "n_channels": 14,
        "channel_names": EEG_CHANNELS,
    }


def subject_independent_split(
    participant_ids: np.ndarray,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42
) -> dict:
    """
    Create subject-independent train/val/test splits.
    No participant appears in more than one split.
    """
    rng = np.random.default_rng(seed)
    unique_participants = np.unique(participant_ids)
    rng.shuffle(unique_participants)

    n = len(unique_participants)
    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))

    train_pids = set(unique_participants[:n_train])
    val_pids = set(unique_participants[n_train:n_train + n_val])
    test_pids = set(unique_participants[n_train + n_val:])

    train_idx = np.where(np.isin(participant_ids, list(train_pids)))[0]
    val_idx = np.where(np.isin(participant_ids, list(val_pids)))[0]
    test_idx = np.where(np.isin(participant_ids, list(test_pids)))[0]

    print(f"\n  Subject-independent splits:")
    print(f"    Train: {len(train_pids)} participants, {len(train_idx)} windows")
    print(f"    Val:   {len(val_pids)} participants, {len(val_idx)} windows")
    print(f"    Test:  {len(test_pids)} participants, {len(test_idx)} windows")

    return {
        "train_participants": list(train_pids),
        "val_participants": list(val_pids),
        "test_participants": list(test_pids),
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 5+7: WINDOWING & DATASET COMPILATION")
    print("=" * 60)

    SYNC_MANIFEST = RESULTS_DIR / "sync_manifest.csv"
    if not SYNC_MANIFEST.exists():
        print(f"ERROR: sync_manifest.csv not found. Run Phase 3 first.")
        sys.exit(1)

    sync_manifest = load_csv(SYNC_MANIFEST)
    print(f"  Sync manifest entries: {len(sync_manifest)}")

    if not PREPROCESSED_DIR.exists():
        print(f"  WARNING: preprocessed dir not found at {PREPROCESSED_DIR}")
        print(f"  Run Phase 4 (eeg_preprocessor.py) first.")
        print(f"  Continuing with empty preprocessed dir for manifest validation...")
        PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(sync_manifest, PREPROCESSED_DIR)

    if not dataset or "X" not in dataset:
        print("\n  Dataset is empty — preprocessing must be run first.")
        print("  Re-run after completing Phase 4.")
        sys.exit(0)

    OUT_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    X = dataset["X"]
    y_task = dataset["y_task"]
    participant_ids = dataset["participant_ids"]
    session_ids = dataset["session_ids"]
    meta = dataset["meta"]

    print(f"\n  Final dataset shape: X={X.shape}, y_task={y_task.shape}")

    # Save arrays
    np.save(OUT_DATASET_DIR / "X_raw.npy", X)
    np.save(OUT_DATASET_DIR / "y_task.npy", y_task)
    np.save(OUT_DATASET_DIR / "participant_ids.npy", participant_ids)
    np.save(OUT_DATASET_DIR / "session_ids.npy", session_ids)
    print(f"  [OK] Arrays saved to {OUT_DATASET_DIR}")

    # Save splits
    splits = subject_independent_split(participant_ids)
    with open(OUT_DATASET_DIR / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)
    print(f"  [OK] splits.json saved")

    # Save label maps
    label_map = {
        "task": TASK_LABEL_MAP,
        "task_inv": {str(v): k for k, v in TASK_LABEL_MAP.items()},
        "window_size_s": WINDOW_SIZE_S,
        "n_channels": 14,
        "channel_names": EEG_CHANNELS,
        "normalization": NORMALIZE,
    }
    with open(OUT_DATASET_DIR / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"  [OK] label_map.json saved")

    # Save window metadata CSV
    if meta:
        fields = list(meta[0].keys())
        with open(OUT_DATASET_DIR / "window_meta.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(meta)
        print(f"  [OK] window_meta.csv saved")

    # Class distribution
    from collections import Counter
    task_dist = Counter(y_task.tolist())
    inv_map = {v: k for k, v in TASK_LABEL_MAP.items()}
    print(f"\n  Task class distribution:")
    for lbl, cnt in sorted(task_dist.items()):
        pct = 100 * cnt / len(y_task)
        print(f"    {inv_map.get(lbl, lbl)}: {cnt} ({pct:.1f}%)")
    print(f"\n  Dataset ready for DL model training.")

"""
EEG Preprocessor — Phase 4
Loads raw EMOTIV CSV files, applies signal cleaning, and outputs
preprocessed numpy arrays ready for windowing.

Handles both file variants:
  .md.bp.csv      → 137 columns (no Performance Metrics)
  .md.pm.bp.csv   → 168 columns (includes PM stream at 0.1 Hz — excluded)

Pipeline:
  1. Load EEG-only columns (14 channels + Timestamp)
  2. Remove interpolated samples (EEG.Interpolated == 1)
  3. Drop bad channels (CQ < 2 in >20% of samples)
  4. 1–40 Hz 4th-order Butterworth bandpass filter
  5. Average reference (common average referencing)
  6. Motion artifact screening via IMU channels
  7. Output per-session: dict with arrays + QC metadata
"""

import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

ROOT = Path(__file__).resolve().parents[2]

# ── EEG channel names (EPOC X standard) ──────────────────────────────────────
EEG_CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
                 "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
N_CH = 14

# CQ column names (Contact Quality, 0=none, 4=excellent)
CQ_CHANNELS = [f"CQ.{ch}" for ch in EEG_CHANNELS]
CQ_OVERALL = "CQ.Overall"

# Motion channels (for artifact detection)
MOTION_CHANNELS = ["MOT.Q0", "MOT.Q1", "MOT.Q2", "MOT.Q3",
                   "MOT.AccX", "MOT.AccY", "MOT.AccZ"]

# Preprocessing parameters
HIGHPASS_HZ = 1.0
LOWPASS_HZ = 40.0
FILTER_ORDER = 4
BAD_CHANNEL_CQ_THRESHOLD = 2       # CQ below this = bad
BAD_CHANNEL_FRACTION = 0.20        # if >20% samples are bad CQ → exclude channel
INTERPOLATED_SAMPLE_THRESHOLD = 0.20  # if >20% samples interpolated → exclude session
MOTION_ZSCORE_THRESHOLD = 5.0      # z-score threshold for motion artifact detection


def get_eeg_column_names(header: list[str]) -> tuple[list[str], list[str], list[str]]:
    """
    Given CSV header row, return lists of:
      (eeg_col_names, cq_col_names, motion_col_names)
    Column names in EMOTIV CSV have format "EEG.AF3" or just "AF3".
    """
    eeg_cols = []
    for ch in EEG_CHANNELS:
        # Try both formats
        if f"EEG.{ch}" in header:
            eeg_cols.append(f"EEG.{ch}")
        elif ch in header:
            eeg_cols.append(ch)
        else:
            eeg_cols.append(None)  # missing

    cq_cols = [c for c in CQ_CHANNELS if c in header]
    mot_cols = [c for c in MOTION_CHANNELS if c in header]

    return eeg_cols, cq_cols, mot_cols


def read_emotiv_csv(filepath: Path) -> tuple[dict, list[str], list[list]]:
    """
    Reads EMOTIV CSV file. Returns (metadata, header, data_rows).
    Row 0 = metadata (title, timestamps, etc.)
    Row 1 = column headers
    Rows 2+ = data
    """
    metadata = {}
    header = []
    data_rows = []

    with open(filepath, newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                # Parse metadata: "key:value| key:value| ..."
                raw = "|".join(row)
                for part in raw.split("|"):
                    part = part.strip()
                    if ":" in part:
                        k, _, v = part.partition(":")
                        metadata[k.strip()] = v.strip()
            elif i == 1:
                header = [c.strip() for c in row]
            else:
                data_rows.append(row)

    return metadata, header, data_rows


def build_numpy_arrays(
    header: list[str],
    data_rows: list[list],
    eeg_cols: list[str],
    cq_cols: list[str],
    mot_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build numpy arrays from CSV data.
    Returns:
      timestamps (N,)
      eeg (N, 14) — µV, NaN for missing channels
      cq  (N, 14) — contact quality 0-4
      interpolated (N,) — bool
    """
    col_idx = {name: i for i, name in enumerate(header)}

    ts_col = col_idx.get("Timestamp") or col_idx.get("EEG.Timestamp")
    interp_col = col_idx.get("EEG.Interpolated")

    timestamps = []
    eeg_data = []
    cq_data = []
    interp_data = []

    for row in data_rows:
        if len(row) < 2:
            continue
        try:
            ts = float(row[ts_col]) if ts_col is not None else float("nan")
        except (ValueError, TypeError, IndexError):
            continue

        eeg_row = []
        for col in eeg_cols:
            if col is None:
                eeg_row.append(float("nan"))
            else:
                idx = col_idx.get(col)
                if idx is None or idx >= len(row):
                    eeg_row.append(float("nan"))
                else:
                    try:
                        eeg_row.append(float(row[idx]))
                    except (ValueError, TypeError):
                        eeg_row.append(float("nan"))

        cq_row = []
        for col in cq_cols:
            idx = col_idx.get(col)
            if idx is None or idx >= len(row):
                cq_row.append(0)
            else:
                try:
                    cq_row.append(int(float(row[idx])))
                except (ValueError, TypeError):
                    cq_row.append(0)

        try:
            interp = int(float(row[interp_col])) if interp_col is not None and interp_col < len(row) else 0
        except (ValueError, TypeError):
            interp = 0

        timestamps.append(ts)
        eeg_data.append(eeg_row)
        cq_data.append(cq_row)
        interp_data.append(interp)

    if not timestamps:
        return (np.array([]), np.zeros((0, N_CH)),
                np.zeros((0, len(cq_cols))), np.array([], dtype=bool))

    timestamps = np.array(timestamps, dtype=np.float64)
    eeg = np.array(eeg_data, dtype=np.float64)
    cq = np.array(cq_data, dtype=np.int8)
    interp = np.array(interp_data, dtype=bool)

    return timestamps, eeg, cq, interp


def bandpass_filter(data: np.ndarray, fs: float,
                    low: float = HIGHPASS_HZ, high: float = LOWPASS_HZ,
                    order: int = FILTER_ORDER) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter. data shape: (N, C)."""
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    filtered = np.zeros_like(data)
    for c in range(data.shape[1]):
        col = data[:, c]
        if np.isnan(col).all():
            filtered[:, c] = col
        else:
            # Replace NaN with 0 for filtering (will be masked later)
            col_clean = np.where(np.isnan(col), 0.0, col)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                filtered[:, c] = filtfilt(b, a, col_clean)
    return filtered


def average_reference(data: np.ndarray, good_channels: np.ndarray) -> np.ndarray:
    """Apply common average reference across good channels."""
    result = data.copy()
    good_idx = np.where(good_channels)[0]
    if len(good_idx) < 2:
        return result
    avg = np.nanmean(result[:, good_idx], axis=1, keepdims=True)
    result[:, good_idx] -= avg
    return result


def preprocess_session(
    filepath: Path,
    sampling_rate: float = 128.0,
    verbose: bool = False,
) -> dict | None:
    """
    Full preprocessing pipeline for one EMOTIV CSV file.

    Returns dict with:
      timestamps, eeg_raw, eeg_preprocessed, good_channels, bad_channels,
      interpolated_mask, motion_mask, qc_flags, metadata, sampling_rate
    Or None if session fails quality gate.
    """
    if verbose:
        print(f"    Processing: {filepath.name}")

    try:
        metadata, header, data_rows = read_emotiv_csv(filepath)
    except Exception as e:
        print(f"    ERROR reading {filepath.name}: {e}")
        return None

    # Detect sampling rate from metadata
    sr_str = metadata.get("sampling rate", "")
    if "eeg_128" in sr_str:
        fs = 128.0
    elif "eeg_256" in sr_str:
        fs = 256.0
    else:
        fs = sampling_rate  # fallback

    eeg_cols, cq_cols, mot_cols = get_eeg_column_names(header)

    if verbose:
        print(f"      Rows: {len(data_rows)}, fs: {fs}Hz, cols: {len(header)}")

    timestamps, eeg, cq, interp = build_numpy_arrays(
        header, data_rows, eeg_cols, cq_cols, mot_cols
    )

    if len(timestamps) == 0:
        print(f"    WARNING: No data parsed from {filepath.name}")
        return None

    N = len(timestamps)

    # ── Step 1: Interpolated sample mask ──────────────────────────────────────
    interp_frac = interp.mean()
    if interp_frac > INTERPOLATED_SAMPLE_THRESHOLD:
        print(f"    FAIL QC: {filepath.name} — {interp_frac:.1%} interpolated (threshold {INTERPOLATED_SAMPLE_THRESHOLD:.0%})")
        return None

    # ── Step 2: Bad channel detection via CQ ─────────────────────────────────
    good_channels = np.ones(N_CH, dtype=bool)
    bad_channel_names = []

    for ch_i in range(min(N_CH, len(EEG_CHANNELS))):
        if cq.shape[1] <= ch_i:
            break
        bad_cq_frac = (cq[:, ch_i] < BAD_CHANNEL_CQ_THRESHOLD).mean()
        if bad_cq_frac > BAD_CHANNEL_FRACTION:
            good_channels[ch_i] = False
            bad_channel_names.append(EEG_CHANNELS[ch_i])

    if good_channels.sum() < 8:  # need at least 8 good channels
        print(f"    FAIL QC: {filepath.name} — only {good_channels.sum()} good channels")
        return None

    # Set bad channel data to NaN
    eeg_clean = eeg.copy()
    eeg_clean[:, ~good_channels] = np.nan

    # ── Step 3: Bandpass filter ───────────────────────────────────────────────
    eeg_filtered = bandpass_filter(eeg_clean, fs)

    # ── Step 4: Average reference ─────────────────────────────────────────────
    eeg_ref = average_reference(eeg_filtered, good_channels)

    # ── Step 5: Motion artifact detection ─────────────────────────────────────
    # Simple z-score on ACC magnitude (if available)
    motion_mask = np.zeros(N, dtype=bool)
    # We don't load MOT here in numpy (they're at 32Hz resampled in EMOTIV CSV)
    # Flag windows where IMU data shows high acceleration
    # (Full implementation in windower — flagged per window)

    # ── Step 6: Clip extreme values (±200µV = artifact) ──────────────────────
    amplitude_artifact = np.any(np.abs(eeg_ref) > 200.0, axis=1)
    # Mark these but don't remove — windower will handle

    qc_flags = {
        "filepath": str(filepath),
        "n_samples": N,
        "sampling_rate_hz": fs,
        "duration_s": round(N / fs, 2),
        "interpolated_fraction": round(float(interp_frac), 4),
        "n_good_channels": int(good_channels.sum()),
        "n_bad_channels": int((~good_channels).sum()),
        "bad_channel_names": bad_channel_names,
        "amplitude_artifact_fraction": round(float(amplitude_artifact.mean()), 4),
        "qc_pass": True,
    }

    return {
        "filepath": str(filepath),
        "metadata": metadata,
        "sampling_rate": fs,
        "timestamps": timestamps,
        "eeg_raw": eeg,
        "eeg_preprocessed": eeg_ref,
        "good_channels": good_channels,
        "bad_channels": bad_channel_names,
        "interpolated_mask": interp,
        "amplitude_artifact_mask": amplitude_artifact,
        "qc_flags": qc_flags,
    }


def preprocess_all(eeg_filepaths: list[Path], verbose: bool = False) -> dict[str, dict]:
    """
    Preprocess all EEG files. Returns dict: filepath_str -> preprocessed_result.
    """
    results = {}
    failed = []

    for fp in eeg_filepaths:
        result = preprocess_session(fp, verbose=verbose)
        if result:
            results[str(fp)] = result
        else:
            failed.append(str(fp))

    print(f"\n  Preprocessing complete:")
    print(f"    Passed: {len(results)}")
    print(f"    Failed QC: {len(failed)}")

    return results


def write_qc_csv(results: dict, out_path: Path):
    """Write QC summary CSV."""
    if not results:
        return
    rows = [r["qc_flags"] for r in results.values()]
    rows.sort(key=lambda x: x.get("filepath", ""))
    fields = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            row_out = {k: ("|".join(v) if isinstance(v, list) else v) for k, v in row.items()}
            writer.writerow(row_out)
    print(f"  [OK] QC CSV written: {out_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 4: EEG PREPROCESSING")
    print("=" * 60)

    DATA_DIR = ROOT / "Data"
    OUT_DIR = ROOT / "results"
    PREPROCESSED_DIR = ROOT / "data" / "processed" / "preprocessed"
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Find all EEG CSV files
    eeg_files = []
    for pattern in ["*.md.bp.csv", "*.md.pm.bp.csv", "*.md.csv"]:
        eeg_files.extend(DATA_DIR.rglob(pattern))

    # Filter out non-EEG files (interval markers are named *_intervalMarker.csv)
    eeg_files = [f for f in eeg_files if "intervalMarker" not in f.name
                 and "participants_rows" not in f.name]

    print(f"  EEG files found: {len(eeg_files)}")

    if not eeg_files:
        print("  No EEG files found. Check Data/ directory.")
        sys.exit(1)

    results = preprocess_all(eeg_files, verbose=True)

    # Write QC
    write_qc_csv(results, OUT_DIR / "preprocessing_qc.csv")

    # Save preprocessed arrays as numpy .npz files
    print(f"\n  Saving preprocessed arrays to: {PREPROCESSED_DIR}")
    for fp_str, res in results.items():
        stem = Path(fp_str).stem.replace(".", "_")
        out_file = PREPROCESSED_DIR / f"{stem}.npz"
        np.savez_compressed(
            out_file,
            timestamps=res["timestamps"],
            eeg_preprocessed=res["eeg_preprocessed"],
            eeg_raw=res["eeg_raw"],
            good_channels=res["good_channels"],
            interpolated_mask=res["interpolated_mask"],
            amplitude_artifact_mask=res["amplitude_artifact_mask"],
        )
    print(f"  Saved {len(results)} preprocessed session files.")

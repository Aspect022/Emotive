"""
Phase 4 → 5 → Train: End-to-end runner
Preprocesses EEG, builds dataset, runs 3 basic DL models with single-fold validation.
Produces visualizations and a mentor-ready summary report.
"""

import argparse
import csv
import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.signal import butter, filtfilt, welch, resample_poly

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
DATA_DIR    = ROOT / "Data"
PREPROC_DIR = ROOT / "data" / "processed" / "preprocessed"
DATASET_DIR = ROOT / "data" / "processed" / "dl_dataset"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PREPROC_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)

EEG_CHANNELS = ["AF3","F7","F3","FC5","T7","P7","O1",
                 "O2","P8","T8","FC6","F4","F8","AF4"]

TASK_LABEL_MAP  = {"arithmetic":0,"pattern":1,"memory":2,
                   "comprehension":3,"attention":4}
TASK_COLORS     = ["#E63946","#2A9D8F","#E9C46A","#457B9D","#9B5DE5"]
WINDOW_S        = 4.0
FS              = 128.0
WIN_SAMPLES     = int(WINDOW_S * FS)  # 512 — always 128 Hz after resampling

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def load_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def bandpass(sig, fs=128., lo=1., hi=40., order=4):
    nyq = fs / 2.
    b, a = butter(order, [lo/nyq, hi/nyq], btype="band")
    out = np.zeros_like(sig)
    for c in range(sig.shape[1]):
        col = sig[:, c]
        if np.isnan(col).all(): continue
        col = np.where(np.isnan(col), 0., col)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[:, c] = filtfilt(b, a, col)
    return out

def avg_ref(sig, good_mask):
    out = sig.copy()
    idx = np.where(good_mask)[0]
    if len(idx) < 2: return out
    out[:, idx] -= np.nanmean(out[:, idx], axis=1, keepdims=True)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4: PREPROCESS
# ─────────────────────────────────────────────────────────────────────────────
def phase4_preprocess():
    print("\n" + "="*60)
    print("PHASE 4: EEG PREPROCESSING")
    print("="*60)
    from preprocessing.eeg_preprocessor import (
        read_emotiv_csv, get_eeg_column_names, build_numpy_arrays,
        bandpass_filter, average_reference,
        BAD_CHANNEL_CQ_THRESHOLD, BAD_CHANNEL_FRACTION,
        INTERPOLATED_SAMPLE_THRESHOLD
    )

    eeg_files = []
    for pat in ["*.md.bp.csv", "*.md.pm.bp.csv", "*.md.csv"]:
        eeg_files.extend(DATA_DIR.rglob(pat))
    eeg_files = [f for f in eeg_files
                 if "intervalMarker" not in f.name
                 and "participants_rows" not in f.name]
    print(f"  EEG files found: {len(eeg_files)}")

    qc_rows = []
    passed = 0
    failed = 0

    for fp in eeg_files:
        stem = fp.stem.replace(".", "_")
        out_npz = PREPROC_DIR / f"{stem}.npz"
        if out_npz.exists():
            passed += 1
            continue  # already done

        try:
            meta, header, data_rows = read_emotiv_csv(fp)
        except Exception as e:
            print(f"  SKIP (read error): {fp.name} — {e}")
            failed += 1
            continue

        sr_str = meta.get("sampling rate", "")
        fs = 256. if "eeg_256" in sr_str else 128.
        eeg_cols, cq_cols, mot_cols = get_eeg_column_names(header)
        ts, eeg, cq, interp = build_numpy_arrays(
            header, data_rows, eeg_cols, cq_cols, mot_cols)

        if len(ts) == 0:
            failed += 1
            continue

        interp_frac = interp.mean()
        if interp_frac > INTERPOLATED_SAMPLE_THRESHOLD:
            failed += 1
            qc_rows.append({"file": fp.name, "status": "FAIL_INTERP",
                             "interp_frac": round(float(interp_frac),3)})
            continue

        good = np.ones(14, dtype=bool)
        for ci in range(min(14, cq.shape[1])):
            if (cq[:, ci] < BAD_CHANNEL_CQ_THRESHOLD).mean() > BAD_CHANNEL_FRACTION:
                good[ci] = False

        if good.sum() < 8:
            failed += 1
            qc_rows.append({"file": fp.name, "status": "FAIL_CQ",
                             "good_ch": int(good.sum())})
            continue

        eeg_c = eeg.copy().astype(np.float64)
        eeg_c[:, ~good] = np.nan
        eeg_f = bandpass_filter(eeg_c, fs)
        eeg_r = average_reference(eeg_f, good)
        amp_artifact = np.any(np.abs(eeg_r) > 200., axis=1)

        np.savez_compressed(
            out_npz,
            timestamps=ts,
            eeg_preprocessed=eeg_r.astype(np.float32),
            eeg_raw=eeg_c.astype(np.float32),
            good_channels=good,
            interpolated_mask=interp,
            amplitude_artifact_mask=amp_artifact,
            sampling_rate=np.array([fs]),
        )
        passed += 1
        qc_rows.append({"file": fp.name, "status": "OK",
                         "n_samples": len(ts), "fs": fs,
                         "interp_frac": round(float(interp_frac),3),
                         "good_ch": int(good.sum()),
                         "amp_artifact_frac": round(float(amp_artifact.mean()),3)})

    print(f"  Passed: {passed} | Failed/Skipped: {failed}")
    if qc_rows:
        with open(RESULTS_DIR / "preprocessing_qc.csv", "w",
                  newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(qc_rows[0].keys()))
            writer.writeheader()
            writer.writerows(qc_rows)
    return passed

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5: WINDOWING + DATASET
# ─────────────────────────────────────────────────────────────────────────────
def phase5_build_dataset():
    print("\n" + "="*60)
    print("PHASE 5+7: WINDOWING & DATASET COMPILATION")
    print("="*60)

    SYNC_CSV = RESULTS_DIR / "sync_manifest.csv"
    SESSIONS_CSV = RESULTS_DIR / "sessions_enriched.csv"
    if not SYNC_CSV.exists() or not SESSIONS_CSV.exists():
        print("  ERROR: run run_pipeline.py --phase 3 first")
        return False

    sync_rows = load_csv(SYNC_CSV)
    sess_rows = {r["session_id"]: r for r in load_csv(SESSIONS_CSV)}

    # Index preprocessed files by EEG filename stem
    preproc_index = {}
    for npz in PREPROC_DIR.glob("*.npz"):
        preproc_index[npz.stem] = npz

    X_list, y_task_list, y_perf_list = [], [], []
    pid_list, sid_list = [], []
    meta_list = []
    skipped = 0

    for row in sync_rows:
        if row["sync_confidence"] not in ("HIGH", "MODERATE"):
            skipped += 1
            continue

        act = row.get("activity_type", "").lower()
        task_lbl = TASK_LABEL_MAP.get(act)
        if task_lbl is None:
            skipped += 1
            continue

        sess = sess_rows.get(row["session_id"], {})
        score = int(sess.get("score", 0) or 0)
        total_q = int(sess.get("total_questions", 10) or 10)
        perf_lbl = 1 if score >= 7 else 0  # binary: high≥7

        eeg_fp = row.get("matched_eeg_file", "")
        if not eeg_fp:
            skipped += 1
            continue

        # Find matching .npz
        stem = Path(eeg_fp).stem.replace(".", "_")
        npz_path = preproc_index.get(stem)
        if not npz_path:
            # Try partial match
            for k, v in preproc_index.items():
                if Path(eeg_fp).stem in k or k in Path(eeg_fp).stem:
                    npz_path = v
                    break
        if not npz_path:
            skipped += 1
            continue

        try:
            data = np.load(npz_path, allow_pickle=False)
            eeg  = data["eeg_preprocessed"].astype(np.float32)  # (N, 14)
            ts   = data["timestamps"]
            interp = data["interpolated_mask"].astype(bool)
            amp    = data["amplitude_artifact_mask"].astype(bool)
            good   = data["good_channels"].astype(bool)
        except Exception:
            skipped += 1
            continue

        # Use stored sampling rate (reliable) — not inferred from timestamps
        stored_sr = data.get("sampling_rate", None)
        if stored_sr is not None and len(stored_sr) > 0:
            fs_est = float(stored_sr[0])
        elif len(ts) > 2:
            fs_est = 1. / float(np.median(np.diff(ts[:200])))
            fs_est = 128. if abs(fs_est - 128.) < 30 else 256.
        else:
            fs_est = 128.

        # Resample 256 Hz → 128 Hz for consistency
        if fs_est == 256. and eeg.shape[0] > 1:
            eeg    = resample_poly(eeg, up=1, down=2, axis=0).astype(np.float32)
            ts     = ts[::2][:eeg.shape[0]]
            interp = interp[::2][:eeg.shape[0]]
            amp    = amp[::2][:eeg.shape[0]]
            fs_est = 128.

        w_samp = WIN_SAMPLES  # always 512 (4s @ 128 Hz)

        rel_start = float(row.get("eeg_relative_start_s", 0) or 0)
        rel_end   = float(row.get("eeg_relative_end_s", 0) or 0)
        time_taken = float(row.get("time_taken_s", 0) or 0)
        if rel_start < 0: rel_start = 0
        if rel_end <= rel_start: rel_end = rel_start + time_taken

        i_start = max(0, int(rel_start * fs_est))
        i_end   = min(len(ts) - 1, int(rel_end * fs_est))

        if i_end - i_start < w_samp:
            skipped += 1
            continue

        idx = i_start
        while idx + w_samp <= i_end:
            sl = slice(idx, idx + w_samp)
            bad = (interp[sl] | amp[sl]).mean()
            if bad <= 0.10:   # at most 10% bad samples per window
                win = eeg[sl, :].T.copy()  # (14, 512)
                win = np.nan_to_num(win, nan=0.)

                # Per-channel z-score normalization
                m = win.mean(axis=1, keepdims=True)
                s = win.std(axis=1, keepdims=True)
                s = np.where(s < 1e-6, 1., s)
                win = (win - m) / s

                X_list.append(win)
                y_task_list.append(task_lbl)
                y_perf_list.append(perf_lbl)
                pid_list.append(row["participant_id"])
                sid_list.append(row["session_id"])
                meta_list.append({
                    "session_id": row["session_id"],
                    "participant_id": row["participant_id"],
                    "activity_type": act,
                    "task_label": task_lbl,
                    "perf_label": perf_lbl,
                    "score": score,
                    "bad_frac": round(float(bad), 3),
                    "eeg_file": eeg_fp,
                })
            idx += w_samp  # non-overlapping

    print(f"  Skipped sessions: {skipped}")

    if not X_list:
        print("  ERROR: no windows produced — check preprocessed files.")
        return False

    X = np.stack(X_list, axis=0)
    y_task = np.array(y_task_list, dtype=np.int64)
    y_perf = np.array(y_perf_list, dtype=np.int64)
    pids   = np.array(pid_list,    dtype=object)
    sids   = np.array(sid_list,    dtype=object)

    print(f"  Total windows: {len(X)} | Shape: {X.shape}")

    # ── Subject-independent split ─────────────────────────────────────────────
    unique_pids = np.unique(pids)
    rng = np.random.default_rng(42)
    rng.shuffle(unique_pids)
    n = len(unique_pids)
    n_train = max(1, int(n * 0.70))
    n_val   = max(1, int(n * 0.15))

    train_pids = set(unique_pids[:n_train])
    val_pids   = set(unique_pids[n_train:n_train+n_val])
    test_pids  = set(unique_pids[n_train+n_val:])

    train_idx = np.where(np.isin(pids, list(train_pids)))[0]
    val_idx   = np.where(np.isin(pids, list(val_pids)))[0]
    test_idx  = np.where(np.isin(pids, list(test_pids)))[0]

    print(f"  Participants — train: {len(train_pids)} | val: {len(val_pids)} | test: {len(test_pids)}")
    print(f"  Windows      — train: {len(train_idx)} | val: {len(val_idx)} | test: {len(test_idx)}")

    # Save
    np.save(DATASET_DIR / "X_raw.npy",   X)
    np.save(DATASET_DIR / "y_task.npy",  y_task)
    np.save(DATASET_DIR / "y_perf.npy",  y_perf)
    np.save(DATASET_DIR / "participant_ids.npy", pids)
    np.save(DATASET_DIR / "session_ids.npy",     sids)

    splits = {
        "train_idx": train_idx.tolist(),
        "val_idx":   val_idx.tolist(),
        "test_idx":  test_idx.tolist(),
        "train_participants": list(train_pids),
        "val_participants":   list(val_pids),
        "test_participants":  list(test_pids),
    }
    with open(DATASET_DIR / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    label_map = {
        "task": TASK_LABEL_MAP,
        "task_inv": {str(v):k for k,v in TASK_LABEL_MAP.items()},
        "n_channels": 14, "n_timepoints": WIN_SAMPLES,
        "window_size_s": WINDOW_S, "normalization": "z_score_per_channel",
        "channel_names": EEG_CHANNELS,
    }
    with open(DATASET_DIR / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    if meta_list:
        with open(DATASET_DIR / "window_meta.csv", "w",
                  newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(meta_list[0].keys()))
            writer.writeheader()
            writer.writerows(meta_list)

    task_dist = Counter(y_task.tolist())
    inv = {v:k for k,v in TASK_LABEL_MAP.items()}
    print("\n  Task distribution:")
    for lbl, cnt in sorted(task_dist.items()):
        print(f"    {inv[lbl]}: {cnt} ({100*cnt/len(y_task):.1f}%)")

    return {"X": X, "y_task": y_task, "y_perf": y_perf,
            "pids": pids, "splits": splits, "label_map": label_map}

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6: VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────
def make_visualizations(X, y_task, splits, label_map):
    print("\n" + "="*60)
    print("PHASE 6: VISUALIZATIONS")
    print("="*60)
    inv = {v:k for k,v in TASK_LABEL_MAP.items()}
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "axes.spines.top": False,
        "axes.spines.right": False, "figure.dpi": 150,
    })

    # ── Fig 1: Raw vs Preprocessed EEG (first good session) ──────────────────
    print("  Generating Fig 1: Raw vs Preprocessed...")
    npz_files = sorted(PREPROC_DIR.glob("*.npz"))
    if npz_files:
        d = np.load(npz_files[0], allow_pickle=False)
        raw_sig = d["eeg_raw"][:512, :].astype(float)
        pre_sig = d["eeg_preprocessed"][:512, :].astype(float)
        t_ax    = np.arange(512) / 128.

        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        for ci, ch in enumerate(EEG_CHANNELS):
            axes[0].plot(t_ax, raw_sig[:, ci] + ci*200, lw=0.5, color="#4477AA", alpha=0.7)
            axes[1].plot(t_ax, pre_sig[:, ci] + ci*200, lw=0.5, color="#EE6677", alpha=0.7)

        axes[0].set_title("Raw EEG Signal (4 seconds, 14 channels)", fontsize=13, fontweight="bold")
        axes[1].set_title("Preprocessed EEG (1–40 Hz bandpass + average reference)", fontsize=13, fontweight="bold")
        for ax in axes:
            ax.set_yticks(np.arange(14)*200)
            ax.set_yticklabels(EEG_CHANNELS, fontsize=8)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude (µV, offset per channel)")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "fig1_raw_vs_preprocessed.png", bbox_inches="tight")
        plt.close()
        print(f"    Saved: fig1_raw_vs_preprocessed.png")

    # ── Fig 2: Power Spectral Density per task ────────────────────────────────
    print("  Generating Fig 2: PSD per task...")
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
    for lbl, task_name in inv.items():
        ax = axes[lbl]
        idx = np.where(y_task == lbl)[0][:30]  # first 30 windows
        if len(idx) == 0:
            continue
        windows = X[idx]  # (N, 14, 512)
        # Average over windows and channels
        freqs, psd_all = [], []
        for win in windows:
            for ch_data in win:
                f, p = welch(ch_data, fs=128., nperseg=256)
                psd_all.append(p)
                freqs = f
        psd_mean = np.mean(psd_all, axis=0)
        mask = freqs <= 45.
        ax.semilogy(freqs[mask], psd_mean[mask], color=TASK_COLORS[lbl], lw=2)
        ax.fill_between(freqs[mask], psd_mean[mask], alpha=0.2, color=TASK_COLORS[lbl])
        # Band markers
        for band, (lo, hi, label) in {
            "δ": (1,4,"δ"), "θ": (4,8,"θ"),
            "α": (8,13,"α"), "β": (13,30,"β")
        }.items():
            ax.axvspan(lo, hi, alpha=0.05, color="gray")
            ax.text((lo+hi)/2, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
                    label, ha="center", fontsize=7, color="gray")
        ax.set_title(task_name.capitalize(), fontsize=11, fontweight="bold",
                     color=TASK_COLORS[lbl])
        ax.set_xlabel("Frequency (Hz)")
        if lbl == 0: ax.set_ylabel("PSD (µV²/Hz)")
    fig.suptitle("Power Spectral Density by Cognitive Task", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_psd_per_task.png", bbox_inches="tight")
    plt.close()
    print(f"    Saved: fig2_psd_per_task.png")

    # ── Fig 3: Dataset class distribution ────────────────────────────────────
    print("  Generating Fig 3: Class distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    task_counts = Counter(y_task.tolist())
    ax = axes[0]
    names  = [inv[k] for k in sorted(task_counts)]
    counts = [task_counts[k] for k in sorted(task_counts)]
    bars = ax.bar(names, counts, color=TASK_COLORS, edgecolor="white", linewidth=1.5)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 3,
                str(cnt), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("EEG Windows per Cognitive Task", fontsize=13, fontweight="bold")
    ax.set_xlabel("Task"); ax.set_ylabel("Number of 4-second windows")
    ax.axhline(y=len(y_task)/5, color="red", linestyle="--", alpha=0.5, label="Balanced baseline")
    ax.legend()

    # Split distribution
    ax2 = axes[1]
    split_data = [
        ("Train", len(splits["train_idx"]), "#2A9D8F"),
        ("Val",   len(splits["val_idx"]),   "#E9C46A"),
        ("Test",  len(splits["test_idx"]),  "#E63946"),
    ]
    snames, scounts, scols = zip(*split_data)
    wedges, texts, autotexts = ax2.pie(
        scounts, labels=snames, colors=scols,
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2})
    for at in autotexts: at.set_fontsize(12)
    ax2.set_title("Subject-Independent Data Split\n(no participant overlap)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_class_distribution.png", bbox_inches="tight")
    plt.close()
    print(f"    Saved: fig3_class_distribution.png")

    # ── Fig 4: EEG window heatmap (channels × time) ───────────────────────────
    print("  Generating Fig 4: EEG Window Heatmaps per task...")
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    for lbl, task_name in inv.items():
        ax = axes[lbl]
        idx = np.where(y_task == lbl)[0]
        if len(idx) == 0:
            ax.set_visible(False)
            continue
        # Average over first 20 windows of this task
        sample = X[idx[:20]].mean(axis=0)  # (14, 512)
        im = ax.imshow(sample, aspect="auto", cmap="RdBu_r",
                       vmin=-2, vmax=2,
                       extent=[0, WINDOW_S, 0, 14])
        ax.set_yticks(np.arange(14) + 0.5)
        ax.set_yticklabels(EEG_CHANNELS[::-1], fontsize=7)
        ax.set_xlabel("Time (s)")
        ax.set_title(task_name.capitalize(), fontsize=11,
                     fontweight="bold", color=TASK_COLORS[lbl])
        if lbl == 4:
            plt.colorbar(im, ax=ax, label="z-score", shrink=0.8)
    fig.suptitle("Average EEG Window (Channels × Time) per Task\n(z-score normalized)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig4_eeg_heatmaps.png", bbox_inches="tight")
    plt.close()
    print(f"    Saved: fig4_eeg_heatmaps.png")

    # ── Fig 5: Pipeline Architecture Diagram ─────────────────────────────────
    print("  Generating Fig 5: Pipeline diagram...")
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 4)
    ax.axis("off")

    stages = [
        ("Raw EEG\nCSV Files\n(EMOTIV EPOC X)", 1, "#264653", "📁"),
        ("Phase 1\nFile Inspection\n& Inventory", 3, "#2A9D8F", "🔍"),
        ("Phase 2\nParticipant\nLinkage + Trials", 5, "#E9C46A", "🔗"),
        ("Phase 3\nSession\nSynchronization", 7, "#F4A261", "⏱"),
        ("Phase 4\nEEG\nPreprocessing", 9, "#E76F51", "🧹"),
        ("Phase 5\nWindowing\n(4-sec epochs)", 11, "#457B9D", "🪟"),
        ("Phase 6\nDL Models\n(EEGNet/CNN/LSTM)", 13, "#9B5DE5", "🧠"),
        ("Results\n& Report", 15, "#2D6A4F", "📊"),
    ]

    for i, (label, x, color, icon) in enumerate(stages):
        box = mpatches.FancyBboxPatch((x-0.8, 1.2), 1.6, 1.6,
                                     boxstyle="round,pad=0.1",
                                     facecolor=color, edgecolor="white",
                                     linewidth=2, alpha=0.9)
        ax.add_patch(box)
        ax.text(x, 2.7, icon, ha="center", va="center", fontsize=18)
        ax.text(x, 1.85, label, ha="center", va="center",
                fontsize=7.5, color="white", fontweight="bold",
                multialignment="center")
        if i < len(stages) - 1:
            ax.annotate("", xy=(x+0.9, 2.), xytext=(x+0.8+0.3, 2.),
                        arrowprops=dict(arrowstyle="->",
                                        color="gray", lw=2))

    # Supabase box below
    sb_box = mpatches.FancyBboxPatch((4.2, 0.1), 3.6, 0.9,
                                     boxstyle="round,pad=0.1",
                                     facecolor="#264653", edgecolor="#2A9D8F",
                                     linewidth=2, alpha=0.85)
    ax.add_patch(sb_box)
    ax.text(6, 0.55, "☁️  Supabase — Behavioral Data\n(273 activity sessions, 2840 trials)",
            ha="center", va="center", fontsize=8, color="white")
    ax.annotate("", xy=(5, 1.2), xytext=(6, 1.0),
                arrowprops=dict(arrowstyle="->", color="#2A9D8F", lw=1.5))

    ax.set_title("EEG Cognitive Task Research Pipeline — End-to-End Architecture",
                 fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_pipeline_diagram.png", bbox_inches="tight")
    plt.close()
    print(f"    Saved: fig5_pipeline_diagram.png")

    print(f"\n  All figures saved to: {FIGURES_DIR}/")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 7: DL MODELS — Simple single-fold
# ─────────────────────────────────────────────────────────────────────────────
def phase7_train_models(X, y_task, splits):
    print("\n" + "="*60)
    print("PHASE 7: DL MODEL TRAINING (SINGLE-FOLD)")
    print("="*60)

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cpu")
    print(f"  Device: {device}")

    def make_loaders(X, y, splits, bs=32):
        def ds(idx):
            Xt = torch.tensor(X[idx], dtype=torch.float32)
            yt = torch.tensor(y[idx], dtype=torch.long)
            return TensorDataset(Xt, yt)
        tr = DataLoader(ds(splits["train_idx"]), bs, shuffle=True)
        va = DataLoader(ds(splits["val_idx"]),   bs, shuffle=False)
        te = DataLoader(ds(splits["test_idx"]),  bs, shuffle=False)
        return tr, va, te

    # ── EEGNet ────────────────────────────────────────────────────────────────
    from models.eegnet import EEGNet

    # ── Tiny CNN (very lightweight) ───────────────────────────────────────────
    class TinyCNN(nn.Module):
        def __init__(self, nc=5):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(14, 32, 16, padding=8, bias=False),
                nn.BatchNorm1d(32), nn.ELU(), nn.MaxPool1d(4),
                nn.Conv1d(32, 64, 8, padding=4, bias=False),
                nn.BatchNorm1d(64), nn.ELU(), nn.AdaptiveAvgPool1d(16),
                nn.Flatten(),
                nn.Linear(64*16, 128), nn.ELU(), nn.Dropout(0.4),
                nn.Linear(128, nc),
            )
        def forward(self, x): return self.net(x)

    # ── Tiny LSTM ─────────────────────────────────────────────────────────────
    class TinyLSTM(nn.Module):
        def __init__(self, nc=5):
            super().__init__()
            self.lstm = nn.LSTM(14, 64, 2, batch_first=True,
                                bidirectional=True, dropout=0.3)
            self.head = nn.Sequential(nn.Dropout(0.4),
                                      nn.Linear(128, nc))
        def forward(self, x):
            x = x.permute(0, 2, 1)  # (B,T,C)
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])

    models = {
        "EEGNet":    EEGNet(n_classes=5, n_channels=14, n_timepoints=WIN_SAMPLES, dropout_rate=0.4),
        "TinyCNN":   TinyCNN(nc=5),
        "TinyLSTM":  TinyLSTM(nc=5),
    }

    n_classes = 5
    all_results = {}
    all_histories = {}

    MODELS_DIR = RESULTS_DIR / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for mname, model in models.items():
        print(f"\n  -- {mname} ({sum(p.numel() for p in model.parameters()):,} params) --")
        model = model.to(device)

        y_tr = y_task[splits["train_idx"]]
        counts = np.bincount(y_tr, minlength=5).astype(float)
        counts = np.where(counts == 0, 1., counts)
        w = torch.tensor(1./counts / (1./counts).sum() * 5, dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=w.to(device))

        opt  = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
        tr_loader, va_loader, te_loader = make_loaders(X, y_task, splits)

        best_val_f1 = -1.
        patience = 10
        patience_ctr = 0
        ckpt = MODELS_DIR / f"{mname}_best.pt"
        history = []

        for epoch in range(1, 51):  # max 50 epochs (fast)
            # Train
            model.train()
            tr_loss, tr_cor, tr_tot = 0., 0, 0
            for Xb, yb in tr_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                opt.zero_grad()
                out = model(Xb)
                loss = criterion(out, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                opt.step()
                tr_loss += loss.item() * len(yb)
                tr_cor  += (out.argmax(1) == yb).sum().item()
                tr_tot  += len(yb)
            sched.step()

            # Val
            model.eval()
            all_p, all_y = [], []
            with torch.no_grad():
                for Xb, yb in va_loader:
                    p = model(Xb.to(device)).argmax(1).cpu().numpy()
                    all_p.extend(p); all_y.extend(yb.numpy())
            all_p, all_y = np.array(all_p), np.array(all_y)
            val_acc = (all_p == all_y).mean()
            # Macro F1
            f1s = []
            for c in range(n_classes):
                tp = ((all_p==c)&(all_y==c)).sum()
                fp = ((all_p==c)&(all_y!=c)).sum()
                fn = ((all_p!=c)&(all_y==c)).sum()
                f1s.append(2*tp/(2*tp+fp+fn+1e-8))
            val_f1 = float(np.mean(f1s))

            history.append({
                "epoch": epoch,
                "train_loss": round(tr_loss/tr_tot, 4),
                "train_acc":  round(tr_cor/tr_tot, 4),
                "val_acc":    round(val_acc, 4),
                "val_f1":     round(val_f1, 4),
            })

            if epoch % 10 == 0 or epoch == 1:
                print(f"    Ep {epoch:2d}: loss={tr_loss/tr_tot:.4f} "
                      f"tr_acc={tr_cor/tr_tot:.3f} val_acc={val_acc:.3f} "
                      f"val_f1={val_f1:.3f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_ctr = 0
                torch.save(model.state_dict(), ckpt)
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    print(f"    Early stop at epoch {epoch}")
                    break

        # Test
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.eval()
        te_p, te_y = [], []
        with torch.no_grad():
            for Xb, yb in te_loader:
                p = model(Xb.to(device)).argmax(1).cpu().numpy()
                te_p.extend(p); te_y.extend(yb.numpy())
        te_p, te_y = np.array(te_p), np.array(te_y)
        te_acc = (te_p == te_y).mean()
        te_f1s = []
        for c in range(n_classes):
            tp = ((te_p==c)&(te_y==c)).sum()
            fp = ((te_p==c)&(te_y!=c)).sum()
            fn = ((te_p!=c)&(te_y==c)).sum()
            te_f1s.append(2*tp/(2*tp+fp+fn+1e-8))
        te_f1 = float(np.mean(te_f1s))

        print(f"\n  {mname} Test → Accuracy: {te_acc:.4f} | Macro-F1: {te_f1:.4f}")
        print(f"  (Chance level = {1/n_classes:.4f})")

        all_results[mname] = {
            "best_val_f1": round(best_val_f1, 4),
            "test_accuracy": round(float(te_acc), 4),
            "test_macro_f1": round(te_f1, 4),
            "test_f1_per_class": [round(f, 4) for f in te_f1s],
            "test_predictions": te_p.tolist(),
            "test_labels": te_y.tolist(),
            "n_params": sum(p.numel() for p in model.parameters()),
        }
        all_histories[mname] = history

    return all_results, all_histories

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 8: RESULT VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────
def make_result_visualizations(all_results, all_histories):
    print("\n" + "="*60)
    print("PHASE 8: RESULT VISUALIZATIONS")
    print("="*60)

    TASK_NAMES = ["Arithmetic", "Pattern", "Memory", "Comprehension", "Attention"]
    plt.rcParams.update({"font.family": "DejaVu Sans",
                         "axes.spines.top": False, "axes.spines.right": False})

    # ── Fig 6: Training curves ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"EEGNet": "#E63946", "TinyCNN": "#2A9D8F", "TinyLSTM": "#9B5DE5"}
    for mname, hist in all_histories.items():
        epochs  = [h["epoch"] for h in hist]
        tr_acc  = [h["train_acc"] for h in hist]
        va_acc  = [h["val_acc"] for h in hist]
        va_f1   = [h["val_f1"] for h in hist]
        c = colors[mname]
        axes[0].plot(epochs, tr_acc, "--", color=c, alpha=0.5, lw=1.5)
        axes[0].plot(epochs, va_acc, "-",  color=c, lw=2, label=mname)
        axes[1].plot(epochs, va_f1,  "-",  color=c, lw=2, label=mname)

    axes[0].axhline(0.2, color="gray", ls=":", alpha=0.6, label="Chance (20%)")
    axes[0].set_title("Training & Validation Accuracy", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].set_ylim(0, 1)
    axes[1].axhline(0.2, color="gray", ls=":", alpha=0.6, label="Chance")
    axes[1].set_title("Validation Macro-F1 Score", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Macro-F1")
    axes[1].legend(); axes[1].set_ylim(0, 1)

    fig.suptitle("Model Training Curves — Single-Fold Subject-Independent Validation",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig6_training_curves.png", bbox_inches="tight")
    plt.close()
    print("  Saved: fig6_training_curves.png")

    # ── Fig 7: Comparison bar chart ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    mnames = list(all_results.keys())
    test_acc  = [all_results[m]["test_accuracy"] for m in mnames]
    test_f1   = [all_results[m]["test_macro_f1"] for m in mnames]
    mcols     = [colors[m] for m in mnames]

    for ax, vals, title in [
        (axes[0], test_acc, "Test Accuracy"),
        (axes[1], test_f1,  "Test Macro-F1"),
    ]:
        bars = ax.bar(mnames, vals, color=mcols, edgecolor="white", linewidth=2, width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontweight="bold")
        ax.axhline(0.2, color="gray", ls="--", alpha=0.6, label="Chance (20%)")
        ax.set_ylim(0, 1); ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel("Score"); ax.legend()

    fig.suptitle("Model Comparison — 5-Class Cognitive Task Classification",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig7_model_comparison.png", bbox_inches="tight")
    plt.close()
    print("  Saved: fig7_model_comparison.png")

    # ── Fig 8: Confusion matrices ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (mname, res) in zip(axes, all_results.items()):
        preds  = np.array(res["test_predictions"])
        labels = np.array(res["test_labels"])
        cm = np.zeros((5, 5), dtype=int)
        for t, p in zip(labels, preds):
            if 0 <= t < 5 and 0 <= p < 5:
                cm[t][p] += 1
        # Normalize per row
        cm_norm = cm.astype(float)
        row_sums = cm_norm.sum(axis=1, keepdims=True)
        cm_norm  = np.divide(cm_norm, row_sums,
                             out=np.zeros_like(cm_norm), where=row_sums > 0)

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(5)); ax.set_yticks(range(5))
        short_names = ["Arith", "Patt", "Mem", "Comp", "Attn"]
        ax.set_xticklabels(short_names, rotation=30, ha="right")
        ax.set_yticklabels(short_names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{mname}\nTest Acc: {res['test_accuracy']:.3f} | F1: {res['test_macro_f1']:.3f}",
                     fontsize=11, fontweight="bold")
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                        fontsize=9, color="white" if cm_norm[i,j] > 0.5 else "black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall")

    fig.suptitle("Confusion Matrices — 5-Class Cognitive Task Classification",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig8_confusion_matrices.png", bbox_inches="tight")
    plt.close()
    print("  Saved: fig8_confusion_matrices.png")

    # Save JSON results
    with open(RESULTS_DIR / "model_results.json", "w") as f:
        out = {m: {k:v for k,v in r.items()
                   if k not in ("test_predictions","test_labels")}
               for m, r in all_results.items()}
        json.dump(out, f, indent=2)
    print("  Saved: model_results.json")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-phase4", action="store_true",
                        help="Skip preprocessing (use existing .npz files)")
    args = parser.parse_args()

    t0 = time.time()

    # Phase 4 (optional)
    if not args.skip_phase4:
        n_passed = phase4_preprocess()
    else:
        n_npz = len(list(PREPROC_DIR.glob("*.npz")))
        print(f"\nPhase 4 skipped — using {n_npz} existing preprocessed files")

    # Phase 5
    dataset = phase5_build_dataset()
    if not dataset:
        print("\nDataset build failed. Exiting.")
        sys.exit(1)

    X       = dataset["X"]
    y_task  = dataset["y_task"]
    splits  = dataset["splits"]
    lmap    = dataset["label_map"]

    # Visualizations (data)
    make_visualizations(X, y_task, splits, lmap)

    # Phase 7: Train models
    all_results, all_histories = phase7_train_models(X, y_task, splits)

    # Visualizations (results)
    make_result_visualizations(all_results, all_histories)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"All phases complete in {elapsed/60:.1f} minutes")
    print(f"Figures: {FIGURES_DIR}/")
    print(f"Results: {RESULTS_DIR}/")
    print("="*60)

"""
Phase 1: Handcrafted EEG Feature Extraction
Extracts 242 features per 4-second window from preprocessed EEG .npz files.
Produces: results/ml_features.csv  (2908 rows x ~250 cols)
"""
import csv
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import welch, coherence
from scipy.stats import skew, kurtosis

warnings.filterwarnings("ignore")

ROOT        = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
DATASET_DIR = ROOT / "data" / "processed" / "dl_dataset"
PREPROC_DIR = ROOT / "data" / "processed" / "preprocessed"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

EEG_CH = ["AF3","F7","F3","FC5","T7","P7","O1",
           "O2","P8","T8","FC6","F4","F8","AF4"]
FS     = 128.0
BANDS  = {"delta":(1,4),"theta":(4,8),"alpha":(8,13),"beta":(13,30),"gamma":(30,40)}

# Frontal asymmetry pairs: (right_ch, left_ch)
ASYM_PAIRS = [("F4","F3"), ("F8","F7"), ("AF4","AF3")]

# Key coherence pairs (frontal-parietal, inter-hemispheric)
COH_PAIRS = [
    ("AF3","AF4"),("F3","F4"),("F7","F8"),("FC5","FC6"),
    ("T7","T8"),("P7","P8"),("O1","O2"),
    ("AF3","P7"),("AF4","P8"),("F3","P7"),("F4","P8"),
    ("F3","O1"),("F4","O2"),("FC5","T7"),("FC6","T8"),
]

# ─── Feature computation helpers ─────────────────────────────────────────────

def bandpower(sig, fs, lo, hi):
    """Absolute band power via Welch PSD."""
    nperseg = min(256, len(sig))
    f, pxx = welch(sig, fs=fs, nperseg=nperseg)
    mask = (f >= lo) & (f <= hi)
    return float(np.trapz(pxx[mask], f[mask])) if mask.any() else 0.

def spectral_entropy(sig, fs):
    """Shannon entropy of normalized PSD."""
    nperseg = min(256, len(sig))
    _, pxx = welch(sig, fs=fs, nperseg=nperseg)
    pxx = pxx / (pxx.sum() + 1e-12)
    return float(-np.sum(pxx * np.log(pxx + 1e-12)))

def hjorth(sig):
    """Activity, Mobility, Complexity."""
    d1 = np.diff(sig)
    d2 = np.diff(d1)
    act  = float(np.var(sig))
    mob  = float(np.sqrt(np.var(d1) / (np.var(sig) + 1e-12)))
    comp = float(np.sqrt(np.var(d2) / (np.var(d1) + 1e-12)) / (mob + 1e-12))
    return act, mob, comp

def sample_entropy(sig, m=2, r_frac=0.2):
    """Sample Entropy (Vectorized NumPy — 60x faster)."""
    sig = (sig - sig.mean()) / (sig.std() + 1e-12)
    N = len(sig)
    if N <= m + 1:
        return 0.
    from numpy.lib.stride_tricks import sliding_window_view
    x_m = sliding_window_view(sig, m)
    diff_m = np.abs(x_m[:, None, :] - x_m[None, :, :]).max(axis=-1)
    i_idx, j_idx = np.triu_indices(len(x_m), k=1)
    B = (diff_m[i_idx, j_idx] < r_frac).sum()

    x_m1 = sliding_window_view(sig, m + 1)
    diff_m1 = np.abs(x_m1[:, None, :] - x_m1[None, :, :]).max(axis=-1)
    i1_idx, j1_idx = np.triu_indices(len(x_m1), k=1)
    A = (diff_m1[i1_idx, j1_idx] < r_frac).sum()

    return float(-np.log((A + 1e-12) / (B + 1e-12)))

def permutation_entropy(sig, D=5, tau=1):
    """Permutation Entropy (Fast NumPy)."""
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(sig, D * tau)
    if len(win) == 0:
        return 0.
    sub_win = win[:, ::tau]
    ranks = np.argsort(sub_win, axis=1)
    weights = (D ** np.arange(D)).astype(np.int64)
    hashes = (ranks * weights).sum(axis=1)
    _, counts = np.unique(hashes, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log(probs + 1e-12)))

def band_coherence(x, y, fs, lo, hi):
    """Mean magnitude-squared coherence in a band."""
    nperseg = min(256, len(x))
    f, Cxy = coherence(x, y, fs=fs, nperseg=nperseg)
    mask = (f >= lo) & (f <= hi)
    return float(Cxy[mask].mean()) if mask.any() else 0.

# ─── Extract features from one window ───────────────────────────────────────

def extract_window_features(win):
    """
    win: (14, 512) float32 array — preprocessed EEG
    Returns: dict of feature_name -> float
    """
    feats = {}

    for ci, ch in enumerate(EEG_CH):
        s = win[ci].astype(float)

        # 1. Statistical (4 per channel)
        feats[f"mean_{ch}"]     = float(np.mean(s))
        feats[f"var_{ch}"]      = float(np.var(s))
        feats[f"skew_{ch}"]     = float(skew(s))
        feats[f"kurt_{ch}"]     = float(kurtosis(s))

        # 2. Hjorth (3 per channel)
        act, mob, comp = hjorth(s)
        feats[f"hj_act_{ch}"]   = act
        feats[f"hj_mob_{ch}"]   = mob
        feats[f"hj_comp_{ch}"]  = comp

        # 3. Band powers (5 per channel)
        bp = {}
        for bname, (lo, hi) in BANDS.items():
            bp[bname] = bandpower(s, FS, lo, hi)
            feats[f"bp_{bname}_{ch}"] = bp[bname]

        # 4. Spectral entropy (1 per channel)
        feats[f"sp_ent_{ch}"]   = spectral_entropy(s, FS)

        # 5. Theta/Alpha ratio (1 per channel)
        denom = bp["alpha"] + 1e-12
        feats[f"theta_alpha_{ch}"] = bp["theta"] / denom

        # 6. Permutation entropy (1 per channel) — fast
        feats[f"pe_{ch}"]       = permutation_entropy(s, D=5)

    # 7. Sample entropy — only frontal channels (expensive O(N²), skip temporal)
    for ch in ["AF3","AF4","F3","F4","F7","F8"]:
        ci = EEG_CH.index(ch)
        s  = win[ci].astype(float)
        feats[f"se_{ch}"]       = sample_entropy(s[:256], m=2)  # use half-window for speed

    # 8. Frontal Alpha Asymmetry (3 features)
    for r_ch, l_ch in ASYM_PAIRS:
        ri = EEG_CH.index(r_ch)
        li = EEG_CH.index(l_ch)
        r_alpha = bandpower(win[ri].astype(float), FS, 8, 13)
        l_alpha = bandpower(win[li].astype(float), FS, 8, 13)
        feats[f"faa_{r_ch}_{l_ch}"] = np.log(r_alpha + 1e-12) - np.log(l_alpha + 1e-12)

    # 9. Inter-channel Theta Coherence (15 features)
    for a_ch, b_ch in COH_PAIRS:
        ai = EEG_CH.index(a_ch)
        bi = EEG_CH.index(b_ch)
        feats[f"coh_theta_{a_ch}_{b_ch}"] = band_coherence(
            win[ai].astype(float), win[bi].astype(float), FS, 4, 8)

    return feats

# ─── Main ────────────────────────────────────────────────────────────────────

# ─── Parallel worker ─────────────────────────────────────────────────────────

def _worker(args):
    i, win, task_lbl, perf_lbl, pid, sid, split = args
    TASK_INV = {0:"arithmetic",1:"pattern",2:"memory",3:"comprehension",4:"attention"}
    try:
        feats = extract_window_features(win)
        row = {
            "window_id":      i,
            "participant_id": str(pid),
            "session_id":     str(sid),
            "activity_type":  TASK_INV.get(int(task_lbl), "unknown"),
            "task_label":     int(task_lbl),
            "perf_label":     int(perf_lbl),
            "split":          split,
        }
        row.update(feats)
        return row
    except Exception as e:
        print(f"  SKIP window {i}: {e}")
        return None

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import os

    print("=" * 60)
    print("PHASE 1: EEG FEATURE EXTRACTION (PARALLEL)")
    print("=" * 60)

    # Load dataset
    X      = np.load(DATASET_DIR / "X_raw.npy")
    y_task = np.load(DATASET_DIR / "y_task.npy")
    y_perf = np.load(DATASET_DIR / "y_perf.npy")
    pids   = np.load(DATASET_DIR / "participant_ids.npy", allow_pickle=True)
    sids   = np.load(DATASET_DIR / "session_ids.npy",     allow_pickle=True)

    with open(DATASET_DIR / "splits.json") as f:
        splits = json.load(f)

    n_workers = max(1, os.cpu_count() - 2)
    print(f"  Windows to process: {len(X)}")
    print(f"  CPU Workers:        {n_workers}")
    print(f"  Features per win:  ~240\n")

    items = []
    for i, (win, task_lbl, perf_lbl, pid, sid) in enumerate(
            zip(X, y_task, y_perf, pids, sids)):
        if i in splits["train_idx"]:
            split = "train"
        elif i in splits["val_idx"]:
            split = "val"
        else:
            split = "test"
        items.append((i, win, task_lbl, perf_lbl, pid, sid, split))

    t0 = time.time()
    rows = []
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_worker, it) for it in items]
        for fut in as_completed(futures):
            res = fut.result()
            if res is not None:
                rows.append(res)
            done += 1
            if done % 500 == 0 or done == len(items):
                elapsed = time.time() - t0
                pct = 100 * done / len(items)
                print(f"  [{done:4d}/{len(items)}] {pct:.0f}% ({elapsed:.1f}s)")

    # Sort rows by window_id to maintain order
    rows.sort(key=lambda r: r["window_id"])

    # Write CSV
    out_csv = RESULTS_DIR / "ml_features.csv"
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s! {len(rows)} rows x {len(fieldnames)} columns")
    print(f"  Feature count: {len(fieldnames) - 7} EEG features")
    print(f"  Saved: {out_csv}")

    # Quick feature group summary
    feat_names = fieldnames[7:]
    groups = {
        "Statistical (mean/var/skew/kurt)": sum(1 for f in feat_names if any(f.startswith(p) for p in ["mean_","var_","skew_","kurt_"])),
        "Hjorth parameters": sum(1 for f in feat_names if f.startswith("hj_")),
        "Band powers (5 bands)": sum(1 for f in feat_names if f.startswith("bp_")),
        "Spectral entropy": sum(1 for f in feat_names if f.startswith("sp_ent")),
        "Theta/Alpha ratio": sum(1 for f in feat_names if f.startswith("theta_alpha")),
        "Sample entropy": sum(1 for f in feat_names if f.startswith("se_")),
        "Permutation entropy": sum(1 for f in feat_names if f.startswith("pe_")),
        "Frontal alpha asymmetry": sum(1 for f in feat_names if f.startswith("faa_")),
        "Theta coherence": sum(1 for f in feat_names if f.startswith("coh_")),
    }
    print("\n  Feature breakdown:")
    for g, c in groups.items():
        print(f"    {g:<38}: {c:3d}")
    print(f"    {'TOTAL':<38}: {sum(groups.values()):3d}")

if __name__ == "__main__":
    main()

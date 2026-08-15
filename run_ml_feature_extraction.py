"""
Phase 1: Handcrafted EEG Feature Extraction
Extracts 242 features per 4-second window from preprocessed EEG .npz files.
Produces: results/ml_features.csv  (2908 rows x ~250 cols)
"""
import csv
import json
import sys
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
    """Sample Entropy (simplified O(N²))."""
    sig = (sig - sig.mean()) / (sig.std() + 1e-12)
    r = r_frac
    N = len(sig)
    def _count(m_):
        count = 0
        for i in range(N - m_):
            for j in range(i+1, N - m_):
                if np.max(np.abs(sig[i:i+m_] - sig[j:j+m_])) < r:
                    count += 1
        return count
    B = _count(m)
    A = _count(m + 1)
    return float(-np.log(A / (B + 1e-12) + 1e-12))

def permutation_entropy(sig, D=5, tau=1):
    """Permutation Entropy."""
    N = len(sig)
    patterns = {}
    for i in range(N - (D-1)*tau):
        s = tuple(np.argsort(sig[i:i+D*tau:tau]))
        patterns[s] = patterns.get(s, 0) + 1
    total = sum(patterns.values())
    probs = np.array([v/total for v in patterns.values()])
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

def main():
    print("=" * 60)
    print("PHASE 1: EEG FEATURE EXTRACTION")
    print("=" * 60)

    # Load dataset
    X      = np.load(DATASET_DIR / "X_raw.npy")
    y_task = np.load(DATASET_DIR / "y_task.npy")
    y_perf = np.load(DATASET_DIR / "y_perf.npy")
    pids   = np.load(DATASET_DIR / "participant_ids.npy", allow_pickle=True)
    sids   = np.load(DATASET_DIR / "session_ids.npy",     allow_pickle=True)

    with open(DATASET_DIR / "splits.json") as f:
        splits = json.load(f)

    TASK_INV = {0:"arithmetic",1:"pattern",2:"memory",3:"comprehension",4:"attention"}

    print(f"  Windows to process: {len(X)}")
    print(f"  Sample entropy computed on first 6 frontal channels only (speed)")
    print(f"  Estimated time: ~8-15 min on CPU\n")

    rows = []
    n    = len(X)
    for i, (win, task_lbl, perf_lbl, pid, sid) in enumerate(
            zip(X, y_task, y_perf, pids, sids)):

        if i % 200 == 0:
            pct = 100 * i / n
            print(f"  [{i:4d}/{n}] {pct:.0f}%...")

        try:
            feats = extract_window_features(win)
        except Exception as e:
            print(f"  SKIP window {i}: {e}")
            continue

        # Determine split
        if i in splits["train_idx"]:
            split = "train"
        elif i in splits["val_idx"]:
            split = "val"
        else:
            split = "test"

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
        rows.append(row)

    if not rows:
        print("ERROR: No features extracted.")
        return

    # Write CSV
    out_csv = RESULTS_DIR / "ml_features.csv"
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  Done! {len(rows)} rows x {len(fieldnames)} columns")
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

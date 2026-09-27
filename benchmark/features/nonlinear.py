"""
benchmark/features/nonlinear.py
Per-window nonlinear features: SampEn, HFD, PermEn, LZC × 14 channels = 56-D.
"""
import numpy as np
from pathlib import Path
from tqdm import tqdm
import antropy as ant
from benchmark.config import N_CH


def _nonlin_window(w: np.ndarray) -> np.ndarray:
    """w: (14, 128) -> (56,) float32"""
    feats = np.zeros((N_CH, 4), dtype=np.float32)
    for ch in range(N_CH):
        sig = w[ch].astype(np.float64)
        if np.std(sig) < 1e-6:
            continue
        try:    feats[ch, 0] = ant.sample_entropy(sig)
        except: pass
        try:    feats[ch, 1] = ant.higuchi_fd(sig, kmax=10)
        except: pass
        try:    feats[ch, 2] = ant.perm_entropy(sig, order=3, delay=1, normalize=True)
        except: pass
        try:
            b = (sig > np.median(sig)).astype(np.int32)
            feats[ch, 3] = ant.lziv_complexity(b, normalize=True)
        except: pass
    return feats.flatten()


def get_nonlinear(cache_dir: str, X_raw: np.ndarray) -> np.ndarray:
    N    = len(X_raw)
    path = Path(cache_dir) / "nonlinear.npy"

    if path.exists():
        try:
            arr = np.load(path)
            if arr.shape == (N, 56):
                print("  [nonlinear] cached — skipping")
                return arr
        except: pass
    path.unlink(missing_ok=True)

    print(f"  [nonlinear] computing {N} windows (slow ~20 min)...")
    out = np.zeros((N, 56), dtype=np.float32)
    for i in tqdm(range(N), desc="  Nonlin"):
        out[i] = _nonlin_window(X_raw[i])

    out = np.nan_to_num(out, nan=0., posinf=5., neginf=-5.)
    m   = np.nanmedian(out, axis=0)
    s   = out.std(axis=0) + 1e-8
    out = np.clip((out - m) / s, -5., 5.)
    np.save(path, out)
    return out

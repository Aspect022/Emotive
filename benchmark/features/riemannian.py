"""
benchmark/features/riemannian.py
Log-Euclidean Riemannian tangent vectors: 5 bands × 105 = 525-D.
Ledoit-Wolf regularised SPD covariance.
"""
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.signal import butter, filtfilt
from sklearn.covariance import ledoit_wolf
from benchmark.config import N_CH, SFREQ, BAND_HZ, DIM_RIEM


def _riem_window(w: np.ndarray) -> np.ndarray:
    """w: (14, 128) -> (525,) float32"""
    vecs = []
    for lo, hi in BAND_HZ:
        b, a = butter(4, [lo/(SFREQ/2), hi/(SFREQ/2)], btype="band")
        filt = filtfilt(b, a, w, axis=1)
        C, _ = ledoit_wolf(filt.T)
        C   += 1e-6 * np.eye(N_CH)
        ev, evec = np.linalg.eigh(C)
        ev   = np.maximum(ev, 1e-6)
        logC = evec @ np.diag(np.log(ev)) @ evec.T
        idx  = np.triu_indices(N_CH)
        vecs.append(logC[idx])
    return np.concatenate(vecs).astype(np.float32)


def get_riemannian(cache_dir: str, X_raw: np.ndarray) -> np.ndarray:
    N    = len(X_raw)
    path = Path(cache_dir) / "riemannian.npy"

    if path.exists():
        try:
            arr = np.load(path)
            if arr.shape == (N, DIM_RIEM):
                print("  [riemannian] cached — skipping")
                return arr
        except: pass
    path.unlink(missing_ok=True)

    print(f"  [riemannian] computing {N} windows (~20 min)...")
    out = np.zeros((N, DIM_RIEM), dtype=np.float32)
    for i in tqdm(range(N), desc="  Riemannian"):
        out[i] = _riem_window(X_raw[i])
    out = np.nan_to_num(out, nan=0., posinf=5., neginf=-5.)
    m = out.mean(0); s = out.std(0) + 1e-8
    out = np.clip((out - m) / s, -5., 5.)
    np.save(path, out)
    return out

"""
benchmark/features/graph_nodes.py
Per-electrode handcrafted node features: Welch PSD × 5 bands, EI, DWT × 4, Hjorth × 3 = 13-D.
Used as GATv2 node inputs in Arch B and C.
"""
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pywt
from scipy.signal import welch
from benchmark.config import N_CH, SFREQ, BAND_HZ


def _node_feats_window(w: np.ndarray) -> np.ndarray:
    """w: (14, 128) -> (14, 13) float32"""
    out = np.zeros((N_CH, 13), dtype=np.float32)
    for ch in range(N_CH):
        sig = w[ch].astype(np.float64)
        f, psd = welch(sig, fs=SFREQ, nperseg=64)
        for bi, (lo, hi) in enumerate(BAND_HZ):
            mask = (f >= lo) & (f < hi)
            out[ch, bi] = psd[mask].mean() if mask.any() else 0.
        a  = psd[(f>=8) & (f<13)].mean()  + 1e-8
        th = psd[(f>=4) & (f<8)].mean()   + 1e-8
        be = psd[(f>=13)& (f<30)].mean()
        out[ch, 5] = be / (a + th)
        coeffs = pywt.wavedec(sig, "db4", level=4)
        for di, c in enumerate(coeffs[:4]):
            out[ch, 6+di] = float(np.sum(c**2))
        d1 = np.diff(sig); d2 = np.diff(d1)
        v0 = np.var(sig)+1e-8; v1 = np.var(d1)+1e-8; v2 = np.var(d2)+1e-8
        out[ch, 10] = v0
        out[ch, 11] = np.sqrt(v1/v0)
        out[ch, 12] = np.sqrt(v2/v1) / np.sqrt(v1/v0)
    return out


def get_node_feats(cache_dir: str, X_raw: np.ndarray) -> np.ndarray:
    N    = len(X_raw)
    path = Path(cache_dir) / "node_feats.npy"

    if path.exists():
        try:
            arr = np.load(path)
            if arr.shape == (N, N_CH, 13):
                print("  [graph_nodes] cached — skipping")
                return arr
        except: pass
    path.unlink(missing_ok=True)

    print(f"  [graph_nodes] computing {N} windows...")
    out = np.zeros((N, N_CH, 13), dtype=np.float32)
    for i in tqdm(range(N), desc="  NodeFeats"):
        out[i] = _node_feats_window(X_raw[i])
    out = np.nan_to_num(out, nan=0., posinf=5., neginf=-5.)
    np.save(path, out)
    return out

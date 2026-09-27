"""
benchmark/features/scalogram.py
Precompute CWT scalograms to a size-validated float16 memmap.
"""
import numpy as np
import pywt
from pathlib import Path
from tqdm import tqdm
from benchmark.config import N_CH, N_FREQ, WIN, PAD, WAVELET, SFREQ


def _build_scales():
    freqs_log  = np.logspace(np.log10(1.0), np.log10(45.0), N_FREQ)
    cf         = pywt.central_frequency(WAVELET)
    scales     = (cf / (freqs_log * (1.0 / SFREQ)))[::-1].copy()
    freqs_asc  = freqs_log[::-1].copy()
    return scales, freqs_asc


CWT_SCALES, FREQS_ASC = _build_scales()


def _cwt_window(w: np.ndarray) -> np.ndarray:
    """w: (14, 128) -> (14, 64, 128) float32 log1p-normalized"""
    out = np.zeros((N_CH, N_FREQ, WIN), dtype=np.float32)
    for ch in range(N_CH):
        sig = np.pad(w[ch], PAD, mode="reflect")
        c, _ = pywt.cwt(sig, CWT_SCALES, WAVELET, sampling_period=1./SFREQ)
        p = np.log1p(np.abs(c[:, PAD:PAD+WIN])**2).astype(np.float32)
        mn, mx = p.min(), p.max()
        if mx > mn:
            p = (p - mn) / (mx - mn)
        out[ch] = p
    return out


def get_scalograms(cache_dir: str, X_raw: np.ndarray) -> np.ndarray:
    """Return (N, 14, 64, 128) float16 memmap, computing if not cached."""
    path  = Path(cache_dir) / "scalograms.dat"
    N     = len(X_raw)
    shape = (N, N_CH, N_FREQ, WIN)
    exp_bytes = int(np.prod(shape)) * 2   # float16 = 2 bytes

    if path.exists() and path.stat().st_size == exp_bytes:
        print("  [scalogram] cache valid — skipping")
        return np.memmap(path, dtype="float16", mode="r", shape=shape)

    path.unlink(missing_ok=True)
    print(f"  [scalogram] computing {N} windows...")
    mmap = np.memmap(path, dtype="float16", mode="w+", shape=shape)
    for i in tqdm(range(N), desc="  CWT"):
        mmap[i] = _cwt_window(X_raw[i]).astype(np.float16)
    mmap.flush()
    return mmap

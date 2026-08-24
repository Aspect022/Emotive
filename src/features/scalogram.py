"""
src/features/scalogram.py
CWT Scalogram generation for CogProfile-Net v2.

Converts raw EEG sub-windows (14, 128) → stacked scalogram tensors (14, F, 128).
Uses complex Morlet wavelet via PyWavelets. Computed on-the-fly in DataLoader.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    print("[WARN] PyWavelets not installed. Run: pip install PyWavelets")


# -----------------------------------------------------------------
# CWT configuration
# -----------------------------------------------------------------
N_FREQ_BINS   = 64           # logarithmically spaced frequency bins
FREQ_MIN      = 1.0          # Hz  (avoid edge effects at 0.5 Hz in 1s window)
FREQ_MAX      = 45.0         # Hz  (Nyquist-safe for 128 Hz)
SFREQ         = 128          # sampling frequency
WAVELET       = "cmor1.5-1.0"  # complex Morlet: bandwidth=1.5, centre freq=1.0
PAD_SAMPLES   = 32           # zero-pad each side before CWT, crop after


def _build_cwt_scales() -> np.ndarray:
    """Return CWT scales corresponding to FREQ_MIN..FREQ_MAX (log-spaced)."""
    freqs = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), N_FREQ_BINS)
    # pywt scale = centre_frequency / (freq * dt)  for complex Morlet
    centre_freq = pywt.central_frequency(WAVELET)
    scales = centre_freq / (freqs * (1.0 / SFREQ))
    return scales[::-1].copy()   # ascending scale order (descending freq)


_CWT_SCALES: np.ndarray | None = None


def get_cwt_scales() -> np.ndarray:
    global _CWT_SCALES
    if _CWT_SCALES is None:
        _CWT_SCALES = _build_cwt_scales()
    return _CWT_SCALES


def compute_scalogram(window: np.ndarray) -> np.ndarray:
    """
    Compute stacked CWT scalogram for a single EEG window.

    Args:
        window: shape (n_channels, n_timepoints) = (14, 128)

    Returns:
        scalogram: shape (n_channels, N_FREQ_BINS, n_timepoints) = (14, 64, 128)
                   Values: log(1 + power), min-max normalised to [0, 1]
    """
    assert PYWT_AVAILABLE, "PyWavelets required: pip install PyWavelets"
    scales = get_cwt_scales()
    n_ch, T = window.shape
    out = np.zeros((n_ch, N_FREQ_BINS, T), dtype=np.float32)

    for ch in range(n_ch):
        sig = window[ch]
        # Zero-pad both sides to reduce edge effects at low frequencies
        sig_padded = np.pad(sig, PAD_SAMPLES, mode="reflect")
        coeffs, _ = pywt.cwt(sig_padded, scales, WAVELET, sampling_period=1.0 / SFREQ)
        # coeffs shape: (N_FREQ_BINS, T+2*PAD)
        power = np.abs(coeffs[:, PAD_SAMPLES: PAD_SAMPLES + T]) ** 2
        # Log-compress
        power = np.log1p(power).astype(np.float32)
        # Per-channel-scalogram min-max normalisation
        vmin, vmax = power.min(), power.max()
        if vmax > vmin:
            power = (power - vmin) / (vmax - vmin)
        out[ch] = power

    return out   # (14, 64, 128)


# -----------------------------------------------------------------
# PyTorch Dataset
# -----------------------------------------------------------------
class ScalogramDataset(Dataset):
    """
    Lazy scalogram dataset. Computes CWT on-the-fly to avoid 17 GB precomputed storage.

    Args:
        X:      Raw EEG windows, shape (N, 14, 128)
        y:      Class labels, shape (N,)
        meta:   Optional dict with 'rt', 'difficulty', 'accuracy' arrays of shape (N,)
        augment: If True, apply basic time-flip augmentation (training only)
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        meta: dict | None = None,
        augment: bool = False,
    ):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.meta = meta
        self.augment = augment

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        window = self.X[idx]   # (14, 128)
        label  = self.y[idx]

        # Optional time-flip augmentation (training only)
        if self.augment and np.random.rand() < 0.3:
            window = window[:, ::-1].copy()

        scalogram = compute_scalogram(window)  # (14, 64, 128)
        scalogram_t = torch.tensor(scalogram, dtype=torch.float32)

        # Behavioural metadata (optional)
        if self.meta is not None:
            rt  = float(self.meta["rt"][idx])
            dif = float(self.meta["difficulty"][idx])
            acc = float(self.meta["accuracy"][idx])
            behav = torch.tensor([rt, dif, acc], dtype=torch.float32)
        else:
            behav = torch.zeros(3, dtype=torch.float32)

        return scalogram_t, torch.tensor(label, dtype=torch.long), behav

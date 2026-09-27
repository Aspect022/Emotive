"""
benchmark/features/erds.py
Precompute per-subject ERDS baseline and ERDS memmap.
"""
import numpy as np
from pathlib import Path
from tqdm import tqdm
from benchmark.config import N_CH, N_FREQ, WIN


def _valid_npy(path, shape):
    if not path.exists(): return False
    try:    return np.load(path).shape == shape
    except: return False


def _valid_mm(path, shape):
    if not path.exists(): return False
    return path.stat().st_size == int(np.prod(shape)) * 2


def get_erds(cache_dir: str, scalograms, subj_ids: np.ndarray):
    """
    Returns
    -------
    erds_mmap : (N, 14, 64, 128) float16 memmap  — ERDS scalogram
    """
    cp     = Path(cache_dir)
    N      = len(subj_ids)
    N_SUBJ = int(subj_ids.max()) + 1

    # ── baselines ────────────────────────────────────────────────────────────
    base_path  = cp / "erds_baselines.npy"
    base_shape = (N_SUBJ, N_CH, N_FREQ)
    if _valid_npy(base_path, base_shape):
        print("  [erds] baselines cached — skipping")
        baselines = np.load(base_path)
    else:
        print("  [erds] computing baselines...")
        baselines = np.zeros(base_shape, dtype=np.float32)
        counts    = np.zeros(N_SUBJ, dtype=np.int32)
        for i in range(N):
            baselines[subj_ids[i]] += scalograms[i].astype(np.float32).mean(axis=-1)
            counts[subj_ids[i]]    += 1
        for s in range(N_SUBJ):
            if counts[s]: baselines[s] /= counts[s]
        np.save(base_path, baselines)

    # ── ERDS memmap ───────────────────────────────────────────────────────────
    erds_path  = cp / "erds.dat"
    erds_shape = (N, N_CH, N_FREQ, WIN)
    if _valid_mm(erds_path, erds_shape):
        print("  [erds] memmap cached — skipping")
        return np.memmap(erds_path, dtype="float16", mode="r", shape=erds_shape)

    erds_path.unlink(missing_ok=True)
    print("  [erds] computing ERDS memmap...")
    mmap = np.memmap(erds_path, dtype="float16", mode="w+", shape=erds_shape)
    for i in tqdm(range(N), desc="  ERDS"):
        sc  = scalograms[i].astype(np.float32)
        bl  = baselines[subj_ids[i]]
        ers = (sc - bl[:, :, None]) / (bl[:, :, None] + 1e-8) * 100.
        mmap[i] = np.clip(ers, -300., 300.).astype(np.float16)
    mmap.flush()
    return mmap

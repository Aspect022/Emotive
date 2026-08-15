import sys, json, csv
from pathlib import Path
import numpy as np

ROOT = Path(".")
PREPROC_DIR = ROOT / "data/processed/preprocessed"

npz = list(PREPROC_DIR.glob("*.npz"))
print(f"Preprocessed files: {len(npz)}")

d = np.load(npz[0], allow_pickle=False)
print(f"Keys: {list(d.keys())}")
print(f"eeg_preprocessed shape: {d['eeg_preprocessed'].shape}")
sr = d.get("sampling_rate", None)
if sr is not None:
    print(f"Stored fs: {sr}")

# Check a few for 256Hz
count_256 = 0
for f in npz:
    d2 = np.load(f, allow_pickle=False)
    ts = d2["timestamps"]
    if len(ts) > 2:
        fs_est = 1. / float(np.median(np.diff(ts[:200])))
        if abs(fs_est - 256.) < 20:
            count_256 += 1
print(f"Files with 256 Hz: {count_256}/{len(npz)}")
print(f"Files with 128 Hz: {len(npz)-count_256}/{len(npz)}")

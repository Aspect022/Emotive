import numpy as np
import json
from pathlib import Path

X      = np.load("data/processed/dl_dataset/X_raw.npy")
y_task = np.load("data/processed/dl_dataset/y_task.npy")
y_perf = np.load("data/processed/dl_dataset/y_perf.npy")

with open("data/processed/dl_dataset/splits.json") as f:
    splits = json.load(f)

TASKS = {0:"Arithmetic", 1:"Pattern", 2:"Memory", 3:"Comprehension", 4:"Attention"}

print("=" * 55)
print("DATASET SIZE SUMMARY")
print("=" * 55)
print(f"Total windows        : {len(X):,}")
print(f"Shape per window     : 14 channels x 512 timepoints")
print(f"Window duration      : 4 seconds @ 128 Hz")
print(f"Array dtype          : {X.dtype}")
uncompressed_mb = X.nbytes / 1e6
print(f"Uncompressed size    : {uncompressed_mb:.1f} MB")
print(f"Participants (total) : 25")
print(f"  Train participants : {len(splits['train_participants'])}")
print(f"  Val   participants : {len(splits['val_participants'])}")
print(f"  Test  participants : {len(splits['test_participants'])}")
print()
print("-" * 55)
print("SPLIT DISTRIBUTION")
print("-" * 55)
for name, key in [("Train","train_idx"),("Val","val_idx"),("Test","test_idx")]:
    idx = splits[key]
    pct = 100*len(idx)/len(X)
    print(f"  {name:<6}: {len(idx):>5,} windows  ({pct:.1f}%)")
print()
print("-" * 55)
print("TASK DISTRIBUTION (all 2,908 windows)")
print("-" * 55)
for lbl, name in TASKS.items():
    cnt = (y_task == lbl).sum()
    pct = 100*cnt/len(y_task)
    bar = "#" * int(pct/2)
    print(f"  {name:<15}: {cnt:>4} windows  ({pct:.1f}%)  {bar}")
print()
print("-" * 55)
print("PERFORMANCE LABEL DISTRIBUTION")
print("-" * 55)
high = (y_perf == 1).sum()
low  = (y_perf == 0).sum()
print(f"  High scorer (>=7/10) : {high:>4} ({100*high/len(y_perf):.1f}%)")
print(f"  Low  scorer  (<7/10) : {low:>4}  ({100*low/len(y_perf):.1f}%)")
print()
print("=" * 55)
print("RAW vs PROCESSED")
print("=" * 55)
raw_size = sum(f.stat().st_size for f in Path("Data").rglob("*.md.bp.csv"))
raw_size += sum(f.stat().st_size for f in Path("Data").rglob("*.md.pm.bp.csv"))
raw_size += sum(f.stat().st_size for f in Path("Data").rglob("*.md.csv")
               if "participants" not in f.name)
proc_size = sum(f.stat().st_size for f in Path("data/processed/preprocessed").glob("*.npz"))
print(f"  Raw EEG files        : {raw_size/1e9:.2f} GB")
print(f"  Preprocessed (.npz)  : {proc_size/1e6:.1f} MB")
print(f"  Final dataset (X)    : {uncompressed_mb:.1f} MB")
print(f"  Compression ratio    : {raw_size/X.nbytes:.0f}x reduction")

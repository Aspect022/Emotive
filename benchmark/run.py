"""
benchmark/run.py  (or run from project root as: python -m benchmark.run)
Entry point.  Orchestrates: feature precomputation -> fold splits -> arch loop.

Usage:
    python run.py
    python run.py --archs A B --epochs 30 --folds 3
    python run.py --data_dir /path/to/data --cache_dir /tmp/eeg_cache
"""
import sys, os, gc, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import GroupKFold

from benchmark.config import get_args, N_CLASSES
from benchmark.data.loader import load_and_subwindow
from benchmark.data.dataset import (
    ArchADataset, ArchBDataset, ArchCDataset, ArchDDataset
)
from benchmark.features.scalogram  import get_scalograms, FREQS_ASC
from benchmark.features.erds       import get_erds
from benchmark.features.nonlinear  import get_nonlinear
from benchmark.features.graph_nodes import get_node_feats
from benchmark.features.riemannian  import get_riemannian
from benchmark.models.arch_a import ArchA
from benchmark.models.arch_b import ArchB
from benchmark.models.arch_c import ArchC
from benchmark.models.arch_d import ArchD
from benchmark.training.trainer    import run_fold
from benchmark.training.leaderboard import save_and_print, ARCH_DESC


ARCH_MODEL = {"A": ArchA, "B": ArchB, "C": ArchC, "D": ArchD}

DATASET_FACTORY = {
    "A": lambda idx, X, y, sc, **kw: ArchADataset(idx, X, y, sc),
    "B": lambda idx, X, y, sc, **kw: ArchBDataset(idx, X, y, sc, kw["nf"], kw["nl"], kw["fa"]),
    "C": lambda idx, X, y, sc, **kw: ArchCDataset(idx, X, y, sc, kw["nf"], kw["rf"], kw["fa"]),
    "D": lambda idx, X, y, sc, **kw: ArchDDataset(idx, X, y, sc, kw["fa"]),
}


def main():
    args = get_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}")
    print(f"  CogProfile-Net  —  Overnight Benchmark")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Device : {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"  Archs  : {args.archs}   Folds: {args.folds}   Epochs: {args.epochs}")
    print(f"{'='*65}\n")

    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading data...")
    X_raw, y, subj_ids = load_and_subwindow(args.data_dir)
    N = len(y)

    # ── Precompute features (cached) ──────────────────────────────────────────
    print("\nPrecomputing features...")
    scalograms = get_scalograms(args.cache_dir, X_raw)
    get_erds(args.cache_dir, scalograms, subj_ids)   # side-effect: writes cache

    needed_b_c = any(a in args.archs for a in ["B","C"])
    node_feats  = get_node_feats(args.cache_dir, X_raw)  if needed_b_c else None
    nonlin_feats= get_nonlinear (args.cache_dir, X_raw)  if "B" in args.archs else None
    riem_feats  = get_riemannian(args.cache_dir, X_raw)  if "C" in args.archs else None

    # shared kwargs for dataset factory
    ds_kw = dict(nf=node_feats, nl=nonlin_feats, rf=riem_feats, fa=FREQS_ASC)

    # ── Fold splits (GroupKFold — subjects never leak) ────────────────────────
    gkf    = GroupKFold(n_splits=args.folds)
    splits = []
    for trn, te in gkf.split(np.arange(N), y, groups=subj_ids):
        n_val  = max(1, int(len(np.unique(subj_ids[trn])) * 0.2))
        val_s  = set(np.unique(subj_ids[trn])[:n_val])
        vm     = np.array([subj_ids[i] in val_s for i in trn])
        splits.append((trn[~vm], trn[vm], te))

    # ── Main benchmark loop ───────────────────────────────────────────────────
    all_results = {}
    for arch in args.archs:
        if arch not in ARCH_MODEL:
            print(f"  Unknown arch '{arch}' — skipping"); continue
        print(f"\n{'='*65}")
        print(f"  ARCHITECTURE {arch}: {ARCH_DESC.get(arch,'')}")
        print(f"{'='*65}")

        fold_results = []
        t0 = time.time()

        for fi, (trn_idx, val_idx, te_idx) in enumerate(splits):
            print(f"\n  Fold {fi+1}/{args.folds}  "
                  f"(trn={len(trn_idx):,}  val={len(val_idx):,}  te={len(te_idx):,})")

            ds_trn = DATASET_FACTORY[arch](trn_idx, X_raw, y, scalograms, **ds_kw)
            ds_val = DATASET_FACTORY[arch](val_idx, X_raw, y, scalograms, **ds_kw)
            ds_te  = DATASET_FACTORY[arch](te_idx,  X_raw, y, scalograms, **ds_kw)

            model = ARCH_MODEL[arch]().to(DEVICE)
            m = run_fold(arch, model, ds_trn, ds_val, ds_te, args, DEVICE)
            fold_results.append(m)

            print(f"\n    Fold {fi+1} result: "
                  f"acc={m['acc']*100:.2f}%  f1={m['f1']:.4f}  kappa={m['kappa']:.4f}")

            del model; gc.collect(); torch.cuda.empty_cache()

        accs = [r["acc"] for r in fold_results]
        all_results[arch] = {
            "description":  ARCH_DESC.get(arch, ""),
            "folds":        fold_results,
            "mean_acc":     float(np.mean(accs)),
            "std_acc":      float(np.std(accs)),
            "mean_f1":      float(np.mean([r["f1"]    for r in fold_results])),
            "mean_kappa":   float(np.mean([r["kappa"] for r in fold_results])),
            "elapsed_h":    (time.time() - t0) / 3600.,
        }
        print(f"\n  Arch {arch} DONE: "
              f"{np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")

    save_and_print(all_results, args.out_dir)


if __name__ == "__main__":
    main()

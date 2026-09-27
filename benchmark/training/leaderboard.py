"""
benchmark/training/leaderboard.py
Save and pretty-print the final benchmark leaderboard.
"""
import json
import numpy as np
from pathlib import Path
from datetime import datetime


ARCH_DESC = {
    "A": "v2  ScalogramCNN + RiemannianSPD        [baseline 81.99%]",
    "B": "v4  Multi-Band TinyViT + GATv2 + Nonlinear",
    "C": "v5  Band-GAT + SpectralTransformer + Riemannian",
    "D": "v6  ConvLSTM Topomap + EEGNet + Engagement Index",
}


def save_and_print(all_results: dict, out_dir: str):
    out = Path(out_dir)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Individual arch files
    for arch, res in all_results.items():
        with open(out / f"arch_{arch}_{ts}.json", "w") as f:
            json.dump(res, f, indent=2)

    # Combined leaderboard
    with open(out / f"leaderboard_{ts}.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Pretty print
    w = 68
    print(f"\n\n{'='*w}")
    print(f"  FINAL LEADERBOARD  —  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*w}")
    print(f"  {'Arch':<6}  {'Accuracy':>14}  {'Macro-F1':>10}  {'Kappa':>8}  {'Time':>6}")
    print(f"  {'─'*58}")
    ranked = sorted(all_results.items(), key=lambda x: x[1]["mean_acc"], reverse=True)
    for arch, r in ranked:
        acc_str = f"{r['mean_acc']*100:.2f}% ±{r['std_acc']*100:.2f}%"
        print(f"  {arch:<6}  {acc_str:>14}  {r['mean_f1']:>10.4f}"
              f"  {r['mean_kappa']:>8.4f}  {r['elapsed_h']:>5.1f}h")
    print(f"{'='*w}")
    for arch, r in ranked:
        print(f"\n  [{arch}] {ARCH_DESC.get(arch,'')}")
        for fold_r in r["folds"]:
            print(f"       Fold  acc={fold_r['acc']*100:.2f}%  "
                  f"f1={fold_r['f1']:.4f}  kappa={fold_r['kappa']:.4f}")
    print()
    print(f"  Saved: {out}/leaderboard_{ts}.json")

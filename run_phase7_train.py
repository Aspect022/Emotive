"""
Run Phase 7 only: Load existing dataset and train 3 DL models.
Assumes X_raw.npy, y_task.npy, splits.json already exist.
"""
import json
import csv
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT        = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

DATASET_DIR = ROOT / "data" / "processed" / "dl_dataset"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
MODELS_DIR  = RESULTS_DIR / "models"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TASK_LABEL_MAP = {"arithmetic":0,"pattern":1,"memory":2,"comprehension":3,"attention":4}
TASK_COLORS    = ["#E63946","#2A9D8F","#E9C46A","#457B9D","#9B5DE5"]
TASK_NAMES     = ["Arithmetic","Pattern","Memory","Comprehension","Attention"]
WIN_SAMPLES    = 512
N_CLASSES      = 5

# ── Load dataset ──────────────────────────────────────────────────────────────
print("Loading dataset...")
X      = np.load(DATASET_DIR / "X_raw.npy")
y_task = np.load(DATASET_DIR / "y_task.npy")

with open(DATASET_DIR / "splits.json") as f:
    splits = json.load(f)

print(f"  X shape: {X.shape}")
print(f"  y_task shape: {y_task.shape}")
print(f"  Train: {len(splits['train_idx'])} | Val: {len(splits['val_idx'])} | Test: {len(splits['test_idx'])}")

task_dist = Counter(y_task.tolist())
print("  Task distribution:")
inv = {v:k for k,v in TASK_LABEL_MAP.items()}
for lbl, cnt in sorted(task_dist.items()):
    print(f"    {inv[lbl]}: {cnt} ({100*cnt/len(y_task):.1f}%)")

# ── Models ────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cpu")
print(f"\nDevice: {device}")


class EEGNet(nn.Module):
    """EEGNet (Lawhern et al. 2018) - the standard lightweight EEG DL model."""
    def __init__(self, n_classes=5, n_ch=14, n_t=512, F1=8, D=2, F2=16, drop=0.4):
        super().__init__()
        kern = 64  # half of fs (128 Hz)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern), padding=(0, kern//2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1*D, (n_ch, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1*D), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(drop),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1*D, F2, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(drop),
        )
        # Compute flat size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_ch, n_t)
            flat  = self.block2(self.block1(dummy)).numel()
        self.fc = nn.Linear(flat, n_classes)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)       # (B,1,C,T)
        return self.fc(self.block2(self.block1(x)).flatten(1))


class TinyCNN(nn.Module):
    """Lightweight 1D-CNN baseline."""
    def __init__(self, n_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(14, 32, 16, padding=8, bias=False),
            nn.BatchNorm1d(32), nn.ELU(), nn.MaxPool1d(4), nn.Dropout(0.3),
            nn.Conv1d(32, 64, 8, padding=4, bias=False),
            nn.BatchNorm1d(64), nn.ELU(), nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(64*16, 128), nn.ELU(), nn.Dropout(0.4),
            nn.Linear(128, n_classes),
        )
    def forward(self, x): return self.net(x)


class TinyLSTM(nn.Module):
    """Bidirectional LSTM baseline."""
    def __init__(self, n_classes=5):
        super().__init__()
        self.lstm = nn.LSTM(14, 64, 2, batch_first=True, bidirectional=True, dropout=0.3)
        self.head  = nn.Sequential(nn.Dropout(0.4), nn.Linear(128, n_classes))
    def forward(self, x):
        x, _ = self.lstm(x.permute(0,2,1))  # (B,T,C)
        return self.head(x[:, -1, :])


MODELS = {
    "EEGNet":   EEGNet(n_classes=N_CLASSES, n_ch=14, n_t=WIN_SAMPLES),
    "TinyCNN":  TinyCNN(n_classes=N_CLASSES),
    "TinyLSTM": TinyLSTM(n_classes=N_CLASSES),
}

# ── Training helpers ──────────────────────────────────────────────────────────
def make_loaders(X, y, splits, bs=32):
    def ds(idx):
        return TensorDataset(torch.tensor(X[idx], dtype=torch.float32),
                             torch.tensor(y[idx], dtype=torch.long))
    return (DataLoader(ds(splits["train_idx"]), bs, shuffle=True),
            DataLoader(ds(splits["val_idx"]),   bs, shuffle=False),
            DataLoader(ds(splits["test_idx"]),  bs, shuffle=False))


def class_weights(y_tr, n=5):
    c = np.bincount(y_tr, minlength=n).astype(float)
    c = np.where(c == 0, 1., c)
    w = 1./c / (1./c).sum() * n
    return torch.tensor(w, dtype=torch.float32)


def macro_f1(preds, labels, n=5):
    f1s = []
    for c in range(n):
        tp = ((preds==c)&(labels==c)).sum()
        fp = ((preds==c)&(labels!=c)).sum()
        fn = ((preds!=c)&(labels==c)).sum()
        f1s.append(float(2*tp / (2*tp+fp+fn+1e-8)))
    return float(np.mean(f1s)), f1s


def run_epoch(model, loader, opt, crit, device, train=True):
    model.train(train)
    loss_sum, cor, tot = 0., 0, 0
    all_p, all_y = [], []
    with torch.set_grad_enabled(train):
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            if train:
                opt.zero_grad()
            out  = model(Xb)
            loss = crit(out, yb)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                opt.step()
            loss_sum += loss.item() * len(yb)
            preds = out.argmax(1)
            cor  += (preds == yb).sum().item()
            tot  += len(yb)
            all_p.extend(preds.cpu().numpy())
            all_y.extend(yb.cpu().numpy())
    all_p, all_y = np.array(all_p), np.array(all_y)
    f1, _ = macro_f1(all_p, all_y)
    return loss_sum/tot, cor/tot, f1, all_p, all_y


# ── Train loop ────────────────────────────────────────────────────────────────
all_results   = {}
all_histories = {}

for mname, model in MODELS.items():
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*55}")
    print(f"  Training: {mname}  ({n_params:,} parameters)")
    print(f"{'='*55}")

    model = model.to(device)
    y_tr  = y_task[splits["train_idx"]]
    crit  = nn.CrossEntropyLoss(weight=class_weights(y_tr).to(device))
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)

    tr_loader, va_loader, te_loader = make_loaders(X, y_task, splits)
    ckpt     = MODELS_DIR / f"{mname}_best.pt"
    best_f1  = -1.
    patience = 10
    wait     = 0
    history  = []

    for ep in range(1, 51):
        tr_loss, tr_acc, tr_f1, _, _ = run_epoch(model, tr_loader, opt, crit, device, True)
        _,       va_acc, va_f1, _, _ = run_epoch(model, va_loader, opt, crit, device, False)
        sched.step()

        history.append({"epoch": ep, "train_loss": round(tr_loss,4),
                         "train_acc": round(tr_acc,4),
                         "val_acc": round(va_acc,4), "val_f1": round(va_f1,4)})

        if ep % 10 == 0 or ep == 1:
            print(f"  Ep {ep:2d}  loss={tr_loss:.4f}  "
                  f"tr_acc={tr_acc:.3f}  val_acc={va_acc:.3f}  val_f1={va_f1:.3f}")

        if va_f1 > best_f1:
            best_f1 = va_f1; wait = 0
            torch.save(model.state_dict(), ckpt)
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stop at epoch {ep}")
                break

    # ── Test evaluation ───────────────────────────────────────────────────────
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    _, te_acc, te_f1, te_preds, te_labels = run_epoch(
        model, te_loader, opt, crit, device, False)
    _, te_f1_final, f1_per_class = te_f1, *macro_f1(te_preds, te_labels)

    print(f"\n  [{mname}] Test Accuracy: {te_acc:.4f}  |  Test Macro-F1: {te_f1:.4f}")
    print(f"  Chance level: {1/N_CLASSES:.4f}  |  Gain over chance: {te_acc-1/N_CLASSES:+.4f}")

    all_results[mname] = {
        "n_params":        n_params,
        "best_val_f1":     round(best_f1, 4),
        "test_accuracy":   round(float(te_acc), 4),
        "test_macro_f1":   round(float(te_f1), 4),
        "test_f1_per_class": [round(f,4) for f in f1_per_class],
        "test_predictions":  te_preds.tolist(),
        "test_labels":       te_labels.tolist(),
    }
    all_histories[mname] = history

# ── Save results JSON ─────────────────────────────────────────────────────────
with open(RESULTS_DIR / "model_results.json", "w") as f:
    out = {m:{k:v for k,v in r.items() if k not in ("test_predictions","test_labels")}
           for m,r in all_results.items()}
    json.dump(out, f, indent=2)
print(f"\n[OK] model_results.json saved")

# ── RESULTS TABLE ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"{'Model':<15} {'Params':>8} {'Val F1':>8} {'Test Acc':>10} {'Test F1':>10}")
print("-"*60)
for mname, r in all_results.items():
    print(f"{mname:<15} {r['n_params']:>8,} {r['best_val_f1']:>8.4f} "
          f"{r['test_accuracy']:>10.4f} {r['test_macro_f1']:>10.4f}")
print(f"{'Chance (20%)':.<15} {'':>8} {'':>8} {'0.2000':>10} {'0.2000':>10}")
print("="*60)

# ── FIGURES ───────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.family":"DejaVu Sans",
                     "axes.spines.top":False, "axes.spines.right":False})
colors = {"EEGNet":"#E63946","TinyCNN":"#2A9D8F","TinyLSTM":"#9B5DE5"}

# Fig 6: Training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for mname, hist in all_histories.items():
    epochs = [h["epoch"] for h in hist]
    tr_acc = [h["train_acc"] for h in hist]
    va_acc = [h["val_acc"] for h in hist]
    va_f1  = [h["val_f1"] for h in hist]
    c = colors[mname]
    axes[0].plot(epochs, tr_acc, "--", color=c, alpha=0.45, lw=1.5)
    axes[0].plot(epochs, va_acc, "-",  color=c, lw=2.5, label=f"{mname} (val)")
    axes[1].plot(epochs, va_f1,  "-",  color=c, lw=2.5, label=mname)

for ax, title, ylabel in [
    (axes[0], "Training (dashed) vs Validation (solid) Accuracy", "Accuracy"),
    (axes[1], "Validation Macro-F1 Score per Epoch", "Macro-F1"),
]:
    ax.axhline(0.2, color="gray", ls=":", alpha=0.6, label="Chance (20%)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
    ax.legend(fontsize=9); ax.set_ylim(0, 1)

fig.suptitle("Model Training Curves -- Single-Fold Subject-Independent Evaluation",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig6_training_curves.png", bbox_inches="tight")
plt.close()
print("Saved: fig6_training_curves.png")

# Fig 7: Bar comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
mnames   = list(all_results.keys())
te_accs  = [all_results[m]["test_accuracy"] for m in mnames]
te_f1s   = [all_results[m]["test_macro_f1"] for m in mnames]
mcols    = [colors[m] for m in mnames]

for ax, vals, title in [(axes[0],te_accs,"Test Accuracy"),(axes[1],te_f1s,"Test Macro-F1")]:
    bars = ax.bar(mnames, vals, color=mcols, edgecolor="white", linewidth=2, width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.007,
                f"{v:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.axhline(0.2, color="gray", ls="--", alpha=0.6, label="Chance (20%)")
    ax.set_ylim(0, 1); ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Score"); ax.legend()

fig.suptitle("Model Comparison -- 5-Class Cognitive Task Classification",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig7_model_comparison.png", bbox_inches="tight")
plt.close()
print("Saved: fig7_model_comparison.png")

# Fig 8: Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
short_names = ["Arith","Patt","Mem","Comp","Attn"]
for ax, (mname, res) in zip(axes, all_results.items()):
    preds  = np.array(res["test_predictions"])
    labels = np.array(res["test_labels"])
    cm = np.zeros((5,5), dtype=int)
    for t,p in zip(labels, preds):
        if 0<=t<5 and 0<=p<5: cm[t][p] += 1

    # Normalize per row (recall matrix)
    rs  = cm.sum(axis=1, keepdims=True)
    cmn = np.divide(cm.astype(float), rs, out=np.zeros_like(cm,dtype=float), where=rs>0)

    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(short_names, rotation=30, ha="right")
    ax.set_yticklabels(short_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{mname}\nAcc: {res['test_accuracy']:.3f} | F1: {res['test_macro_f1']:.3f}",
                 fontsize=11, fontweight="bold")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if cmn[i,j]>0.5 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall")

fig.suptitle("Confusion Matrices -- 5-Class Cognitive Task Classification",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig8_confusion_matrices.png", bbox_inches="tight")
plt.close()
print("Saved: fig8_confusion_matrices.png")

print(f"\nAll done! Figures saved to: {FIGURES_DIR}/")

"""
train_cogprofile.py
Main training script for CogProfile-Net v2.

Three-phase training:
  Phase 1: Pre-train encoder only (DEC off) — establish good embeddings
  Phase 2: K-Means init DEC centroids, then joint training (DEC on)
  Phase 3: EDL fine-tuning with behavioral gate

Evaluation across all three tiers (window / trial / subject split).
Run: python train_cogprofile.py
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent

# Add src to path
import sys
sys.path.insert(0, str(ROOT / "src"))

from features.scalogram import ScalogramDataset
from models.cogprofile import CogProfileNet
from training.splits import tier1_window_split, tier2_trial_split, tier3_subject_split

# ------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------ #
CFG = {
    "n_classes":     5,
    "n_channels":    14,
    "n_freq":        64,
    "n_time":        128,     # 1 second @ 128 Hz
    "emb_dim":       128,
    "n_clusters":    8,       # 5 cognitive + 3 noise clusters
    "dropout":       0.5,
    "batch_size":    64,
    "lr":            1e-3,
    "weight_decay":  1e-4,
    "n_epochs_pretrain": 15,  # Phase 1: pre-train encoder
    "n_epochs_joint":    50,  # Phase 2: joint DEC + EDL
    "n_epochs_finetune": 20,  # Phase 3: EDL fine-tune
    "patience":      12,
    "alpha_dec":     0.1,
    "alpha_orth":    0.05,
    "alpha_aux":     0.3,
    "seed":          42,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[CogProfileNet] Device: {DEVICE}")


# ------------------------------------------------------------------ #
# Load dataset
# ------------------------------------------------------------------ #
def load_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load raw EEG trials and sub-window them to (N_windows, 14, 128)."""
    DL = ROOT / "data" / "processed" / "dl_dataset"

    X_raw = np.load(DL / "X_raw.npy")           # (2908, 14, 512)
    y     = np.load(DL / "y_task.npy")           # (2908,)
    pids  = np.load(DL / "participant_ids.npy", allow_pickle=True)  # (2908,)

    print(f"[Data] Loaded X={X_raw.shape}, y={y.shape}, subjects={len(np.unique(pids))}")

    # Sub-window each 4s trial into 1s windows with 75% overlap
    W, STEP = 128, 32    # window=1s, step=0.25s
    windows, labels, trial_ids, subject_ids = [], [], [], []

    for i, (trial, label, pid) in enumerate(zip(X_raw, y, pids)):
        T = trial.shape[1]   # 512
        for start in range(0, T - W + 1, STEP):
            windows.append(trial[:, start: start + W])
            labels.append(label)
            trial_ids.append(i)
            subject_ids.append(pid)

    X_win = np.stack(windows).astype(np.float32)   # (N, 14, 128)
    y_win = np.array(labels, dtype=np.int64)
    t_ids = np.array(trial_ids, dtype=np.int64)
    s_ids = np.array(subject_ids)

    print(f"[Data] Sub-windows: {X_win.shape}, class dist: {dict(zip(*np.unique(y_win, return_counts=True)))}")

    # Build metadata (placeholder — use window_meta.csv if available)
    meta = {
        "rt":         np.ones(len(X_win)) * 0.5,
        "difficulty": np.ones(len(X_win)) * 0.5,
        "accuracy":   np.ones(len(X_win)) * 0.5,
    }

    return X_win, y_win, t_ids, s_ids, meta


# ------------------------------------------------------------------ #
# Evaluation helper
# ------------------------------------------------------------------ #
@torch.no_grad()
def evaluate_model(model, loader, device, n_classes=5):
    model.eval()
    all_preds, all_labels, all_u = [], [], []

    for batch in loader:
        scalogram, raw_eeg, behav, labels = batch
        scalogram = scalogram.to(device)
        raw_eeg   = raw_eeg.to(device)
        behav     = behav.to(device)
        labels    = labels.to(device)

        out = model(scalogram, raw_eeg, behav)
        probs = out["edl"]["prob"]
        preds = probs.argmax(dim=1)
        u     = out["edl"]["uncertainty"].squeeze(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_u.extend(u.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_u      = np.array(all_u)
    acc = (all_preds == all_labels).mean()

    # Macro F1
    f1s = []
    for c in range(n_classes):
        tp = ((all_preds == c) & (all_labels == c)).sum()
        fp = ((all_preds == c) & (all_labels != c)).sum()
        fn = ((all_preds != c) & (all_labels == c)).sum()
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f1s.append(2 * p * r / (p + r + 1e-8))

    return {
        "accuracy": float(acc),
        "macro_f1": float(np.mean(f1s)),
        "mean_uncertainty": float(all_u.mean()),
    }


# ------------------------------------------------------------------ #
# DualBranchDataset — returns (scalogram, raw_eeg, behav, label)
# ------------------------------------------------------------------ #
class DualDataset(torch.utils.data.Dataset):
    """Returns scalogram AND raw EEG for the dual-branch model."""

    def __init__(self, X, y, meta, augment=False):
        from features.scalogram import compute_scalogram
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.meta = meta
        self.augment = augment
        self._scalogram_fn = compute_scalogram

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        window = self.X[idx]   # (14, 128)
        if self.augment and np.random.rand() < 0.3:
            window = window[:, ::-1].copy()

        scalogram = self._scalogram_fn(window)   # (14, 64, 128)
        raw       = window                        # (14, 128)

        behav = np.array([
            self.meta["rt"][idx],
            self.meta["difficulty"][idx],
            self.meta["accuracy"][idx],
        ], dtype=np.float32)

        return (
            torch.tensor(scalogram, dtype=torch.float32),
            torch.tensor(raw, dtype=torch.float32),
            torch.tensor(behav, dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


# ------------------------------------------------------------------ #
# Main training
# ------------------------------------------------------------------ #
def main():
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])

    # Load data
    X, y, trial_ids, subject_ids, meta = load_dataset()

    # All three split tiers
    splits = {
        "tier1": tier1_window_split(y, seed=CFG["seed"]),
        "tier2": tier2_trial_split(
            trial_ids=np.arange(len(np.unique(trial_ids))),
            y_trial=y[np.array([np.where(trial_ids == t)[0][0] for t in np.unique(trial_ids)])],
            window_to_trial=trial_ids,
            y_window=y,
            seed=CFG["seed"],
        ),
        "tier3": tier3_subject_split(subject_ids, y, seed=CFG["seed"]),
    }

    # Use Tier 2 (clean trial-level) for training
    sp = splits["tier2"]
    print(f"[Split Tier2] Train windows={len(sp['train_idx'])} | "
          f"Val={len(sp['val_idx'])} | Test={len(sp['test_idx'])}")

    train_ds = DualDataset(X[sp["train_idx"]], y[sp["train_idx"]],
                           {k: v[sp["train_idx"]] for k, v in meta.items()}, augment=True)
    val_ds   = DualDataset(X[sp["val_idx"]], y[sp["val_idx"]],
                           {k: v[sp["val_idx"]] for k, v in meta.items()})
    test_ds  = DualDataset(X[sp["test_idx"]], y[sp["test_idx"]],
                           {k: v[sp["test_idx"]] for k, v in meta.items()})

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"], shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=CFG["batch_size"], shuffle=False, num_workers=0)

    # Model
    model = CogProfileNet(
        n_classes=CFG["n_classes"], n_channels=CFG["n_channels"],
        n_freq=CFG["n_freq"], n_time=CFG["n_time"],
        emb_dim=CFG["emb_dim"], n_clusters=CFG["n_clusters"],
        dropout=CFG["dropout"], use_behavioral=True,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] CogProfileNet — {n_params:,} trainable parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["n_epochs_pretrain"] + CFG["n_epochs_joint"]
    )

    best_val_acc = 0.0
    history = []
    CKPT = ROOT / "results" / "models" / "cogprofile_net_best.pt"
    CKPT.parent.mkdir(parents=True, exist_ok=True)

    # ---- Phase 1: Pre-train encoder (DEC off) ----
    print("\n=== Phase 1: Pre-training encoder ===")
    for epoch in range(1, CFG["n_epochs_pretrain"] + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for scalogram, raw_eeg, behav, labels in train_loader:
            scalogram, raw_eeg = scalogram.to(DEVICE), raw_eeg.to(DEVICE)
            behav, labels = behav.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            out = model(scalogram, raw_eeg, behav)
            # Phase 1: only CE + aux (no DEC)
            ce_loss  = nn.CrossEntropyLoss()(out["edl"]["prob"], labels)
            aux_loss = nn.CrossEntropyLoss()(out["aux_logits"], labels) if out["aux_logits"] is not None else 0
            loss = ce_loss + CFG["alpha_aux"] * aux_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()
        if epoch % 5 == 0:
            val_m = evaluate_model(model, val_loader, DEVICE)
            print(f"  Pre-train Epoch {epoch:3d}: loss={total_loss/n_batches:.4f} "
                  f"val_acc={val_m['accuracy']:.4f} u={val_m['mean_uncertainty']:.3f}")

    # ---- Phase 2: Initialise DEC centroids via K-Means ----
    print("\n=== Phase 2a: Initialising DEC centroids via K-Means ===")
    model.eval()
    all_emb = []
    with torch.no_grad():
        for scalogram, raw_eeg, behav, _ in train_loader:
            scalogram, raw_eeg, behav = scalogram.to(DEVICE), raw_eeg.to(DEVICE), behav.to(DEVICE)
            out = model(scalogram, raw_eeg, behav)
            all_emb.append(out["embedding"].cpu().numpy())
    all_emb = np.concatenate(all_emb, axis=0)
    model.dec.initialise_from_kmeans(all_emb, seed=CFG["seed"])

    # ---- Phase 2: Joint DEC + EDL training ----
    print("\n=== Phase 2b: Joint training (DEC + EDL) ===")
    patience_counter = 0
    for epoch in range(1, CFG["n_epochs_joint"] + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for scalogram, raw_eeg, behav, labels in train_loader:
            scalogram, raw_eeg = scalogram.to(DEVICE), raw_eeg.to(DEVICE)
            behav, labels = behav.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            out  = model(scalogram, raw_eeg, behav)
            losses = model.compute_loss(
                out, labels, epoch,
                alpha_dec=CFG["alpha_dec"],
                alpha_orth=CFG["alpha_orth"],
                alpha_aux=CFG["alpha_aux"],
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += losses["total"].item()
            n_batches += 1
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            val_m = evaluate_model(model, val_loader, DEVICE)
            print(f"  Joint Epoch {epoch:3d}: loss={total_loss/n_batches:.4f} "
                  f"val_acc={val_m['accuracy']:.4f} val_f1={val_m['macro_f1']:.4f} "
                  f"u={val_m['mean_uncertainty']:.3f}")
            history.append({"epoch": epoch, **val_m})

            if val_m["accuracy"] > best_val_acc:
                best_val_acc = val_m["accuracy"]
                patience_counter = 0
                torch.save(model.state_dict(), CKPT)
                print(f"    ✓ New best saved ({best_val_acc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= CFG["patience"] // 5:
                    print(f"  Early stopping at joint epoch {epoch}")
                    break

    # ---- Final evaluation ----
    print("\n=== Final Evaluation ===")
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True))

    for tier_name, sp_tier in splits.items():
        tier_ds = DualDataset(
            X[sp_tier["test_idx"]], y[sp_tier["test_idx"]],
            {k: v[sp_tier["test_idx"]] for k, v in meta.items()},
        )
        tier_loader = DataLoader(tier_ds, batch_size=CFG["batch_size"], num_workers=0)
        m = evaluate_model(model, tier_loader, DEVICE)
        print(f"  [{tier_name.upper()}] acc={m['accuracy']:.4f} "
              f"f1={m['macro_f1']:.4f} u={m['mean_uncertainty']:.3f}")

    # Save results
    results = {"config": CFG, "best_val_acc": best_val_acc, "history": history}
    out_path = ROOT / "results" / "cogprofile_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Done] Results saved to {out_path}")


if __name__ == "__main__":
    main()

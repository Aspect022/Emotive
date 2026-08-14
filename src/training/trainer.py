"""
Training Engine — Common training loop for all EEG DL models

Features:
  - Subject-independent evaluation (no participant leakage)
  - Class-weighted cross-entropy (handles imbalance)
  - Cosine annealing LR schedule
  - Early stopping (patience-based)
  - Best model checkpointing
  - Per-epoch metrics logging (loss, accuracy, F1)
"""

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "results" / "models"
LOGS_DIR = ROOT / "results" / "training_logs"


def compute_class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse frequency class weights for imbalanced datasets."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * n_classes
    return torch.tensor(weights, dtype=torch.float32)


def make_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    splits: dict,
    batch_size: int = 32,
    use_weighted_sampler: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders from subject-independent splits.
    X shape: (N, 14, T)
    """
    def make_tensor_dataset(indices):
        X_sub = torch.tensor(X[indices], dtype=torch.float32)
        y_sub = torch.tensor(y[indices], dtype=torch.long)
        return TensorDataset(X_sub, y_sub), y_sub.numpy()

    train_ds, y_train = make_tensor_dataset(splits["train_idx"])
    val_ds, _ = make_tensor_dataset(splits["val_idx"])
    test_ds, _ = make_tensor_dataset(splits["test_idx"])

    # Weighted sampler for training (handles class imbalance)
    train_sampler = None
    if use_weighted_sampler and len(y_train) > 0:
        n_classes = len(np.unique(y_train))
        class_weights = compute_class_weights(y_train, n_classes)
        sample_weights = class_weights[y_train]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights.tolist(),
            num_samples=len(y_train),
            replacement=True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    n_classes: int,
) -> dict:
    """Returns dict with loss, accuracy, per-class metrics."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    N = len(all_labels)

    acc = (all_preds == all_labels).mean()

    # Macro F1 (unweighted mean across classes)
    f1_per_class = []
    for c in range(n_classes):
        tp = ((all_preds == c) & (all_labels == c)).sum()
        fp = ((all_preds == c) & (all_labels != c)).sum()
        fn = ((all_preds != c) & (all_labels == c)).sum()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        f1_per_class.append(float(f1))

    macro_f1 = float(np.mean(f1_per_class))

    return {
        "loss": total_loss / N if N > 0 else 0,
        "accuracy": float(acc),
        "macro_f1": macro_f1,
        "f1_per_class": f1_per_class,
        "predictions": all_preds,
        "labels": all_labels,
    }


def train_model(
    model: nn.Module,
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    splits: dict,
    n_classes: int,
    task_name: str = "task",
    n_epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 15,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Full training run for one model on one task.

    Returns summary dict with best val metrics and test metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n  Training {model_name} | Task: {task_name} | Device: {device}")
    print(f"  Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    model = model.to(device)

    # Class-weighted loss
    y_train = y[splits["train_idx"]]
    class_weights = compute_class_weights(y_train, n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    train_loader, val_loader, test_loader = make_dataloaders(
        X, y, splits, batch_size=batch_size
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = MODELS_DIR / f"{model_name}_{task_name}_best.pt"

    best_val_f1 = -1.0
    patience_counter = 0
    history = []

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device, n_classes)
        scheduler.step()

        elapsed = time.time() - t0
        val_f1 = val_metrics["macro_f1"]

        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_metrics["loss"], 4),
            "val_acc": round(val_metrics["accuracy"], 4),
            "val_f1": round(val_f1, 4),
            "lr": round(scheduler.get_last_lr()[0], 6),
            "elapsed_s": round(elapsed, 2),
        }
        history.append(log_entry)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: "
                  f"loss={train_loss:.4f} acc={train_acc:.3f} | "
                  f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.3f} "
                  f"val_f1={val_f1:.3f}")

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    # Load best model and evaluate on test
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    test_metrics = evaluate(model, test_loader, criterion, device, n_classes)

    print(f"\n  === {model_name} | {task_name} Results ===")
    print(f"  Best val F1:  {best_val_f1:.4f}")
    print(f"  Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Test macro-F1: {test_metrics['macro_f1']:.4f}")

    # Save training log
    log_path = LOGS_DIR / f"{model_name}_{task_name}_history.json"
    with open(log_path, "w") as f:
        json.dump({"model": model_name, "task": task_name,
                   "history": history,
                   "best_val_f1": best_val_f1,
                   "test_accuracy": test_metrics["accuracy"],
                   "test_macro_f1": test_metrics["macro_f1"],
                   "test_f1_per_class": test_metrics["f1_per_class"]}, f, indent=2)

    return {
        "model_name": model_name,
        "task_name": task_name,
        "best_val_f1": best_val_f1,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_f1_per_class": test_metrics["f1_per_class"],
        "checkpoint_path": str(ckpt_path),
        "history": history,
    }

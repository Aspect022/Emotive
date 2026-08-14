"""
Model Evaluation & Comparison — Phase 9
Runs all models on all tasks, produces comparison tables and confusion matrices.

Usage:
  python run_models.py                   # train + evaluate all models
  python run_models.py --task task       # only task-type classification
  python run_models.py --model eegnet    # only EEGNet
  python run_models.py --eval-only       # skip training, load checkpoints
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

DATASET_DIR = ROOT / "data" / "processed" / "dl_dataset"
RESULTS_DIR = ROOT / "results"


def load_dataset():
    """Load compiled DL dataset."""
    required = ["X_raw.npy", "y_task.npy", "participant_ids.npy",
                "splits.json", "label_map.json"]
    for f in required:
        if not (DATASET_DIR / f).exists():
            print(f"ERROR: {f} not found. Run run_pipeline.py first.")
            sys.exit(1)

    X = np.load(DATASET_DIR / "X_raw.npy")
    y_task = np.load(DATASET_DIR / "y_task.npy")

    with open(DATASET_DIR / "splits.json") as f:
        splits = json.load(f)

    with open(DATASET_DIR / "label_map.json") as f:
        label_map = json.load(f)

    print(f"  Dataset: X={X.shape}, y_task={y_task.shape}")
    print(f"  Train windows: {len(splits['train_idx'])}")
    print(f"  Val windows:   {len(splits['val_idx'])}")
    print(f"  Test windows:  {len(splits['test_idx'])}")

    return X, y_task, splits, label_map


def build_model_zoo(n_channels: int, n_timepoints: int, n_classes: int) -> dict:
    """Returns dict of model_name -> model instance."""
    from models.eegnet import EEGNet
    from models.cnn_lstm import CNN1D, CNNLSTM, LSTM
    from models.transformer import EEGTransformer, MLPBaseline

    zoo = {
        "EEGNet": EEGNet(
            n_classes=n_classes, n_channels=n_channels, n_timepoints=n_timepoints
        ),
        "CNN1D": CNN1D(
            n_classes=n_classes, n_channels=n_channels, n_timepoints=n_timepoints
        ),
        "CNN_LSTM": CNNLSTM(
            n_classes=n_classes, n_channels=n_channels, n_timepoints=n_timepoints
        ),
        "LSTM": LSTM(
            n_classes=n_classes, n_channels=n_channels, n_timepoints=n_timepoints
        ),
        "EEGTransformer": EEGTransformer(
            n_classes=n_classes, n_channels=n_channels, n_timepoints=n_timepoints
        ),
    }
    return zoo


def print_comparison_table(results: list[dict]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 70)
    print(f"{'Model':<20} {'Task':<12} {'Val F1':>8} {'Test Acc':>10} {'Test F1':>10}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: -x["test_macro_f1"]):
        print(
            f"{r['model_name']:<20} {r['task_name']:<12} "
            f"{r['best_val_f1']:>8.4f} {r['test_accuracy']:>10.4f} "
            f"{r['test_macro_f1']:>10.4f}"
        )
    print("=" * 70)

    # Chance level
    # 5-class: 20%, Binary: 50%
    print("\nChance level: 5-class = 20.0%, Binary = 50.0%")


def save_comparison_csv(results: list[dict], path: Path):
    if not results:
        return
    fields = ["model_name", "task_name", "best_val_f1", "test_accuracy", "test_macro_f1",
              "checkpoint_path"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fields})
    print(f"  [OK] Comparison saved: {path}")


def plot_confusion_matrix(labels, preds, class_names, model_name, task_name):
    """Save confusion matrix as text table (no matplotlib dependency required)."""
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for true, pred in zip(labels, preds):
        if 0 <= true < n and 0 <= pred < n:
            cm[true][pred] += 1

    cm_dir = RESULTS_DIR / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)
    out_path = cm_dir / f"{model_name}_{task_name}_cm.txt"

    with open(out_path, "w") as f:
        f.write(f"Confusion Matrix: {model_name} | {task_name}\n\n")
        header = f"{'':>14} " + " ".join(f"{c[:8]:>10}" for c in class_names)
        f.write(header + "\n")
        for i, row in enumerate(cm):
            row_str = f"{class_names[i][:13]:>14} " + " ".join(f"{v:>10}" for v in row)
            f.write(row_str + "\n")

        # Per-class accuracy
        f.write("\nPer-class recall:\n")
        for i, c in enumerate(class_names):
            total = cm[i].sum()
            recall = cm[i][i] / total if total > 0 else 0
            f.write(f"  {c}: {recall:.3f} ({cm[i][i]}/{total})\n")

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EEG Model Zoo")
    parser.add_argument("--task", choices=["task", "all"], default="all")
    parser.add_argument("--model", choices=["eegnet", "cnn1d", "cnn_lstm",
                                             "lstm", "transformer", "all"],
                        default="all")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, only evaluate from checkpoints")
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 8+9: MODEL TRAINING & EVALUATION")
    print("=" * 60)

    from training.trainer import train_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    X, y_task, splits, label_map = load_dataset()

    n_channels, n_timepoints = X.shape[1], X.shape[2]
    n_task_classes = len(label_map["task"])

    print(f"\n  EEG shape per window: ({n_channels} channels, {n_timepoints} timepoints)")
    print(f"  Task classes: {n_task_classes}")

    # ── Task-type classification ──────────────────────────────────────────────
    all_results = []

    if args.task in ("task", "all"):
        print("\n" + "─" * 60)
        print("TASK: 5-class task-type classification")
        print("─" * 60)

        model_zoo = build_model_zoo(n_channels, n_timepoints, n_task_classes)
        task_class_names = list(label_map["task"].keys())

        for model_name, model in model_zoo.items():
            if args.model != "all" and model_name.lower() != args.model:
                continue

            if args.eval_only:
                ckpt = RESULTS_DIR / "models" / f"{model_name}_task_best.pt"
                if not ckpt.exists():
                    print(f"  Skipping {model_name} (no checkpoint found)")
                    continue
                model = model.to(device)
                model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

                from training.trainer import make_dataloaders, evaluate
                import torch.nn as nn
                _, _, test_loader = make_dataloaders(X, y_task, splits, args.batch_size)
                criterion = nn.CrossEntropyLoss()
                test_metrics = evaluate(model, test_loader, criterion, device, n_task_classes)
                result = {
                    "model_name": model_name, "task_name": "task",
                    "best_val_f1": 0.0,
                    "test_accuracy": test_metrics["accuracy"],
                    "test_macro_f1": test_metrics["macro_f1"],
                    "test_f1_per_class": test_metrics["f1_per_class"],
                    "checkpoint_path": str(ckpt),
                    "history": [],
                }
            else:
                result = train_model(
                    model=model,
                    model_name=model_name,
                    X=X, y=y_task,
                    splits=splits,
                    n_classes=n_task_classes,
                    task_name="task",
                    n_epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    device=device,
                )

            all_results.append(result)

            # Confusion matrix
            if result.get("history") or args.eval_only:
                # Re-evaluate to get predictions
                from training.trainer import make_dataloaders, evaluate as eval_fn
                import torch.nn as nn
                _, _, test_loader = make_dataloaders(X, y_task, splits, args.batch_size)
                criterion = nn.CrossEntropyLoss()
                model = model.to(device)
                test_m = eval_fn(model, test_loader, criterion, device, n_task_classes)
                cm_path = plot_confusion_matrix(
                    test_m["labels"], test_m["predictions"],
                    task_class_names, model_name, "task"
                )
                print(f"  Confusion matrix: {cm_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print_comparison_table(all_results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_comparison_csv(all_results, RESULTS_DIR / "model_comparison.csv")

    # Full JSON dump for reproducibility
    with open(RESULTS_DIR / "model_results_full.json", "w") as f:
        out = [{k: v for k, v in r.items() if k != "history"} for r in all_results]
        json.dump(out, f, indent=2)
    print(f"  [OK] Full results: {RESULTS_DIR / 'model_results_full.json'}")

"""
benchmark/training/evaluate.py
Evaluation helpers: per-fold metric computation.
"""
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report


@torch.no_grad()
def evaluate_loader(model, loader, forward_fn, device) -> dict:
    model.eval()
    preds, labs, uncs = [], [], []
    for batch in loader:
        out = forward_fn(model, batch, device)
        preds.extend(out["prob"].argmax(1).cpu().numpy())
        labs.extend(batch[-1].numpy())
        uncs.extend(out["uncertainty"].squeeze(1).cpu().numpy())
    preds = np.array(preds); labs = np.array(labs)
    return {
        "acc":         accuracy_score(labs, preds),
        "f1":          f1_score(labs, preds, average="macro", zero_division=0),
        "kappa":       cohen_kappa_score(labs, preds),
        "uncertainty": float(np.mean(uncs)),
        "report":      classification_report(labs, preds, zero_division=0),
    }

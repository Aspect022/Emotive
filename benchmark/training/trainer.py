"""
benchmark/training/trainer.py
Training loop and fold runner.  Arch-agnostic: receives model + forward_fn.
"""
import gc
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from benchmark.models.shared import edl_loss, dec_loss
from benchmark.training.evaluate import evaluate_loader


# ── arch-specific forward dispatchers ────────────────────────────────────────
def forward_a(model, batch, device):
    x_sc, x_raw, _ = batch
    return model(x_sc.to(device), x_raw.to(device))

def forward_b(model, batch, device):
    x_b, x_n, x_a, x_nl, _ = batch
    return model(x_b.to(device), x_n.to(device), x_a.to(device), x_nl.to(device))

def forward_c(model, batch, device):
    x_b, x_n, x_a, x_r, _ = batch
    return model(x_b.to(device), x_n.to(device), x_a.to(device), x_r.to(device))

def forward_d(model, batch, device):
    x_t, x_r, x_ei, _ = batch
    return model(x_t.to(device), x_r.to(device), x_ei.to(device))

FORWARD_MAP = {"A": forward_a, "B": forward_b, "C": forward_c, "D": forward_d}


def run_fold(arch_name, model, dataset_trn, dataset_val, dataset_te,
             args, device) -> dict:
    """Train one fold, return test metrics dict."""
    fwd = FORWARD_MAP[arch_name]
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    make_dl = lambda ds, shuf: DataLoader(
        ds, batch_size=args.batch, shuffle=shuf,
        num_workers=args.workers, pin_memory=True, drop_last=shuf)

    train_dl = make_dl(dataset_trn, True)
    val_dl   = make_dl(dataset_val, False)
    test_dl  = make_dl(dataset_te,  False)

    opt   = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    phase1 = int(args.epochs * 0.7)

    best_acc, best_state = 0., None

    for ep in range(1, args.epochs + 1):
        model.train()
        tot, n_samp = 0., 0
        for batch in train_dl:
            opt.zero_grad()
            out  = fwd(model, batch, device)
            lab  = batch[-1].to(device)
            l    = (edl_loss(out, lab, ep)
                    + F.cross_entropy(out["prob"].log().clamp(-20,20), lab) * 0.1
                    + dec_loss(out["q"]) * (0.05 if ep > phase1 else 0.))
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += l.item() * len(lab); n_samp += len(lab)
        sched.step()

        # K-Means init for DEC at phase boundary
        if ep == phase1 + 1:
            model.eval()
            embs = []
            with torch.no_grad():
                for batch in train_dl:
                    embs.append(fwd(model, batch, device)["z"].cpu().numpy())
            model.dec.init_kmeans(np.concatenate(embs))

        if ep % 5 == 0 or ep == 1:
            m = evaluate_loader(model, val_dl, fwd, device)
            print(f"    [E{ep:3d}] loss={tot/n_samp:.4f}  "
                  f"val_acc={m['acc']*100:.2f}%  f1={m['f1']:.3f}  unc={m['uncertainty']:.3f}")
            if m["acc"] > best_acc:
                best_acc = m["acc"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_m = evaluate_loader(model, test_dl, fwd, device)
    test_m["params"] = n_params
    return test_m

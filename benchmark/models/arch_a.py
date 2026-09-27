"""
benchmark/models/arch_a.py
Architecture A — CogProfile-Net v2: ScalogramCNN + RiemannianSPD.
81.99% baseline.  Keep unchanged as anchor.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from benchmark.config import N_CH, N_FREQ, WIN, EMB_DIM, N_CLUSTERS, DROPOUT
from benchmark.models.shared import EDLHead, DECLayer


class ScalogramCNN(nn.Module):
    def __init__(self, n_ch=N_CH, n_freq=N_FREQ, n_time=WIN,
                 emb=EMB_DIM, drop=DROPOUT):
        super().__init__()
        self.blk1 = nn.Sequential(
            nn.Conv2d(n_ch, 32, (1, 8), padding=(0, 4), bias=False),
            nn.BatchNorm2d(32), nn.ELU())
        self.blk2 = nn.Sequential(
            nn.Conv2d(32, 64, (8, 1), groups=32, padding=(4, 0), bias=False),
            nn.BatchNorm2d(64), nn.ELU(), nn.AvgPool2d((2, 4)), nn.Dropout(drop * 0.5))
        self.blk3 = nn.Sequential(
            nn.Conv2d(64, 128, (4, 4), padding=(2, 2), bias=False),
            nn.BatchNorm2d(128), nn.ELU(), nn.AvgPool2d((4, 4)), nn.Dropout(drop))
        with torch.no_grad():
            d = torch.zeros(1, n_ch, n_freq, n_time)
            flat = self.blk3(self.blk2(self.blk1(d))).flatten(1).shape[1]
        self.proj = nn.Sequential(
            nn.Flatten(), nn.Linear(flat, emb), nn.LayerNorm(emb))

    def forward(self, x):
        return self.proj(self.blk3(self.blk2(self.blk1(x))))


class RiemannianSPD(nn.Module):
    def __init__(self, n_ch=N_CH, emb=EMB_DIM // 2):
        super().__init__()
        in_d = n_ch * (n_ch + 1) // 2
        self.net = nn.Sequential(
            nn.Linear(in_d, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, emb), nn.LayerNorm(emb))

    def forward(self, x):
        B = x.shape[0]
        vecs = []
        for b in range(B):
            C = torch.cov(x[b]) + 1e-5 * torch.eye(x.shape[1], device=x.device)
            idx = torch.triu_indices(x.shape[1], x.shape[1], device=x.device)
            vecs.append(C[idx[0], idx[1]])
        return self.net(torch.stack(vecs))


class ArchA(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn  = ScalogramCNN()
        self.spd  = RiemannianSPD()
        self.gate = nn.Sequential(nn.Linear(EMB_DIM + EMB_DIM//2, EMB_DIM), nn.Sigmoid())
        self.norm = nn.LayerNorm(EMB_DIM)
        self.dec  = DECLayer()
        self.edl  = EDLHead()

    def forward(self, x_sc, x_raw):
        za  = self.cnn(x_sc)              # (B,128)
        zb  = self.spd(x_raw)             # (B,64)
        g   = self.gate(torch.cat([za, zb], 1))
        z   = self.norm(g * za + (1 - g) * za)   # gated residual
        return {"z": z, "q": self.dec(z), **self.edl(z)}

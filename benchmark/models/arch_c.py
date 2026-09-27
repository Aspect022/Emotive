"""
benchmark/models/arch_c.py
Architecture C — CogProfile-Net v5: Band-GAT + Spectral Transformer + Riemannian.
Inspired by: WL-GraphTrax spectral encoding (Paper 5), GNN-Riemannian (Paper 2).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from benchmark.config import EMB_DIM, N_CLUSTERS, DROPOUT, GAT_HEADS, N_BANDS, DIM_RIEM, BAND_HZ
from benchmark.models.shared import EDLHead, DECLayer, GATv2Layer


class SpectralEncoder(nn.Module):
    """
    Physics-informed spectral position encoding.
    Each band described by [f_lo, f_hi, bandwidth] -> dim.
    """
    def __init__(self, dim=EMB_DIM):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, dim))
        band_info = torch.tensor(
            [(lo, hi, hi-lo) for lo, hi in BAND_HZ], dtype=torch.float32)
        self.register_buffer("band_info", band_info)   # (5,3)

    def forward(self, B):
        return self.fc(self.band_info).unsqueeze(0).expand(B, -1, -1)  # (B,5,dim)


class SpectralTransformer(nn.Module):
    """5 band embeddings -> spectral encoding -> Transformer -> CLS."""
    def __init__(self, dim=EMB_DIM, heads=4, depth=2):
        super().__init__()
        self.spec_enc = SpectralEncoder(dim)
        self.cls  = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos  = nn.Parameter(torch.zeros(1, N_BANDS + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        enc  = nn.TransformerEncoderLayer(dim, heads, dim*2, 0.1, batch_first=True, norm_first=True)
        self.tfm  = nn.TransformerEncoder(enc, depth)
        self.norm = nn.LayerNorm(dim)
        self.out  = nn.Linear(dim, dim)

    def forward(self, band_embs: torch.Tensor) -> torch.Tensor:
        """band_embs: (B, 5, dim)"""
        B = band_embs.shape[0]
        x = band_embs + self.spec_enc(B)
        x = torch.cat([self.cls.expand(B, -1, -1), x], 1) + self.pos
        return self.out(self.norm(self.tfm(x))[:, 0])


class ArchC(nn.Module):
    """
    Branch A: Per-band GATv2 -> 5 embeddings -> SpectralTransformer -> z_A (128)
    Branch B: Riemannian tangent MLP (525-D) -> z_B (128)
    Fusion:   Learnable soft 3-weight gate over (z_A, z_B, blend)
    """
    def __init__(self):
        super().__init__()
        # Branch A: one shared-weight GATv2 per band, then spectral transformer
        self.band_gat  = nn.ModuleList([GATv2Layer(13, 64, GAT_HEADS) for _ in range(N_BANDS)])
        self.band_proj = nn.ModuleList([nn.Linear(64, EMB_DIM) for _ in range(N_BANDS)])
        self.stfm = SpectralTransformer()

        # Branch B: Riemannian MLP
        self.mlp_r = nn.Sequential(
            nn.Linear(DIM_RIEM, 256), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(256, EMB_DIM))

        # Gated 2-way fusion
        self.gate = nn.Sequential(nn.Linear(EMB_DIM * 2, 2), nn.Softmax(dim=1))
        self.proj = nn.Linear(EMB_DIM, EMB_DIM)
        self.norm = nn.LayerNorm(EMB_DIM)
        self.dec  = DECLayer()
        self.edl  = EDLHead()

    def forward(self, x_band, x_node, x_adj, x_riem):
        # A: per-band GAT
        band_embs = []
        for bi in range(N_BANDS):
            h = self.band_gat[bi](x_node, x_adj)    # (B,14,64)
            z = self.band_proj[bi](h.mean(1))        # (B,128)
            band_embs.append(z)
        band_embs = torch.stack(band_embs, dim=1)    # (B,5,128)
        za = self.stfm(band_embs)                    # (B,128)

        # B: Riemannian
        zb = self.mlp_r(x_riem)                      # (B,128)

        # Soft gate fusion
        g  = self.gate(torch.cat([za, zb], 1))       # (B,2)
        z  = g[:, 0:1] * za + g[:, 1:2] * zb
        z  = self.norm(self.proj(z))

        return {"z": z, "q": self.dec(z), **self.edl(z)}

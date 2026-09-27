"""
benchmark/models/arch_b.py
Architecture B — CogProfile-Net v4: Multi-Band TinyViT + GATv2 + Nonlinear MLP.
Inspired by: WL-GraphTrax (Paper 5), EEGGNN-XAI (Paper 1).
"""
import torch
import torch.nn as nn
from benchmark.config import EMB_DIM, N_CLUSTERS, DROPOUT, GAT_HEADS, N_BANDS
from benchmark.models.shared import EDLHead, DECLayer, GATv2Layer, TinyViT


class BandSEAttn(nn.Module):
    """Squeeze-and-Excitation attention over the band axis."""
    def __init__(self, n=N_BANDS):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(n, n), nn.Sigmoid())

    def forward(self, x):
        w = self.fc(x).view(x.shape[0], x.shape[1], 1, 1)
        return x * w


class ArchB(nn.Module):
    """
    Branch A: SE-attended multi-band images -> TinyViT -> z_A (128)
    Branch B: GATv2 on electrode graph (Pearson adj, node feats 13-D) -> z_B (128)
    Branch C: Nonlinear MLP (56-D) -> z_C (64)
    Fusion:   cross-gate AB, then gate in C
    """
    def __init__(self):
        super().__init__()
        # Branch A
        self.se   = BandSEAttn()
        self.vit  = TinyViT(N_BANDS, 16, 32, patch=4, dim=EMB_DIM, depth=3)

        # Branch B
        self.gat1 = GATv2Layer(13, 64,        n_heads=GAT_HEADS)
        self.gat2 = GATv2Layer(64, EMB_DIM,   n_heads=GAT_HEADS)
        self.pool_b = nn.Linear(EMB_DIM, EMB_DIM)

        # Branch C
        self.mlp_c = nn.Sequential(
            nn.Linear(56, EMB_DIM), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(EMB_DIM, 64))

        # Fusion
        self.gate_ab = nn.Sequential(nn.Linear(EMB_DIM * 2, EMB_DIM), nn.Sigmoid())
        self.gate_c  = nn.Sequential(nn.Linear(EMB_DIM + 64, EMB_DIM), nn.Sigmoid())
        self.norm    = nn.LayerNorm(EMB_DIM)
        self.dec     = DECLayer()
        self.edl     = EDLHead()

    def forward(self, x_band, x_node, x_adj, x_nonlin):
        # A
        za = self.vit(self.se(x_band))

        # B: GATv2 graph
        h  = self.gat2(self.gat1(x_node, x_adj), x_adj)
        zb = self.pool_b(h.mean(1))

        # AB cross-gate
        g  = self.gate_ab(torch.cat([za, zb], 1))
        z  = g * za + (1 - g) * zb

        # C gate-in
        zc = self.mlp_c(x_nonlin)
        gc = self.gate_c(torch.cat([z, zc], 1))
        z  = self.norm(gc * z)

        return {"z": z, "q": self.dec(z), **self.edl(z)}

"""
benchmark/models/shared.py
Shared building blocks: EDLHead, DECLayer, losses, GATv2Layer, TinyViT.
Imported by every arch module.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from benchmark.config import N_CLASSES, N_CLUSTERS, EMB_DIM


# ── EDL ───────────────────────────────────────────────────────────────────────
class EDLHead(nn.Module):
    def __init__(self, in_dim=EMB_DIM, n_classes=N_CLASSES):
        super().__init__()
        self.K = n_classes
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2), nn.ReLU(),
            nn.Linear(in_dim // 2, n_classes), nn.Softplus(),
        )

    def forward(self, x):
        e     = self.net(x)
        alpha = e + 1.
        S     = alpha.sum(1, keepdim=True)
        return {
            "evidence":    e,
            "alpha":       alpha,
            "S":           S,
            "prob":        alpha / S,
            "uncertainty": self.K / S,
        }


def edl_loss(out: dict, labels: torch.Tensor, epoch: int, anneal: int = 15) -> torch.Tensor:
    K    = out["prob"].shape[1]
    y_oh = F.one_hot(labels, K).float()
    alpha, S, prob = out["alpha"], out["S"], out["prob"]

    # MSE component
    var  = alpha * (S - alpha) / (S * S * (S + 1))
    mse  = (y_oh - prob).pow(2) + var

    # KL annealing
    lam  = min(1., epoch / max(1, anneal))
    at   = y_oh + (1 - y_oh) * alpha
    bt   = torch.ones_like(at)
    kl   = (
        torch.lgamma(at.sum(1, keepdim=True))
        - torch.lgamma(bt.sum(1, keepdim=True))
        - (torch.lgamma(at) - torch.lgamma(bt)).sum(1, keepdim=True)
        + ((at - bt) * (torch.digamma(at)
           - torch.digamma(at.sum(1, keepdim=True)))).sum(1, keepdim=True)
    ).mean()
    return mse.sum(1).mean() + lam * kl


# ── DEC ───────────────────────────────────────────────────────────────────────
class DECLayer(nn.Module):
    def __init__(self, n_clusters=N_CLUSTERS, emb_dim=EMB_DIM, nu=1.):
        super().__init__()
        self.centroids = nn.Parameter(torch.randn(n_clusters, emb_dim) * 0.1)
        self.nu = nu

    def forward(self, z):
        d2 = ((z.unsqueeze(1) - self.centroids.unsqueeze(0)) ** 2).sum(2)
        q  = (1 + d2 / self.nu).pow(-(self.nu + 1) / 2.)
        return q / q.sum(1, keepdim=True)

    @torch.no_grad()
    def init_kmeans(self, embeddings: np.ndarray):
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=self.centroids.shape[0], n_init=10, random_state=42)
        km.fit(embeddings)
        self.centroids.data = torch.tensor(km.cluster_centers_, dtype=torch.float32)


def dec_loss(q: torch.Tensor) -> torch.Tensor:
    f = q.sum(0, keepdim=True)
    p = (q.pow(2) / f)
    p = (p / p.sum(1, keepdim=True)).detach()
    return F.kl_div(q.log().clamp(-20, 20), p, reduction="batchmean")


# ── GATv2 (no torch_geometric dep) ───────────────────────────────────────────
class GATv2Layer(nn.Module):
    """
    Batched GATv2.  Inputs: x (B,N,in_dim),  adj (B,N,N) weighted adjacency.
    """
    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.H = n_heads
        self.d = out_dim // n_heads
        self.W    = nn.Linear(in_dim, out_dim, bias=False)
        self.a    = nn.Parameter(torch.randn(n_heads, self.d * 2) * 0.02)
        self.act  = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(dropout)
        self.bn   = nn.BatchNorm1d(out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        h = self.W(x).view(B, N, self.H, self.d)
        hi  = h.unsqueeze(2).expand(-1,-1,N,-1,-1)
        hj  = h.unsqueeze(1).expand(-1,N,-1,-1,-1)
        eij = torch.cat([hi, hj], dim=-1)               # (B,N,N,H,2d)
        attn = self.act((eij * self.a).sum(-1))          # (B,N,N,H)
        mask = (adj.unsqueeze(-1) > 0)
        attn = attn.masked_fill(~mask, -1e9)
        attn = self.drop(F.softmax(attn, dim=2))
        out  = (attn.unsqueeze(-1) * hj.permute(0,2,1,3,4)).sum(2)  # (B,N,H,d)
        out  = out.reshape(B, N, -1)
        return self.bn(out.reshape(B*N, -1)).view(B, N, -1)


# ── TinyViT ───────────────────────────────────────────────────────────────────
class TinyViT(nn.Module):
    def __init__(self, in_c: int, h: int, w: int,
                 patch: int = 4, dim: int = 128, depth: int = 3,
                 heads: int = 4, drop: float = 0.2):
        super().__init__()
        n_patches = (h // patch) * (w // patch)
        self.pe   = nn.Conv2d(in_c, dim, patch, stride=patch)
        self.cls  = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos  = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.drop = nn.Dropout(drop)
        enc = nn.TransformerEncoderLayer(
            dim, heads, dim * 2, drop, batch_first=True, norm_first=True)
        self.tfm  = nn.TransformerEncoder(enc, depth)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.pe(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(B, -1, -1), x], 1)
        x = self.drop(x + self.pos)
        return self.norm(self.tfm(x))[:, 0]

"""
src/models/cogprofile/riemannian_spd.py
Branch B of CogProfile-Net v2 — Riemannian SPD Manifold Encoder.

Maps raw EEG windows (batch, 14, 128) → covariance matrix Σ ∈ S++^14
→ BiMap/ReEig/LogEig layers → Euclidean embedding z_B (batch, 64)
"""

import torch
import torch.nn as nn
import numpy as np


class CovarianceLayer(nn.Module):
    """
    Computes regularised sample covariance from a raw EEG window.
    Input:  (batch, n_ch, T)
    Output: (batch, n_ch, n_ch)  — symmetric positive-definite matrix
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        B, C, T = x.shape
        # Zero-mean per channel
        x = x - x.mean(dim=2, keepdim=True)
        # Covariance: (B, C, C)
        cov = torch.bmm(x, x.transpose(1, 2)) / (T - 1)
        # Tikhonov regularisation: Σ + εI
        eye = self.eps * torch.eye(C, device=x.device, dtype=x.dtype).unsqueeze(0)
        return cov + eye


class BiMapLayer(nn.Module):
    """
    Stiefel manifold projection: Σ' = W Σ Wᵀ
    Reduces dimensionality while preserving SPD structure.

    in_dim  → out_dim (in_dim > out_dim)
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # Initialise W on Stiefel manifold (columns orthonormal)
        W = torch.zeros(out_dim, in_dim)
        nn.init.orthogonal_(W)
        self.W = nn.Parameter(W)

    def forward(self, Sigma: torch.Tensor) -> torch.Tensor:
        # Sigma: (B, in, in)
        # Σ' = W Σ Wᵀ
        return self.W @ Sigma @ self.W.t()  # (B, out, out)


class ReEigLayer(nn.Module):
    """
    Rectified eigenvalue layer — clips near-zero eigenvalues to eps.
    Ensures SPD property is preserved through the network.
    """

    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, Sigma: torch.Tensor) -> torch.Tensor:
        # Sigma: (B, C, C)
        try:
            L, V = torch.linalg.eigh(Sigma)       # eigendecomposition (symmetric)
        except RuntimeError:
            # Fallback: add noise and retry
            Sigma = Sigma + 1e-3 * torch.eye(Sigma.shape[-1], device=Sigma.device)
            L, V = torch.linalg.eigh(Sigma)
        L_clipped = torch.clamp(L, min=self.eps)   # rectify
        return V @ torch.diag_embed(L_clipped) @ V.transpose(-1, -2)


class LogEigLayer(nn.Module):
    """
    Matrix logarithm via eigendecomposition: log(Σ) = V·log(Λ)·Vᵀ
    Maps SPD manifold → tangent space (Euclidean) for linear layers.
    """

    def forward(self, Sigma: torch.Tensor) -> torch.Tensor:
        L, V = torch.linalg.eigh(Sigma)
        log_L = torch.log(torch.clamp(L, min=1e-8))
        log_Sigma = V @ torch.diag_embed(log_L) @ V.transpose(-1, -2)
        # Vectorise upper triangle (symmetric → no redundancy)
        B, C, _ = log_Sigma.shape
        idx = torch.triu_indices(C, C, offset=0)
        return log_Sigma[:, idx[0], idx[1]]   # (B, C*(C+1)/2)


class RiemannianSPD(nn.Module):
    """
    Full Riemannian SPD encoder for CogProfile-Net Branch B.

    Pipeline:
        raw EEG (B, 14, 128)
        → CovarianceLayer → Σ ∈ S++^14
        → BiMap(14→8)  → ReEig → BiMap(8→6) → ReEig
        → LogEig → vec ∈ R^21
        → Linear(21→64) → z_B

    Args:
        n_channels: EEG channels (14)
        emb_dim:    Output embedding dimension (64)
    """

    def __init__(self, n_channels: int = 14, emb_dim: int = 64):
        super().__init__()
        # SPD pipeline dimensions
        dim1, dim2 = 8, 6
        log_dim = dim2 * (dim2 + 1) // 2   # = 21

        self.cov       = CovarianceLayer()
        self.bimap1    = BiMapLayer(n_channels, dim1)
        self.reeig1    = ReEigLayer()
        self.bimap2    = BiMapLayer(dim1, dim2)
        self.reeig2    = ReEigLayer()
        self.logeig    = LogEigLayer()
        self.proj      = nn.Sequential(
            nn.Linear(log_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_channels, n_timepoints)
        returns: (batch, emb_dim)
        """
        Sigma = self.cov(x)           # (B, 14, 14)
        Sigma = self.bimap1(Sigma)    # (B, 8, 8)
        Sigma = self.reeig1(Sigma)    # (B, 8, 8)
        Sigma = self.bimap2(Sigma)    # (B, 6, 6)
        Sigma = self.reeig2(Sigma)    # (B, 6, 6)
        vec   = self.logeig(Sigma)    # (B, 21)
        z_B   = self.proj(vec)        # (B, 64)
        return z_B

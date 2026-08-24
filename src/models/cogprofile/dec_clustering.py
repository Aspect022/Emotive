"""
src/models/cogprofile/dec_clustering.py
Deep Embedded Clustering (DEC) for CogProfile-Net.

Jointly optimises the feature encoder and cluster assignments using
Student-t soft assignments + KL divergence self-supervision.

Based on: Xie et al. (2016) "Unsupervised Deep Embedding for Clustering Analysis"
Extended with noise cluster concept for label-noise sub-windows.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DECLayer(nn.Module):
    """
    Soft cluster assignment layer using Student-t kernel.

    q_ij = (1 + ||z_i - μ_j||² / ν)^(-(ν+1)/2)
           ─────────────────────────────────────────
           Σ_j' (1 + ||z_i - μ_j'||² / ν)^(-(ν+1)/2)

    where ν = degrees of freedom (1 for Cauchy-like heavy tails).
    """

    def __init__(self, n_clusters: int, emb_dim: int, nu: float = 1.0):
        super().__init__()
        self.n_clusters = n_clusters
        self.nu = nu
        # Cluster centroids — learnable
        self.centroids = nn.Parameter(torch.randn(n_clusters, emb_dim))

    @torch.no_grad()
    def initialise_from_kmeans(self, embeddings: np.ndarray, seed: int = 42):
        """Pre-initialise centroids using K-Means on pre-trained embeddings."""
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=seed)
        km.fit(embeddings)
        self.centroids.data = torch.tensor(km.cluster_centers_, dtype=torch.float32)
        print(f"[DEC] Centroids initialised via K-Means. Inertia={km.inertia_:.2f}")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (batch, emb_dim)
        returns: q (batch, n_clusters) — soft assignments
        """
        # Pairwise squared distances to centroids
        diff = z.unsqueeze(1) - self.centroids.unsqueeze(0)   # (B, K, D)
        dist2 = (diff ** 2).sum(dim=2)                         # (B, K)
        q = (1.0 + dist2 / self.nu).pow(-(self.nu + 1.0) / 2.0)
        q = q / q.sum(dim=1, keepdim=True)
        return q   # (B, K)


def target_distribution(q: torch.Tensor) -> torch.Tensor:
    """
    Sharpened self-supervised target distribution.
    p_ij = (q_ij² / f_j) / Σ_j' (q_ij'² / f_j')
    where f_j = Σ_i q_ij  (soft cluster frequencies)
    """
    f = q.sum(dim=0, keepdim=True)   # (1, K)
    p = q.pow(2) / f
    p = p / p.sum(dim=1, keepdim=True)
    return p.detach()   # stop gradient


def dec_loss(q: torch.Tensor) -> torch.Tensor:
    """KL divergence DEC loss: KL(P || Q)"""
    p = target_distribution(q)
    loss = F.kl_div(q.log(), p, reduction="batchmean")
    return loss


class ClusterWeightScheduler:
    """
    Down-weights sub-windows assigned to noise clusters during CE loss.
    Windows in clusters >= n_task_clusters are considered noise.
    """

    def __init__(self, n_task_clusters: int = 5, noise_weight: float = 0.1):
        self.n_task = n_task_clusters
        self.noise_w = noise_weight

    def get_sample_weights(self, q: torch.Tensor) -> torch.Tensor:
        """
        Returns per-sample weights in [noise_weight, 1.0].
        Hard-assigned cluster via argmax; noise clusters get noise_weight.
        """
        with torch.no_grad():
            assigned = q.argmax(dim=1)   # (B,)
            weights = torch.ones(len(assigned), device=q.device)
            noise_mask = assigned >= self.n_task
            weights[noise_mask] = self.noise_w
        return weights

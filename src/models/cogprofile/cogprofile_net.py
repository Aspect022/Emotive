"""
src/models/cogprofile/cogprofile_net.py
CogProfile-Net v2 — Full Dual-Branch Cognitive Profiling Model.

Architecture:
  Branch A: ScalogramCNN  (14, 64, 128) scalogram → z_A ∈ R^128
  Branch B: RiemannianSPD (14, 128) raw EEG  → z_B ∈ R^64
  Fusion:   Attention-gated concat → z ∈ R^128
  Heads:    DEC clustering + EDL (Dirichlet) classification
  Optional: Behavioral gate (RT, difficulty, accuracy)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .scalogram_cnn  import ScalogramCNN
from .riemannian_spd import RiemannianSPD
from .dec_clustering import DECLayer, dec_loss, ClusterWeightScheduler
from .edl_head       import EDLHead, edl_mse_loss


class AttentionFusion(nn.Module):
    """Gated attention fusion for two embeddings."""

    def __init__(self, dim_a: int, dim_b: int, out_dim: int):
        super().__init__()
        self.proj_a = nn.Linear(dim_a, out_dim)
        self.proj_b = nn.Linear(dim_b, out_dim)
        self.gate   = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        a = self.proj_a(z_a)
        b = self.proj_b(z_b)
        g = self.gate(torch.cat([a, b], dim=1))
        z = g * a + (1.0 - g) * b
        return self.norm(z)


class BehavioralGate(nn.Module):
    """
    Late-fusion gating from trial behavioral metadata.
    Inputs: [rt_norm, difficulty, accuracy] → gate applied to embedding.
    Anti-shortcut: auxiliary head forces EEG to be predictive without metadata.
    """

    def __init__(self, emb_dim: int = 128, behav_dim: int = 3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(behav_dim, 16),
            nn.ReLU(),
            nn.Linear(16, emb_dim),
            nn.Sigmoid(),   # gate in [0,1]
        )
        # Auxiliary head — trained on EEG embedding only (anti-shortcut)
        self.aux_head = nn.Linear(emb_dim, 5)

    def forward(
        self, z: torch.Tensor, behav: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        z:     (batch, emb_dim)
        behav: (batch, 3)
        Returns (z_gated, aux_logits)
        """
        gate    = self.mlp(behav)                  # (B, emb_dim)
        z_gated = z + gate * z                     # residual gate
        aux_logits = self.aux_head(z)              # EEG-only logits
        return z_gated, aux_logits


class CogProfileNet(nn.Module):
    """
    CogProfile-Net v2.

    Args:
        n_classes:       Cognitive classes (5)
        n_channels:      EEG channels (14)
        n_freq:          Scalogram frequency bins (64)
        n_time:          Time samples per window (128)
        emb_dim:         Main embedding dimension (128)
        n_clusters:      DEC clusters (8 = 5 cognitive + 3 noise)
        dropout:         Dropout rate (0.5 for N=25 subjects)
        use_behavioral:  Enable behavioral gating (True in training, optional)
    """

    def __init__(
        self,
        n_classes: int = 5,
        n_channels: int = 14,
        n_freq: int = 64,
        n_time: int = 128,
        emb_dim: int = 128,
        n_clusters: int = 8,
        dropout: float = 0.5,
        use_behavioral: bool = True,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.use_behavioral = use_behavioral

        # Branch A — Scalogram CNN
        self.branch_a = ScalogramCNN(
            n_channels=n_channels, n_freq=n_freq, n_time=n_time,
            emb_dim=emb_dim, dropout=dropout,
        )

        # Branch B — Riemannian SPD (takes raw EEG)
        self.branch_b = RiemannianSPD(n_channels=n_channels, emb_dim=emb_dim // 2)

        # Fusion (attention-gated)
        self.fusion = AttentionFusion(
            dim_a=emb_dim, dim_b=emb_dim // 2, out_dim=emb_dim
        )

        # Behavioral gate (optional)
        if use_behavioral:
            self.behav_gate = BehavioralGate(emb_dim=emb_dim)

        # DEC clustering head
        self.dec = DECLayer(n_clusters=n_clusters, emb_dim=emb_dim)
        self.cluster_scheduler = ClusterWeightScheduler(n_task_clusters=n_classes)

        # EDL classification head
        self.edl = EDLHead(in_dim=emb_dim, n_classes=n_classes)

    def forward(
        self,
        scalogram: torch.Tensor,   # (B, 14, 64, 128)
        raw_eeg: torch.Tensor,     # (B, 14, 128)
        behav: torch.Tensor | None = None,  # (B, 3)
    ) -> dict:
        """
        Returns dict with all outputs needed for loss computation.
        """
        # --- Encoders ---
        z_a = self.branch_a(scalogram)    # (B, 128)
        z_b = self.branch_b(raw_eeg)      # (B, 64)

        # --- Fusion ---
        z = self.fusion(z_a, z_b)         # (B, 128)

        # --- Behavioral gate ---
        aux_logits = None
        if self.use_behavioral and behav is not None:
            z, aux_logits = self.behav_gate(z, behav)

        # --- DEC clustering ---
        q = self.dec(z)                    # (B, n_clusters)

        # --- EDL classification ---
        edl_out = self.edl(z)              # dict with prob, uncertainty, etc.

        return {
            "embedding":   z,
            "q_cluster":   q,
            "edl":         edl_out,
            "aux_logits":  aux_logits,
            "z_a":         z_a,
            "z_b":         z_b,
        }

    def compute_loss(
        self,
        output: dict,
        labels: torch.Tensor,
        epoch: int,
        alpha_dec: float = 0.1,
        alpha_orth: float = 0.05,
        alpha_aux: float = 0.3,
    ) -> dict:
        """
        Compute composite training loss.

        L_total = L_EDL + α_dec·L_DEC + α_orth·L_orth + α_aux·L_aux
        """
        # Sample weights from DEC (down-weight noise clusters)
        sample_w = self.cluster_scheduler.get_sample_weights(output["q_cluster"])

        # EDL loss (weighted by cluster noise mask)
        edl_loss_val = edl_mse_loss(
            output["edl"], labels, epoch, self.n_classes
        )

        # DEC self-supervision loss
        dec_loss_val = dec_loss(output["q_cluster"])

        # Orthogonality regularisation between branch embeddings
        # z_a is 128-dim, z_b is 64-dim — compare per-sample dot product instead
        z_a_n = F.normalize(output["z_a"], dim=1)   # (B, 128)
        z_b_n = F.normalize(output["z_b"], dim=1)   # (B, 64)
        # Project z_b to z_a dim for cosine comparison (element-wise on shared min-dim)
        min_dim = min(z_a_n.shape[1], z_b_n.shape[1])
        orth_loss = (z_a_n[:, :min_dim] * z_b_n[:, :min_dim]).sum(dim=1).pow(2).mean()

        # Auxiliary anti-shortcut loss (EEG-only head)
        aux_loss = torch.tensor(0.0, device=labels.device)
        if output["aux_logits"] is not None:
            aux_loss = F.cross_entropy(output["aux_logits"], labels)

        total = (
            edl_loss_val
            + alpha_dec  * dec_loss_val
            + alpha_orth * orth_loss
            + alpha_aux  * aux_loss
        )

        return {
            "total":    total,
            "edl":      edl_loss_val.item(),
            "dec":      dec_loss_val.item(),
            "orth":     orth_loss.item(),
            "aux":      aux_loss.item() if isinstance(aux_loss, torch.Tensor) else 0.0,
        }

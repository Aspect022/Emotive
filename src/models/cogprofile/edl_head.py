"""
src/models/cogprofile/edl_head.py
Evidential Deep Learning (EDL) classification head for CogProfile-Net.

Replaces standard softmax with a Dirichlet output, providing:
  - Calibrated class probabilities: p̂_k = α_k / S
  - Epistemic uncertainty (vacuity): u = K / S
  - Full Dirichlet distribution over the simplex

References:
  Sensoy et al. (2018) "Evidential Deep Learning to Quantify Classification Uncertainty"
  Jürgens et al. (2024) - EDL loss consistency caveats
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EDLHead(nn.Module):
    """
    Evidential classification head.

    Args:
        in_dim:    Input embedding dimension
        n_classes: Number of cognitive classes (5)
    """

    def __init__(self, in_dim: int = 128, n_classes: int = 5):
        super().__init__()
        self.n_classes = n_classes
        self.evidence_layer = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, n_classes),
            nn.Softplus(),   # ensures non-negative evidence
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: embedding (batch, in_dim)

        Returns dict with:
            evidence:     (batch, K) — non-negative evidence per class
            alpha:        (batch, K) — Dirichlet parameters α_k = e_k + 1
            S:            (batch, 1) — Dirichlet strength S = Σ α_k
            prob:         (batch, K) — expected probability p̂_k = α_k / S
            uncertainty:  (batch, 1) — vacuity u = K / S ∈ (0, 1]
        """
        e   = self.evidence_layer(x)           # (B, K)
        alpha = e + 1.0                         # Dirichlet params
        S     = alpha.sum(dim=1, keepdim=True)  # (B, 1)
        prob  = alpha / S                        # (B, K)
        u     = self.n_classes / S               # (B, 1)
        return {
            "evidence":    e,
            "alpha":       alpha,
            "S":           S,
            "prob":        prob,
            "uncertainty": u,
        }


def edl_mse_loss(
    output: dict,
    labels: torch.Tensor,
    epoch: int,
    n_classes: int = 5,
    annealing_epochs: int = 10,
) -> torch.Tensor:
    """
    EDL MSE loss with KL divergence annealing.

    L = MSE(p̂, y) + λ_t · KL[ Dir(ã) || Dir(1) ]
    where ã = y + (1-y)⊙α (keep correct class evidence, remove wrong-class)
    λ_t = min(1, t / annealing_epochs)
    """
    alpha = output["alpha"]
    S     = output["S"]
    prob  = output["prob"]
    B, K  = alpha.shape

    # One-hot target
    y_oh = F.one_hot(labels, K).float()

    # MSE term: E[(p̂ - y)²] = Σ_k [y_k(1 - p̂_k)² + (1-y_k)p̂_k²]
    # Expanded analytically: Var(p_k) + (p̂_k - y_k)²
    var  = alpha * (S - alpha) / (S * S * (S + 1))
    mse  = (y_oh - prob).pow(2) + var
    loss_mse = mse.sum(dim=1).mean()

    # KL regularisation: remove evidence for wrong classes
    alpha_tilde = y_oh + (1.0 - y_oh) * alpha   # keep correct, penalise wrong
    loss_kl = dirichlet_kl(alpha_tilde, n_classes)

    # Annealing coefficient
    lam = min(1.0, epoch / annealing_epochs)

    return loss_mse + lam * loss_kl


def dirichlet_kl(alpha: torch.Tensor, n_classes: int) -> torch.Tensor:
    """KL divergence KL[Dir(alpha) || Dir(1,...,1)]."""
    beta = torch.ones_like(alpha)
    S_a = alpha.sum(dim=1, keepdim=True)
    S_b = beta.sum(dim=1, keepdim=True)
    kl = (
        torch.lgamma(S_a) - torch.lgamma(S_b)
        - (torch.lgamma(alpha) - torch.lgamma(beta)).sum(dim=1, keepdim=True)
        + ((alpha - beta) * (torch.digamma(alpha) - torch.digamma(S_a))).sum(dim=1, keepdim=True)
    )
    return kl.mean()

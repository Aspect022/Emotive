"""
src/models/cogprofile/digital_twin.py
EEG Digital Twin — Variational Recurrent State-Space Model.

Maintains a personalized latent cognitive state z_t per subject.
Enables simulation of how cognitive profiles evolve over future sessions.

Design: lightweight GRU-based state-space model, NOT a biophysical
brain simulator. Operates at the profile level (Dir(alpha) trajectories).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path


class DigitalTwin(nn.Module):
    """
    Per-subject latent cognitive state-space model.

    State update:
        z_t = GRU(z_{t-1}, [x_t || b_t])
        ŷ_t = Dirichlet evidence head(z_t)

    Args:
        eeg_emb_dim:  Dimension of EEG embedding input (128)
        behav_dim:    Behavioral metadata dimension (3: rt, difficulty, acc)
        state_dim:    Latent state dimension (32)
        n_classes:    Number of cognitive classes (5)
    """

    def __init__(
        self,
        eeg_emb_dim: int = 128,
        behav_dim: int = 3,
        state_dim: int = 32,
        n_classes: int = 5,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_classes = n_classes

        # Input projection: [EEG_emb || behavioral] → GRU input
        self.input_proj = nn.Linear(eeg_emb_dim + behav_dim, state_dim)

        # Recurrent state update
        self.gru = nn.GRUCell(state_dim, state_dim)

        # Profile decoder: state → Dirichlet evidence
        self.decoder = nn.Sequential(
            nn.Linear(state_dim, state_dim * 2),
            nn.ELU(),
            nn.Linear(state_dim * 2, n_classes),
            nn.Softplus(),  # non-negative evidence
        )

        # Subject state registry (persistent per subject, not a network param)
        self._subject_states: dict[str, torch.Tensor] = {}

    def get_state(self, subject_id: str) -> torch.Tensor:
        """Return current latent state for a subject (zeros if new)."""
        if subject_id not in self._subject_states:
            self._subject_states[subject_id] = torch.zeros(1, self.state_dim)
        return self._subject_states[subject_id]

    def update_state(self, subject_id: str, z_new: torch.Tensor):
        """Persist updated state for a subject."""
        self._subject_states[subject_id] = z_new.detach()

    def forward(
        self,
        eeg_emb: torch.Tensor,     # (batch, eeg_emb_dim)
        behav: torch.Tensor,        # (batch, behav_dim)
        z_prev: torch.Tensor,       # (batch, state_dim)
    ) -> dict:
        """
        Single step forward: update state and decode profile.

        Returns:
            z_new:      Updated latent state (batch, state_dim)
            evidence:   Non-negative class evidence (batch, n_classes)
            alpha:      Dirichlet params α_k = e_k + 1 (batch, n_classes)
            prob:       Expected profile p̂_k = α_k / S (batch, n_classes)
            uncertainty: Vacuity u = K/S (batch, 1)
        """
        inp = torch.cat([eeg_emb, behav], dim=1)  # (B, emb+behav)
        inp = self.input_proj(inp)                  # (B, state_dim)
        z_new = self.gru(inp, z_prev)               # (B, state_dim)

        evidence = self.decoder(z_new)              # (B, K)
        alpha    = evidence + 1.0
        S        = alpha.sum(dim=1, keepdim=True)
        prob     = alpha / S
        u        = self.n_classes / S

        return {
            "z_new":       z_new,
            "evidence":    evidence,
            "alpha":       alpha,
            "prob":        prob,
            "uncertainty": u,
        }

    # ------------------------------------------------------------------
    # Simulation API (the novel output of the Digital Twin)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def simulate_trajectory(
        self,
        subject_id: str,
        n_future_sessions: int = 50,
        learning_rate: float = 0.02,
        noise_std: float = 0.1,
        device: torch.device | None = None,
    ) -> dict:
        """
        Simulate how a subject's cognitive profile evolves over N future sessions.

        Uses:
        - Current persisted state z_0 for the subject
        - Learning-curve prior: behavioral improvement over sessions
        - Cluster-level VAE sampling (simplified: Gaussian noise around current state)

        Args:
            subject_id:         Subject identifier
            n_future_sessions:  Number of sessions to simulate (e.g. 50, 100, 200)
            learning_rate:      Rate of behavioral improvement per session
            noise_std:          State transition noise (individual variability)

        Returns dict with:
            profiles:      (n_future_sessions, n_classes) — expected profile per session
            uncertainties: (n_future_sessions, 1) — uncertainty per session
            sessions:      list of session indices
        """
        if device is None:
            device = next(self.parameters()).device

        z = self.get_state(subject_id).to(device)    # (1, state_dim)

        profiles      = []
        uncertainties = []

        for s in range(n_future_sessions):
            # Simulate behavioral improvement (simple linear learning curve prior)
            improvement = learning_rate * s
            behav_sim = torch.tensor(
                [[max(0.3, 1.0 - improvement), 0.5, min(1.0, 0.5 + improvement)]],
                dtype=torch.float32, device=device,
            )  # [rt decreases, medium difficulty, accuracy improves]

            # Simulate EEG embedding: Gaussian noise around current state projection
            eeg_sim = z + noise_std * torch.randn_like(z)
            eeg_sim_padded = F.pad(eeg_sim, (0, 128 - self.state_dim))[:, :128]

            out = self.forward(eeg_sim_padded, behav_sim, z)
            z = out["z_new"]

            profiles.append(out["prob"].squeeze(0).cpu().numpy())
            uncertainties.append(out["uncertainty"].squeeze(0).cpu().numpy())

        return {
            "subject_id":   subject_id,
            "sessions":     list(range(1, n_future_sessions + 1)),
            "profiles":     np.stack(profiles),         # (N, 5)
            "uncertainties": np.stack(uncertainties),   # (N, 1)
            "class_names":  [
                "Mental Arithmetic", "Pattern Recognition",
                "Working Memory", "Reading Comprehension", "Sustained Attention"
            ],
        }

    def save_states(self, path: Path):
        """Persist all subject states to disk."""
        states = {sid: s.cpu().numpy() for sid, s in self._subject_states.items()}
        np.save(str(path), states)

    def load_states(self, path: Path):
        """Load persisted subject states from disk."""
        states = np.load(str(path), allow_pickle=True).item()
        self._subject_states = {
            sid: torch.tensor(s, dtype=torch.float32)
            for sid, s in states.items()
        }

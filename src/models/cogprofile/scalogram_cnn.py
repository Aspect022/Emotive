"""
src/models/cogprofile/scalogram_cnn.py
Branch A of CogProfile-Net v2.

EEGNet-inspired 2D CNN that processes stacked CWT scalograms.
Input:  (batch, 14, 64, 128) — (batch, channels, freq_bins, time)
Output: embedding z_A of shape (batch, 128)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScalogramCNN(nn.Module):
    """
    Lightweight 2D CNN for stacked EEG scalograms.

    Designed for small N (25 subjects): heavy regularization, depthwise
    separable convolutions, aggressive pooling.

    Args:
        n_channels:  EEG channels (14 for Emotiv EPOC+)
        n_freq:      Frequency bins in scalogram (64)
        n_time:      Time samples in scalogram (128)
        emb_dim:     Output embedding dimension (128)
        dropout:     Dropout rate (0.5 recommended for N=25)
    """

    def __init__(
        self,
        n_channels: int = 14,
        n_freq: int = 64,
        n_time: int = 128,
        emb_dim: int = 128,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_freq = n_freq
        self.n_time = n_time

        # Block 1: Temporal convolution across time axis (shared across channels)
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=(1, 8), padding=(0, 4), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
        )

        # Block 2: Depthwise spatial convolution across frequency bins
        self.freq_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(8, 1), groups=32, padding=(4, 0), bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(2, 4)),   # (64, 32, 32)
            nn.Dropout(dropout * 0.5),
        )

        # Block 3: Separable conv — fuse frequency-time features
        self.sep_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(4, 4), padding=(2, 2), bias=False),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(4, 4)),   # heavily pool: (128, ~8, ~8)
            nn.Dropout(dropout),
        )

        # Compute flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, n_freq, n_time)
            dummy = self.temporal_conv(dummy)
            dummy = self.freq_conv(dummy)
            dummy = self.sep_conv(dummy)
            flat_size = dummy.flatten(1).shape[1]

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, emb_dim),
            nn.LayerNorm(emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_channels, n_freq, n_time)
        returns: (batch, emb_dim)
        """
        x = self.temporal_conv(x)
        x = self.freq_conv(x)
        x = self.sep_conv(x)
        z = self.projection(x)
        return z   # (batch, 128)

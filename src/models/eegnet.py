"""
EEGNet — Lightweight BCI Deep Learning Model
Based on: Lawhern et al. (2018) "EEGNet: A Compact Convolutional Neural Network
for EEG-based Brain-Computer Interfaces"

Input: (batch, 1, n_channels, n_timepoints)
Output: (batch, n_classes)

EEGNet is the de-facto standard lightweight EEG DL model.
It has ~2,000 parameters and is designed for small EEG datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    """
    EEGNet: Compact CNN for EEG.

    Args:
        n_classes:      Number of output classes
        n_channels:     EEG channels (14 for EPOC X)
        n_timepoints:   Samples per window (512 for 4s @ 128Hz)
        dropout_rate:   Dropout probability
        kern_length:    Length of temporal convolution kernel (default: fs/2 = 64)
        F1:             Number of temporal filters
        D:              Depth multiplier (spatial filters per temporal filter)
        F2:             Number of pointwise filters (= F1 * D)
    """

    def __init__(
        self,
        n_classes: int = 5,
        n_channels: int = 14,
        n_timepoints: int = 512,
        dropout_rate: float = 0.5,
        kern_length: int = 64,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints

        # Block 1: Temporal + Depthwise Spatial Convolution
        self.block1 = nn.Sequential(
            # Temporal convolution
            nn.Conv2d(1, F1, (1, kern_length), padding=(0, kern_length // 2), bias=False),
            nn.BatchNorm2d(F1),
            # Depthwise spatial convolution
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate),
        )

        # Block 2: Separable Convolution
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate),
        )

        # Compute flatten size
        self._flat_size = self._get_flat_size(n_channels, n_timepoints, F1, D, F2)

        # Classifier
        self.classifier = nn.Linear(self._flat_size, n_classes)

    def _get_flat_size(self, n_channels, n_timepoints, F1, D, F2):
        """Compute flattened feature map size after conv blocks."""
        with torch.no_grad():
            x = torch.zeros(1, 1, n_channels, n_timepoints)
            x = self.block1(x)
            x = self.block2(x)
            return x.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_channels, n_timepoints) OR (batch, 1, n_channels, n_timepoints)
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)  # add channel dim: (B, 1, C, T)
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(1)
        return self.classifier(x)


class EEGNetBinary(EEGNet):
    """EEGNet variant for binary classification (performance: high/low)."""
    def __init__(self, n_channels=14, n_timepoints=512, **kwargs):
        super().__init__(n_classes=2, n_channels=n_channels,
                         n_timepoints=n_timepoints, **kwargs)

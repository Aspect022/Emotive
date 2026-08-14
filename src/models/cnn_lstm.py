"""
CNN-LSTM Hybrid — EEG Deep Learning Model
Uses 1D-CNN to extract local temporal features, then BiLSTM for
long-range temporal dependencies across the EEG window.

Input: (batch, n_channels, n_timepoints)
Output: (batch, n_classes)
"""

import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    """
    Hybrid CNN + BiLSTM model for EEG classification.

    Architecture:
      1. Per-channel 1D convolutions (channel-wise feature extraction)
      2. Cross-channel mixing (depthwise)
      3. BiLSTM over time
      4. Attention pooling
      5. Linear classifier
    """

    def __init__(
        self,
        n_classes: int = 5,
        n_channels: int = 14,
        n_timepoints: int = 512,
        cnn_filters: int = 32,
        cnn_kernel: int = 16,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        dropout_rate: float = 0.4,
    ):
        super().__init__()
        self.n_channels = n_channels

        # ── CNN feature extractor ─────────────────────────────────────────────
        self.cnn = nn.Sequential(
            # Temporal convolution across all channels
            nn.Conv1d(n_channels, cnn_filters, kernel_size=cnn_kernel,
                      padding=cnn_kernel // 2, bias=False),
            nn.BatchNorm1d(cnn_filters),
            nn.ELU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout_rate / 2),

            nn.Conv1d(cnn_filters, cnn_filters * 2, kernel_size=8,
                      padding=4, bias=False),
            nn.BatchNorm1d(cnn_filters * 2),
            nn.ELU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout_rate / 2),
        )

        cnn_out_size = cnn_filters * 2
        cnn_out_len = n_timepoints // 16  # after two MaxPool1d(4)

        # ── BiLSTM ────────────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=cnn_out_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if lstm_layers > 1 else 0.0,
        )

        lstm_out_size = lstm_hidden * 2  # bidirectional

        # ── Attention pooling ─────────────────────────────────────────────────
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_size, lstm_out_size // 2),
            nn.Tanh(),
            nn.Linear(lstm_out_size // 2, 1),
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(lstm_out_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T)"""
        # CNN: (B, C, T) -> (B, F, T')
        feat = self.cnn(x)

        # Rearrange for LSTM: (B, T', F)
        feat = feat.permute(0, 2, 1)

        # LSTM: (B, T', F) -> (B, T', 2*hidden)
        lstm_out, _ = self.lstm(feat)

        # Attention pooling
        attn_weights = self.attention(lstm_out)          # (B, T', 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)  # (B, 2*hidden)

        out = self.dropout(context)
        return self.classifier(out)


class CNN1D(nn.Module):
    """
    Pure 1D-CNN for EEG classification.
    Simpler baseline compared to CNN-LSTM.
    """

    def __init__(
        self,
        n_classes: int = 5,
        n_channels: int = 14,
        n_timepoints: int = 512,
        dropout_rate: float = 0.4,
    ):
        super().__init__()

        self.net = nn.Sequential(
            # Block 1
            nn.Conv1d(n_channels, 32, kernel_size=16, padding=8, bias=False),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout_rate / 2),

            # Block 2
            nn.Conv1d(32, 64, kernel_size=8, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout_rate / 2),

            # Block 3
            nn.Conv1d(64, 128, kernel_size=4, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Dropout(dropout_rate),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8, 256),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T)"""
        return self.classifier(self.net(x))


class LSTM(nn.Module):
    """Pure LSTM baseline (without CNN front-end)."""

    def __init__(
        self,
        n_classes: int = 5,
        n_channels: int = 14,
        n_timepoints: int = 512,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout_rate: float = 0.4,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) -> transpose to (B, T, C) for LSTM"""
        x = x.permute(0, 2, 1)  # (B, T, C)
        out, _ = self.lstm(x)
        # Use last timestep output
        out = self.dropout(out[:, -1, :])
        return self.classifier(out)

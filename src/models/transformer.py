"""
EEG Transformer — Multi-Head Self-Attention for EEG Classification
Treats each EEG channel as a token and applies self-attention over time.

Two variants:
  - ChannelTransformer: attention over channels at each timepoint
  - TemporalTransformer: attention over time patches per channel

Input: (batch, n_channels, n_timepoints)
Output: (batch, n_classes)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Split EEG signal into patches and project to embedding dim."""

    def __init__(self, n_channels: int, n_timepoints: int,
                 patch_size: int = 64, embed_dim: int = 64):
        super().__init__()
        assert n_timepoints % patch_size == 0, \
            f"n_timepoints ({n_timepoints}) must be divisible by patch_size ({patch_size})"
        self.n_patches = n_timepoints // patch_size
        self.patch_proj = nn.Conv1d(
            n_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=False
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.n_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) -> (B, n_patches, embed_dim)"""
        x = self.patch_proj(x)        # (B, embed_dim, n_patches)
        x = x.permute(0, 2, 1)       # (B, n_patches, embed_dim)
        x = x + self.pos_embed
        return x


class TransformerBlock(nn.Module):
    """Standard Transformer encoder block with pre-norm."""

    def __init__(self, embed_dim: int, n_heads: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        # FFN with residual
        x = x + self.mlp(self.norm2(x))
        return x


class EEGTransformer(nn.Module):
    """
    Transformer encoder for EEG classification.
    Splits the temporal signal into patches and uses self-attention over patches.

    Args:
        n_classes:      Number of output classes
        n_channels:     EEG channels
        n_timepoints:   Samples per window
        patch_size:     Samples per patch (default 64 = 0.5s @ 128Hz)
        embed_dim:      Embedding dimension
        n_heads:        Number of attention heads
        n_layers:       Number of transformer blocks
        dropout_rate:   Dropout probability
    """

    def __init__(
        self,
        n_classes: int = 5,
        n_channels: int = 14,
        n_timepoints: int = 512,
        patch_size: int = 64,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        # Adjust patch_size to be divisor of n_timepoints
        while n_timepoints % patch_size != 0 and patch_size > 1:
            patch_size -= 1

        self.patch_embed = PatchEmbedding(n_channels, n_timepoints, patch_size, embed_dim)

        self.transformer = nn.Sequential(*[
            TransformerBlock(embed_dim, n_heads, dropout=dropout_rate)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # CLS token for classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.classifier = nn.Linear(embed_dim, n_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T)"""
        B = x.shape[0]

        # Patch embedding: (B, n_patches, embed_dim)
        x = self.patch_embed(x)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Apply transformer blocks
        x = self.transformer(x)
        x = self.norm(x)

        # Use CLS token representation
        cls_out = x[:, 0]
        return self.classifier(self.dropout(cls_out))


class MLPBaseline(nn.Module):
    """
    MLP baseline operating on hand-crafted features.
    Used to establish the classical ML performance ceiling.

    Input: (batch, n_features)
    Output: (batch, n_classes)
    """

    def __init__(
        self,
        n_features: int = 280,   # 14ch × (5 bands × 2 + 3 hjorth + entropy) ≈ typical
        n_classes: int = 5,
        hidden_sizes: list[int] = None,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]

        layers = []
        in_dim = n_features
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, n_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

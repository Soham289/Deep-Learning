"""1D-CNN model — Member B (Bhavya), Model 2 of the HAR comparison.

Architecture, per the synopsis (Model 2: 1D Convolutional Neural Network):
three stacked Conv1D blocks (64, 128, 128 filters; kernel size 5-7), each
followed by batch normalisation, ReLU and max pooling, then global average
pooling and a dense classification head.

The convolutional structure imposes translation invariance and locality: a
kernel that has learned to recognise one phase of a gait cycle recognises it
anywhere in the 2.56s window, which is the structural assumption an MLP
cannot express.

Input convention
-----------------
Expects (N, 9, 128) — channels-first, matching `har_data.make_loaders(...,
channels_first=True)`. nn.Conv1d slides its kernel along the last dimension
(time), treating the 9 channels as input feature maps.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Conv1d -> BatchNorm1d -> ReLU -> MaxPool1d, with 'same' padding.

    Padding is set to kernel_size // 2 so the block only changes sequence
    length via the pooling step, keeping the receptive-field arithmetic
    predictable across the three stacked blocks.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool_size: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HARCnn1D(nn.Module):
    """Three-block 1D-CNN for 6-class HAR over (N, 9, 128) windows.

    128 timesteps -> /2 -> 64 -> /2 -> 32 -> /2 -> 16, then global average
    pooling collapses the time axis entirely, so the dense head sees a fixed
    128-dim vector regardless of window length. This is what lets the same
    architecture generalise to a different sampling rate or window size
    without changing the classifier head.
    """

    def __init__(
        self,
        in_channels: int = 9,
        n_classes: int = 6,
        channel_sizes: tuple[int, int, int] = (64, 128, 128),
        kernel_sizes: tuple[int, int, int] = (7, 5, 5),
        dropout: float = 0.3,
    ):
        super().__init__()
        c1, c2, c3 = channel_sizes
        k1, k2, k3 = kernel_sizes

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, c1, k1),
            ConvBlock(c1, c2, k2),
            ConvBlock(c2, c3, k3),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(c3, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 9, 128)
        x = self.blocks(x)               # (N, c3, T // 8)
        x = self.global_pool(x)          # (N, c3, 1)
        x = x.squeeze(-1)                # (N, c3)
        return self.classifier(x)        # (N, n_classes) logits


if __name__ == "__main__":
    model = HARCnn1D()
    dummy = torch.randn(8, 9, 128)
    out = model(dummy)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"output shape: {tuple(out.shape)}")
    print(f"trainable parameters: {n_params:,}")

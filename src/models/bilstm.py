"""BiLSTM model implementation — Member C (Avni), Model 3 of the HAR comparison.

Architecture, per the synopsis:
Two stacked bidirectional LSTM layers (64-128 hidden units per direction),
inter-layer dropout, mean pooling over timesteps (collapsing the 128 time steps),
followed by a dense classification head returning raw logits for 6 classes.

Input convention:
Expects (N, 128, 9) — channels-last, matching `har_data.make_loaders(..., channels_first=False)`.
"""

from __future__ import annotations

import torch
from torch import nn


class BiLSTMClassifier(nn.Module):
    """Two-layer bidirectional LSTM classifier for 6-class HAR.

    Input shape:  (N, 128, 9) where 128 = window timesteps, 9 = inertial channels.
    Output shape: (N, 6) raw logits (no softmax).
    """

    def __init__(
        self,
        input_size: int = 9,
        hidden_size: int = 128,
        num_layers: int = 2,
        n_classes: int = 6,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 128, 9)
        lstm_out, _ = self.lstm(x)           # (N, 128, 2 * hidden_size)
        pooled = lstm_out.mean(dim=1)        # Mean pooling over time -> (N, 2 * hidden_size)
        return self.classifier(pooled)       # (N, n_classes) raw logits


# Alias for compatibility if referenced by HAR prefix
HARBiLSTM = BiLSTMClassifier


if __name__ == "__main__":
    model = BiLSTMClassifier()
    dummy = torch.randn(8, 128, 9)
    out = model(dummy)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"output shape: {tuple(out.shape)}")
    print(f"trainable parameters: {n_params:,}")

"""MLP baseline — Member A (Soham), Model 1 of the HAR comparison.

=============================================================================
SCAFFOLD. The architecture body below is deliberately left for you to write.
You are individually vivaed on this model at the Final Presentation, so the
twenty lines that define it should be yours. Everything else here — the
docstrings, the contract, the self-test — is spec, not answer.
=============================================================================

What this model is for
----------------------
This is the CONTROL. The 1D-CNN assumes locality and translation invariance;
the BiLSTM assumes order-sensitive temporal dependency. This model assumes
NOTHING about the structure of its input: it flattens the window and treats
all 1,152 numbers as an unordered feature vector. Shuffle the timesteps and
its accuracy is unchanged — the CNN's and the LSTM's would collapse.

That is exactly why it belongs in the comparison. Without it, "the CNN scored
93%" is a number with nothing to be measured against. With it, the gap between
this model and the other two IS the measured value of architectural prior.

Two input representations (this is your distinctive contribution)
----------------------------------------------------------------
You run the SAME architecture on two different representations of the same
windows, which neither teammate can do:

  (a) RAW SIGNALS   — (N, 128, 9) flattened to 1,152 dims. Learned
                      representation, no human feature engineering.
  (b) 561 FEATURES  — the dataset's hand-crafted time- and frequency-domain
                      descriptors (mean, std, energy, FFT skewness, ...)
                      computed by domain experts in the Anguita et al. paper.

Expect (b) to beat (a) by a wide margin. That single comparison answers the
question the whole project is really about: the CNN's advantage over your raw
baseline is it LEARNING the kind of features that (b) got handed to it. Be
ready to say that sentence in the viva — it is the best answer in the project.

Input convention
----------------
Use `har_data.make_loaders(..., channels_first=False)` -> (N, 128, 9).
`nn.Flatten()` collapses that to (N, 1152). Channel order is irrelevant to
this model (that is the point), but keep it the same as the others anyway so
the protocol claim in the report is literally true.
"""

from __future__ import annotations

import torch
from torch import nn


class HARMlp(nn.Module):
    """Fully connected baseline for 6-class HAR.

    Accepts either (N, 128, 9) raw windows or (N, 561) feature vectors — set
    `input_dim` accordingly. The leading nn.Flatten() makes both work through
    one class, which is what lets you run the dual-representation study
    without maintaining two models.

    Architecture to implement (from the synopsis — do not silently change it,
    the synopsis is already submitted):

        Flatten
        Linear(input_dim -> 256)   BatchNorm1d(256)   ReLU   Dropout(p)
        Linear(256 -> 128)         BatchNorm1d(128)   ReLU   Dropout(p)
        Linear(128 -> n_classes)                              <- logits

    Four things to get right, each of which you may be asked about:

    1. NO softmax on the output. `nn.CrossEntropyLoss` in src/train.py applies
       log-softmax internally. Adding your own means applying it twice, which
       flattens the gradients and quietly costs you several points of
       accuracy with no error message.

    2. BatchNorm goes BEFORE the activation, after the Linear. Know why it is
       there at all: the flattened raw signal has wildly different scales
       across the 1,152 inputs, and without normalisation the first layer's
       gradients are dominated by the few largest-magnitude channels.

    3. Dropout goes AFTER the activation, not before.

    4. Build it as a single `nn.Sequential` assigned to `self.net`. The
       self-test at the bottom and the driver script both expect that name.

    Parameters
    ----------
    input_dim : 1152 for raw windows (128 * 9), 561 for the feature vectors.
    hidden_sizes : (256, 128) per the synopsis.
    dropout : start at 0.3 and tune on VALIDATION loss only, never on test.
    """

    def __init__(
        self,
        input_dim: int = 128 * 9,
        n_classes: int = 6,
        hidden_sizes: tuple[int, int] = (256, 128),
        dropout: float = 0.3,
    ):
        super().__init__()
        h1, h2 = hidden_sizes

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, 128, 9) or (N, 561) -> (N, n_classes) raw logits."""
        return self.net(x)


# ---------------------------------------------------------------------------
# Self-test — run `python src/models/mlp.py` to check your implementation.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Raw-signal variant")
    m_raw = HARMlp(input_dim=128 * 9)
    out = m_raw(torch.randn(8, 128, 9))
    p_raw = sum(p.numel() for p in m_raw.parameters() if p.requires_grad)
    print(f"  output shape : {tuple(out.shape)}   (expected (8, 6))")
    print(f"  parameters   : {p_raw:,}        (expect 329,606)")

    print("\n561-feature variant")
    m_feat = HARMlp(input_dim=561)
    out_f = m_feat(torch.randn(8, 561))
    p_feat = sum(p.numel() for p in m_feat.parameters() if p.requires_grad)
    print(f"  output shape : {tuple(out_f.shape)}   (expected (8, 6))")
    print(f"  parameters   : {p_feat:,}        (expect 178,310)")

    # Permutation invariance: the property that defines this baseline.
    # Shuffling timesteps leaves the output distribution unchanged in kind --
    # the model never saw time as an axis. Mention this in the viva.
    m_raw.eval()
    x = torch.randn(4, 128, 9)
    with torch.no_grad():
        a = m_raw(x)
        b = m_raw(x[:, torch.randperm(128), :])
    print(f"\nOutputs differ after shuffling timesteps: {not torch.allclose(a, b)}")
    print("  (True is expected -- weights are position-specific. The point is")
    print("   that the model has no reason to prefer the real ordering: retrain")
    print("   on shuffled data and accuracy is unchanged. The CNN's would not be.)")

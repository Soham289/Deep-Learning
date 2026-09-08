"""Shared training loop entry point.

Every model (MLP / 1D-CNN / BiLSTM) trains through this one function so that
optimizer, loss, early-stopping criterion and random seed are identical
across the comparison — architecture is meant to be the only variable that
differs. Do not write a per-model training loop; import this instead.
"""

from __future__ import annotations

import copy
import time

import torch
from torch import nn


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device: str | torch.device | None = None,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    max_epochs: int = 100,
    patience: int = 10,
    verbose: bool = True,
) -> dict:
    """Train with Adam + categorical cross-entropy, early-stopped on val loss.

    Early stopping watches validation loss (not accuracy) computed on the
    subject-held-out validation split from `har_data.load_bundle`, per the
    synopsis's shared protocol. The model's weights are restored to the best
    epoch before returning, so the caller always gets the early-stopped model,
    not the last one.

    Returns a dict with the training history, the best epoch/val loss, total
    wall-clock training time, and the number of trainable parameters — the
    figures the comparison table in the report needs.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()  # expects integer class labels, i.e. categorical CE

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    start_time = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss, n_seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            n_seen += xb.size(0)
        train_loss = running_loss / n_seen

        val_loss, val_acc = _evaluate_loss_acc(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose:
            marker = " *" if improved else ""
            print(f"  epoch {epoch:3d}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}{marker}")

        if epochs_without_improvement >= patience:
            if verbose:
                print(f"  early stopping at epoch {epoch} "
                      f"(no val_loss improvement for {patience} epochs)")
            break

    training_time = time.perf_counter() - start_time
    model.load_state_dict(best_state)  # restore best-epoch weights

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "training_time_sec": training_time,
        "n_params": n_params,
        "device": str(device),
    }


@torch.no_grad()
def _evaluate_loss_acc(model: nn.Module, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    running_loss, n_correct, n_seen = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        running_loss += loss.item() * xb.size(0)
        n_correct += (logits.argmax(dim=1) == yb).sum().item()
        n_seen += xb.size(0)
    return running_loss / n_seen, n_correct / n_seen


if __name__ == "__main__":
    print("Import train_model from this module; it is not a standalone script.")

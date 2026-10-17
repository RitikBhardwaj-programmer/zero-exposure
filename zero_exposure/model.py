"""The shared model and parameter helpers.

A small MLP (13 -> 16 -> 8 -> 1, 369 parameters): enough for 13 clinical features, small enough that SecAgg+ masking and the
inspector stay readable. Parameters cross the network as numpy arrays; an array may arrive in the other byte order
(an s390x hospital is big-endian, an amd64 server little-endian), so every incoming array is converted to native order first.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

N_PARAMS = 369


class HeartNet(nn.Module):
    def __init__(self, n_features: int = 13):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))

    def forward(self, x):
        return self.net(x)  # logits


def get_weights(model: nn.Module) -> list[np.ndarray]:
    return [p.detach().cpu().numpy().astype(np.float32, copy=True) for p in model.state_dict().values()]


def native(a) -> np.ndarray:
    """Same values, native byte order, float32 (torch refuses non-native arrays; numpy converts '>f4' on the cast)."""
    return np.ascontiguousarray(a, dtype=np.float32)


def set_weights(model: nn.Module, weights) -> None:
    keys = list(model.state_dict().keys())
    if len(keys) != len(weights):
        raise ValueError(f"expected {len(keys)} arrays, got {len(weights)}")
    model.load_state_dict({k: torch.from_numpy(native(w).copy()) for k, w in zip(keys, weights)}, strict=True)


def flat(weights) -> np.ndarray:
    return np.concatenate([native(w).ravel() for w in weights])


@torch.no_grad()
def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """(BCE loss, accuracy, F1) on these rows; threshold 0.5."""
    model.eval()
    logits = model(torch.from_numpy(X))
    yt = torch.from_numpy(y)
    loss = float(nn.functional.binary_cross_entropy_with_logits(logits, yt))
    pred = (logits > 0).float()
    acc = float((pred == yt).float().mean())
    tp = float(((pred == 1) & (yt == 1)).sum())
    fp = float(((pred == 1) & (yt == 0)).sum())
    fn = float(((pred == 0) & (yt == 1)).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    return loss, acc, f1


@torch.no_grad()
def example_losses(weights, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-patient BCE loss of a model with these weights (used by the membership-inference attack)."""
    model = HeartNet()
    set_weights(model, weights)
    model.eval()
    return nn.functional.binary_cross_entropy_with_logits(model(torch.from_numpy(X)), torch.from_numpy(y), reduction="none").numpy().ravel()

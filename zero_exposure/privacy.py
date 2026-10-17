"""Differentially private local training (DP-SGD with Opacus) and the privacy budget each hospital spends.

DP-SGD: every step samples patients independently (Poisson sampling, rate q), clips each patient's gradient to norm `clip`,
adds Gaussian noise of scale `noise * clip`, then averages. The budget epsilon after r rounds is computed with the RDP
accountant from (noise, q, number of steps); it depends only on public numbers, so the server can report it.

FedProx under DP: Opacus builds the update from per-patient gradients, so a proximal term added to the loss would be lost.
The pull towards the global model is applied as its own step after each DP step instead (it reads no patient data).
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
from torch.utils.data import DataLoader, TensorDataset

from .model import native

BATCH_SIZE = 16  # with 5 local epochs and 20 rounds: max epsilon 12.2 at noise 2, 166 at noise 0.5
LR = 0.05
MOMENTUM = 0.9


def steps_per_epoch(n: int) -> int:
    return math.ceil(n / BATCH_SIZE)


def sample_rate(n: int) -> float:
    return 1.0 / steps_per_epoch(n)  # what Opacus uses for a loader of this length


def epsilon_after(n: int, rounds_done: int, local_epochs: int, noise: float, delta: float) -> float:
    """Epsilon a hospital with n training patients has spent after `rounds_done` rounds (inf without noise, 0 before training)."""
    if rounds_done <= 0:
        return 0.0
    if noise <= 0:
        return math.inf
    acc = RDPAccountant()
    acc.history = [(noise, sample_rate(n), rounds_done * local_epochs * steps_per_epoch(n))]
    return float(acc.get_epsilon(delta))


def train_private(model: nn.Module, X: np.ndarray, y: np.ndarray, *, epochs: int, noise: float, clip: float, seed: int,
                  prox_mu: float = 0.0, global_weights=None) -> nn.Module:
    """Train `model` in place with DP-SGD. `model` must be a fresh module (Opacus attaches hooks to it)."""
    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)), batch_size=BATCH_SIZE, shuffle=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    anchors = [torch.from_numpy(native(w)) for w in global_weights] if (prox_mu and global_weights is not None) else None
    loss_fn = nn.BCEWithLogitsLoss()
    with warnings.catch_warnings():
        # Opacus warns about its non-secure RNG (we seed on purpose, for reproducible experiments) and torch about
        # backward hooks on inputs that need no gradient: both are expected here.
        warnings.simplefilter("ignore")
        engine = PrivacyEngine(accountant="rdp")
        dp_model, dp_opt, dp_loader = engine.make_private(
            module=model, optimizer=optimizer, data_loader=loader, noise_multiplier=noise, max_grad_norm=clip,
            poisson_sampling=True, noise_generator=torch.Generator().manual_seed(seed))
        dp_loader.batch_sampler.generator = torch.Generator().manual_seed(seed + 1)  # make_private leaves it on the global RNG
        dp_model.train()
        for _ in range(epochs):
            for xb, yb in dp_loader:
                if len(xb) == 0:  # Poisson sampling can draw an empty batch
                    continue
                dp_opt.zero_grad()
                loss_fn(dp_model(xb), yb).backward()
                dp_opt.step()
                if anchors is not None:
                    with torch.no_grad():
                        for p, a in zip(model.parameters(), anchors):
                            p -= LR * prox_mu * (p - a)
    return model

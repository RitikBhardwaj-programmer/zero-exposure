"""The attack lab: a hospital that poisons its update, and a curious server that tests who was in the training data.

Poisoning (by the hospital `attacker`, every round):
  labelflip  trains on flipped labels (sick <-> healthy); same DP as everyone else.
  boost      sends global + boost * (local - global): its update counts `boost` times more in the average.
  flipboost  both: amplifies a harmful direction (plain boost only amplifies an honest one).
Membership inference (by the server, after training): a loss-threshold attack (Yeom et al., 2018). Patients the model trained
on tend to have a lower loss. Score = -loss; the AUC says how well that separates members (train rows) from non-members (test
rows) of one hospital. AUC 0.5 = a coin flip = nothing leaks to this attack.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from .config import MAX_WEIGHT, RunConfig
from .model import example_losses, native


def flips_labels(cfg: RunConfig, pid: int) -> bool:
    return pid == cfg.attacker and cfg.attack in ("labelflip", "flipboost")


def boosts(cfg: RunConfig, pid: int) -> bool:
    return pid == cfg.attacker and cfg.attack in ("boost", "flipboost")


def flip(y: np.ndarray) -> np.ndarray:
    return (1.0 - y).astype(np.float32)


def boost_update(local, global_weights, factor: float) -> list[np.ndarray]:
    return [(native(g) + factor * (native(p) - native(g))).astype(np.float32) for p, g in zip(local, global_weights)]


def max_scaled(weights, n_examples: int) -> float:
    """The largest value SecAgg+ sees for this update: it multiplies by n / max_weight before clipping to +-clipping_range."""
    return max(float(np.abs(native(w)).max()) for w in weights) * n_examples / MAX_WEIGHT


def membership_attack(weights, X_in, y_in, X_out, y_out, seed: int) -> tuple[float, float, int]:
    """(AUC, advantage = best TPR - FPR, n per side). Balanced: n = min(members, non-members), seeded subsample.

    The advantage picks the best threshold on the same rows it scores, so it is optimistic; AUC is the number to quote.
    """
    rng = np.random.default_rng(seed)
    n = min(len(y_in), len(y_out))
    i_in = rng.choice(len(y_in), n, replace=False)
    i_out = rng.choice(len(y_out), n, replace=False)
    scores = -np.concatenate([example_losses(weights, X_in[i_in], y_in[i_in]), example_losses(weights, X_out[i_out], y_out[i_out])])
    labels = np.concatenate([np.ones(n), np.zeros(n)])
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(roc_auc_score(labels, scores)), float(np.max(tpr - fpr)), int(n)

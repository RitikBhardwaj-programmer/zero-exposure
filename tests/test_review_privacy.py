"""Review tests: DP-SGD randomness, the epsilon accountant and FedProx under DP (no Flower run, ~1 min).

Run: python -m pytest tests/test_review_privacy.py -q
"""
import math

import numpy as np
import pytest
import torch

from zero_exposure import privacy
from zero_exposure.client_app import train_seed
from zero_exposure.config import RunConfig
from zero_exposure.data import prepare
from zero_exposure.model import HeartNet, flat, get_weights


def _local_update(X, y, seed, noise=1.0, epochs=1):
    torch.manual_seed(0)  # the initial (global) model is public: the server sent it
    m = HeartNet()
    start = get_weights(m)
    privacy.train_private(m, X, y, epochs=epochs, noise=noise, clip=1.0, seed=seed, global_weights=start)
    return flat(get_weights(m))


# ---------------------------------------------------------------- the DP noise must be secret
@pytest.mark.xfail(strict=True, raises=AssertionError, reason="FINDING 1: DP-SGD noise and Poisson sampling are derived only from the public run "
                                       "seed, so the update is a deterministic function of the data and epsilon is void")
def test_dp_update_is_not_a_deterministic_function_of_public_values():
    """Differential privacy needs randomness the adversary cannot reproduce. The server knows the run seed (it is in
    run_config, the baseline event and RESULT), the hospital id and the round, i.e. everything train_seed() uses.

    Neighbouring datasets D and D' = D minus one patient. A server that recomputes both updates from the public seed and
    compares them with the update it received names the right dataset every time. For any finite epsilon, DP requires
    P[guess D | D] <= e^eps * P[guess D | D'] + delta; here that reads 1 <= e^eps * 0 + 1e-5, which is false.
    """
    h = prepare(2, 0)  # switzerland, 92 training patients
    cfg = RunConfig(seed=0)
    seed = train_seed(cfg, 2, 1)
    X_d, y_d = h.X_train, h.y_train
    X_n, y_n = h.X_train[1:], h.y_train[1:]
    eps_one_round = privacy.epsilon_after(len(y_d), 1, 1, 1.0, 1e-5)
    assert math.isfinite(eps_one_round)  # the run reports a finite budget for this round

    correct = 0
    for truth in ("D", "D'"):
        received = _local_update(X_d, y_d, seed) if truth == "D" else _local_update(X_n, y_n, seed)
        # attacker: replay both candidates from public values only
        guess = "D" if np.array_equal(received, _local_update(X_d, y_d, seed)) else "D'"
        correct += guess == truth
    assert correct < 2, f"the server identified the dataset in 2/2 trials although epsilon = {eps_one_round:.2f}"


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="FINDING 1b: the Poisson-sampling stream of round r reuses the seed of the noise "
                                       "stream of round r+1 (train_seed + 1)")
def test_sampling_and_noise_streams_never_share_a_seed():
    """privacy.train_private seeds the noise generator with `seed` and the batch sampler with `seed + 1`, and
    client_app.train_seed gives consecutive rounds consecutive seeds. So round r's sampler and round r+1's noise start from
    the same generator state. DP noise must be independent of everything else."""
    cfg = RunConfig(seed=0, rounds=20)
    noise_seeds = {train_seed(cfg, pid, r) for pid in range(4) for r in range(1, cfg.rounds + 1)}
    sampler_seeds = {train_seed(cfg, pid, r) + 1 for pid in range(4) for r in range(1, cfg.rounds + 1)}
    assert not noise_seeds & sampler_seeds, sorted(noise_seeds & sampler_seeds)[:5]


def test_train_seeds_are_distinct_per_hospital_round_and_run_seed():
    seen = {}
    for s in range(3):
        for pid in range(4):
            for r in range(1, 31):  # the dashboard allows up to 30 rounds
                k = train_seed(RunConfig(seed=s), pid, r)
                assert k not in seen, (k, seen.get(k), (s, pid, r))
                seen[k] = (s, pid, r)


# ---------------------------------------------------------------- accountant
def test_sample_rate_and_steps_match_opacus_dataloader():
    """epsilon_after assumes q = 1/ceil(n/B) and ceil(n/B) steps per epoch: what Opacus's DPDataLoader uses."""
    from opacus.data_loader import DPDataLoader
    from torch.utils.data import DataLoader, TensorDataset
    for n in (92, 150, 220, 227, 16, 17):
        loader = DataLoader(TensorDataset(torch.zeros(n, 13), torch.zeros(n, 1)), batch_size=privacy.BATCH_SIZE)
        dp = DPDataLoader.from_data_loader(loader)
        assert dp.sample_rate == pytest.approx(privacy.sample_rate(n))
        assert len(dp) == privacy.steps_per_epoch(n)


@pytest.mark.parametrize("n", [92, 150, 220, 227])
def test_epsilon_is_monotone_in_rounds_epochs_and_noise(n):
    rounds = [privacy.epsilon_after(n, r, 5, 1.0, 1e-5) for r in range(0, 21)]
    assert rounds[0] == 0.0 and all(a < b for a, b in zip(rounds, rounds[1:]))
    epochs = [privacy.epsilon_after(n, 3, e, 1.0, 1e-5) for e in (1, 2, 5, 10)]
    assert all(a < b for a, b in zip(epochs, epochs[1:]))
    noises = [privacy.epsilon_after(n, 3, 5, s, 1e-5) for s in (0.5, 1.0, 2.0, 4.0)]
    assert all(a > b for a, b in zip(noises, noises[1:]))
    deltas = [privacy.epsilon_after(n, 3, 5, 1.0, d) for d in (1e-7, 1e-5, 1e-3)]
    assert all(a > b for a, b in zip(deltas, deltas[1:]))


def test_smaller_hospital_spends_more_epsilon():
    """Same number of rounds: a small hospital has a larger sampling rate q, so it spends more."""
    eps = [privacy.epsilon_after(n, 20, 5, 1.0, 1e-5) for n in (92, 150, 220)]
    assert eps[0] > eps[1] > eps[2]


def test_noise_zero_is_infinite_and_null_in_events():
    from zero_exposure.config import finite_or_none
    assert math.isinf(privacy.epsilon_after(150, 1, 5, 0.0, 1e-5))
    assert finite_or_none(privacy.epsilon_after(150, 1, 5, 0.0, 1e-5)) is None
    assert privacy.epsilon_after(150, 0, 5, 0.0, 1e-5) == 0.0  # round 0: nothing spent yet
    assert finite_or_none(None) is None and finite_or_none(float("nan")) is None and finite_or_none(2) == 2.0


def test_noise_zero_training_still_clips_and_runs():
    h = prepare(2, 0)
    a = _local_update(h.X_train, h.y_train, seed=3, noise=0.0)
    b = _local_update(h.X_train, h.y_train, seed=3, noise=0.0)
    assert np.isfinite(a).all() and np.array_equal(a, b)


# ---------------------------------------------------------------- FedProx under DP
def test_fedprox_mu_zero_equals_fedavg_training():
    h = prepare(1, 0)
    torch.manual_seed(0)
    start = get_weights(HeartNet())

    def run(mu):
        torch.manual_seed(0)
        m = HeartNet()
        privacy.train_private(m, h.X_train, h.y_train, epochs=1, noise=1.0, clip=1.0, seed=9, prox_mu=mu,
                              global_weights=start)
        return flat(get_weights(m))
    assert np.array_equal(run(0.0), run(0.0))
    assert not np.array_equal(run(0.0), run(1.0))


def test_fedprox_anchor_accepts_big_endian_global_weights():
    """The anchor is the global model as received; from an s390x server it is big-endian."""
    h = prepare(1, 0)
    torch.manual_seed(0)
    start = get_weights(HeartNet())

    def run(anchor):
        torch.manual_seed(0)
        m = HeartNet()
        privacy.train_private(m, h.X_train, h.y_train, epochs=1, noise=1.0, clip=1.0, seed=9, prox_mu=0.5,
                              global_weights=anchor)
        return flat(get_weights(m))
    assert np.array_equal(run(start), run([w.astype(">f4") for w in start]))

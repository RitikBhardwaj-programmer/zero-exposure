"""Fast checks of the core pieces (no Flower run). Run: python -m pytest tests/test_core_units.py -q"""
import json
import math
import warnings

import numpy as np
import pytest
import torch

from zero_exposure import attacks, events
from zero_exposure.config import ConfigError, RunConfig, from_context
from zero_exposure.data import load_all, prepare
from zero_exposure.model import N_PARAMS, HeartNet, flat, get_weights, set_weights
from zero_exposure import privacy


# ---------------------------------------------------------------- settings
def test_defaults_are_valid_and_match_pyproject():
    from zero_exposure.sim import defaults
    cfg = RunConfig.from_mapping(defaults())
    assert cfg == RunConfig()


@pytest.mark.parametrize("bad", [
    {"agg": "median", "secagg": 1},
    {"agg": "trimmed", "secagg": 1},
    {"agg": "median", "secagg": 0, "strategy": "fedprox"},
    {"attack": "labelflip", "attacker": 0, "mia-hospital": 0},
    {"attacker": 4}, {"attack": "steal"}, {"strategy": "sgd"}, {"noise": -1.0}, {"rounds": 0}, {"unknown-key": 1},
])
def test_refused_settings(bad):
    with pytest.raises(ConfigError):
        RunConfig.from_mapping(bad)


def test_allowed_combinations():
    RunConfig.from_mapping({"agg": "median", "secagg": 0, "attack": "flipboost"})
    RunConfig.from_mapping({"attack": "labelflip", "attacker": 0, "mia": 0, "mia-hospital": 0})


def test_types_are_coerced_and_sim_env_overrides(monkeypatch):
    cfg = RunConfig.from_mapping({"rounds": "3", "noise": "2", "secagg": 0})
    assert (cfg.rounds, cfg.noise, cfg.secagg) == (3, 2.0, 0)
    monkeypatch.setenv("ZE_RUN_CONFIG", json.dumps({"rounds": 7}))
    assert from_context({"rounds": 3}).rounds == 7


def test_mia_needs_the_side_channel():
    assert not RunConfig(mia=1).mia_active
    assert RunConfig(mia=1, side_dir="x").mia_active


# ---------------------------------------------------------------- data
def test_split_sizes_match_the_contract():
    data = load_all(0)
    assert [len(h.y_train) for h in data] == [227, 220, 92, 150]
    assert [len(h.y_test) for h in data] == [76, 74, 31, 50]
    for h in data:
        assert h.X_train.dtype == np.float32 and not np.isnan(h.X_train).any() and not np.isnan(h.X_test).any()
        assert np.allclose(h.X_train.mean(axis=0), 0, atol=1e-4)  # standardised with its own train statistics


def test_a_supernode_with_one_file(tmp_path):
    import shutil
    from zero_exposure.data import data_dir
    shutil.copy(data_dir() / "processed.hungarian.data", tmp_path)
    data = load_all(0, allow_missing=True, folder=tmp_path)
    assert data[0] is None and data[1] is not None and data[2] is None and data[3] is None
    with pytest.raises(FileNotFoundError):
        load_all(0, allow_missing=False, folder=tmp_path)


def test_seed_changes_the_split():
    assert not np.array_equal(prepare(0, 0).y_test, prepare(0, 1).y_test) or not np.array_equal(prepare(0, 0).X_test, prepare(0, 1).X_test)


# ---------------------------------------------------------------- model
def test_parameter_count_and_round_trip():
    m = HeartNet()
    w = get_weights(m)
    assert flat(w).size == N_PARAMS == 369
    m2 = HeartNet()
    set_weights(m2, w)
    assert all(np.array_equal(a, b) for a, b in zip(w, get_weights(m2)))


def test_big_endian_arrays_are_accepted():
    """An s390x hospital sends big-endian arrays; torch refuses those unless converted."""
    w = get_weights(HeartNet())
    big = [a.astype(">f4") for a in w]
    m = HeartNet()
    set_weights(m, big)
    assert all(np.array_equal(a, b) for a, b in zip(w, get_weights(m)))


# ---------------------------------------------------------------- privacy
def test_epsilon_grows_with_rounds_and_is_zero_or_infinite_at_the_edges():
    eps = [privacy.epsilon_after(150, r, 5, 1.0, 1e-5) for r in range(0, 6)]
    assert eps[0] == 0 and all(a < b for a, b in zip(eps, eps[1:]))
    assert math.isinf(privacy.epsilon_after(150, 3, 5, 0.0, 1e-5))
    assert privacy.epsilon_after(150, 3, 5, 2.0, 1e-5) < privacy.epsilon_after(150, 3, 5, 1.0, 1e-5)


def test_epsilon_matches_the_opacus_accountant_after_real_training():
    """Our stateless formula equals what Opacus's own RDP accountant records while training."""
    from opacus import PrivacyEngine
    from torch.utils.data import DataLoader, TensorDataset
    h = prepare(2, 0)
    torch.manual_seed(0)
    model = HeartNet()
    loader = DataLoader(TensorDataset(torch.from_numpy(h.X_train), torch.from_numpy(h.y_train)), batch_size=privacy.BATCH_SIZE)
    engine = PrivacyEngine(accountant="rdp")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m, opt, dl = engine.make_private(module=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.05), data_loader=loader,
                                         noise_multiplier=1.3, max_grad_norm=1.0, poisson_sampling=True)
        for _ in range(2):
            for xb, yb in dl:
                if len(xb):
                    opt.zero_grad()
                    torch.nn.functional.binary_cross_entropy_with_logits(m(xb), yb).backward()
                opt.step()  # an empty batch is still a step for the accountant
    expected = privacy.epsilon_after(len(h.y_train), 1, 2, 1.3, 1e-5)
    assert engine.get_epsilon(1e-5) == pytest.approx(expected, rel=1e-6)


def _train(seed, mu=0.0, anchor=None, noise=1.0):
    h = prepare(1, 0)
    torch.manual_seed(0)
    m = HeartNet()
    start = get_weights(m)
    privacy.train_private(m, h.X_train, h.y_train, epochs=2, noise=noise, clip=1.0, seed=seed, prox_mu=mu,
                          global_weights=anchor if anchor is not None else start)
    return start, get_weights(m)


def test_training_is_reproducible_per_seed():
    _, a = _train(5)
    _, b = _train(5)
    _, c = _train(6)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not all(np.array_equal(x, y) for x, y in zip(a, c))


def test_fedprox_pulls_towards_the_global_model():
    start, plain = _train(5, mu=0.0)
    _, prox = _train(5, mu=5.0, anchor=start)
    dist = lambda w: float(np.linalg.norm(flat(w) - flat(start)))
    assert dist(prox) < dist(plain)


# ---------------------------------------------------------------- attacks
def test_attack_roles():
    cfg = RunConfig(attack="flipboost", attacker=2)
    assert attacks.flips_labels(cfg, 2) and attacks.boosts(cfg, 2)
    assert not attacks.flips_labels(cfg, 1) and not attacks.boosts(RunConfig(attack="labelflip", attacker=2), 2)
    assert np.array_equal(attacks.flip(np.array([[0.0], [1.0]], np.float32)), np.array([[1.0], [0.0]], np.float32))


def test_boost_scales_the_step():
    g = [np.zeros(3, np.float32)]
    p = [np.ones(3, np.float32)]
    assert np.allclose(attacks.boost_update(p, g, 10.0)[0], 10.0)


def test_membership_attack_is_balanced_and_bounded():
    h = prepare(0, 0)
    auc, adv, n = attacks.membership_attack(get_weights(HeartNet()), h.X_train, h.y_train, h.X_test, h.y_test, 1)
    assert n == min(len(h.y_train), len(h.y_test)) and 0 <= auc <= 1 and 0 <= adv <= 1


# ---------------------------------------------------------------- events
def test_events_are_json_with_null_for_infinity(tmp_path):
    f = tmp_path / "e" / "events.jsonl"
    events.emit(str(f), "round", epsilon=[1.0, math.inf], acc=np.float32(0.5))
    row = json.loads(f.read_text())
    assert row["type"] == "round" and row["epsilon"] == [1.0, None] and row["acc"] == 0.5 and "t" in row
    events.emit("", "round")  # no file set: nothing happens

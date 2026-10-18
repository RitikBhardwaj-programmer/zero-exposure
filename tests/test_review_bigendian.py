"""Review tests: big-endian (s390x) arrays through every server and client function that touches parameters (no Flower run).

An s390x hospital or server serialises '>f4' arrays; an amd64 peer must give bit-identical results. Run:
python -m pytest tests/test_review_bigendian.py -q
"""
import json

import numpy as np
import pytest
import torch
from flwr.server.strategy.aggregate import aggregate, aggregate_median, aggregate_trimmed_avg

from zero_exposure import attacks
from zero_exposure.client_app import HospitalClient
from zero_exposure.config import RunConfig
from zero_exposure.model import HeartNet, flat, get_weights
from zero_exposure.server_app import Run


def big(ws):
    return [w.astype(">f4") for w in ws]


@pytest.fixture(scope="module")
def initial():
    torch.manual_seed(0)
    return get_weights(HeartNet())


def test_big_endian_helper_really_swaps_bytes(initial):
    b = big(initial)
    assert b[0].dtype.byteorder == ">" and b[0].tobytes() != initial[0].tobytes()


@pytest.mark.parametrize("attack", ["none", "flipboost"])
def test_client_fit_gives_the_same_update_for_big_endian_globals(initial, attack, tmp_path):
    cfg = RunConfig(rounds=1, local_epochs=1, attack=attack, attacker=0, mia=0, side_dir=str(tmp_path), reproducible=1)
    client = HospitalClient(cfg, 0)
    little, n1, _ = client.fit(initial, {"server_round": 1})
    bigres, n2, _ = client.fit(big(initial), {"server_round": 1})
    assert n1 == n2 and np.array_equal(flat(little), flat(bigres))
    assert all(w.dtype == np.float32 for w in bigres)


def test_client_evaluate_accepts_big_endian(initial):
    client = HospitalClient(RunConfig(), 1)
    assert client.evaluate(initial, {}) == client.evaluate(big(initial), {})


@pytest.mark.parametrize("fn", ["fedavg", "median", "trimmed"])
def test_flower_aggregation_of_mixed_byte_orders_matches_little_endian(initial, fn):
    rng = np.random.default_rng(0)
    ups = [[(w + rng.normal(0, 0.1, w.shape)).astype(np.float32) for w in initial] for _ in range(4)]
    n = [227, 220, 92, 150]
    mixed = [big(u) if i == 1 else u for i, u in enumerate(ups)]  # hungarian on s390x
    agg = {"fedavg": lambda us: aggregate(list(zip(us, n))), "median": lambda us: aggregate_median(list(zip(us, n))),
           "trimmed": lambda us: aggregate_trimmed_avg(list(zip(us, n)), 0.25)}[fn]
    assert np.allclose(flat(agg(ups)), flat(agg(mixed)), rtol=0, atol=1e-6)


def test_server_global_eval_and_inspector_accept_big_endian(initial, tmp_path):
    cfg = RunConfig(rounds=1, events_file=str(tmp_path / "e.jsonl"), inspect_dir=str(tmp_path / "insp"), inspect_round=0)
    a = Run(cfg).global_eval(0, initial, {})
    b = Run(cfg).global_eval(0, big(initial), {})
    assert a == b
    saved = np.load(tmp_path / "insp" / "aggregate.npy")
    assert np.array_equal(saved, flat(initial)) and saved.dtype == np.float32
    rows = [json.loads(l) for l in (tmp_path / "e.jsonl").read_text().splitlines()]
    assert rows[0]["acc"] == rows[1]["acc"] and rows[0]["max_abs_weight"] == rows[1]["max_abs_weight"]


def test_attack_helpers_accept_big_endian(initial):
    local = [(w + 0.5).astype(np.float32) for w in initial]
    assert np.array_equal(flat(attacks.boost_update(local, initial, 10.0)), flat(attacks.boost_update(big(local), big(initial), 10.0)))
    assert attacks.max_scaled(local, 150) == attacks.max_scaled(big(local), 150)


def test_membership_attack_reads_a_big_endian_side_channel_update(initial, tmp_path):
    """mia_update.npz is written by the MIA hospital; on s390x it holds '>f4' arrays."""
    cfg = RunConfig(rounds=1, mia=1, side_dir=str(tmp_path))
    run = Run(cfg)
    run.global_eval(0, initial, {})
    np.savez(tmp_path / "mia_update.npz", *big(initial))
    res_big = run.attack_summary()
    np.savez(tmp_path / "mia_update.npz", *initial)
    res_little = run.attack_summary()
    assert res_big["mia_auc_update"] == res_little["mia_auc_update"] is not None
    assert res_big["mia_n"] == 76

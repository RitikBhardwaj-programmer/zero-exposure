"""Review tests: every refused setting (CONTRACTS.md section 1) raises ConfigError; edge values (no Flower run, seconds).

Run: python -m pytest tests/test_review_config.py -q
"""
import json

import pytest

from zero_exposure.config import AGGS, ATTACKS, ConfigError, RunConfig, from_context

REFUSED = [
    # robust aggregation needs every update in the clear
    ({"agg": "median", "secagg": 1}, "SecAgg+"),
    ({"agg": "trimmed", "secagg": 1}, "SecAgg+"),
    # fedprox is a FedAvg variant
    ({"agg": "median", "secagg": 0, "strategy": "fedprox"}, "fedprox"),
    ({"agg": "trimmed", "secagg": 0, "strategy": "fedprox"}, "fedprox"),
    # the MIA hospital must not be the attacker, for every attack and every hospital
    *[({"attack": a, "attacker": h, "mia-hospital": h, "mia": 1}, "mia-hospital") for a in ATTACKS if a != "none" for h in range(4)],
    # ids outside 0..3
    ({"attacker": -1}, "hospital ids"), ({"attacker": 4}, "hospital ids"),
    ({"mia-hospital": -1}, "hospital ids"), ({"mia-hospital": 4}, "hospital ids"),
    # unknown strings
    ({"strategy": "FedAvg"}, "strategy"), ({"agg": "mean"}, "agg"), ({"attack": "poison"}, "attack"),
    # bad numbers
    ({"rounds": 0}, "rounds"), ({"local-epochs": 0}, "local-epochs"), ({"noise": -0.1}, "noise"), ({"clip": 0.0}, "clip"),
    ({"delta": 0.0}, "delta"), ({"delta": 1.0}, "delta"), ({"secagg": 2}, "secagg"), ({"mia": -1}, "mia"),
    ({"trim-beta": 0.5}, "trim-beta"), ({"trim-beta": -0.1}, "trim-beta"), ({"boost": 0.0}, "boost"), ({"mu": -1.0}, "mu"),
    # type errors and unknown keys
    ({"rounds": "three"}, "rounds"), ({"noise": "high"}, "noise"), ({"rounds": "1.5"}, "rounds"), ({"no-such-key": 1}, "unknown"),
]


@pytest.mark.parametrize("bad,words", REFUSED, ids=[json.dumps(b) for b, _ in REFUSED])
def test_every_refused_setting_raises_config_error(bad, words):
    with pytest.raises(ConfigError, match=words.replace("+", r"\+")):
        RunConfig.from_mapping(bad)


def test_config_error_is_a_value_error():
    assert issubclass(ConfigError, ValueError)


ALLOWED = [
    {"agg": a, "secagg": 0} for a in AGGS
] + [
    {"attack": a, "attacker": h, "mia-hospital": (h + 1) % 4} for a in ATTACKS for h in range(4)
] + [
    {"attack": "none", "attacker": 0, "mia-hospital": 0},          # no attack: same id is fine
    {"attack": "labelflip", "attacker": 2, "mia": 0, "mia-hospital": 2},  # no MIA: same id is fine
    {"noise": 0.0}, {"strategy": "fedprox", "secagg": 1}, {"trim-beta": 0.0, "agg": "trimmed", "secagg": 0},
]


@pytest.mark.parametrize("ok", ALLOWED, ids=[json.dumps(o) for o in ALLOWED])
def test_allowed_settings(ok):
    RunConfig.from_mapping(ok)


@pytest.mark.parametrize("attacker", [1, 2, 3])
def test_every_attacker_id_is_accepted_by_every_attack(attacker):
    for a in ("labelflip", "boost", "flipboost"):
        cfg = RunConfig.from_mapping({"attack": a, "attacker": attacker})
        assert cfg.attacker == attacker


@pytest.mark.parametrize("key", ["noise", "clip", "mu", "boost"])
@pytest.mark.parametrize("value", ["nan", "inf"])
def test_non_finite_numbers_are_refused(key, value):
    with pytest.raises(ConfigError):
        RunConfig.from_mapping({key: value})


def test_fractional_int_is_refused():
    with pytest.raises(ConfigError):
        RunConfig.from_mapping({"rounds": 2.7})


@pytest.mark.parametrize("r", [0, 4])
def test_inspect_round_outside_the_run_is_refused(r):
    with pytest.raises(ConfigError):
        RunConfig.from_mapping({"rounds": 3, "inspect-round": r, "inspect-dir": "x"})


def test_public_dict_has_every_contract_key_except_paths():
    keys = set(RunConfig().public_dict())
    assert keys == {"rounds", "local-epochs", "noise", "clip", "delta", "strategy", "mu", "seed", "secagg", "agg", "trim-beta",
                    "attack", "attacker", "boost", "mia", "mia-hospital", "inspect-round", "reproducible"}


def test_sim_env_overrides_run_config_and_bad_env_is_refused(monkeypatch):
    monkeypatch.setenv("ZE_RUN_CONFIG", json.dumps({"agg": "median"}))
    with pytest.raises(ConfigError):
        from_context({"secagg": 1})  # env makes the combination refused
    monkeypatch.delenv("ZE_RUN_CONFIG")
    assert from_context({"secagg": 1}).agg == "fedavg"

"""Review tests: events, RESULT and inspector files of real simulations checked against CONTRACTS.md sections 3-5.

The expected keys and JSON types are read from the examples in CONTRACTS.md itself, so the contract and the code cannot drift
apart unnoticed. Three 2-round simulations cover SecAgg+ on/off, FedAvg/trimmed mean, attackers 1, 2 and 3, noise 0.
Run: python -m pytest tests/test_review_contract.py -q    (about 4-6 minutes)
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
from test_sim import sim  # noqa: E402

from zero_exposure.config import HOSPITALS, RunConfig  # noqa: E402

ROUNDS = 2


def contract_examples() -> dict:
    """{event type: example object} from the JSON block of CONTRACTS.md section 3."""
    text = (ROOT / "CONTRACTS.md").read_text(encoding="utf-8")
    sec3 = text[text.index("## 3."):text.index("## 4.")]
    block = re.search(r"```json\n(.*?)```", sec3, re.S).group(1)
    objs, buf = {}, ""
    for line in block.splitlines():
        buf += line
        try:
            o = json.loads(buf)
        except json.JSONDecodeError:
            continue
        objs[o["type"]] = o
        buf = ""
    assert set(objs) == {"baseline", "round", "attack", "done", "error"}, objs.keys()
    return objs


def result_contract_keys(attack_example: dict) -> set:
    text = (ROOT / "CONTRACTS.md").read_text(encoding="utf-8")
    sec5 = text[text.index("## 5."):text.index("## 6.")]
    extra = set(re.findall(r"`([a-z0-9_]+)`", sec5.split("plus", 1)[1].split("and every field")[0]))
    config = {k.replace("-", "_") for k in RunConfig().public_dict()}
    return config | extra | (set(attack_example) - {"type", "t"})


def check_type(path: str, example, actual) -> None:
    """Example null = nullable of any type; example int = int; example float = int or float; lists element-wise."""
    if example is None:
        return
    if isinstance(example, bool) or isinstance(actual, bool):
        assert type(example) is type(actual), path
    elif isinstance(example, int):
        assert type(actual) is int, f"{path}: expected int, got {actual!r}"
    elif isinstance(example, float):
        assert isinstance(actual, (int, float)), f"{path}: expected number, got {actual!r}"
    elif isinstance(example, str):
        assert isinstance(actual, str), f"{path}: expected str, got {actual!r}"
    elif isinstance(example, list):
        assert isinstance(actual, list) and len(actual) == len(example), f"{path}: {actual!r}"
        for i, (e, a) in enumerate(zip(example, actual)):
            check_type(f"{path}[{i}]", e, a)


def check_event(ev: dict, examples: dict) -> None:
    ex = examples[ev["type"]]
    if ev["type"] == "baseline":
        assert set(ev["config"]) == set(RunConfig().public_dict()), "config must hold all section 1 keys (no paths)"
        ex = {k: v for k, v in ex.items() if k != "config"}
        ev = {k: v for k, v in ev.items() if k != "config"}
    assert set(ev) == set(ex), (ev["type"], sorted(set(ev) ^ set(ex)))
    for k in ex:
        check_type(f"{ev['type']}.{k}", ex[k], ev[k])
    assert isinstance(ev["t"], float)


@pytest.fixture(scope="module")
def tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("review_contract")


@pytest.fixture(scope="module")
def runs(tmp):
    return {
        "secagg_boost_h1": sim(tmp, "secagg_boost_h1", rounds=ROUNDS, secagg=1, attack="boost", attacker=1),
        "trimmed_flip_h2": sim(tmp, "trimmed_flip_h2", rounds=ROUNDS, secagg=0, agg="trimmed", attack="labelflip", attacker=2),
        "nonoise_flipboost_h3": sim(tmp, "nonoise_flipboost_h3", rounds=ROUNDS, secagg=0, noise=0.0, attack="flipboost",
                                    attacker=3, **{"mia-hospital": 1}),
    }


TAGS = ["secagg_boost_h1", "trimmed_flip_h2", "nonoise_flipboost_h3"]


@pytest.mark.parametrize("tag", TAGS)
def test_run_finished(runs, tag):
    res, ev, _, p = runs[tag]
    assert res is not None, p.stdout[-3000:] + p.stderr[-3000:]
    assert [e["type"] for e in ev] == ["baseline"] + ["round"] * (ROUNDS + 1) + ["attack", "done"]


@pytest.mark.parametrize("tag", TAGS)
def test_every_event_matches_contract_section_3(runs, tag):
    examples = contract_examples()
    for e in runs[tag][1]:
        check_event(e, examples)


@pytest.mark.parametrize("tag", TAGS)
def test_round_events_semantics(runs, tag):
    res, ev, _, _ = runs[tag]
    rounds = [e for e in ev if e["type"] == "round"]
    assert [e["round"] for e in rounds] == list(range(ROUNDS + 1))
    assert all(e["total_rounds"] == ROUNDS for e in rounds)
    assert rounds[0]["epsilon"] == [0.0] * 4, "round 0: nothing spent yet"
    noise = res["noise"]
    for e in rounds[1:]:
        if noise == 0:
            assert e["epsilon"] == [None] * 4, "noise 0: epsilon is null"
        else:
            assert all(isinstance(x, float) and x > 0 for x in e["epsilon"])
    assert all(0 <= e["acc"] <= 1 and 0 <= e["f1"] <= 1 and e["loss"] >= 0 for e in rounds)


@pytest.mark.parametrize("tag", TAGS)
def test_result_keys_match_contract_section_5(runs, tag):
    res = runs[tag][0]
    expected = result_contract_keys(contract_examples()["attack"])
    assert set(res) == expected, sorted(set(res) ^ expected)
    for k in ("events_file", "inspect_dir", "side_dir"):
        assert k not in res


@pytest.mark.parametrize("tag", TAGS)
def test_result_agrees_with_events(runs, tag):
    res, ev, _, _ = runs[tag]
    last = [e for e in ev if e["type"] == "round"][-1]
    base, att = ev[0], [e for e in ev if e["type"] == "attack"][0]
    assert res["fed_acc"] == last["acc"] and res["fed_per_hospital_acc"] == last["per_hospital_acc"]
    assert res["epsilon_per_hospital"] == last["epsilon"]
    if res["noise"] == 0:
        assert res["epsilon_max"] is None
    else:
        assert res["epsilon_max"] == max(last["epsilon"])
    assert res["central_acc"] == base["central_acc"] and res["local_per_hospital_acc"] == base["local_per_hospital_acc"]
    assert res["max_abs_weight"] == max(e["max_abs_weight"] for e in ev if e["type"] == "round")
    for k, v in att.items():
        if k not in ("type", "t"):
            assert res[k] == v, k


@pytest.mark.parametrize("tag", TAGS)
def test_attack_event_semantics(runs, tag):
    res, ev, _, _ = runs[tag]
    att = [e for e in ev if e["type"] == "attack"][0]
    last = [e for e in ev if e["type"] == "round"][-1]
    a = att["attacker"]
    assert att["attacker_name"] == HOSPITALS[a] and att["mia_hospital_name"] == HOSPITALS[att["mia_hospital"]]
    assert att["honest_acc"] == pytest.approx(np.mean([x for i, x in enumerate(last["per_hospital_acc"]) if i != a]))
    assert att["attacker_acc"] == last["per_hospital_acc"][a]
    assert att["boost"] == (10.0 if att["attack"] in ("boost", "flipboost") else None)
    assert isinstance(att["clipped_rounds"], int) and 0 <= att["clipped_rounds"] <= ROUNDS
    assert att["max_scaled_attacker"] > 0
    if not att["secagg"]:
        assert att["clipped_rounds"] == 0
    h = att["mia_hospital"]
    assert att["mia_n"] == min(ev[0]["n_train"][h], ev[0]["n_test"][h]), "balanced: n = min(members, non-members)"
    for k in ("mia_auc_update", "mia_auc_global", "mia_adv_update", "mia_adv_global"):
        assert 0 <= att[k] <= 1


def test_attacker_ids_reach_the_clients(runs):
    """Each run wrote the attacker side channel once per round, for the configured attacker only."""
    for tag in TAGS:
        _, _, d, _ = runs[tag]
        rows = [json.loads(l) for l in (d / "side" / "attacker.jsonl").read_text().splitlines()]
        assert [r["round"] for r in rows] == list(range(1, ROUNDS + 1)), tag


def test_inspector_with_secagg(runs):
    _, _, d, _ = runs["secagg_boost_h1"]
    insp = d / "insp"
    meta = json.loads((insp / "inspect.json").read_text())
    assert set(meta) == {"round", "pid", "n_params", "secagg", "n_examples", "clipping_range", "quantization_range"}
    assert meta["round"] == 1 and meta["pid"] == 0 and meta["n_params"] == 369 and meta["secagg"] == 1 and meta["n_examples"] == 227
    assert np.load(insp / "masked.npy").dtype == np.uint32
    assert np.load(insp / "plain.npy").shape == (369,) and np.load(insp / "aggregate.npy").shape == (369,)
    for pid in range(4):
        lines = [json.loads(l) for l in (insp / f"stages_h{pid}.jsonl").read_text().splitlines()]
        assert all(set(l) == {"pid", "round", "stage", "t"} and l["pid"] == pid and l["round"] == 1 for l in lines)
        assert [l["stage"] for l in lines] == ["received", "trained", "masked", "sent"]
        assert [l["t"] for l in lines] == sorted(l["t"] for l in lines)


def test_inspector_without_secagg_has_no_masked_stage(runs):
    _, _, d, _ = runs["trimmed_flip_h2"]
    insp = d / "insp"
    assert not (insp / "masked.npy").exists()
    assert json.loads((insp / "inspect.json").read_text())["secagg"] == 0
    for pid in range(4):
        stages = [json.loads(l)["stage"] for l in (insp / f"stages_h{pid}.jsonl").read_text().splitlines()]
        assert stages == ["received", "trained", "sent"], (pid, stages)


def test_trimmed_mean_is_what_ran(runs):
    res = runs["trimmed_flip_h2"][0]
    assert res["agg"] == "trimmed" and res["trim_beta"] == 0.25 and res["secagg"] == 0


def test_relative_paths_land_in_the_launch_folder(tmp):
    """The dashboard and the docs pass paths; a relative one must resolve against the folder the run was started in, for
    the server AND the hospitals (Ray workers)."""
    cwd = tmp / "relative"
    cwd.mkdir()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONPATH": str(ROOT)}
    p = subprocess.run([sys.executable, "-m", "zero_exposure.sim", "rounds=1", "secagg=0", "events-file=out/events.jsonl",
                        "inspect-dir=out/insp", "side-dir=out/side"], cwd=cwd, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=900)
    assert "RESULT " in p.stdout, p.stdout[-2000:] + p.stderr[-2000:]
    out = cwd / "out"
    assert (out / "events.jsonl").exists()
    assert (out / "insp" / "plain.npy").exists() and (out / "insp" / "stages_h3.jsonl").exists()
    assert (out / "side" / "mia_update.npz").exists()

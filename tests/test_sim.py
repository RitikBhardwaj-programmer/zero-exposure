"""End-to-end checks of the core through real Flower simulations (each run is a subprocess, ~30-60 s).
Run: python -m pytest tests/test_sim.py -q    (about 8-10 minutes)
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
ROUNDS = 3


def sim(tmp: Path, tag: str, **settings):
    """Run one simulation; return (RESULT dict or None, events list, run folder, process)."""
    d = tmp / tag
    d.mkdir(parents=True, exist_ok=True)
    args = {"rounds": ROUNDS, "events-file": (d / "events.jsonl").as_posix(), "inspect-dir": (d / "insp").as_posix(),
            "side-dir": (d / "side").as_posix(), "reproducible": 1, **settings}
    cmd = [sys.executable, "-m", "zero_exposure.sim", *[f"{k}={v}" for k, v in args.items()]]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    lines = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    result = json.loads(lines[-1][7:]) if lines else None
    ev_file = d / "events.jsonl"
    ev = [json.loads(l) for l in ev_file.read_text(encoding="utf-8").splitlines()] if ev_file.exists() else []
    return result, ev, d, p


@pytest.fixture(scope="module")
def tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("sim")


@pytest.fixture(scope="module")
def plain_run(tmp):
    return sim(tmp, "plain", secagg=0)


@pytest.fixture(scope="module")
def secagg_run(tmp):
    return sim(tmp, "secagg", secagg=1)


def test_a_run_writes_the_contract_events(plain_run):
    res, ev, _, p = plain_run
    assert res is not None, p.stdout[-3000:] + p.stderr[-3000:]
    kinds = [e["type"] for e in ev]
    assert kinds[0] == "baseline" and kinds[-2:] == ["attack", "done"]
    rounds = [e for e in ev if e["type"] == "round"]
    assert [e["round"] for e in rounds] == list(range(ROUNDS + 1))
    assert all(len(e["per_hospital_acc"]) == 4 and len(e["epsilon"]) == 4 for e in rounds)
    assert ev[0]["n_train"] == [227, 220, 92, 150]
    assert res["fed_acc"] > 0.65, "federated training should beat a coin flip clearly after 3 rounds"
    eps = [e["epsilon"] for e in rounds]
    assert all(a <= b for r1, r2 in zip(eps, eps[1:]) for a, b in zip(r1, r2)), "epsilon only grows"


def test_the_same_seed_gives_the_same_result(tmp, plain_run):
    again, _, _, _ = sim(tmp, "plain_again", secagg=0)
    for k in ("fed_acc", "fed_f1", "fed_per_hospital_acc", "epsilon_per_hospital", "central_acc", "local_per_hospital_acc"):
        assert again[k] == plain_run[0][k], k


def test_secagg_changes_only_the_rounding(plain_run, secagg_run):
    """The server gets the same aggregate with SecAgg+ on, up to its quantization (stochastic rounding: < 1 cell per hospital)."""
    res, ev, d, p = secagg_run
    assert res is not None, p.stdout[-3000:] + p.stderr[-3000:]
    assert res["epsilon_per_hospital"] == plain_run[0]["epsilon_per_hospital"]
    on = np.load(d / "insp" / "aggregate.npy")
    off = np.load(plain_run[2] / "insp" / "aggregate.npy")
    n = np.array([227, 220, 92, 150])
    cell = 2 * 8.0 / 4194304  # clipping range 8, quantization range 2**22
    bound = 4 * cell / (n / 300).sum()  # each hospital's error is < 1 cell in the scaled domain; the server divides by sum(n/max_weight)
    assert float(np.abs(on.astype(np.float64) - off).max()) < bound


def test_the_inspector_sees_masked_noise(secagg_run):
    from scipy.stats import chi2
    _, _, d, _ = secagg_run
    insp = d / "insp"
    masked = np.load(insp / "masked.npy")
    plain = np.load(insp / "plain.npy")
    meta = json.loads((insp / "inspect.json").read_text())
    assert masked.dtype == np.uint32 and masked.size == plain.size + 1 == 370 and meta["n_params"] == 369
    counts, _ = np.histogram(masked.astype(np.float64), bins=32, range=(0, 2**32))
    expected = masked.size / 32
    stat = float(((counts - expected) ** 2 / expected).sum())
    assert stat < chi2.ppf(0.99, 31), "the masked vector should look uniform (random) to the server"
    for pid in range(4):
        stages = [json.loads(l)["stage"] for l in (insp / f"stages_h{pid}.jsonl").read_text().splitlines()]
        assert stages == ["received", "trained", "masked", "sent"], (pid, stages)


def test_label_flip_hurts_and_the_side_channel_works(tmp, plain_run):
    res, ev, d, p = sim(tmp, "flip", secagg=0, attack="flipboost", attacker=3)
    assert res is not None, p.stdout[-3000:] + p.stderr[-3000:]
    assert res["honest_acc"] < plain_run[0]["honest_acc"] - 0.1, "flip + boost should hurt the honest hospitals"
    assert res["max_scaled_attacker"] is not None and res["clipped_rounds"] == 0  # SecAgg+ is off: nothing is clipped
    assert 0 <= res["mia_auc_update"] <= 1 and 0 <= res["mia_auc_global"] <= 1 and res["mia_n"] == 76


def test_median_runs_and_fedprox_runs(tmp):
    med, _, _, p = sim(tmp, "median", secagg=0, agg="median", attack="labelflip")
    assert med is not None and med["agg"] == "median", p.stdout[-3000:]
    prox, _, _, p = sim(tmp, "prox", secagg=1, strategy="fedprox", mu=0.1)
    assert prox is not None and prox["strategy"] == "fedprox", p.stdout[-3000:]


def test_refused_settings_stop_before_training(tmp):
    res, ev, _, p = sim(tmp, "refused", secagg=1, agg="median")
    assert res is None and p.returncode != 0 and "SecAgg+ hides exactly that" in (p.stdout + p.stderr)

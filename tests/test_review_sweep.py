"""Review tests: the sweep runner's plans and resume logic on a temporary results folder (no simulation is started).

Run: python -m pytest tests/test_review_sweep.py -q
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import sweep  # noqa: E402

from zero_exposure.config import RunConfig  # noqa: E402


def fake_result(settings: dict, name: str) -> dict:
    """What a real RESULT line holds for these settings: every config key with its coerced type, plus a few metrics."""
    cfg = RunConfig.from_mapping({**settings, "side-dir": "x"})
    return {**{k.replace("-", "_"): v for k, v in cfg.public_dict().items()}, "fed_acc": 0.7, "honest_acc": 0.7,
            "sweep": name, "wall_s": 1.0}


@pytest.fixture
def results(tmp_path, monkeypatch):
    calls = []

    def run_one(name, settings, out):
        calls.append((name, settings))
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(fake_result(settings, name)) + "\n")
        return f"ok {name}"

    monkeypatch.setattr(sweep, "RESULTS", tmp_path)
    monkeypatch.setattr(sweep, "run_one", run_one)
    return tmp_path, calls


def main(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["sweep.py", *args])
    return sweep.main()


def test_plan_sizes_and_every_plan_is_a_valid_setting():
    p = sweep.plans(20, 5)
    assert {k: len(v) for k, v in p.items()} == {"strategy": 10, "privacy": 40, "attacks": 60}
    for runs in p.values():
        for r in runs:
            RunConfig.from_mapping(r)  # no planned run is refused
        assert len({sweep.key(r) for r in runs}) == len(runs), "two planned runs share a resume key"


def test_a_fresh_sweep_runs_everything_and_a_second_call_runs_nothing(results, monkeypatch):
    tmp, calls = results
    assert main(monkeypatch, "all", "--rounds", "2", "--seeds", "2", "--jobs", "1") == 0
    assert len(calls) == 4 + 16 + 24
    calls.clear()
    assert main(monkeypatch, "all", "--rounds", "2", "--seeds", "2", "--jobs", "1") == 0
    assert calls == [], f"resume re-ran {len(calls)} finished runs, e.g. {calls[:2]}"


def test_an_interrupted_sweep_runs_only_the_missing_runs(results, monkeypatch):
    tmp, calls = results
    runs = sweep.plans(2, 2)["privacy"]
    with (tmp / "privacy.jsonl").open("w", encoding="utf-8") as f:
        for r in runs[:5]:
            f.write(json.dumps(fake_result(r, "privacy")) + "\n")
    main(monkeypatch, "privacy", "--rounds", "2", "--seeds", "2", "--jobs", "1")
    assert [s for _, s in calls] == runs[5:]


def test_other_rounds_or_seeds_are_not_mistaken_for_done(results, monkeypatch):
    tmp, calls = results
    main(monkeypatch, "strategy", "--rounds", "2", "--seeds", "1", "--jobs", "1")
    calls.clear()
    main(monkeypatch, "strategy", "--rounds", "3", "--seeds", "2", "--jobs", "1")
    assert len(calls) == 4


def test_failed_runs_are_retried_next_time(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "RESULTS", tmp_path)
    monkeypatch.setattr(sweep, "run_one", lambda n, s, o: f"FAILED {s}")
    assert main(monkeypatch, "strategy", "--rounds", "2", "--seeds", "1", "--jobs", "1") == 1
    assert not (tmp_path / "strategy.jsonl").exists() or (tmp_path / "strategy.jsonl").read_text() == ""


def test_a_half_written_last_line_does_not_block_resume(results, monkeypatch):
    tmp, calls = results
    runs = sweep.plans(2, 1)["strategy"]
    line = json.dumps(fake_result(runs[0], "strategy"))
    (tmp / "strategy.jsonl").write_text(line + "\n" + line[: len(line) // 2], encoding="utf-8")
    main(monkeypatch, "strategy", "--rounds", "2", "--seeds", "1", "--jobs", "1")
    assert [s for _, s in calls] == runs[1:]

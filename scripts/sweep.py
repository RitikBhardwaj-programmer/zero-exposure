"""Run the experiment sweeps behind the dashboard's charts: many 20-round simulations, one RESULT line each.

Sweeps (5 seeds each, 20 rounds):
  strategy  FedAvg vs FedProx, SecAgg+ on, noise 1.0                                   (10 runs)
  privacy   noise 0 / 0.5 / 1 / 2  x  SecAgg+ off / on, FedAvg                         (40 runs)
  attacks   none / labelflip / boost / flipboost  x  (FedAvg + SecAgg+, FedAvg, median), noise 1.0  (60 runs)
Output: results/<sweep>.jsonl, one RESULT object per line plus "sweep" (CONTRACTS.md section 5).
Resumable: a run whose settings already have a line is skipped, so an interrupted sweep continues where it stopped.

Use:  python scripts/sweep.py [strategy|privacy|attacks|all] [--jobs 2] [--rounds 20] [--seeds 5]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
KEY_FIELDS = ("rounds", "noise", "strategy", "secagg", "agg", "attack", "seed")


def plans(rounds: int, seeds: int) -> dict[str, list[dict]]:
    s = range(seeds)
    base = {"rounds": rounds}
    return {
        "strategy": [{**base, "strategy": st, "secagg": 1, "noise": 1.0, "seed": i} for st, i in itertools.product(("fedavg", "fedprox"), s)],
        "privacy": [{**base, "noise": n, "secagg": sa, "seed": i}
                    for n, sa, i in itertools.product((0.0, 0.5, 1.0, 2.0), (0, 1), s)],
        "attacks": [{**base, "attack": a, "agg": agg, "secagg": sa, "noise": 1.0, "seed": i}
                    for a, (agg, sa), i in itertools.product(("none", "labelflip", "boost", "flipboost"),
                                                             (("fedavg", 1), ("fedavg", 0), ("median", 0)), s)],
    }


def key(d: dict) -> tuple:
    full = {"strategy": "fedavg", "agg": "fedavg", "attack": "none", **d}
    return tuple(json.dumps(full.get(k)) for k in KEY_FIELDS)


def done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.add(key(json.loads(line)))
    return out


_lock = threading.Lock()


def run_one(sweep: str, settings: dict, out: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="ze-sweep-") as tmp:
        args = {**settings, "side-dir": Path(tmp).as_posix()}  # side channel: membership test + attacker stats
        cmd = [sys.executable, "-m", "zero_exposure.sim", *[f"{k}={v}" for k, v in args.items()]]
        t0 = time.time()
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"}, timeout=3600)
    lines = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    if not lines:
        return f"FAILED {settings}: {(p.stdout + p.stderr)[-500:]}"
    row = {**json.loads(lines[-1][7:]), "sweep": sweep, "wall_s": round(time.time() - t0, 1)}
    with _lock, out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return f"ok {sweep} {settings} acc {row['fed_acc']:.3f} honest {row['honest_acc']:.3f} ({row['wall_s']} s)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep", nargs="?", default="all", choices=["strategy", "privacy", "attacks", "all"])
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    todo = []
    for name, runs in plans(a.rounds, a.seeds).items():
        if a.sweep not in ("all", name):
            continue
        out = RESULTS / f"{name}.jsonl"
        have = done_keys(out)
        todo += [(name, r, out) for r in runs if key(r) not in have]
    print(f"{len(todo)} runs to do ({a.jobs} at a time)", flush=True)
    failures = 0
    with ThreadPoolExecutor(a.jobs) as pool:
        for msg in pool.map(lambda t: run_one(*t), todo):
            failures += msg.startswith("FAILED")
            print(msg, flush=True)
    print(f"finished: {len(todo) - failures} ok, {failures} failed", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

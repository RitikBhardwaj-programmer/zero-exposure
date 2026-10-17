"""Run one federated experiment as a Flower simulation (laptop, sweeps): the SAME ServerApp and ClientApp as the deployment.

Use:  python -m zero_exposure.sim rounds=3 noise=1.0 secagg=1 attack=labelflip events-file=runs/x/events.jsonl
Keys: CONTRACTS.md section 1 (unset keys keep the defaults in pyproject.toml). The last stdout line is `RESULT {json}`.
"""
from __future__ import annotations

import json
import os
import sys
import tomllib
from pathlib import Path

from .config import NUM_HOSPITALS, SIM_ENV, RunConfig

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def defaults() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["tool"]["flwr"]["app"]["config"]


def parse_args(argv: list[str]) -> dict:
    out = {}
    for arg in argv:
        key, sep, value = arg.partition("=")
        if not sep:
            raise SystemExit(f"expected key=value, got {arg!r}")
        out[key.strip()] = value.strip().strip('"')
    return out


def main(argv: list[str]) -> int:
    settings = {**defaults(), **parse_args(argv)}
    RunConfig.from_mapping(settings)  # refuse bad settings before starting Ray
    os.environ[SIM_ENV] = json.dumps(settings)  # inherited by the Ray workers that run the hospitals
    os.environ.setdefault("RAY_DEDUP_LOGS", "0")
    from flwr.simulation import run_simulation  # late: imports Ray

    from .client_app import app as client_app
    from .server_app import app as server_app

    backend = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}
    run_simulation(server_app=server_app, client_app=client_app, num_supernodes=NUM_HOSPITALS, backend_config=backend)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

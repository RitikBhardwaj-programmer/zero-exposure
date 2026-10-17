"""Files the dashboard reads: the events file and the inspector folder (formats: CONTRACTS.md sections 3-4).

All writers are no-ops when their path is not set. They create folders first: an exception inside a Flower client mod would
make that hospital drop out, and with SecAgg+ a dropped hospital only shows up as a stalled model.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .config import finite_or_none


def _clean(v):
    if isinstance(v, float):
        return finite_or_none(v)
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, np.generic):
        return _clean(v.item())
    return v


def emit(events_file: str, kind: str, **fields) -> None:
    if not events_file:
        return
    path = Path(events_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_clean({"type": kind, "t": time.time(), **fields})) + "\n")
        f.flush()


def stage(inspect_dir: str, pid: int, rnd: int, name: str) -> None:
    if not inspect_dir:
        return
    d = Path(inspect_dir)
    d.mkdir(parents=True, exist_ok=True)
    with (d / f"stages_h{pid}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"pid": pid, "round": rnd, "stage": name, "t": time.time()}) + "\n")


def save_array(inspect_dir: str, name: str, arr: np.ndarray) -> None:
    if not inspect_dir:
        return
    d = Path(inspect_dir)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / name, arr)


def save_json(inspect_dir: str, name: str, obj: dict) -> None:
    if not inspect_dir:
        return
    d = Path(inspect_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(_clean(obj)), encoding="utf-8")

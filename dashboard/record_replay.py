"""Record a REAL engine run as a dashboard replay (dashboard/replays/<name>/: events.jsonl, insp/, about.json).

Use:  python -m dashboard.record_replay                      (the golden replay: 20 rounds, SecAgg+, label flip, seed 0)
      python -m dashboard.record_replay --name x rounds=5 attack=none
Runs the simulation exactly as the dashboard does (same command builder, events-file / inspect-dir / side-dir in a run folder)
and keeps the result only when the events file ends with `done`.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .server import REPLAY_NAME, ROOT, RunParams, Settings, build_command, child_env, read_events, refusal

GOLDEN = {"rounds": 20, "secagg": True, "attack": "labelflip", "seed": 0}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default="golden")
    ap.add_argument("--out", default=str(ROOT / "dashboard" / "replays"))
    ap.add_argument("settings", nargs="*", help="key=value overrides of the golden settings (POST /api/runs keys)")
    args = ap.parse_args(argv)
    if not REPLAY_NAME.fullmatch(args.name):
        raise SystemExit(f"replay name must match {REPLAY_NAME.pattern}")
    body = dict(GOLDEN)
    for kv in args.settings:
        key, _, value = kv.partition("=")
        try:
            body[key] = json.loads(value)  # numbers and true/false
        except json.JSONDecodeError:
            body[key] = value  # names such as labelflip
    params = RunParams.model_validate(body)
    if (msg := refusal(params)):
        raise SystemExit(msg)

    with tempfile.TemporaryDirectory(prefix="ze-replay-") as tmp:
        folder = Path(tmp)
        (folder / "insp").mkdir()
        (folder / "side").mkdir()
        run_config = {**params.run_config(), "events-file": (folder / "events.jsonl").as_posix(),
                      "inspect-dir": (folder / "insp").as_posix(), "side-dir": (folder / "side").as_posix()}
        cmd = build_command(Settings(mode="sim"), run_config)
        print("running:", " ".join(cmd), flush=True)
        started = time.time()
        proc = subprocess.run(cmd, cwd=ROOT, env=child_env(Settings()), check=False)
        events = read_events(folder / "events.jsonl")
        if not events or events[-1].get("type") != "done":
            print(f"not recorded: the run did not end with a done event (exit code {proc.returncode})", file=sys.stderr)
            return 1
        dest = Path(args.out) / args.name
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        shutil.copy(folder / "events.jsonl", dest / "events.jsonl")
        shutil.copytree(folder / "insp", dest / "insp")
        about = {"recorded": time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime(started)), "mode": "sim",
                 "params": params.model_dump(), "command": "python -m zero_exposure.sim " + " ".join(cmd[3:]).replace(tmp, "<run>"),
                 "seconds": round(time.time() - started, 1)}
        (dest / "about.json").write_text(json.dumps(about, indent=2) + "\n", encoding="utf-8")
        print(f"replay written to {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

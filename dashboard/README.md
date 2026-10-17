# Dashboard

Live view of a Zero Exposure run: start / stop training, watch accuracy and privacy per round, see what the server actually
receives, and compare runs. API: [`CONTRACTS.md`](../CONTRACTS.md) section 6.

```bash
python -m dashboard.server            # http://127.0.0.1:8000  (MODE=sim, the Flower simulation on this machine)
MODE=deploy FLWR_CONNECTION=deploy python -m dashboard.server    # runs go to a SuperLink through `flwr run`
```

| Env | Default | Meaning |
|---|---|---|
| `MODE` | `sim` | `sim`: `python -m zero_exposure.sim ...`; `deploy`: `flwr run <repo> <FLWR_CONNECTION> --stream --run-config "..."` |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | listen address; keep loopback unless you know who can reach it (it starts training runs) |
| `ALLOWED_HOSTS` | | extra `Host` names to accept, comma separated (needed for a non-loopback `HOST`) |
| `RUNS_DIR` | `dashboard/runs` | one folder per run: `events.jsonl`, `insp/`, `side/`, `stdout.log`, `stderr.log` |
| `FLWR_CONNECTION` / `FLWR_HOME` | `deploy` / unset | deploy mode: SuperLink connection name, Flower config home |
| `RESULTS_DIR` | `results` | sweep results read by `/api/sweeps` |
| `REPLAYS_DIR` | `dashboard/replays` | recorded runs |

Relative paths are relative to the repository root.

A run is `done` only when its events file has a `done` event (the exit code is unreliable on s390x); otherwise it is `failed`,
or `stopped` after Stop. Stop and server shutdown kill the whole process tree (and, in deploy mode, call `flwr stop <run id>`).

## Replays

`dashboard/replays/<name>/` holds a recorded real run (`events.jsonl`, `insp/`, `about.json`), streamed with its original
timing. On first load the page plays one (`golden` if present) so every panel has data without training. Record it with the
engine (needs the UCI data in `data/heart_disease/`):

```bash
python -m dashboard.record_replay          # 20 rounds, SecAgg+ on, label-flip attack, seed 0, inspector + side channel
```

## Tests

```bash
python -m pytest tests/test_dashboard.py -q
```

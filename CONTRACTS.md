# Contracts

Fixed interfaces between the parts of Zero Exposure. Parallel work builds against this file. Change it only through a PR that
changes nothing else, and merge that PR before the code that depends on it.

Parts and owners:
| Part | Path | Owner |
|---|---|---|
| Core engine (simulation + deployment apps) | `zero_exposure/` | core |
| Dashboard | `dashboard/` | dashboard |
| Deployment (launcher, Docker, TLS, runbook) | `deploy/`, `scripts/run_local_deploy.py` | deploy |
| Sweeps and results | `scripts/sweep.py`, `results/` | sweeps |

## 1. Run configuration

One set of keys everywhere: Flower `run_config` (`[tool.flwr.app.config]` in `pyproject.toml`), the simulation CLI and the
dashboard. Floats must be written with a decimal point (`noise=1.0`). An empty string means "not set".

| Key | Type | Default | Meaning |
|---|---|---|---|
| `rounds` | int | 20 | federated rounds |
| `local-epochs` | int | 5 | local epochs per round |
| `noise` | float | 1.0 | DP-SGD noise multiplier (0.0 = no noise; epsilon is then reported as null) |
| `clip` | float | 1.0 | DP-SGD per-example gradient clipping norm |
| `delta` | float | 1e-5 | DP delta |
| `strategy` | str | `fedavg` | `fedavg` or `fedprox` |
| `mu` | float | 0.1 | FedProx proximal weight (used only with `fedprox`) |
| `seed` | int | 0 | seeds the data split, the initial model and every DP random stream |
| `secagg` | int | 1 | 1 = Flower SecAgg+ on, 0 = off |
| `agg` | str | `fedavg` | `fedavg` (uses `strategy`), `median` (FedMedian) or `trimmed` (FedTrimmedAvg) |
| `trim-beta` | float | 0.25 | FedTrimmedAvg beta |
| `attack` | str | `none` | `none`, `labelflip`, `boost` or `flipboost` |
| `attacker` | int | 3 | hospital id of the malicious hospital |
| `boost` | float | 10.0 | boost factor for `boost` / `flipboost` |
| `mia` | int | 1 | 1 = run the membership-inference attack after training (needs `side-dir`) |
| `mia-hospital` | int | 0 | hospital whose patients the curious server tests |
| `events-file` | str | `""` | JSONL events file (section 3) |
| `inspect-dir` | str | `""` | inspector folder (section 4) |
| `inspect-round` | int | 1 | round the inspector records |
| `side-dir` | str | `""` | folder shared by all processes for the attack side channel (attacker stats, MIA hospital's last update) |

Refused combinations (the run stops with a clear message before training; the dashboard returns 422 with the same message):
- `agg` != `fedavg` with `secagg` = 1: robust aggregation needs every update in the clear; SecAgg+ hides them.
- `agg` != `fedavg` with `strategy` = `fedprox`.
- `mia` = 1 and `attack` != `none` and `mia-hospital` == `attacker`.
- ids outside 0..3, unknown strings.
Without `side-dir` the run still trains and attacks; `mia` is treated as 0 and the attacker stats are null.

Hospitals (id: name): 0 `cleveland`, 1 `hungarian`, 2 `switzerland`, 3 `va`. Data files
`processed.<name>.data` from the UCI Heart Disease set in `DATA_DIR` (default `data/heart_disease`). A deployment SuperNode holds only
its own file.

SecAgg+ settings (fixed): `num_shares` 4, `reconstruction_threshold` 3, `max_weight` 300, `clipping_range` 8.0,
`quantization_range` 4194304. The model is the MLP 13 -> 16 -> 8 -> 1 (**P = 369 parameters**).

## 2. Entry points

- Simulation: `python -m zero_exposure.sim key=value ...` (keys of section 1). Uses Flower's simulation engine (Ray).
- Deployment: `flwr run . <connection> --stream --run-config "key=value ..."` against a SuperLink with 4 SuperNodes started with
  `--node-config "partition-id=<id> num-partitions=4"`. Native launcher: `python scripts/run_local_deploy.py key=value ...`.
- Both print, as the ServerApp's last stdout line, `RESULT {json}` (section 5). Both write the same events and inspector files.
- Success = the events file ends with a `done` event. The process exit code is not reliable on s390x (exit-time segfault).

## 3. Events file (JSONL, one object per line, appended by the ServerApp)

Every event has `type` and `t` (Unix time, float). Numbers that are infinite or NaN are written as `null`.

```json
{"type": "baseline", "t": 0.0, "hospitals": ["cleveland", "hungarian", "switzerland", "va"],
 "n_train": [227, 220, 92, 150], "n_test": [76, 74, 31, 50],
 "central_acc": 0.0, "central_f1": 0.0, "local_per_hospital_acc": [0.0, 0.0, 0.0, 0.0], "config": {"rounds": 20, "...": "all section 1 keys"}}
{"type": "round", "t": 0.0, "round": 0, "total_rounds": 20, "acc": 0.0, "f1": 0.0, "loss": 0.0,
 "per_hospital_acc": [0.0, 0.0, 0.0, 0.0], "epsilon": [null, null, null, null], "max_abs_weight": 0.0}
{"type": "attack", "t": 0.0, "attack": "none", "attacker": 3, "attacker_name": "va", "boost": null, "agg": "fedavg", "secagg": 1,
 "honest_acc": 0.0, "attacker_acc": 0.0, "mia_hospital": 0, "mia_hospital_name": "cleveland", "mia_n": null,
 "mia_auc_update": null, "mia_adv_update": null, "mia_auc_global": null, "mia_adv_global": null,
 "clipped_rounds": null, "max_scaled_attacker": null}
{"type": "done", "t": 0.0}
{"type": "error", "t": 0.0, "message": "text"}
```
- `round` events: round 0 is the untrained initial model, then 1..`rounds`. `epsilon[i]` = hospital i's epsilon after this round
  (0 at round 0; null when `noise` = 0). Accuracy/F1/loss are on the pooled test rows of all 4 hospitals.
- `attack` is written once after the last round, also when `attack` = `none` (then `honest_acc` = mean of hospitals != `attacker`).
- `error` replaces `done` when the ServerApp fails after it started writing.

## 4. Inspector folder (written only when `inspect-dir` is set; round = `inspect-round`)

| File | Writer | Content |
|---|---|---|
| `stages_h<id>.jsonl` | each hospital | lines `{"pid", "round", "stage", "t"}`, stage in `received`, `trained`, `masked` (only with SecAgg+), `sent` |
| `plain.npy` | hospital 0 | float32 (P,): its trained parameters (what it would send without SecAgg+) |
| `masked.npy` | hospital 0 | uint32 (P+1,): what actually left the hospital with SecAgg+ (slot 0 = example count, then the P parameters); absent when `secagg` = 0 |
| `inspect.json` | hospital 0 | `{"round", "pid", "n_params": 369, "secagg", "n_examples", "clipping_range", "quantization_range"}` |
| `aggregate.npy` | server | float32 (P,): the new global model after that round |

Arrays may be big-endian when written on s390x; readers must use the dtype stored in the `.npy` header.

## 5. RESULT line

`RESULT ` + one JSON object: every section 1 key with `-` replaced by `_` (except the file/folder paths), plus
`fed_acc`, `fed_f1`, `fed_per_hospital_acc`, `epsilon_per_hospital`, `epsilon_max`, `max_abs_weight`, `sec_per_round`,
`central_acc`, `central_f1`, `local_per_hospital_acc`, and every field of the `attack` event (without `type`, `t`).
Sweep result files (`results/<sweep>.jsonl`) hold one RESULT object per line plus `"sweep": "<name>"`.

## 6. Dashboard HTTP API (served by `python -m dashboard.server`; env `MODE` sim|deploy, `HOST`, `PORT`, `RUNS_DIR`)

| Method, path | Answer |
|---|---|
| `GET /api/config` | `{"mode", "hospitals", "limits": {"rounds_max": 30, "noise_max": 5.0, "seed_max": 99}}` |
| `POST /api/runs` | body `{"noise", "strategy", "secagg": bool, "rounds", "seed", "attack", "attacker", "agg"}` -> 200 run; 409 a run is active; 422 invalid or refused (detail = message) |
| `GET /api/runs/current` | the latest run or `null` |
| `POST /api/runs/current/stop` | stops the active run and its whole process tree -> run |
| `GET /api/runs/{id}/events` | server-sent events: each existing line, then new ones as they are appended; ends after `done` / `error` / stop |
| `GET /api/runs/{id}/inspect` | `{"meta", "plain", "masked", "aggregate", "stages", "chi2": {"stat", "df", "critical_99"}}` (nulls while missing) |
| `GET /api/sweeps` | data for the privacy-vs-accuracy and attack charts, read from `results/*.jsonl` |
| `GET /api/replays` and `GET /api/replays/{name}/events` | recorded runs in `dashboard/replays/<name>/` streamed with their original timing (no training) |

Run object: `{"id": "YYYYMMDD-HHMMSS-xxxxxx", "params", "status": "running|done|failed|stopped", "exit_code", "started", "finished"}`.
A run's folder is `RUNS_DIR/<id>/` with `events.jsonl`, `insp/`, `side/`, `stdout.log`, `stderr.log`.
The dashboard passes `events-file`, `inspect-dir` and `side-dir` pointing into that folder.

## 7. Fixture

`fixtures/sample_run/` holds an events file and inspector files that follow sections 3-4 (made by `scripts/make_fixture.py`,
random numbers, not a real run). The dashboard is built and tested against it before the core engine exists.

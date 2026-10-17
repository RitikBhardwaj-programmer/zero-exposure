"""Zero Exposure dashboard backend: starts one training run at a time and streams what it writes (CONTRACTS.md section 6).

Run:  python -m dashboard.server        then open http://127.0.0.1:8000
Env:  MODE sim|deploy (default sim), HOST (127.0.0.1), PORT (8000), RUNS_DIR (dashboard/runs), FLWR_CONNECTION (deploy),
      FLWR_HOME (passed to `flwr`), ALLOWED_HOSTS (extra Host names, comma separated, for a non-loopback HOST),
      RESULTS_DIR (results), REPLAYS_DIR (dashboard/replays). Relative paths are relative to the repository root.
      DASHBOARD_TRAIN_CMD (tests only): JSON list that replaces the training command; the key=value arguments are appended.

The run itself is a child process (simulation CLI or `flwr run`). The dashboard never trusts its exit code (s390x can
segfault at exit): a run is `done` only when its events file has a `done` event.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from zero_exposure.config import HOSPITALS, ConfigError, RunConfig

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"

LIMITS = {"rounds_max": 30, "noise_max": 5.0, "seed_max": 99}
RUN_ID = re.compile(r"\d{8}-\d{6}-[0-9a-f]{6}")
REPLAY_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
FLOWER_RUN_ID = re.compile(r"started run\D{0,20}(\d+)", re.IGNORECASE)
LOOPBACK = {"127.0.0.1", "localhost", "::1"}

P = 369  # model parameters (CONTRACTS.md section 1)
CHI2_BINS = 32
CHI2_DF = CHI2_BINS - 1
EXIT_GRACE = 90.0  # seconds a child may linger after writing done/error before the dashboard kills it
SSE_POLL = 0.25
SSE_PING = 15.0
MAX_BODY = 4096


# ---------------------------------------------------------------------------------------------------------- settings

def _path(env: str, default: str) -> Path:
    p = Path(os.environ.get(env) or default)
    return p if p.is_absolute() else ROOT / p


@dataclass
class Settings:
    mode: str = "sim"
    host: str = "127.0.0.1"
    port: int = 8000
    runs_dir: Path = ROOT / "dashboard" / "runs"
    results_dir: Path = ROOT / "results"
    replays_dir: Path = ROOT / "dashboard" / "replays"
    connection: str = "deploy"
    flwr_home: str = ""
    train_cmd: list[str] | None = None
    allowed_hosts: list[str] = field(default_factory=lambda: sorted(LOOPBACK))

    @classmethod
    def from_env(cls) -> "Settings":
        mode = os.environ.get("MODE", "sim")
        if mode not in ("sim", "deploy"):
            raise SystemExit(f"MODE must be sim or deploy, got {mode!r}")
        train_cmd = None
        if os.environ.get("DASHBOARD_TRAIN_CMD"):
            train_cmd = json.loads(os.environ["DASHBOARD_TRAIN_CMD"])
            if not (isinstance(train_cmd, list) and train_cmd and all(isinstance(a, str) for a in train_cmd)):
                raise SystemExit("DASHBOARD_TRAIN_CMD must be a JSON list of strings")
        host = os.environ.get("HOST", "127.0.0.1")
        extra = [h.strip() for h in os.environ.get("ALLOWED_HOSTS", "").split(",") if h.strip()]
        return cls(mode=mode, host=host, port=int(os.environ.get("PORT", "8000")),
                   runs_dir=_path("RUNS_DIR", "dashboard/runs"), results_dir=_path("RESULTS_DIR", "results"),
                   replays_dir=_path("REPLAYS_DIR", "dashboard/replays"),
                   connection=os.environ.get("FLWR_CONNECTION", "deploy"), flwr_home=os.environ.get("FLWR_HOME", ""),
                   train_cmd=train_cmd, allowed_hosts=sorted(LOOPBACK | {host} | set(extra)))


# ------------------------------------------------------------------------------------------------------- run request

class RunParams(BaseModel):
    """Body of POST /api/runs. Ranges come from /api/config; hospital 0 is the membership-test target, so it is never the attacker."""
    model_config = ConfigDict(extra="forbid", strict=True)

    noise: float = Field(1.0, ge=0.0, le=LIMITS["noise_max"], allow_inf_nan=False)
    strategy: Literal["fedavg", "fedprox"] = "fedavg"
    secagg: bool = True
    rounds: int = Field(20, ge=1, le=LIMITS["rounds_max"])
    seed: int = Field(0, ge=0, le=LIMITS["seed_max"])
    attack: Literal["none", "labelflip", "boost", "flipboost"] = "none"
    attacker: int = Field(3, ge=1, le=3)
    agg: Literal["fedavg", "median", "trimmed"] = "fedavg"

    def run_config(self) -> dict:
        """The section 1 keys this request sets (everything else keeps the engine's defaults)."""
        return {"rounds": self.rounds, "noise": float(self.noise), "strategy": self.strategy, "seed": self.seed,
                "secagg": int(self.secagg), "agg": self.agg, "attack": self.attack, "attacker": self.attacker}


def refusal(params: RunParams) -> str | None:
    """The engine's own message for a combination CONTRACTS.md section 1 refuses, or None."""
    rc = params.run_config()
    try:
        RunConfig(rounds=rc["rounds"], noise=rc["noise"], strategy=rc["strategy"], seed=rc["seed"], secagg=rc["secagg"],
                  agg=rc["agg"], attack=rc["attack"], attacker=rc["attacker"]).validate()
    except ConfigError as exc:
        return str(exc)
    return None


def validation_message(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        where = ".".join(str(x) for x in err.get("loc", ())) or "body"
        parts.append(f"{where}: {err.get('msg', 'invalid')}")
    return "; ".join(parts)


# ----------------------------------------------------------------------------------------------------------- command

def float_text(x: float) -> str:
    """A float that always reads as a float (CONTRACTS.md section 1: `noise=1.0`, never `noise=1`)."""
    s = repr(float(x))
    if "." in s or "inf" in s or "nan" in s:
        return s
    if "e" in s:
        mant, exp = s.split("e")
        return f"{mant}.0e{exp}"
    return s + ".0"


def _arg_value(v, toml: bool) -> str:
    if isinstance(v, bool):
        raise TypeError("run_config values are int, float or str")
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return float_text(v)
    return json.dumps(str(v)) if toml else str(v)


def build_command(settings: Settings, run_config: dict) -> list[str]:
    """The one child process of a run: simulation CLI, `flwr run` against a SuperLink, or the test stand-in."""
    if settings.train_cmd:
        return [*settings.train_cmd, *(f"{k}={_arg_value(v, False)}" for k, v in run_config.items())]
    if settings.mode == "sim":
        return [sys.executable, "-m", "zero_exposure.sim", *(f"{k}={_arg_value(v, False)}" for k, v in run_config.items())]
    overrides = " ".join(f"{k}={_arg_value(v, True)}" for k, v in run_config.items())
    return ["flwr", "run", str(ROOT), settings.connection, "--stream", "--run-config", overrides]


def child_env(settings: Settings) -> dict:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    if settings.flwr_home:
        env["FLWR_HOME"] = settings.flwr_home
    return env


# ------------------------------------------------------------------------------------------------------- process tree

def _descendants(pid: int) -> list:
    try:
        import psutil
        return psutil.Process(pid).children(recursive=True)
    except Exception:
        return []


def kill_tree(proc: subprocess.Popen) -> None:
    """Kill the child and everything it started (Ray / Flower spawn many processes)."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, check=False)
        return
    stragglers = _descendants(proc.pid)  # before the kill: children that left our process group are found by parentage
    try:
        os.killpg(proc.pid, signal.SIGKILL)  # start_new_session=True made the child the leader of its own group
    except (ProcessLookupError, PermissionError):
        pass
    for p in stragglers:
        try:
            p.kill()
        except Exception:
            pass


# ------------------------------------------------------------------------------------------------------- event files

def read_events(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue  # a line the writer has not finished yet
        if isinstance(obj, dict):
            out.append(obj)
    return out


def terminal_event(path: Path) -> str | None:
    """`error` if the run wrote one, else `done` if it wrote one, else None."""
    kinds = {e.get("type") for e in read_events(path)}
    return "error" if "error" in kinds else "done" if "done" in kinds else None


# --------------------------------------------------------------------------------------------------------------- runs

@dataclass
class Run:
    id: str
    params: dict
    folder: Path
    started: float
    status: str = "running"
    exit_code: int | None = None
    finished: float | None = None
    proc: subprocess.Popen | None = None
    stop_requested: bool = False
    flower_run_id: str | None = None
    watcher: threading.Thread | None = None

    @property
    def events_file(self) -> Path:
        return self.folder / "events.jsonl"

    def public(self) -> dict:
        return {"id": self.id, "params": self.params, "status": self.status, "exit_code": self.exit_code,
                "started": self.started, "finished": self.finished}


class Busy(Exception):
    pass


class RunManager:
    """At most one active run. All state changes happen under one lock."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.lock = threading.Lock()
        self.runs: dict[str, Run] = {}
        self.latest: Run | None = None

    def active(self) -> Run | None:
        run = self.latest
        return run if run is not None and run.status == "running" else None

    def start(self, params: RunParams) -> Run:
        s = self.settings
        with self.lock:
            if self.active() is not None:
                raise Busy("a run is already active: stop it first")
            run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
            folder = s.runs_dir / run_id
            (folder / "insp").mkdir(parents=True)
            (folder / "side").mkdir()
            run_config = {**params.run_config(), "events-file": (folder / "events.jsonl").as_posix(),
                          "inspect-dir": (folder / "insp").as_posix(), "side-dir": (folder / "side").as_posix()}
            cmd = build_command(s, run_config)
            run = Run(id=run_id, params=params.model_dump(), folder=folder, started=time.time())
            out = (folder / "stdout.log").open("wb")
            err = (folder / "stderr.log").open("wb")
            kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
            try:
                run.proc = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                            env=child_env(s), **kw)
            except OSError as exc:
                run.status, run.finished = "failed", time.time()
                err.write(f"could not start {cmd[0]!r}: {exc}\n".encode())
                raise RuntimeError(f"could not start the training process: {exc}") from exc
            finally:
                out.close()
                err.close()
                self.runs[run_id] = run
                self.latest = run
            run.watcher = threading.Thread(target=self._watch, args=(run,), name=f"run-{run_id}", daemon=True)
            run.watcher.start()
            return run

    def _flower_run_id(self, run: Run) -> None:
        if self.settings.mode != "deploy" or run.flower_run_id:
            return
        try:
            text = (run.folder / "stdout.log").read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        m = FLOWER_RUN_ID.search(text)
        if m:
            run.flower_run_id = m.group(1)

    def _watch(self, run: Run) -> None:
        proc = run.proc
        deadline = None
        while proc.poll() is None:
            self._flower_run_id(run)
            if terminal_event(run.events_file) is not None:
                deadline = deadline or time.monotonic() + EXIT_GRACE
                if time.monotonic() > deadline:  # finished its work but does not exit (e.g. a hung Ray shutdown)
                    kill_tree(proc)
                    break
            time.sleep(0.5)
        code = proc.wait()
        self._flower_run_id(run)
        if os.name != "nt":
            kill_tree(proc)  # leftovers of the group (Ray workers) must not outlive the run
        end = terminal_event(run.events_file)
        with self.lock:
            run.exit_code = code
            run.finished = time.time()
            run.status = "stopped" if run.stop_requested else "done" if end == "done" else "failed"

    def stop(self) -> Run:
        with self.lock:
            run = self.active()
            if run is None:
                raise LookupError("no active run")
            run.stop_requested = True
        self._flower_run_id(run)
        if self.settings.mode == "deploy" and run.flower_run_id and not self.settings.train_cmd:
            try:
                subprocess.run(["flwr", "stop", run.flower_run_id, self.settings.connection], cwd=ROOT,
                               capture_output=True, timeout=30, env=child_env(self.settings), check=False)
            except (OSError, subprocess.TimeoutExpired):
                pass  # the process tree is killed below either way
        kill_tree(run.proc)
        if run.watcher is not None:
            run.watcher.join(timeout=15)
        return run

    def shutdown(self) -> None:
        run = self.active()
        if run is not None:
            try:
                self.stop()
            except LookupError:
                pass


# ----------------------------------------------------------------------------------------------------------- inspect

def _load_npy(path: Path):
    if not path.is_file():
        return None
    try:
        arr = np.load(path, allow_pickle=False)  # keeps the dtype in the header, big-endian included
    except (OSError, ValueError):
        return None  # still being written
    return arr.astype(arr.dtype.newbyteorder("=")).ravel()


def chi2_uniform(masked) -> dict:
    """Chi-square test that the masked parameters are uniform over the uint32 range (32 equal bins, df 31).

    Slot 0 of the masked vector is the example count, so only the P parameters are tested. Below the 99% critical value,
    the vector is statistically indistinguishable from random noise.
    """
    from scipy.stats import chi2
    critical = float(chi2.ppf(0.99, CHI2_DF))
    if masked is None or masked.size < 2:
        return {"stat": None, "df": CHI2_DF, "critical_99": critical}
    params = masked[1:].astype(np.uint64)
    counts = np.bincount((params * CHI2_BINS) >> np.uint64(32), minlength=CHI2_BINS)[:CHI2_BINS].astype(float)
    expected = params.size / CHI2_BINS
    return {"stat": float(((counts - expected) ** 2 / expected).sum()), "df": CHI2_DF, "critical_99": critical}


def _finite_list(arr):
    if arr is None:
        return None
    if arr.dtype.kind == "f":
        return [float(x) if math.isfinite(x) else None for x in arr.tolist()]
    return arr.tolist()


def read_inspect(insp: Path) -> dict:
    meta = None
    if (insp / "inspect.json").is_file():
        try:
            meta = json.loads((insp / "inspect.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = None
    masked = _load_npy(insp / "masked.npy")
    stages = []
    for pid in range(len(HOSPITALS)):
        rows = read_events(insp / f"stages_h{pid}.jsonl")
        stages.append([{k: r.get(k) for k in ("pid", "round", "stage", "t")} for r in rows])
    return {"meta": meta, "plain": _finite_list(_load_npy(insp / "plain.npy")), "masked": _finite_list(masked),
            "aggregate": _finite_list(_load_npy(insp / "aggregate.npy")), "stages": stages, "chi2": chi2_uniform(masked)}


# ------------------------------------------------------------------------------------------------------------ sweeps

GROUP_KEYS = ("noise", "secagg", "strategy", "attack", "agg")
METRICS = ("fed_acc", "fed_f1", "epsilon_max", "honest_acc", "attacker_acc", "mia_auc_update", "mia_adv_update",
           "mia_auc_global", "mia_adv_global", "central_acc", "sec_per_round")


def _mean_std(values: list) -> dict:
    vals = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return {"mean": float(np.mean(vals)), "std": std, "n": len(vals)}


def read_sweeps(results_dir: Path) -> dict:
    """Mean and standard deviation over seeds of every RESULT row, grouped by sweep + noise / secagg / strategy / attack / agg."""
    files, rows, skipped = [], [], 0
    if results_dir.is_dir():
        for path in sorted(results_dir.glob("*.jsonl")):
            if not path.is_file():
                continue
            files.append(path.name)
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                if isinstance(row, dict) and all(k in row for k in GROUP_KEYS):
                    rows.append({**row, "sweep": str(row.get("sweep") or path.stem)})
                else:
                    skipped += 1
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["sweep"], *(row[k] for k in GROUP_KEYS)), []).append(row)
    out = []
    for key, members in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        g = {"sweep": key[0], **dict(zip(GROUP_KEYS, key[1:])), "n": len(members),
             "seeds": sorted({m.get("seed") for m in members if isinstance(m.get("seed"), int)}),
             "rounds": sorted({m.get("rounds") for m in members if isinstance(m.get("rounds"), int)})}
        g.update({m: _mean_std([r.get(m) for r in members]) for m in METRICS})
        out.append(g)
    return {"files": files, "n_rows": len(rows), "skipped": skipped, "groups": out}


# ----------------------------------------------------------------------------------------------------------- replays

def replay_dir(settings: Settings, name: str) -> Path:
    if not REPLAY_NAME.fullmatch(name):
        raise HTTPException(404, "unknown replay")
    d = settings.replays_dir / name
    if not (d / "events.jsonl").is_file():
        raise HTTPException(404, "unknown replay")
    return d


def list_replays(settings: Settings) -> list[dict]:
    out = []
    if not settings.replays_dir.is_dir():
        return out
    for d in sorted(settings.replays_dir.iterdir()):
        if not (d.is_dir() and REPLAY_NAME.fullmatch(d.name) and (d / "events.jsonl").is_file()):
            continue
        ev = read_events(d / "events.jsonl")
        base = next((e for e in ev if e.get("type") == "baseline"), {})
        ts = [e["t"] for e in ev if isinstance(e.get("t"), (int, float))]
        about = None
        if (d / "about.json").is_file():
            try:
                about = json.loads((d / "about.json").read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                about = None
        out.append({"name": d.name, "n_events": len(ev), "duration": (max(ts) - min(ts)) if ts else 0.0,
                    "config": base.get("config"), "complete": any(e.get("type") == "done" for e in ev),
                    "has_inspect": (d / "insp").is_dir(), "about": about})
    return out


# --------------------------------------------------------------------------------------------------------------- SSE

def _sse(data: str, event: str | None = None) -> str:
    head = f"event: {event}\n" if event else ""
    return head + "".join(f"data: {line}\n" for line in data.splitlines() or [""]) + "\n"


async def tail_file(path: Path, is_live, final=None):
    """Each complete line of a growing JSONL file as one SSE message; ends after done / error, or when the writer stopped."""
    pos, buf, last_ping = 0, b"", time.monotonic()
    while True:
        live = is_live()  # read AFTER this check, so the last lines of a finished run are never missed
        chunk = b""
        if path.is_file():
            with path.open("rb") as f:
                f.seek(pos)
                chunk = f.read()
            pos += len(chunk)
        buf += chunk
        *lines, buf = buf.split(b"\n")
        for raw in lines:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            yield _sse(line)
            try:
                kind = json.loads(line).get("type")
            except (json.JSONDecodeError, AttributeError):
                kind = None
            if kind in ("done", "error"):
                if final:
                    yield _sse(json.dumps(final()), "end")
                return
        if not chunk and not live:
            if final:
                yield _sse(json.dumps(final()), "end")
            return
        if time.monotonic() - last_ping > SSE_PING:
            last_ping = time.monotonic()
            yield ": ping\n\n"
        if not chunk:
            await asyncio.sleep(SSE_POLL)


async def replay_stream(path: Path, speed: float):
    events = [l.strip() for l in path.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    prev_t = None
    for line in events:
        try:
            t = json.loads(line).get("t")
        except (json.JSONDecodeError, AttributeError):
            continue
        if isinstance(t, (int, float)) and prev_t is not None and t > prev_t:
            await asyncio.sleep((t - prev_t) / speed)
        if isinstance(t, (int, float)):
            prev_t = t
        yield _sse(line)
    yield _sse(json.dumps({"replay": path.parent.name}), "end")


SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


# --------------------------------------------------------------------------------------------------------------- app

CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; "
       "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    manager = RunManager(settings)

    @asynccontextmanager
    async def lifespan(_app):
        yield
        manager.shutdown()  # a server that goes away takes its training process with it

    app = FastAPI(title="Zero Exposure dashboard", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.settings = settings
    app.state.manager = manager
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        # A web page on another site must not be able to start or stop runs (the browser sends its Origin).
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if origin and urlsplit(origin).hostname not in settings.allowed_hosts:
                return JSONResponse({"detail": "cross-origin request refused"}, status_code=403)
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    def get_run(run_id: str) -> Run | None:
        if not RUN_ID.fullmatch(run_id):
            raise HTTPException(404, "unknown run")
        run = manager.runs.get(run_id)
        if run is None and not (settings.runs_dir / run_id).is_dir():
            raise HTTPException(404, "unknown run")
        return run

    @app.get("/api/config")
    def config():
        return {"mode": settings.mode, "hospitals": list(HOSPITALS), "limits": LIMITS}

    @app.post("/api/runs")
    async def start_run(request: Request):
        body = await request.body()
        if len(body) > MAX_BODY:
            raise HTTPException(413, "request body too large")
        try:
            params = RunParams.model_validate_json(body or b"{}")
        except ValidationError as exc:
            raise HTTPException(422, validation_message(exc)) from None
        msg = refusal(params)
        if msg:
            raise HTTPException(422, msg)
        try:
            run = await asyncio.to_thread(manager.start, params)
        except Busy as exc:
            raise HTTPException(409, str(exc)) from None
        except RuntimeError as exc:
            raise HTTPException(500, str(exc)) from None
        return run.public()

    @app.get("/api/runs/current")
    def current():
        return manager.latest.public() if manager.latest else None

    @app.post("/api/runs/current/stop")
    def stop():
        try:
            return manager.stop().public()
        except LookupError as exc:
            raise HTTPException(409, str(exc)) from None

    @app.get("/api/runs/{run_id}/events")
    def run_events(run_id: str):
        run = get_run(run_id)
        path = settings.runs_dir / run_id / "events.jsonl"
        is_live = (lambda: run.status == "running") if run else (lambda: False)
        final = (lambda: run.public()) if run else None
        return StreamingResponse(tail_file(path, is_live, final), media_type="text/event-stream", headers=SSE_HEADERS)

    @app.get("/api/runs/{run_id}/inspect")
    def run_inspect(run_id: str):
        get_run(run_id)
        return read_inspect(settings.runs_dir / run_id / "insp")

    @app.get("/api/sweeps")
    def sweeps():
        return read_sweeps(settings.results_dir)

    @app.get("/api/replays")
    def replays():
        return list_replays(settings)

    @app.get("/api/replays/{name}/events")
    def replay_events(name: str, speed: float = Query(1.0, gt=0.0, le=1000.0)):
        d = replay_dir(settings, name)
        return StreamingResponse(replay_stream(d / "events.jsonl", speed), media_type="text/event-stream",
                                 headers=SSE_HEADERS)

    @app.get("/api/replays/{name}/inspect")
    def replay_inspect(name: str):
        return read_inspect(replay_dir(settings, name) / "insp")

    @app.get("/api/{rest:path}", include_in_schema=False)
    def api_404(rest: str):
        raise HTTPException(404, "not found")

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app


def main() -> None:
    import uvicorn
    settings = Settings.from_env()
    if settings.host not in LOOPBACK:
        print(f"WARNING: HOST={settings.host} exposes the dashboard (it can start training runs) beyond this machine; "
              f"allowed Host headers: {settings.allowed_hosts}", file=sys.stderr)
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    print(f"Zero Exposure dashboard ({settings.mode} mode): http://{settings.host}:{settings.port}", flush=True)
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()

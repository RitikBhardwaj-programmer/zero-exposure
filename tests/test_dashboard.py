"""Dashboard API checks (CONTRACTS.md section 6). The training child is a stand-in that replays fixtures/sample_run
(DASHBOARD_TRAIN_CMD), except test_real_short_simulation, which runs the real engine for 2 rounds (needs the UCI data).
Run: python -m pytest tests/test_dashboard.py -q
"""
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from dashboard import server
from dashboard.server import RunParams, Settings, build_command, chi2_uniform, create_app, float_text

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "fixtures" / "sample_run"
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data" / "heart_disease"))
HOSPITAL_FILES = [DATA_DIR / f"processed.{n}.data" for n in ("cleveland", "hungarian", "switzerland", "va")]

# Stand-in for the training process: same arguments, same files, fixture numbers. FAKE_MODE: ok | nodone | error | hang.
FAKE_CHILD = textwrap.dedent('''
    import json, os, shutil, subprocess, sys, time
    from pathlib import Path
    fixture = Path(sys.argv[1])
    args = dict(a.split("=", 1) for a in sys.argv[2:])
    print("ARGS " + json.dumps(args), flush=True)
    mode = os.environ.get("FAKE_MODE", "ok")
    side = Path(args["side-dir"])
    grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    (side / "grandchild.pid").write_text(str(grandchild.pid))
    events = [json.loads(l) for l in (fixture / "events.jsonl").read_text().splitlines() if l.strip()]
    out = Path(args["events-file"])
    insp = Path(args["inspect-dir"])
    for ev in events:
        if mode == "hang" and ev["type"] != "baseline":
            time.sleep(300)
        if ev["type"] == "done":
            if mode == "nodone":
                break
            if mode == "error":
                ev = {"type": "error", "message": "boom"}
        if ev["type"] == "round" and ev["round"] == 1:
            for f in (fixture / "insp").iterdir():
                shutil.copy(f, insp / f.name)
        ev["t"] = time.time()
        with out.open("a") as f:
            f.write(json.dumps(ev) + "\\n")
        time.sleep(0.3)
    print("RESULT {}", flush=True)
''')


def alive(pid: int) -> bool:
    if os.name == "nt":  # os.kill(pid, 0) would TERMINATE the process on Windows; ask tasklist instead
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"], capture_output=True, text=True).stdout
        return f'"{pid}"' in out
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # a zombie is dead for our purposes
        return Path(f"/proc/{pid}/stat").read_text().split(")")[1].split()[0] != "Z"
    except OSError:
        return True


def wait_for(pred, timeout=20.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.1)
    return False


@pytest.fixture
def env(tmp_path):
    fake = tmp_path / "fake_child.py"
    fake.write_text(FAKE_CHILD, encoding="utf-8")
    settings = Settings(mode="sim", runs_dir=tmp_path / "runs", results_dir=tmp_path / "results",
                        replays_dir=tmp_path / "replays", train_cmd=[sys.executable, str(fake), str(FIXTURE)],
                        allowed_hosts=["127.0.0.1", "localhost", "testserver"])
    return settings


@pytest.fixture
def client(env):
    with TestClient(create_app(env)) as c:
        yield c


BODY = {"noise": 1.0, "strategy": "fedavg", "secagg": True, "rounds": 3, "seed": 0, "attack": "labelflip", "attacker": 3,
        "agg": "fedavg"}


def sse_messages(client, url, timeout=30):
    """(event name, data) pairs of a server-sent event stream, read until the server ends it."""
    out, event, data = [], None, []
    with client.stream("GET", url, timeout=timeout) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if line == "":
                if data:
                    out.append((event or "message", "\n".join(data)))
                event, data = None, []
            elif line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data.append(line[6:])
    return out


# ---------------------------------------------------------------------------------------------------------- contract

def test_config(client):
    r = client.get("/api/config")
    assert r.status_code == 200
    assert r.json() == {"mode": "sim", "hospitals": ["cleveland", "hungarian", "switzerland", "va"],
                        "limits": {"rounds_max": 30, "noise_max": 5.0, "seed_max": 99}}


def test_no_run_yet(client):
    assert client.get("/api/runs/current").json() is None
    assert client.post("/api/runs/current/stop").status_code == 409


@pytest.mark.parametrize("patch, needle", [
    ({"extra": 1}, "extra"),
    ({"rounds": 31}, "rounds"),
    ({"rounds": 0}, "rounds"),
    ({"rounds": 3.0}, "rounds"),
    ({"rounds": "3"}, "rounds"),
    ({"noise": 5.5}, "noise"),
    ({"noise": -0.1}, "noise"),
    ({"noise": "1.0"}, "noise"),
    ({"seed": 100}, "seed"),
    ({"secagg": 1}, "secagg"),
    ({"attacker": 0}, "attacker"),
    ({"attacker": 4}, "attacker"),
    ({"attack": "evil"}, "attack"),
    ({"strategy": "fedsgd"}, "strategy"),
    ({"agg": "mean"}, "agg"),
])
def test_invalid_bodies_are_422_with_a_message(client, patch, needle):
    r = client.post("/api/runs", json={**BODY, **patch})
    assert r.status_code == 422
    assert isinstance(r.json()["detail"], str) and needle in r.json()["detail"]
    assert client.get("/api/runs/current").json() is None


def test_malformed_json_is_422(client):
    r = client.post("/api/runs", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 422


@pytest.mark.parametrize("patch", [
    {"agg": "median", "secagg": True},
    {"agg": "trimmed", "secagg": True},
    {"agg": "median", "secagg": False, "strategy": "fedprox"},
])
def test_refused_combinations_use_the_engine_message(client, patch):
    from zero_exposure.config import ConfigError, RunConfig
    body = {**BODY, **patch}
    r = client.post("/api/runs", json=body)
    assert r.status_code == 422
    with pytest.raises(ConfigError) as exc:
        RunConfig(noise=body["noise"], strategy=body["strategy"], secagg=int(body["secagg"]), agg=body["agg"],
                  attack=body["attack"], attacker=body["attacker"]).validate()
    assert r.json()["detail"] == str(exc.value)


def test_int_noise_is_accepted_as_float():
    assert RunParams.model_validate_json(b'{"noise": 2}').noise == 2.0


# ---------------------------------------------------------------------------------------------------------- commands

def test_command_builder_sim_and_deploy(env):
    rc = {**RunParams(noise=2, rounds=3).run_config(), "events-file": "/tmp/a b/events.jsonl"}
    sim = build_command(Settings(mode="sim"), rc)
    assert sim[:3] == [sys.executable, "-m", "zero_exposure.sim"]
    assert "noise=2.0" in sim and "rounds=3" in sim and "secagg=1" in sim and "strategy=fedavg" in sim
    assert "events-file=/tmp/a b/events.jsonl" in sim
    dep = build_command(Settings(mode="deploy", connection="superlink-x"), rc)
    assert dep[:6] == ["flwr", "run", str(ROOT), "superlink-x", "--stream", "--run-config"]
    assert len(dep) == 7
    assert dep[6].split(" ")[:3] == ["rounds=3", "noise=2.0", 'strategy="fedavg"']
    assert 'events-file="/tmp/a b/events.jsonl"' in dep[6]
    assert float_text(1e-5) == "1.0e-05" and float_text(0.0) == "0.0"
    fake = build_command(env, {"rounds": 3, "noise": 1.0})
    assert fake[:2] == env.train_cmd[:2] and fake[-2:] == ["rounds=3", "noise=1.0"]


# ---------------------------------------------------------------------------------------------------------- full run

def test_full_run_with_the_fake_child(client, env):
    r = client.post("/api/runs", json=BODY)
    assert r.status_code == 200, r.text
    run = r.json()
    assert set(run) == {"id", "params", "status", "exit_code", "started", "finished"}
    assert server.RUN_ID.fullmatch(run["id"]) and run["status"] == "running" and run["params"] == BODY
    assert client.post("/api/runs", json=BODY).status_code == 409, "one run at a time"

    msgs = sse_messages(client, f"/api/runs/{run['id']}/events")
    events = [json.loads(d) for e, d in msgs if e == "message"]
    expected = [json.loads(l)["type"] for l in (FIXTURE / "events.jsonl").read_text().splitlines()]
    assert [e["type"] for e in events] == expected
    assert [e["round"] for e in events if e["type"] == "round"] == [0, 1, 2, 3]

    assert wait_for(lambda: client.get("/api/runs/current").json()["status"] != "running")
    cur = client.get("/api/runs/current").json()
    assert cur["status"] == "done" and cur["id"] == run["id"] and cur["finished"] >= cur["started"]

    folder = env.runs_dir / run["id"]
    assert {p.name for p in folder.iterdir()} >= {"events.jsonl", "insp", "side", "stdout.log", "stderr.log"}
    args = json.loads((folder / "stdout.log").read_text().splitlines()[0][5:])
    assert args["events-file"] == (folder / "events.jsonl").as_posix()
    assert args["inspect-dir"] == (folder / "insp").as_posix() and args["side-dir"] == (folder / "side").as_posix()
    assert args["noise"] == "1.0" and args["secagg"] == "1" and args["attack"] == "labelflip"

    # the finished run streams again and ends by itself
    again = sse_messages(client, f"/api/runs/{run['id']}/events")
    assert [json.loads(d)["type"] for e, d in again if e == "message"] == expected
    assert json.loads(again[-1][1])["status"] == "done"

    # the run's grandchild is cleaned up with the process group
    gpid = int((folder / "side" / "grandchild.pid").read_text())
    if os.name != "nt":  # Windows cannot find an orphaned grandchild by tree once its parent has exited; the servers are Linux
        assert wait_for(lambda: not alive(gpid), 10)

    insp = client.get(f"/api/runs/{run['id']}/inspect").json()
    assert insp["meta"]["n_params"] == 369
    assert len(insp["plain"]) == 369 and len(insp["masked"]) == 370 and len(insp["aggregate"]) == 369
    assert [len(s) for s in insp["stages"]] == [4, 4, 4, 4]
    assert insp["chi2"]["df"] == 31 and abs(insp["chi2"]["critical_99"] - 52.191) < 1e-3
    assert insp["chi2"]["stat"] < insp["chi2"]["critical_99"]

    # a new run is allowed once the previous one finished
    r2 = client.post("/api/runs", json={**BODY, "attack": "none"})
    assert r2.status_code == 200
    client.post("/api/runs/current/stop")


def test_sse_streams_lines_as_they_are_written(env):
    """Through a real uvicorn server (TestClient buffers streamed bodies): events arrive while the run writes them."""
    import socket
    import threading

    import httpx
    import uvicorn
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(create_app(env), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    try:
        assert wait_for(lambda: srv.started, 10)
        base = f"http://127.0.0.1:{port}"
        with httpx.Client(base_url=base, timeout=30, trust_env=False) as h:
            run = h.post("/api/runs", json=BODY).json()
            t0 = time.monotonic()
            stamps, types = [], []
            with h.stream("GET", f"/api/runs/{run['id']}/events") as r:
                for line in r.iter_lines():
                    if line.startswith("data: "):
                        stamps.append(time.monotonic() - t0)
                        types.append(json.loads(line[6:]).get("type"))
            assert "done" in types
            assert stamps[-1] - stamps[0] > 1.0, "events arrive over time, not in one block at the end"
    finally:
        srv.should_exit = True
        thread.join(15)


@pytest.mark.parametrize("mode", ["nodone", "error"])
def test_a_run_without_done_is_failed(client, monkeypatch, mode):
    monkeypatch.setenv("FAKE_MODE", mode)
    run = client.post("/api/runs", json=BODY).json()
    msgs = sse_messages(client, f"/api/runs/{run['id']}/events")
    assert wait_for(lambda: client.get("/api/runs/current").json()["status"] != "running")
    cur = client.get("/api/runs/current").json()
    assert cur["status"] == "failed"
    assert cur["exit_code"] == 0, "the exit code is not trusted: no done event means failed"
    if mode == "error":
        assert json.loads([d for e, d in msgs if e == "message"][-1])["type"] == "error"


def test_stop_kills_the_whole_process_tree(client, env, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "hang")
    run = client.post("/api/runs", json=BODY).json()
    folder = env.runs_dir / run["id"]
    assert wait_for(lambda: (folder / "side" / "grandchild.pid").is_file())
    gpid = int((folder / "side" / "grandchild.pid").read_text())
    child = client.app.state.manager.latest.proc.pid
    assert alive(child) and alive(gpid)

    r = client.post("/api/runs/current/stop")
    assert r.status_code == 200 and r.json()["status"] == "stopped" and r.json()["finished"] is not None
    assert wait_for(lambda: not alive(child) and not alive(gpid), 10)
    msgs = sse_messages(client, f"/api/runs/{run['id']}/events")
    assert [json.loads(d)["type"] for e, d in msgs if e == "message"] == ["baseline"]
    assert json.loads(msgs[-1][1])["status"] == "stopped"
    assert client.post("/api/runs/current/stop").status_code == 409


def test_server_shutdown_kills_the_child(env, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "hang")
    with TestClient(create_app(env)) as c:
        c.post("/api/runs", json=BODY)
        folder = env.runs_dir / c.get("/api/runs/current").json()["id"]
        assert wait_for(lambda: (folder / "side" / "grandchild.pid").is_file())
        child = c.app.state.manager.latest.proc.pid
        gpid = int((folder / "side" / "grandchild.pid").read_text())
    assert wait_for(lambda: not alive(child) and not alive(gpid), 10)


# ---------------------------------------------------------------------------------------------------------- inspect

def test_inspect_reads_big_endian_and_missing_files(client, env):
    run_id = "20261017-120000-abcdef"
    insp = env.runs_dir / run_id / "insp"
    insp.mkdir(parents=True)
    empty = client.get(f"/api/runs/{run_id}/inspect").json()
    assert empty["plain"] is None and empty["masked"] is None and empty["meta"] is None
    assert empty["chi2"]["stat"] is None and empty["stages"] == [[], [], [], []]

    rng = np.random.default_rng(0)
    plain = rng.normal(0, 0.3, 369).astype(">f4")
    masked = rng.integers(0, 2**32, 370, dtype=np.uint64).astype(">u4")
    np.save(insp / "plain.npy", plain)
    np.save(insp / "masked.npy", masked)
    got = client.get(f"/api/runs/{run_id}/inspect").json()
    assert np.allclose(got["plain"], plain.astype("<f4"))
    assert got["masked"] == masked.astype("<u4").tolist()
    assert got["chi2"]["stat"] < got["chi2"]["critical_99"], "uniform random numbers pass"


def test_chi2_flags_structure():
    structured = np.zeros(370, dtype=np.uint32)
    structured[1:] = np.arange(369) % 4  # every value in the first bin
    res = chi2_uniform(structured)
    assert res["stat"] == pytest.approx(369 * 31)  # all in one bin: sum = (n - e)^2/e + 31 e = 31 n
    assert res["stat"] > res["critical_99"]


# ---------------------------------------------------------------------------------------------------------- sweeps

def result_row(**kw):
    row = {"noise": 1.0, "secagg": 1, "strategy": "fedavg", "attack": "none", "agg": "fedavg", "seed": 0, "rounds": 20,
           "fed_acc": 0.8, "fed_f1": 0.8, "epsilon_max": 3.0, "honest_acc": 0.8, "attacker_acc": 0.7, "mia_auc_update": 0.5,
           "mia_adv_update": 0.05, "mia_auc_global": 0.5, "mia_adv_global": 0.04, "central_acc": 0.83, "sec_per_round": 2.0}
    return {**row, **kw}


def test_sweeps_empty(client, env):
    assert client.get("/api/sweeps").json() == {"files": [], "n_rows": 0, "skipped": 0, "groups": []}
    env.results_dir.mkdir()
    (env.results_dir / "empty.jsonl").write_text("")
    assert client.get("/api/sweeps").json()["groups"] == []


def test_sweeps_mean_and_std_over_seeds(client, env):
    env.results_dir.mkdir()
    rows = [result_row(seed=0, fed_acc=0.80), result_row(seed=1, fed_acc=0.84), result_row(seed=2, fed_acc=0.82),
            result_row(noise=0.0, epsilon_max=None, fed_acc=0.86),
            result_row(attack="labelflip", agg="median", secagg=0, honest_acc=0.75)]
    (env.results_dir / "privacy.jsonl").write_text("".join(json.dumps({**r, "sweep": "privacy"}) + "\n" for r in rows) + "garbage\n")
    data = client.get("/api/sweeps").json()
    assert data["files"] == ["privacy.jsonl"] and data["n_rows"] == 5 and data["skipped"] == 1
    g = next(g for g in data["groups"] if g["noise"] == 1.0 and g["attack"] == "none")
    assert g["n"] == 3 and g["seeds"] == [0, 1, 2]
    assert g["fed_acc"]["mean"] == pytest.approx(0.82) and g["fed_acc"]["std"] == pytest.approx(0.02)
    g0 = next(g for g in data["groups"] if g["noise"] == 0.0)
    assert g0["epsilon_max"] == {"mean": None, "std": None, "n": 0} and g0["fed_acc"]["std"] == 0.0
    ga = next(g for g in data["groups"] if g["attack"] == "labelflip")
    assert (ga["agg"], ga["secagg"], ga["honest_acc"]["mean"]) == ("median", 0, 0.75)


# ---------------------------------------------------------------------------------------------------------- replays

def test_replays(client, env):
    assert client.get("/api/replays").json() == []
    shutil.copytree(FIXTURE, env.replays_dir / "sample")
    listing = client.get("/api/replays").json()
    assert [r["name"] for r in listing] == ["sample"]
    assert listing[0]["n_events"] == 7 and listing[0]["duration"] == 126.0 and listing[0]["complete"]

    t0 = time.monotonic()
    msgs = sse_messages(client, "/api/replays/sample/events?speed=100")
    took = time.monotonic() - t0
    assert [json.loads(d)["type"] for e, d in msgs if e == "message"] == \
        [json.loads(l)["type"] for l in (FIXTURE / "events.jsonl").read_text().splitlines()]
    assert msgs[-1][0] == "end"
    assert 1.0 < took < 10, "original timing (126 s) divided by the speed"
    assert client.get("/api/replays/sample/inspect").json()["meta"]["n_params"] == 369
    assert client.get("/api/replays/sample/events?speed=0").status_code == 422
    assert client.get("/api/replays/nope/events").status_code == 404


# ---------------------------------------------------------------------------------------------------------- security

@pytest.mark.parametrize("path", ["/api/runs/..%2F..%2Fetc/inspect", "/api/runs/20261017-120000-ABCDEF/events",
                                  "/api/runs/x/inspect", "/api/replays/..%2Ffixtures/events", "/api/replays/UP/inspect",
                                  "/api/runs/20261017-120000-abcdef/inspect", "/api/nothing"])
def test_ids_are_validated(client, path):
    assert client.get(path).status_code == 404


def test_untrusted_host_and_cross_origin_post_are_refused(client):
    assert client.get("/api/config", headers={"host": "evil.example"}).status_code == 400
    r = client.post("/api/runs", json=BODY, headers={"origin": "https://evil.example"})
    assert r.status_code == 403
    assert client.get("/api/runs/current").json() is None


def test_static_page_and_headers(client):
    r = client.get("/")
    assert r.status_code == 200 and "Zero Exposure" in r.text
    assert "script-src 'self'" in r.headers["content-security-policy"]
    for asset in ("/app.js", "/style.css", "/vendor/chart.umd.min.js", "/vendor/fonts/IBMPlexSans-Regular-Latin1.woff2"):
        assert client.get(asset).status_code == 200, asset


# ---------------------------------------------------------------------------------------------------------- real engine

@pytest.mark.skipif(not all(f.is_file() for f in HOSPITAL_FILES), reason=f"UCI Heart Disease files missing in {DATA_DIR}")
def test_real_short_simulation(tmp_path):
    settings = Settings(mode="sim", runs_dir=tmp_path / "runs", results_dir=tmp_path / "results",
                        replays_dir=tmp_path / "replays", allowed_hosts=["testserver"])
    with TestClient(create_app(settings)) as c:
        r = c.post("/api/runs", json={**BODY, "rounds": 2})
        assert r.status_code == 200, r.text
        run = r.json()
        msgs = sse_messages(c, f"/api/runs/{run['id']}/events", timeout=900)
        types = [json.loads(d)["type"] for e, d in msgs if e == "message"]
        folder = settings.runs_dir / run["id"]
        log = (folder / "stdout.log").read_text(errors="replace")[-3000:] + (folder / "stderr.log").read_text(errors="replace")[-3000:]
        assert types == ["baseline", "round", "round", "round", "attack", "done"], log
        assert wait_for(lambda: c.get("/api/runs/current").json()["status"] != "running", 120)
        assert c.get("/api/runs/current").json()["status"] == "done"
        insp = c.get(f"/api/runs/{run['id']}/inspect").json()
        assert len(insp["plain"]) == 369 and len(insp["masked"]) == 370 and len(insp["aggregate"]) == 369
        assert insp["chi2"]["stat"] < insp["chi2"]["critical_99"], "SecAgg+ output looks like noise"
        assert all(len(s) == 4 for s in insp["stages"])

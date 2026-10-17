"""The deployment (1 SuperLink + 4 SuperNode processes, TLS) computes exactly what the simulation computes.
Run: python -m pytest tests/test_deploy.py -q     (about 10 minutes; needs ports 9092-9097 free)
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import run_local_deploy as L  # noqa: E402
from test_sim import sim  # noqa: E402

ROUNDS = 3
SAME = ("fed_acc", "fed_f1", "fed_per_hospital_acc", "epsilon_per_hospital", "central_acc", "local_per_hospital_acc")
PROC_RE = r"flower-(superlink|supernode|superexec)|flwr-(serverapp|clientapp)"


def flower_processes() -> list[str]:
    if sys.platform == "win32":
        ps = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match '%s' } | "
              "ForEach-Object { $_.CommandLine }" % PROC_RE)
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=60).stdout
    else:
        out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True, timeout=60).stdout
        out = "\n".join(l for l in out.splitlines() if re.search(PROC_RE, l))
    return [l for l in out.splitlines() if l.strip()]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """One deployment serves every run below; the simulation references run first."""
    assert not flower_processes(), "another Flower deployment is running on this machine"
    tmp = tmp_path_factory.mktemp("deploy")
    ref = {
        "plain": sim(tmp, "sim_plain", secagg=0),
        "prox": sim(tmp, "sim_prox", secagg=0, strategy="fedprox"),
        "flip": sim(tmp, "sim_flip", secagg=0, attack="labelflip"),
        "median": sim(tmp, "sim_median", secagg=0, agg="median", attack="boost"),
    }
    dep = {}
    with L.LocalDeployment() as d:
        for tag, extra in (("plain", {"secagg": 0}), ("prox", {"secagg": 0, "strategy": "fedprox"}),
                           ("flip", {"secagg": 0, "attack": "labelflip"}),
                           ("median", {"secagg": 0, "agg": "median", "attack": "boost"}), ("secagg", {"secagg": 1})):
            folder = tmp / f"dep_{tag}"
            settings = {"rounds": ROUNDS, "events-file": (folder / "events.jsonl").as_posix(),
                        "inspect-dir": (folder / "insp").as_posix(), "side-dir": (folder / "side").as_posix(), **extra}
            dep[tag] = (L.parse_result(d.run(settings)), folder)
        dep["refused"] = d.run({"rounds": ROUNDS, "secagg": 1, "agg": "median"}, timeout=300)
    return ref, dep


@pytest.mark.parametrize("tag", ["plain", "prox", "flip", "median"])
def test_deployment_equals_simulation(runs, tag):
    ref, dep = runs
    r, d = ref[tag][0], dep[tag][0]
    assert d is not None
    for k in SAME + (("honest_acc", "attacker_acc", "mia_auc_update", "mia_auc_global") if tag in ("flip", "median") else ()):
        assert d[k] == r[k], (tag, k, d[k], r[k])


def test_secagg_deployment_within_the_quantization_bound(runs):
    ref, dep = runs
    on = np.load(dep["secagg"][1] / "insp" / "aggregate.npy").astype(np.float64)
    off = np.load(dep["plain"][1] / "insp" / "aggregate.npy").astype(np.float64)
    bound = 4 * (2 * 8.0 / 4194304) / (np.array([227, 220, 92, 150]) / 300).sum()
    assert np.abs(on - off).max() < bound
    assert dep["secagg"][0]["epsilon_per_hospital"] == dep["plain"][0]["epsilon_per_hospital"]


def test_deployment_writes_the_dashboard_files(runs):
    _, dep = runs
    folder = dep["secagg"][1]
    kinds = [json.loads(l)["type"] for l in (folder / "events.jsonl").read_text().splitlines()]
    assert kinds[0] == "baseline" and kinds[-1] == "done" and kinds.count("round") == ROUNDS + 1
    assert np.load(folder / "insp" / "masked.npy").size == 370
    assert all((folder / "insp" / f"stages_h{i}.jsonl").exists() for i in range(4))


def test_refused_settings_in_the_deployment(runs):
    _, dep = runs
    assert L.parse_result(dep["refused"]) is None and "SecAgg+ hides exactly that" in dep["refused"]


def test_tls_refuses_plaintext_and_foreign_certificates(tmp_path):
    import grpc
    sys.path.insert(0, str(ROOT / "deploy"))
    from gen_certs import generate
    with L.LocalDeployment(tls=True) as d:
        ca = (d.work / "certs" / "ca.crt").read_bytes()
        other = generate(tmp_path / "other")  # a CA nobody trusts

        def ready(creds):
            ch = grpc.insecure_channel(f"127.0.0.1:{L.FLEET_PORT}") if creds is None else \
                grpc.secure_channel(f"127.0.0.1:{L.FLEET_PORT}", creds)
            try:
                grpc.channel_ready_future(ch).result(timeout=5)
                return True
            except grpc.FutureTimeoutError:
                return False
            finally:
                ch.close()

        assert ready(grpc.ssl_channel_credentials(ca))
        assert not ready(grpc.ssl_channel_credentials((other / "ca.crt").read_bytes()))
        assert not ready(None)


def test_nothing_is_left_running(runs):
    assert not flower_processes()

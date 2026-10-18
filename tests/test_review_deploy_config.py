"""Review tests: static checks of the deployment configuration (compose files, certificates, launcher helpers). No Docker needed.

Run: python -m pytest tests/test_review_deploy_config.py -q
"""
import re
import subprocess
import sys
from pathlib import Path

import pytest
from cryptography import x509

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "deploy"))
sys.path.insert(0, str(ROOT / "scripts"))
import run_local_deploy as L  # noqa: E402
from gen_certs import generate  # noqa: E402

COMPOSE = (ROOT / "deploy" / "docker-compose.yml").read_text(encoding="utf-8")


def service_block(name: str) -> str:
    m = re.search(rf"^  {re.escape(name)}:\n(.*?)(?=^  \S|^\S|\Z)", COMPOSE, re.S | re.M)
    assert m, name
    return m.group(1)


# ---------------------------------------------------------------- which container gets which key
def test_only_the_certs_service_mounts_the_ca_key_volume():
    users = [s for s in ("certs", "superlink", "dashboard", *[f"supernode-{i}" for i in range(4)]) if "ca-key:" in service_block(s)]
    assert users == ["certs"]


def test_only_the_superlink_mounts_the_server_key_volume():
    users = [s for s in ("superlink", "dashboard", *[f"supernode-{i}" for i in range(4)]) if "certs-server:" in service_block(s)]
    assert users == ["superlink"]


@pytest.mark.parametrize("i,name", list(enumerate(["cleveland", "hungarian", "switzerland", "va"])))
def test_each_supernode_mounts_only_its_own_data_file_read_only(i, name):
    block = service_block(f"supernode-{i}")
    mounts = re.findall(r"heart_disease/([^:\"]+):", block)
    assert mounts == [f"processed.{name}.data"], mounts
    assert f"processed.{name}.data:ro" in block
    assert f"partition-id={i} " in block


def test_only_the_dashboard_publishes_a_port_and_only_on_loopback():
    for s in ("certs", "superlink", *[f"supernode-{i}" for i in range(4)]):
        assert "ports:" not in service_block(s), s
    assert re.search(r'ports: \[ "127\.0\.0\.1:', service_block("dashboard"))


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="FINDING (suspected, not run in Docker): every SuperNode shares the default network with the "
                          "SuperLink, whose Control API listens on 0.0.0.0:9093 without user authentication, so a hospital "
                          "container can reach the API that submits and stops runs")
def test_supernodes_cannot_reach_the_control_api():
    """A hospital only needs the Fleet API (9092). Putting the Control API on a network shared only with the dashboard
    (or enabling Flower's user authentication) would make this pass."""
    has_networks = "networks:" in COMPOSE
    assert has_networks, "no network separation in deploy/docker-compose.yml"
    for i in range(4):
        assert "control" not in service_block(f"supernode-{i}")


# ---------------------------------------------------------------- certificates
def test_server_certificate_names_and_chain(tmp_path):
    out = generate(tmp_path)
    ca = x509.load_pem_x509_certificate((out / "ca.crt").read_bytes())
    cert = x509.load_pem_x509_certificate((out / "server.pem").read_bytes())
    san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert set(san.get_values_for_type(x509.DNSName)) == {"superlink", "localhost"}
    assert {str(a) for a in san.get_values_for_type(x509.IPAddress)} == {"127.0.0.1", "::1"}
    assert cert.issuer == ca.subject
    cert.verify_directly_issued_by(ca)
    assert ca.extensions.get_extension_for_class(x509.BasicConstraints).value.ca
    assert not cert.extensions.get_extension_for_class(x509.BasicConstraints).value.ca


def test_compose_superlink_address_is_in_the_certificate():
    names = {"superlink", "localhost", "127.0.0.1", "::1"}
    for f in ("flwr_home/config.toml", "flwr_home_insecure/config.toml"):
        host = re.search(r'address = "([^":]+):', (ROOT / "deploy" / f).read_text()).group(1)
        assert host in names
    for i in range(4):
        assert "--superlink superlink:9092" in service_block(f"supernode-{i}")


def test_existing_certificates_are_kept_and_force_regenerates(tmp_path):
    generate(tmp_path)
    before = (tmp_path / "server.pem").read_bytes()
    generate(tmp_path)
    assert (tmp_path / "server.pem").read_bytes() == before
    generate(tmp_path, force=True)
    assert (tmp_path / "server.pem").read_bytes() != before


# ---------------------------------------------------------------- launcher helpers
@pytest.mark.parametrize("v,toml", [(3, "3"), (1.0, "1.0"), (1e-5, "1e-05"), (True, "true"), ("fedavg", '"fedavg"'),
                                    ("C:\\runs\\x", '"C:\\\\runs\\\\x"')])
def test_toml_values_keep_their_type(v, toml):
    assert L.toml_value(v) == toml


def test_run_config_arg_round_trips_through_a_toml_parser():
    import tomllib
    settings = {"rounds": 3, "noise": 1.0, "delta": 1e-5, "strategy": "fedprox", "events-file": "C:\\a b\\e.jsonl"}
    parsed = tomllib.loads("\n".join(f'"{k}" = {L.toml_value(v)}' for k, v in settings.items()))
    assert parsed == settings


def test_parse_settings_types():
    s = L.parse_settings(["rounds=3", "noise=1.0", "strategy=fedprox", "secagg=1"])
    assert s == {"rounds": 3, "noise": 1.0, "strategy": "fedprox", "secagg": 1}
    assert type(s["noise"]) is float and type(s["rounds"]) is int


def test_launcher_scrubs_the_simulation_channel(monkeypatch):
    monkeypatch.setenv("ZE_RUN_CONFIG", '{"agg": "median"}')
    assert "ZE_RUN_CONFIG" not in L.LocalDeployment(workdir=".").env()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows process-tree kill")
def test_kill_tree_kills_grandchildren():
    code = ("import subprocess, sys, time; "
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); time.sleep(120)")
    p = subprocess.Popen([sys.executable, "-c", code], creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    import time
    time.sleep(2)
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          f"(Get-CimInstance Win32_Process -Filter 'ParentProcessId={p.pid}').ProcessId"],
                         capture_output=True, text=True).stdout.split()
    assert out, "grandchild did not start"
    L.kill_tree(p)
    left = subprocess.run(["powershell", "-NoProfile", "-Command",
                           f"Get-Process -Id {out[0]} -ErrorAction SilentlyContinue | Measure-Object | % Count"],
                          capture_output=True, text=True).stdout.strip()
    assert p.poll() is not None and left == "0"

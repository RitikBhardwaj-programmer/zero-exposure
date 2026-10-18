"""Run the federation as a real Flower deployment on one machine, without Docker: 1 SuperLink + 4 SuperNode processes.

Each SuperNode gets its own data folder holding ONLY its hospital's file (DATA_DIR + ALLOW_MISSING_DATA=1), the same isolation
the Docker setup gives with one mount per container. The SuperLink (which starts the ServerApp) sees all four files: the
ServerApp evaluates on every hospital's test rows (a stated demo limit). This isolation is by configuration only: all
processes run as the same user under one work folder, so one could read another's folder. Real isolation is the Docker
deployment (deploy/docker-compose.yml: separate containers, one mounted file each). TLS is on by default (env TLS=0 turns it off):
SuperNodes and the flwr CLI get only ca.crt. Everything started here is killed at the end, also on Ctrl+C or a crash.

Use:  python scripts/run_local_deploy.py rounds=3 secagg=1 noise=1.0        (keys: CONTRACTS.md section 1)
Prints the ServerApp's `RESULT {json}` line.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from zero_exposure.config import HOSPITALS  # noqa: E402

BIN = Path(sys.executable).parent  # the SuperLink starts `flower-superexec` by name: this folder must be on PATH
CONTROL_PORT, FLEET_PORT, NODE_PORT0 = 9093, 9092, 9094
CONNECTION = "local"


def exe(name: str) -> str:
    """Full path of a console script (on Windows, Popen looks a bare name up on the PARENT's PATH, not on env=)."""
    return str(BIN / (name + (".exe" if sys.platform == "win32" else "")))


def kill_tree(proc: subprocess.Popen) -> None:
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, timeout=30)
        else:
            os.killpg(proc.pid, signal.SIGKILL)  # started with start_new_session: its group id is its pid
    except (ProcessLookupError, PermissionError, subprocess.TimeoutExpired):
        pass
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass


def toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)  # keeps the decimal point: an override must have the default's type
    return json.dumps(str(v))


def run_config_arg(settings: dict) -> str:
    return " ".join(f"{k}={toml_value(v)}" for k, v in settings.items())


def port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


class LocalDeployment:
    """Context manager: starts the SuperLink and 4 SuperNodes; `run(settings)` submits one run and waits for it."""

    def __init__(self, workdir=None, data_dir=None, tls=None):
        self.owns_work = workdir is None  # a folder we made is deleted on close (it holds the CA and server private keys)
        self.work = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="ze-deploy-"))
        self.data_dir = Path(data_dir) if data_dir else ROOT / "data" / "heart_disease"
        self.tls = (os.environ.get("TLS", "1").lower() not in ("0", "false", "off", "no")) if tls is None else tls
        self.logs = self.work / "logs"
        self.procs: list[subprocess.Popen] = []

    def env(self, **extra) -> dict:
        e = {**os.environ, "PATH": str(BIN) + os.pathsep + os.environ.get("PATH", ""), "FLWR_HOME": str(self.work / "flwr_home"),
             "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", "FLWR_TELEMETRY_ENABLED": "0", "FLWR_DISABLE_UPDATE_CHECK": "1"}
        e.pop("ZE_RUN_CONFIG", None)  # simulation-only channel: a deployment takes its settings from run_config
        e.update(extra)
        return e

    def spawn(self, name: str, cmd: list[str], env: dict) -> None:
        kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if sys.platform == "win32" else {"start_new_session": True}
        with open(self.logs / f"{name}.log", "wb") as out:
            # cwd = work dir, NOT the repo: the apps must come from the shipped app bundle, never from the checkout
            self.procs.append(subprocess.Popen(cmd, cwd=self.work, env=env, stdin=subprocess.DEVNULL, stdout=out,
                                               stderr=subprocess.STDOUT, **kw))

    def __enter__(self):
        try:
            self.logs.mkdir(parents=True, exist_ok=True)
            busy = [p for p in (CONTROL_PORT, FLEET_PORT, *[NODE_PORT0 + i for i in range(4)]) if port_open(p)]
            if busy:
                raise RuntimeError(f"ports already in use: {busy} (is another deployment running?)")
            if self.tls:
                sys.path.insert(0, str(ROOT / "deploy"))
                from gen_certs import generate
                certs = generate(self.work / "certs")
                link_tls = ["--ssl-ca-certfile", str(certs / "ca.crt"), "--ssl-certfile", str(certs / "server.pem"),
                            "--ssl-keyfile", str(certs / "server.key")]
                node_tls = ["--root-certificates", str(certs / "ca.crt")]
                cli_tls = f'root-certificates = "{(certs / "ca.crt").as_posix()}"\n'
            else:
                link_tls, node_tls, cli_tls = ["--insecure"], ["--insecure"], "insecure = true\n"
            home = self.work / "flwr_home"
            home.mkdir(exist_ok=True)  # our own Flower config: the user's ~/.flwr is never touched
            (home / "config.toml").write_text(f'[superlink]\ndefault = "{CONNECTION}"\n\n[superlink.{CONNECTION}]\n'
                                              f'address = "127.0.0.1:{CONTROL_PORT}"\n{cli_tls}', encoding="utf-8")
            self.spawn("superlink", [exe("flower-superlink"), *link_tls, "--disable-runtime-dependency-installation",
                                     "--host", "127.0.0.1", "--port", str(CONTROL_PORT),
                                     "--fleet-api-address", f"127.0.0.1:{FLEET_PORT}"], self.env(DATA_DIR=str(self.data_dir)))
            for i, name in enumerate(HOSPITALS):
                own = self.work / f"data_{name}"  # this SuperNode's whole world: one file
                own.mkdir(exist_ok=True)
                shutil.copy2(self.data_dir / f"processed.{name}.data", own)
                self.spawn(f"supernode-{i}", [exe("flower-supernode"), *node_tls, "--superlink", f"127.0.0.1:{FLEET_PORT}",
                                              "--node-config", f"partition-id={i} num-partitions={len(HOSPITALS)}",
                                              "--port", str(NODE_PORT0 + i)], self.env(DATA_DIR=str(own), ALLOW_MISSING_DATA="1"))
            self.wait_ready()
            return self
        except BaseException:
            self.close()
            raise

    def wait_ready(self, timeout: float = 90) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout:
            for p in self.procs:
                if p.poll() is not None:
                    raise RuntimeError(f"a process exited early (code {p.returncode}); logs in {self.logs}")
            if port_open(CONTROL_PORT) and all(port_open(NODE_PORT0 + i) for i in range(4)):
                time.sleep(3)  # let the SuperNodes finish registering
                return
            time.sleep(0.5)
        raise RuntimeError(f"deployment not ready after {timeout}s; logs in {self.logs}")

    def run(self, settings: dict, timeout: float = 1200) -> str:
        """Submit one run with `flwr run --stream` and return its combined output."""
        cmd = [exe("flwr"), "run", str(ROOT), CONNECTION, "--stream", "--run-config", run_config_arg(settings)]
        p = subprocess.run(cmd, cwd=self.work, env=self.env(), capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout)
        text = p.stdout + "\n" + p.stderr
        (self.logs / "flwr_run.log").write_text(text, encoding="utf-8")
        return text

    def close(self) -> None:
        for p in reversed(self.procs):
            kill_tree(p)
        self.procs = []
        if self.owns_work and os.environ.get("KEEP") != "1":  # KEEP=1 keeps logs and certificates for debugging
            shutil.rmtree(self.work, ignore_errors=True)

    def __exit__(self, *exc):
        self.close()


def parse_result(text: str):
    for line in reversed(text.splitlines()):
        if "RESULT {" in line:
            return json.loads(line[line.index("RESULT ") + 7:])
    return None


def parse_settings(argv: list[str]) -> dict:
    out = {}
    for a in argv:
        k, _, v = a.partition("=")
        try:
            out[k] = json.loads(v)  # numbers, true/false; write floats with a decimal point (noise=1.0)
        except ValueError:
            out[k] = v
    return out


def main(argv: list[str]) -> int:
    with LocalDeployment() as dep:
        print(f"work dir {dep.work} (logs in {dep.logs})", flush=True)
        text = dep.run(parse_settings(argv))
    res = parse_result(text)
    print(text if res is None else "RESULT " + json.dumps(res))
    return 0 if res else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

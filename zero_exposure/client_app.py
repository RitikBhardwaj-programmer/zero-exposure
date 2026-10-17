"""The hospital side (Flower ClientApp). One app for the simulation and the deployment.

Each hospital reads only its own data file, trains with DP-SGD, optionally misbehaves (attack lab), and sends its update.
Mods (Flower message wrappers; mods[0] is the outermost):
  1. inspector_mod  OUTSIDE SecAgg+: sees the reply after masking, so the dashboard can show what actually left the hospital.
  2. secagg_mod     Flower's secaggplus_mod when the run uses SecAgg+, else a pass-through.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

from flwr.app import Context
from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod
from flwr.common.secure_aggregation.secaggplus_constants import RECORD_KEY_CONFIGS, Key, Stage

from . import attacks, events
from .config import CLIPPING_RANGE, QUANTIZATION_RANGE, RunConfig, from_context
from .data import Hospital, prepare
from .model import HeartNet, evaluate, flat, get_weights, set_weights
from .privacy import train_private

_hospitals: dict[tuple[int, int], Hospital] = {}


def hospital(pid: int, seed: int) -> Hospital:
    """This hospital's data for this seed (the split depends on the seed). Only its own file is read."""
    key = (pid, seed)
    if key not in _hospitals:
        _hospitals[key] = prepare(pid, seed)
    return _hospitals[key]


def train_seed(cfg: RunConfig, pid: int, rnd: int) -> int:
    """Distinct, reproducible random stream per run seed, hospital and round."""
    return cfg.seed * 1_000_000 + pid * 1_000 + rnd


class HospitalClient(NumPyClient):
    def __init__(self, cfg: RunConfig, pid: int):
        self.cfg, self.pid, self.h = cfg, pid, hospital(pid, cfg.seed)

    def fit(self, parameters, config):
        cfg, pid, h = self.cfg, self.pid, self.h
        rnd = int(config.get("server_round", 0))
        y = attacks.flip(h.y_train) if attacks.flips_labels(cfg, pid) else h.y_train
        model = HeartNet()
        set_weights(model, parameters)
        mu = cfg.mu if cfg.strategy == "fedprox" else 0.0
        train_private(model, h.X_train, y, epochs=cfg.local_epochs, noise=cfg.noise, clip=cfg.clip,
                      seed=train_seed(cfg, pid, rnd), prox_mu=mu, global_weights=parameters)
        update = get_weights(model)
        if attacks.boosts(cfg, pid):
            update = attacks.boost_update(update, parameters, cfg.boost)
        n = len(h.y_train)
        self._side_channel(rnd, update, n)
        if cfg.inspect_dir and rnd == cfg.inspect_round:
            events.stage(cfg.inspect_dir, pid, rnd, "trained")
            if pid == 0:
                events.save_array(cfg.inspect_dir, "plain.npy", flat(update).astype(np.float32))
                events.save_json(cfg.inspect_dir, "inspect.json", {
                    "round": rnd, "pid": pid, "n_params": int(flat(update).size), "secagg": cfg.secagg, "n_examples": n,
                    "clipping_range": CLIPPING_RANGE, "quantization_range": QUANTIZATION_RANGE})
        return update, n, {}

    def evaluate(self, parameters, config):
        model = HeartNet()
        set_weights(model, parameters)
        loss, acc, _ = evaluate(model, self.h.X_test, self.h.y_test)
        return loss, len(self.h.y_test), {"accuracy": acc}

    def _side_channel(self, rnd: int, update, n: int) -> None:
        """What the attack summary needs from the hospitals (CONTRACTS.md: side-dir)."""
        cfg, pid = self.cfg, self.pid
        if not cfg.side_dir:
            return
        d = Path(cfg.side_dir)
        d.mkdir(parents=True, exist_ok=True)
        if cfg.attack != "none" and pid == cfg.attacker:
            scaled = attacks.max_scaled(update, n)
            row = {"round": rnd, "max_scaled": scaled, "clipped": bool(cfg.secagg and scaled >= CLIPPING_RANGE)}
            with (d / "attacker.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        if cfg.mia_active and pid == cfg.mia_hospital and rnd == cfg.rounds:
            np.savez(d / "mia_update.npz", *update)  # what the server would see of this hospital with SecAgg+ off


def client_fn(context: Context):
    cfg = from_context(context.run_config)
    return HospitalClient(cfg, int(context.node_config["partition-id"])).to_client()


def _round(msg) -> int:
    try:
        return int(msg.metadata.group_id)
    except (TypeError, ValueError):
        return -1


def inspector_mod(msg, ctxt, call_next):
    cfg = from_context(ctxt.run_config)
    if not cfg.inspect_dir or _round(msg) != cfg.inspect_round or msg.metadata.message_type != "train":
        return call_next(msg, ctxt)
    pid = int(ctxt.node_config["partition-id"])
    stage = None
    if cfg.secagg:
        sa = msg.content.config_records.get(RECORD_KEY_CONFIGS)
        stage = sa.get(Key.STAGE) if sa is not None else None  # read BEFORE call_next: secaggplus_mod pops it
        if stage != Stage.COLLECT_MASKED_VECTORS:
            return call_next(msg, ctxt)
    events.stage(cfg.inspect_dir, pid, cfg.inspect_round, "received")
    reply = call_next(msg, ctxt)
    if cfg.secagg:
        events.stage(cfg.inspect_dir, pid, cfg.inspect_round, "masked")
        if pid == 0:
            parts = reply.content.config_records[RECORD_KEY_CONFIGS][Key.MASKED_PARAMETERS]
            masked = np.concatenate([np.load(io.BytesIO(b), allow_pickle=False).ravel() for b in parts])
            events.save_array(cfg.inspect_dir, "masked.npy", masked.astype(np.uint32))
    events.stage(cfg.inspect_dir, pid, cfg.inspect_round, "sent")
    return reply


def secagg_mod(msg, ctxt, call_next):
    if from_context(ctxt.run_config).secagg:
        return secaggplus_mod(msg, ctxt, call_next)
    return call_next(msg, ctxt)


app = ClientApp(client_fn=client_fn, mods=[inspector_mod, secagg_mod])

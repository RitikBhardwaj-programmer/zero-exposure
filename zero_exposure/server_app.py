"""The server side (Flower ServerApp). One app for the simulation and the deployment.

Order of a run: settings -> non-private reference baselines -> initial model -> rounds (FedAvg / FedProx / median / trimmed
mean, with or without SecAgg+) -> attack summary (poisoning effect, membership test) -> RESULT line -> `done` event.
Evaluation needs every hospital's TEST rows, so this process reads all four files: a stated demo limit (evaluation only;
no training row of any hospital is used here except for the reference baselines and the membership test).
"""
from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from flwr.app import Context
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerConfig
from flwr.server.compat.legacy_context import LegacyContext
from flwr.server.strategy import FedAvg, FedMedian, FedProx, FedTrimmedAvg
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow

from . import attacks, events
from .config import (CLIPPING_RANGE, HOSPITALS, MAX_WEIGHT, NUM_HOSPITALS, NUM_SHARES, QUANTIZATION_RANGE,
                     RECONSTRUCTION_THRESHOLD, SIM_ENV, RunConfig, finite_or_none, from_context)
from .data import load_all
from .model import HeartNet, evaluate, flat, get_weights, native, set_weights
from .privacy import epsilon_after

BASELINE_EPOCHS = 60
BASELINE_LR = 0.01


def train_plain(X, y, seed: int) -> nn.Module:
    """Non-private reference training (Adam, full passes, seeded shuffling)."""
    torch.manual_seed(seed)
    model = HeartNet()
    opt = torch.optim.Adam(model.parameters(), lr=BASELINE_LR)
    loss_fn = nn.BCEWithLogitsLoss()
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    gen = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(BASELINE_EPOCHS):
        for idx in torch.randperm(len(Xt), generator=gen).split(32):
            opt.zero_grad()
            loss_fn(model(Xt[idx]), yt[idx]).backward()
            opt.step()
    return model


def baselines(data, seed: int):
    """What each hospital gets alone, and what pooling all records would give (both without privacy): the two references."""
    local = [evaluate(train_plain(h.X_train, h.y_train, seed * 100 + h.pid), h.X_test, h.y_test)[1] for h in data]
    X = np.concatenate([h.X_train for h in data])
    y = np.concatenate([h.y_train for h in data])
    central = train_plain(X, y, seed * 100 + 99)
    _, c_acc, c_f1 = evaluate(central, np.concatenate([h.X_test for h in data]), np.concatenate([h.y_test for h in data]))
    return local, c_acc, c_f1


class Run:
    """State of one run on the server: settings, data, per-round history."""

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.data = load_all(cfg.seed, allow_missing=False)
        self.X_test = np.concatenate([h.X_test for h in self.data])
        self.y_test = np.concatenate([h.y_test for h in self.data])
        self.history: list[dict] = []
        self.stamps: list[float] = []
        self.last_weights = None

    def global_eval(self, rnd: int, weights, _config):
        cfg = self.cfg
        self.stamps.append(time.perf_counter())
        model = HeartNet()
        set_weights(model, weights)
        loss, acc, f1 = evaluate(model, self.X_test, self.y_test)
        per = [evaluate(model, h.X_test, h.y_test)[1] for h in self.data]
        eps = [epsilon_after(len(h.y_train), rnd, cfg.local_epochs, cfg.noise, cfg.delta) for h in self.data]
        max_abs = float(np.abs(flat(weights)).max())
        if rnd > 0 and cfg.secagg and cfg.attack == "none" and max_abs >= CLIPPING_RANGE:
            raise RuntimeError(f"round {rnd}: max |weight| {max_abs:.2f} reached the SecAgg+ clipping range {CLIPPING_RANGE}")
        if rnd > 0 and self.last_weights is not None and np.array_equal(flat(self.last_weights), flat(weights)):
            raise RuntimeError(f"round {rnd}: the model did not change (a hospital dropped out and SecAgg+ halted?)")
        self.last_weights = [native(w) for w in weights]
        if cfg.inspect_dir and rnd == cfg.inspect_round:
            events.save_array(cfg.inspect_dir, "aggregate.npy", flat(weights).astype(np.float32))
        row = {"round": rnd, "total_rounds": cfg.rounds, "acc": acc, "f1": f1, "loss": loss, "per_hospital_acc": per,
               "epsilon": [finite_or_none(e) for e in eps], "max_abs_weight": max_abs}
        self.history.append(row)
        events.emit(cfg.events_file, "round", **row)
        print(f"round {rnd:2d} | acc {acc:.3f} F1 {f1:.3f} loss {loss:.3f} | per-hospital {[round(p, 2) for p in per]} | "
              f"eps {[None if e is None else round(e, 2) for e in row['epsilon']]} | max|w| {max_abs:.2f}", flush=True)
        return loss, {"accuracy": acc, "f1": f1}

    def strategy(self, initial):
        cfg = self.cfg
        kw = dict(fraction_fit=1.0, fraction_evaluate=0.0, min_fit_clients=NUM_HOSPITALS, min_evaluate_clients=0,
                  min_available_clients=NUM_HOSPITALS, initial_parameters=ndarrays_to_parameters(initial),
                  evaluate_fn=lambda r, p, c: self.global_eval(r, p, c), on_fit_config_fn=lambda r: {"server_round": r})
        if cfg.agg == "median":
            return FedMedian(**kw)
        if cfg.agg == "trimmed":
            return FedTrimmedAvg(beta=cfg.trim_beta, **kw)
        if cfg.strategy == "fedprox":
            return FedProx(proximal_mu=cfg.mu, **kw)
        return FedAvg(**kw)

    def attack_summary(self) -> dict:
        cfg = self.cfg
        last = self.history[-1]
        honest = [a for i, a in enumerate(last["per_hospital_acc"]) if i != cfg.attacker]
        res = {"attack": cfg.attack, "attacker": cfg.attacker, "attacker_name": HOSPITALS[cfg.attacker],
               "boost": cfg.boost if cfg.attack in ("boost", "flipboost") else None, "agg": cfg.agg, "secagg": cfg.secagg,
               "honest_acc": float(np.mean(honest)), "attacker_acc": last["per_hospital_acc"][cfg.attacker],
               "mia_hospital": cfg.mia_hospital, "mia_hospital_name": HOSPITALS[cfg.mia_hospital], "mia_n": None,
               "mia_auc_update": None, "mia_adv_update": None, "mia_auc_global": None, "mia_adv_global": None,
               "clipped_rounds": None, "max_scaled_attacker": None}
        side = Path(cfg.side_dir) if cfg.side_dir else None
        if side and cfg.attack != "none" and (side / "attacker.jsonl").exists():
            rows = [json.loads(l) for l in (side / "attacker.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
            res["clipped_rounds"] = sum(r["clipped"] for r in rows)
            res["max_scaled_attacker"] = max(r["max_scaled"] for r in rows)
        if cfg.mia_active:
            h = self.data[cfg.mia_hospital]
            with np.load(side / "mia_update.npz") as z:
                update = [z[k] for k in z.files]
            mseed = cfg.seed * 1_000_003 + 17  # own stream: the attack never touches the training randomness
            res["mia_auc_update"], res["mia_adv_update"], res["mia_n"] = attacks.membership_attack(
                update, h.X_train, h.y_train, h.X_test, h.y_test, mseed)
            res["mia_auc_global"], res["mia_adv_global"], _ = attacks.membership_attack(
                self.last_weights, h.X_train, h.y_train, h.X_test, h.y_test, mseed)
        return res


app = ServerApp()


@app.main()
def main(grid, context: Context) -> None:
    raw = dict(context.run_config or {})
    if os.environ.get(SIM_ENV):
        raw.update(json.loads(os.environ[SIM_ENV]))
    events_file = str(raw.get("events-file", ""))  # known even when the settings are refused
    try:
        cfg = from_context(context.run_config)
        _run(grid, context, cfg)
    except BaseException as exc:
        events.emit(events_file, "error", message=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise


def _run(grid, context: Context, cfg: RunConfig) -> None:
    side = Path(cfg.side_dir) if cfg.side_dir else None
    if side:
        side.mkdir(parents=True, exist_ok=True)
        for name in ("attacker.jsonl", "mia_update.npz"):  # a reused folder must not leak an older run's files
            (side / name).unlink(missing_ok=True)
    run = Run(cfg)
    local, c_acc, c_f1 = baselines(run.data, cfg.seed)
    events.emit(cfg.events_file, "baseline", hospitals=list(HOSPITALS), n_train=[len(h.y_train) for h in run.data],
                n_test=[len(h.y_test) for h in run.data], central_acc=c_acc, central_f1=c_f1,
                local_per_hospital_acc=local, config=cfg.public_dict())
    print(f"baselines: local-only {[round(a, 3) for a in local]} | centralized acc {c_acc:.3f} F1 {c_f1:.3f}", flush=True)
    print(f"settings: {cfg.public_dict()}", flush=True)

    torch.manual_seed(cfg.seed)  # the initial model depends only on the seed
    initial = get_weights(HeartNet())
    ctx = LegacyContext(context=context, config=ServerConfig(num_rounds=cfg.rounds), strategy=run.strategy(initial))
    if cfg.secagg:
        fit = SecAggPlusWorkflow(num_shares=NUM_SHARES, reconstruction_threshold=RECONSTRUCTION_THRESHOLD,
                                 max_weight=MAX_WEIGHT, clipping_range=CLIPPING_RANGE, quantization_range=QUANTIZATION_RANGE)
        DefaultWorkflow(fit_workflow=fit)(grid, ctx)
    else:
        DefaultWorkflow()(grid, ctx)
    if len(run.history) != cfg.rounds + 1:
        raise RuntimeError(f"expected {cfg.rounds + 1} evaluations, got {len(run.history)}")

    summary = run.attack_summary()
    events.emit(cfg.events_file, "attack", **summary)
    last = run.history[-1]
    eps = last["epsilon"]
    result = {**{k.replace("-", "_"): v for k, v in cfg.public_dict().items()},
              "fed_acc": last["acc"], "fed_f1": last["f1"], "fed_per_hospital_acc": last["per_hospital_acc"],
              "epsilon_per_hospital": eps, "epsilon_max": None if any(e is None for e in eps) else max(eps),
              "max_abs_weight": max(h["max_abs_weight"] for h in run.history),
              "sec_per_round": (run.stamps[-1] - run.stamps[0]) / cfg.rounds,
              "central_acc": c_acc, "central_f1": c_f1, "local_per_hospital_acc": local, **summary}
    print("RESULT " + json.dumps(events._clean(result)), flush=True)
    events.emit(cfg.events_file, "done")

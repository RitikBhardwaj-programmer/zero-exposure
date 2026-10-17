"""Run settings: one dataclass for the simulation, the deployment and the dashboard (keys: CONTRACTS.md section 1)."""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, fields

HOSPITALS = ("cleveland", "hungarian", "switzerland", "va")
NUM_HOSPITALS = len(HOSPITALS)

# Flower SecAgg+ settings (fixed for every run)
NUM_SHARES = 4
RECONSTRUCTION_THRESHOLD = 3
MAX_WEIGHT = 300
CLIPPING_RANGE = 8.0
QUANTIZATION_RANGE = 4194304  # 2**22

STRATEGIES = ("fedavg", "fedprox")
AGGS = ("fedavg", "median", "trimmed")
ATTACKS = ("none", "labelflip", "boost", "flipboost")


class ConfigError(ValueError):
    """A setting or a combination of settings the system refuses to run."""


@dataclass(frozen=True)
class RunConfig:
    rounds: int = 20
    local_epochs: int = 5
    noise: float = 1.0
    clip: float = 1.0
    delta: float = 1e-5
    strategy: str = "fedavg"
    mu: float = 0.1
    seed: int = 0
    secagg: int = 1
    agg: str = "fedavg"
    trim_beta: float = 0.25
    attack: str = "none"
    attacker: int = 3
    boost: float = 10.0
    mia: int = 1
    mia_hospital: int = 0
    events_file: str = ""
    inspect_dir: str = ""
    inspect_round: int = 1
    side_dir: str = ""

    @classmethod
    def from_mapping(cls, raw) -> "RunConfig":
        """Build from Flower run_config / CLI keys (dashes or underscores). Unknown keys are an error; values are coerced to the field type."""
        known = {f.name: f.type for f in fields(cls)}
        kwargs = {}
        for key, value in dict(raw).items():
            name = key.replace("-", "_")
            if name not in known:
                raise ConfigError(f"unknown setting {key!r}")
            typ = known[name]
            try:
                kwargs[name] = int(value) if typ == "int" else float(value) if typ == "float" else str(value)
            except (TypeError, ValueError):
                raise ConfigError(f"{key}: expected {typ}, got {value!r}") from None
        cfg = cls(**kwargs)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Raise ConfigError for a bad value or a combination the system refuses (CONTRACTS.md section 1)."""
        if self.rounds < 1 or self.local_epochs < 1:
            raise ConfigError("rounds and local-epochs must be at least 1")
        if self.noise < 0 or self.clip <= 0 or not 0 < self.delta < 1:
            raise ConfigError("noise must be >= 0, clip > 0 and 0 < delta < 1")
        if self.strategy not in STRATEGIES:
            raise ConfigError(f"strategy must be one of {STRATEGIES}")
        if self.agg not in AGGS:
            raise ConfigError(f"agg must be one of {AGGS}")
        if self.attack not in ATTACKS:
            raise ConfigError(f"attack must be one of {ATTACKS}")
        if self.secagg not in (0, 1) or self.mia not in (0, 1):
            raise ConfigError("secagg and mia must be 0 or 1")
        if not (0 <= self.attacker < NUM_HOSPITALS and 0 <= self.mia_hospital < NUM_HOSPITALS):
            raise ConfigError(f"attacker and mia-hospital must be hospital ids 0..{NUM_HOSPITALS - 1}")
        if not 0 <= self.trim_beta < 0.5 or self.boost <= 0 or self.mu < 0:
            raise ConfigError("trim-beta must be in [0, 0.5), boost > 0, mu >= 0")
        if self.agg != "fedavg" and self.secagg:
            raise ConfigError(f"agg={self.agg} needs every hospital's update in the clear, and SecAgg+ hides exactly that: "
                              "turn SecAgg+ off, or use agg=fedavg")
        if self.agg != "fedavg" and self.strategy == "fedprox":
            raise ConfigError("fedprox is a FedAvg variant: use it with agg=fedavg")
        if self.mia and self.attack != "none" and self.mia_hospital == self.attacker:
            raise ConfigError("mia-hospital must not be the attacker: a poisoned update says nothing about an honest hospital")

    @property
    def mia_active(self) -> bool:
        """The membership test needs the side channel to get the honest hospital's last update."""
        return bool(self.mia and self.side_dir)

    def public_dict(self) -> dict:
        """Settings with contract key names (dashes), without the file/folder paths."""
        return {k.replace("_", "-"): v for k, v in asdict(self).items() if k not in ("events_file", "inspect_dir", "side_dir")}


SIM_ENV = "ZE_RUN_CONFIG"  # simulation only: flwr.simulation.run_simulation takes no run_config, so sim.py passes it here


def from_context(run_config) -> RunConfig:
    """Settings of this run: Flower's run_config, overridden by the simulation's ZE_RUN_CONFIG (JSON) when sim.py set it."""
    mapping = dict(run_config or {})
    if os.environ.get(SIM_ENV):
        mapping.update(json.loads(os.environ[SIM_ENV]))
    return RunConfig.from_mapping(mapping)


def finite_or_none(x):
    """JSON-safe number: inf / nan become None (CONTRACTS.md section 3)."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None

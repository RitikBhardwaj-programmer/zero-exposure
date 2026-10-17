"""The four UCI Heart Disease hospitals, each kept as its own federated client.

Each hospital prepares its data alone, as a real one would: it fills missing values with ITS OWN training medians and
standardises with ITS OWN training mean and spread. Nothing about one hospital's data is used by another.
Label: 1 = heart disease (UCI "num" > 0), 0 = none. Split: 75 % train / 25 % test, stratified, seeded.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .config import HOSPITALS

N_FEATURES = 13
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "heart_disease"


@dataclass
class Hospital:
    pid: int
    name: str
    X_train: np.ndarray  # float32 (n, 13), standardised with this hospital's train statistics
    y_train: np.ndarray  # float32 (n, 1)
    X_test: np.ndarray
    y_test: np.ndarray


def data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR") or DEFAULT_DATA_DIR)


def read_raw(name: str, folder: Path | None = None) -> np.ndarray:
    """The 14 columns of processed.<name>.data as float, '?' -> nan."""
    path = (folder or data_dir()) / f"processed.{name}.data"
    rows = [[np.nan if v.strip() == "?" else float(v) for v in line.split(",")]
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    arr = np.asarray(rows, dtype=np.float64)
    if arr.shape[1] != N_FEATURES + 1:
        raise ValueError(f"{path}: expected {N_FEATURES + 1} columns, got {arr.shape[1]}")
    return arr


def prepare(pid: int, seed: int, folder: Path | None = None) -> Hospital:
    """Load, split and preprocess one hospital (only its own file is read)."""
    raw = read_raw(HOSPITALS[pid], folder)
    X, y = raw[:, :N_FEATURES], (raw[:, N_FEATURES] > 0).astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    med = np.nanmedian(X_tr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)  # a column missing for every train row (e.g. Switzerland's cholesterol)
    X_tr = np.where(np.isnan(X_tr), med, X_tr)
    X_te = np.where(np.isnan(X_te), med, X_te)
    mean, std = X_tr.mean(axis=0), X_tr.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)  # constant columns stay 0 after centring
    f32 = lambda a: np.ascontiguousarray(a, dtype=np.float32)
    return Hospital(pid, HOSPITALS[pid], f32((X_tr - mean) / std), f32(y_tr.reshape(-1, 1)),
                    f32((X_te - mean) / std), f32(y_te.reshape(-1, 1)))


def load_all(seed: int, allow_missing: bool | None = None, folder: Path | None = None) -> list[Hospital | None]:
    """All hospitals whose file exists. A deployment SuperNode holds only its own file: then the others are None."""
    if allow_missing is None:
        allow_missing = os.environ.get("ALLOW_MISSING_DATA") == "1"
    out = []
    for pid, name in enumerate(HOSPITALS):
        if not ((folder or data_dir()) / f"processed.{name}.data").exists():
            if not allow_missing:
                raise FileNotFoundError(f"missing processed.{name}.data in {folder or data_dir()} (see README: get the data)")
            out.append(None)
            continue
        out.append(prepare(pid, seed, folder))
    return out

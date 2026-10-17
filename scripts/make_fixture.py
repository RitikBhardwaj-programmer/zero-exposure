"""Write fixtures/sample_run/: a fake 3-round run that follows CONTRACTS.md sections 3-4 (random numbers, no training).

The dashboard is built against it before the core engine exists. Run: python scripts/make_fixture.py
"""
import json
from pathlib import Path

import numpy as np

P = 369  # model parameters (CONTRACTS.md section 1)
HOSPITALS = ["cleveland", "hungarian", "switzerland", "va"]
OUT = Path(__file__).resolve().parent.parent / "fixtures" / "sample_run"


def main():
    rng = np.random.default_rng(7)
    insp = OUT / "insp"
    insp.mkdir(parents=True, exist_ok=True)
    t = 1_790_000_000.0
    config = {"rounds": 3, "local-epochs": 5, "noise": 1.0, "clip": 1.0, "delta": 1e-5, "strategy": "fedavg", "mu": 0.1,
              "seed": 0, "secagg": 1, "agg": "fedavg", "trim-beta": 0.25, "attack": "labelflip", "attacker": 3, "boost": 10.0,
              "mia": 1, "mia-hospital": 0, "inspect-round": 1}
    events = [{"type": "baseline", "t": t, "hospitals": HOSPITALS, "n_train": [227, 220, 92, 150], "n_test": [76, 74, 31, 50],
               "central_acc": 0.81, "central_f1": 0.82, "local_per_hospital_acc": [0.79, 0.8, 0.84, 0.66], "config": config}]
    acc = 0.45
    for r in range(4):
        acc = acc if r == 0 else min(0.8, acc + 0.1)
        eps = [0.0] * 4 if r == 0 else [round(2.1 * r + i * 0.4, 3) for i in range(4)]
        events.append({"type": "round", "t": t + 1 + 40 * r, "round": r, "total_rounds": 3, "acc": acc, "f1": acc - 0.02,
                       "loss": 0.7 - 0.05 * r, "per_hospital_acc": [acc + d for d in (0.03, 0.01, -0.05, -0.1)],
                       "epsilon": eps, "max_abs_weight": 0.3 + 0.1 * r})
    events.append({"type": "attack", "t": t + 125, "attack": "labelflip", "attacker": 3, "attacker_name": "va", "boost": None,
                   "agg": "fedavg", "secagg": 1, "honest_acc": 0.68, "attacker_acc": 0.31, "mia_hospital": 0,
                   "mia_hospital_name": "cleveland", "mia_n": 76, "mia_auc_update": 0.49, "mia_adv_update": 0.08,
                   "mia_auc_global": 0.5, "mia_adv_global": 0.07, "clipped_rounds": 0, "max_scaled_attacker": 2.1})
    events.append({"type": "done", "t": t + 126})
    (OUT / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")

    plain = rng.normal(0, 0.3, P).astype(np.float32)
    np.save(insp / "plain.npy", plain)
    np.save(insp / "masked.npy", rng.integers(0, 2**32, P + 1, dtype=np.uint64).astype(np.uint32))
    np.save(insp / "aggregate.npy", (plain * 0.9).astype(np.float32))
    (insp / "inspect.json").write_text(json.dumps({"round": 1, "pid": 0, "n_params": P, "secagg": 1, "n_examples": 227,
                                                   "clipping_range": 8.0, "quantization_range": 4194304}), encoding="utf-8")
    for pid in range(4):
        stages = ["received", "trained", "masked", "sent"]
        lines = [json.dumps({"pid": pid, "round": 1, "stage": s, "t": t + 2 + pid + 5 * i}) for i, s in enumerate(stages)]
        (insp / f"stages_h{pid}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"fixture written to {OUT}")


if __name__ == "__main__":
    main()

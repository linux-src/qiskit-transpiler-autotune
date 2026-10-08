"""Predicting how much a circuit gains from extra transpiler seeds, and spending a
shared seed budget accordingly.

    python -m qta.headroom --runs results/dataset --circuits-dir results/circuits \
        --out results/headroom

Headroom of a (device, circuit) pair is the log-improvement of the best of all
recorded level 3 seeds over a typical single seed. A model trained on circuit
features predicts it, and the prediction decides how many seeds each circuit of a
batch receives. Evaluation is leave-one-family-out, so a family is never seen in
training when its circuits are scored.
"""

from __future__ import annotations

import argparse
import time
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut

from qta.circuits import load_suite
from qta.features import extract

OBJECTIVES = {
    # value whose larger is better
    "n2q": lambda runs: -np.log(runs.n_2q.clip(lower=1)),
    "esp": lambda runs: runs.log_esp,
}


def family(circuit_id: str) -> str:
    return circuit_id.rsplit("_", 1)[0]


def expected_best(values: np.ndarray, k: int) -> float:
    """Expected maximum of ``k`` values drawn without replacement from ``values``."""
    v = np.sort(values)
    n = len(v)
    k = min(k, n)
    # P(the i-th smallest is the maximum of the draw) = C(i-1, k-1) / C(n, k), 1-based i
    weights = np.array([comb(i - 1, k - 1) for i in range(1, n + 1)], dtype=float) / comb(n, k)
    return float(weights @ v)


def targets(runs: pd.DataFrame, objective: str) -> pd.DataFrame:
    score = OBJECTIVES[objective]
    rows = []
    for (device, circuit), r in runs.groupby(["device", "circuit"], sort=False):
        values = score(r).to_numpy()
        rows.append({
            "device": device,
            "circuit": circuit,
            "family": family(circuit),
            "headroom": values.max() - values.mean(),
            "values": values,
        })
    return pd.DataFrame(rows)


def feature_table(circuits: dict) -> pd.DataFrame:
    rows = []
    for cid, qc in circuits.items():
        start = time.perf_counter()
        f = extract(qc)
        rows.append({"circuit": cid, **f, "feature_time_s": time.perf_counter() - start})
    return pd.DataFrame(rows)


def cross_val_predict(table: pd.DataFrame, feature_cols: list[str], seed: int = 0) -> np.ndarray:
    X = table[feature_cols].to_numpy(dtype=float)
    y = table.headroom.to_numpy()
    pred = np.zeros_like(y)
    for train, test in LeaveOneGroupOut().split(X, y, table.family):
        model = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=seed)
        model.fit(X[train], y[train])
        pred[test] = model.predict(X[test])
    return pred


def allocate(priority: np.ndarray, total: int, cap: int) -> np.ndarray:
    """Give every circuit one seed, then hand out the rest greedily by ``priority / seeds``."""
    n = len(priority)
    seeds = np.ones(n, dtype=int)
    priority = np.clip(priority, 1e-9, None)
    for _ in range(max(total - n, 0)):
        gain = np.where(seeds < cap, priority / seeds, -np.inf)
        seeds[int(np.argmax(gain))] += 1
    return seeds


def budget_curve(table: pd.DataFrame, strategies: dict[str, np.ndarray],
                 per_circuit: list[int], cap: int) -> pd.DataFrame:
    rows = []
    for device, t in table.groupby("device"):
        values = list(t["values"])
        single = np.array([v.mean() for v in values])
        for k in per_circuit:
            total = k * len(t)
            plans = {"uniform": np.full(len(t), min(k, cap))}
            for name, priority in strategies.items():
                plans[name] = allocate(priority[t.index], total, cap)
            for name, seeds in plans.items():
                best = np.array([expected_best(v, s) for v, s in zip(values, seeds)])
                rows.append({
                    "device": device,
                    "seeds_per_circuit": k,
                    "strategy": name,
                    "mean_gain": float((best - single).mean()),
                    "seeds_used": int(seeds.sum()),
                })
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True, help="directory with baseline.csv")
    parser.add_argument("--circuits-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--objective", choices=list(OBJECTIVES), default="esp")
    parser.add_argument("--min-esp", type=float, default=0.0,
                        help="keep only pairs whose median level 3 ESP reaches this value")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    runs = pd.read_csv(args.runs / "baseline.csv")
    runs = runs[runs.level == 3]
    if args.min_esp > 0:
        median_esp = runs.groupby(["device", "circuit"]).esp.transform("median")
        runs = runs[median_esp >= args.min_esp]
    cap = int(runs.groupby(["device", "circuit"]).size().min())

    features_path = args.out / "features.csv"
    if features_path.exists():
        features = pd.read_csv(features_path)
    else:
        circuits = load_suite(args.circuits_dir, [(family(c), int(c.rsplit("_", 1)[1]))
                                                  for c in runs.circuit.unique()])
        features = feature_table(circuits)
        features.to_csv(features_path, index=False)
    feature_cols = [c for c in features.columns if c not in ("circuit", "feature_time_s")]

    table = targets(runs, args.objective).merge(features, on="circuit")
    devices = sorted(table.device.unique())
    for d in devices:
        table[f"device_{d}"] = (table.device == d).astype(float)
    model_cols = feature_cols + [f"device_{d}" for d in devices]

    table["pred"] = cross_val_predict(table, model_cols)
    table["pred_n2q_only"] = cross_val_predict(table, ["n_2q", "num_qubits"] + model_cols[-len(devices):])

    quality = []
    for device, t in table.groupby("device"):
        for col in ["pred", "pred_n2q_only"]:
            rho = spearmanr(t[col], t.headroom).statistic
            quality.append({"device": device, "predictor": col, "spearman": rho,
                            "mae": float((t[col] - t.headroom).abs().mean())})
    quality = pd.DataFrame(quality)

    curve = budget_curve(
        table,
        {
            "by_size": table.n_2q.to_numpy(dtype=float),
            "predicted": table.pred.to_numpy(),
            "oracle": table.headroom.to_numpy(),
        },
        per_circuit=[1, 2, 3, 4, 6, 8, 12],
        cap=cap,
    )

    tag = args.objective + (f"_minesp{args.min_esp:g}" if args.min_esp > 0 else "")
    table.drop(columns=["values"]).to_csv(args.out / f"headroom_{tag}.csv", index=False)
    quality.to_csv(args.out / f"quality_{tag}.csv", index=False)
    curve.to_csv(args.out / f"budget_{tag}.csv", index=False)

    with pd.option_context("display.width", 200):
        print(quality.round(3).to_string(index=False))
        print(curve.pivot_table(index=["device", "seeds_per_circuit"], columns="strategy",
                                values="mean_gain").round(4))
        print("feature extraction, s: median", round(features.feature_time_s.median(), 4),
              "max", round(features.feature_time_s.max(), 4))


if __name__ == "__main__":
    main()

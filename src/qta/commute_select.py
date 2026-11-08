"""Decide per circuit whether the commuting SABRE mode is worth half of the budget.

    python -m qta.commute_select features --out results/select --circuits-dir results/circuits
    python -m qta.commute_select report --features results/select/features.csv \
        --train results/sabre --test results/holdout --out results/select

The features come from the circuit after the init stage of level 3, which is what the router
sees. The threshold is picked on the training runs only and then applied unchanged to the
test runs. A circuit that is not selected spends the whole budget on plain seeds.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager

from qta.circuits import SUITES, get_device, load_suite
from qta.commute_order import levels, wire_axes
from qta.sabre_commute_report import BUDGETS, compare, geometric_mean

FEATURES = {"depth_ratio": -1, "commuting_share": 1}  # sign: which direction means more freedom


def is_two_qubit(inst) -> bool:
    return len(inst.qubits) == 2 and not inst.clbits and inst.operation.name != "barrier"


def depth_ratio(qc: QuantumCircuit) -> float:
    """Two-qubit depth when commuting pairs are unordered, relative to the written order."""
    _, preds = levels(qc)
    depth = [0] * len(preds)
    for i, need in enumerate(preds):
        depth[i] = max((depth[j] for j in need), default=0) + int(is_two_qubit(qc.data[i]))
    plain = qc.depth(is_two_qubit)
    return max(depth, default=0) / plain if plain else 1.0


def commuting_share(qc: QuantumCircuit) -> float:
    """Share of two-qubit gates that sit next to another one along the same axis on some wire."""
    groups: dict = {}
    marked: set[int] = set()
    for i, inst in enumerate(qc.data):
        for bit, axis in wire_axes(qc, inst).items():
            cur_axis, members = groups.get(bit, (None, []))
            if axis is None or axis != cur_axis:
                if len(members) > 1:
                    marked.update(members)
                cur_axis, members = axis, []
                groups[bit] = (cur_axis, members)
            if is_two_qubit(inst):
                members.append(i)
    marked.update(i for _, members in groups.values() if len(members) > 1 for i in members)
    total = sum(1 for inst in qc.data if is_two_qubit(inst))
    return len(marked) / total if total else 0.0


def features(args) -> None:
    rows = []
    backend = get_device("torino")
    for suite in ("dataset", "holdout"):
        circuits = load_suite(args.circuits_dir, [(n, s) for n, s in SUITES[suite] if s <= args.max_qubits])
        for cid, qc in circuits.items():
            init = generate_preset_pass_manager(3, backend=backend, seed_transpiler=0).init
            start = time.perf_counter()
            prepared = init.run(qc)
            ratio, share = depth_ratio(prepared), commuting_share(prepared)
            rows.append({"circuit": cid, "suite": suite, "depth_ratio": ratio, "commuting_share": share,
                         "time_s": time.perf_counter() - start})
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / "features.csv", index=False)


def load_table(runs: Path, feats: pd.DataFrame) -> dict[int, pd.DataFrame]:
    off = pd.read_csv(runs / "off" / "runs.csv")
    on = pd.read_csv(runs / "on" / "runs.csv")
    joined = off.merge(on, on=["device", "circuit", "seed", "num_qubits"], suffixes=("_off", "_on"))
    joined["family"] = joined.circuit.str.replace(r"_\d+$", "", regex=True)
    return {k: compare(joined, k).merge(feats, on="circuit") for k in BUDGETS}


def selected(table: pd.DataFrame, feature: str, threshold: float) -> pd.Series:
    return FEATURES[feature] * table[feature] >= FEATURES[feature] * threshold


def strategy_ratio(table: pd.DataFrame, mask: pd.Series) -> pd.Series:
    return table.portfolio_n.where(mask, table.plain_n) / table.plain_n


def fit(tables: dict[int, pd.DataFrame], feature: str) -> float:
    """Threshold with the lowest mean log ratio over all budgets; the middle one of a tie."""
    values = np.sort(tables[BUDGETS[0]][feature].unique())
    candidates = np.concatenate([[values[0] - 1e-9], (values[1:] + values[:-1]) / 2, [values[-1] + 1e-9]])
    scores = np.array([np.mean([np.log(strategy_ratio(t, selected(t, feature, c))).mean() for t in tables.values()])
                       for c in candidates])
    best = candidates[scores <= scores.min() + 1e-12]
    return float(best[len(best) // 2])


def evaluate(tables: dict[int, pd.DataFrame], feature: str, threshold: float, name: str) -> list[dict]:
    rows = []
    for k, table in tables.items():
        mask = selected(table, feature, threshold)
        ratio = strategy_ratio(table, mask)
        always = table.portfolio_n / table.plain_n
        rows.append({"set": name, "feature": feature, "threshold": threshold, "k": k, "circuits": len(table),
                     "selected": int(mask.sum()), "rule_n2q": geometric_mean(ratio), "always_n2q": geometric_mean(always),
                     "wins": int((ratio < 1).sum()), "losses": int((ratio > 1).sum()), "ties": int((ratio == 1).sum()),
                     "always_wins": int((always < 1).sum()), "always_losses": int((always > 1).sum())})
    return rows


def leave_one_family_out(tables: dict[int, pd.DataFrame], feature: str) -> list[dict]:
    rows = []
    for family in sorted(tables[BUDGETS[0]].family.unique()):
        train = {k: t[t.family != family] for k, t in tables.items()}
        held = {k: t[t.family == family] for k, t in tables.items()}
        threshold = fit(train, feature)
        for row in evaluate(held, feature, threshold, f"lofo:{family}"):
            rows.append({**row, "family": family})
    return rows


def report(args) -> None:
    feats = pd.read_csv(args.features)
    train = load_table(args.train, feats[feats.suite == "dataset"].drop(columns="suite"))
    test = load_table(args.test, feats[feats.suite == "holdout"].drop(columns="suite"))
    rows, cv = [], []
    for feature in FEATURES:
        threshold = fit(train, feature)
        rows += evaluate(train, feature, threshold, "train") + evaluate(test, feature, threshold, "test")
        cv += leave_one_family_out(train, feature)
    args.out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "summary.csv", index=False)
    print(table.round(4).to_string(index=False))
    folds = pd.DataFrame(cv)
    folds.to_csv(args.out / "leave_one_family_out.csv", index=False)
    pooled = folds.groupby(["feature", "k"]).apply(
        lambda g: pd.Series({"rule": np.exp(np.average(np.log(g.rule_n2q), weights=g.circuits)),
                             "always": np.exp(np.average(np.log(g.always_n2q), weights=g.circuits)),
                             "wins": g.wins.sum(), "losses": g.losses.sum()}), include_groups=False)
    print(pooled.round(4).to_string())
    print(folds.groupby("feature").threshold.describe().round(4).to_string())


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("features")
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--circuits-dir", type=Path, required=True)
    f.add_argument("--max-qubits", type=int, default=40)
    f.set_defaults(func=features)
    r = sub.add_parser("report")
    r.add_argument("--features", type=Path, required=True)
    r.add_argument("--train", type=Path, required=True)
    r.add_argument("--test", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.set_defaults(func=report)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

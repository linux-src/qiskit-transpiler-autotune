"""Equal-time comparison of commutation-aware SABRE with plain seed search.

    python -m qta.sabre_commute_report --runs results/sabre --out results/sabre/report

For each circuit the budget is the compile time of K seeds of plain level 3. A strategy runs
until the next compilation would not fit; every strategy gets at least one compilation.
The portfolio alternates the commuting and the plain mode, starting with the commuting one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BUDGETS = (2, 4, 10, 20)
EXECUTABLE_LOG_ESP = np.log(1e-3)


def geometric_mean(values) -> float:
    return float(np.exp(np.log(values).mean()))


def within_budget(runs: list[tuple[float, int, float]], budget: float) -> tuple[int, float]:
    spent = 0.0
    n_2q, log_esp = [], []
    for seconds, n, esp in runs:
        if n_2q and spent + seconds > budget:
            break
        spent += seconds
        n_2q.append(n)
        log_esp.append(esp)
    return min(n_2q), max(log_esp)


def circuit_row(group: pd.DataFrame, k: int) -> dict:
    group = group.sort_values("seed")
    on = list(zip(group.time_s_on, group.n_2q_on, group.log_esp_on))
    off = list(zip(group.time_s_off, group.n_2q_off, group.log_esp_off))
    budget = sum(t for t, _, _ in off[:k])
    alternating = [run for pair in zip(on, off) for run in pair]
    plain_n, plain_esp = min(n for _, n, _ in off[:k]), max(e for _, _, e in off[:k])
    portfolio_n, portfolio_esp = within_budget(alternating, budget)
    commuting_n, commuting_esp = within_budget(on, budget)
    return {"plain_n": plain_n, "plain_esp": plain_esp, "portfolio_n": portfolio_n, "portfolio_esp": portfolio_esp,
            "commuting_n": commuting_n, "commuting_esp": commuting_esp}


def compare(joined: pd.DataFrame, k: int) -> pd.DataFrame:
    rows = []
    for (device, circuit, qubits, family), group in joined.groupby(["device", "circuit", "num_qubits", "family"]):
        rows.append({"device": device, "circuit": circuit, "num_qubits": qubits, "family": family,
                     **circuit_row(group, k)})
    table = pd.DataFrame(rows)
    return table[table.plain_n > 0].copy()


def summary(table: pd.DataFrame, k: int) -> dict:
    ratio = table.portfolio_n / table.plain_n
    gain = table.portfolio_esp - table.plain_esp
    executable = table.plain_esp > EXECUTABLE_LOG_ESP
    return {"k": k, "circuits": len(table),
            "portfolio_n2q": geometric_mean(ratio),
            "wins": int((ratio < 1).sum()), "losses": int((ratio > 1).sum()), "ties": int((ratio == 1).sum()),
            "commuting_n2q": geometric_mean(table.commuting_n / table.plain_n),
            "portfolio_esp_median": float(np.exp(gain).median()),
            "portfolio_esp_executable": float(np.exp(gain[executable].mean()))}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    off = pd.read_csv(args.runs / "off" / "runs.csv")
    on = pd.read_csv(args.runs / "on" / "runs.csv")
    joined = off.merge(on, on=["device", "circuit", "seed", "num_qubits"], suffixes=("_off", "_on"))
    joined["family"] = joined.circuit.str.replace(r"_\d+$", "", regex=True)

    args.out.mkdir(parents=True, exist_ok=True)
    overall = pd.DataFrame([summary(compare(joined, k), k) for k in BUDGETS])
    overall.to_csv(args.out / "summary.csv", index=False)
    print(overall.round(4).to_string(index=False))

    table = compare(joined, 20)
    table.to_csv(args.out / "per_circuit_k20.csv", index=False)
    table["ratio"] = table.portfolio_n / table.plain_n
    table["commuting_ratio"] = table.commuting_n / table.plain_n
    table["esp_gain"] = np.exp(table.portfolio_esp - table.plain_esp)
    families = table.groupby("family").agg(
        circuits=("ratio", "size"), portfolio=("ratio", geometric_mean), commuting=("commuting_ratio", geometric_mean),
        wins=("ratio", lambda s: int((s < 1).sum())), losses=("ratio", lambda s: int((s > 1).sum())),
        esp_median=("esp_gain", "median")).sort_values("portfolio")
    families.to_csv(args.out / "families_k20.csv")
    print(families.round(3).to_string())
    sizes = table.groupby(pd.cut(table.num_qubits, [0, 10, 20, 30, 40]), observed=True).agg(
        circuits=("ratio", "size"), portfolio=("ratio", geometric_mean), commuting=("commuting_ratio", geometric_mean))
    print(sizes.round(3).to_string())
    print(table.groupby("device").ratio.apply(geometric_mean).round(4).to_dict())
    print("total compile time on/off:", round(joined.time_s_on.sum() / joined.time_s_off.sum(), 3))


if __name__ == "__main__":
    main()

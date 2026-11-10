"""Equal-time comparison of several modes of the patched SABRE core with plain seed search.

    python -m qta.sabre_modes_report --runs results/sabre_absorb --out results/sabre_absorb/report

`--runs` holds one directory per mode (`off`, `absorb`, `commute`, `commute_absorb`), each with
the `runs.csv` of `qta.sabre_commute run`. For each circuit the budget is the compile time of
K seeds of plain level 3 (`off`). A strategy cycles through its modes, taking seeds in order,
until the next compilation would not fit; every strategy gets at least one compilation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BUDGETS = (1, 2, 4, 10, 20)
EXECUTABLE_LOG_ESP = np.log(1e-3)
STRATEGIES = {
    "absorb": ("absorb",),
    "commute_absorb": ("commute_absorb",),
    "commute+plain": ("commute", "off"),
    "absorb+commute_absorb": ("absorb", "commute_absorb"),
}


def geometric_mean(values) -> float:
    return float(np.exp(np.log(values).mean()))


def bootstrap_interval(ratios, repeats: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95 % interval of the geometric mean, resampling circuits."""
    logs = np.log(np.asarray(ratios, dtype=float))
    rng = np.random.default_rng(seed)
    means = logs[rng.integers(0, len(logs), (repeats, len(logs)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(np.exp(low)), float(np.exp(high))


def within_budget(runs: list[tuple[float, int, float]], budget: float) -> tuple[int, float, int]:
    spent = 0.0
    n_2q, log_esp = [], []
    for seconds, n, esp in runs:
        if n_2q and spent + seconds > budget:
            break
        spent += seconds
        n_2q.append(n)
        log_esp.append(esp)
    return min(n_2q), max(log_esp), len(n_2q)


def interleave(per_mode: dict[str, list], modes: tuple[str, ...]) -> list:
    return [run for group in zip(*(per_mode[m] for m in modes)) for run in group]


def compare(runs: dict[str, pd.DataFrame], strategies: dict[str, tuple[str, ...]], k: int) -> pd.DataFrame:
    keys = ["device", "circuit", "num_qubits"]
    grouped = {mode: dict(iter(frame.sort_values("seed").groupby(keys))) for mode, frame in runs.items()}
    rows = []
    for key, plain in grouped["off"].items():
        per_mode = {mode: list(zip(groups[key].time_s, groups[key].n_2q, groups[key].log_esp))
                    for mode, groups in grouped.items()}
        off = per_mode["off"][:k]
        row = dict(zip(keys, key), plain_n=min(n for _, n, _ in off), plain_esp=max(e for _, _, e in off))
        budget = sum(t for t, _, _ in off)
        for name, modes in strategies.items():
            row[f"{name}_n"], row[f"{name}_esp"], row[f"{name}_runs"] = within_budget(interleave(per_mode, modes), budget)
        rows.append(row)
    table = pd.DataFrame(rows)
    table["family"] = table.circuit.str.replace(r"_\d+$", "", regex=True)
    return table[table.plain_n > 0].copy()


def summary(table: pd.DataFrame, strategies, k: int) -> list[dict]:
    rows = []
    executable = table.plain_esp > EXECUTABLE_LOG_ESP
    for name in strategies:
        ratio = table[f"{name}_n"] / table.plain_n
        gain = table[f"{name}_esp"] - table.plain_esp
        low, high = bootstrap_interval(ratio)
        rows.append({"k": k, "strategy": name, "circuits": len(table), "n2q": geometric_mean(ratio),
                     "n2q_low": low, "n2q_high": high,
                     "wins": int((ratio < 1).sum()), "losses": int((ratio > 1).sum()), "ties": int((ratio == 1).sum()),
                     "esp_executable": float(np.exp(gain[executable].mean())),
                     "esp_median": float(np.exp(gain).median()),
                     "runs_per_seed": float((table[f"{name}_runs"] / k).mean())})
    return rows


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--main-strategy", default="absorb+commute_absorb")
    args = parser.parse_args(argv)

    runs = {path.parent.name: pd.read_csv(path) for path in sorted(args.runs.glob("*/runs.csv"))}
    strategies = {name: modes for name, modes in STRATEGIES.items() if all(m in runs for m in modes)}
    args.out.mkdir(parents=True, exist_ok=True)

    overall = pd.DataFrame([row for k in BUDGETS for row in summary(compare(runs, strategies, k), strategies, k)])
    overall.to_csv(args.out / "summary.csv", index=False)
    print(overall.round(4).to_string(index=False))

    total = {mode: frame.time_s.sum() for mode, frame in runs.items()}
    times = pd.Series({mode: seconds / total["off"] for mode, seconds in total.items()}, name="time_to_plain")
    times.to_csv(args.out / "time.csv")
    print(times.round(3).to_dict())

    table = compare(runs, strategies, 20)
    table.to_csv(args.out / "per_circuit_k20.csv", index=False)
    name = args.main_strategy
    table["ratio"] = table[f"{name}_n"] / table.plain_n
    table["esp_gain"] = np.exp(table[f"{name}_esp"] - table.plain_esp)
    columns = {s: (f"{s}_ratio", geometric_mean) for s in strategies}
    for s in strategies:
        table[f"{s}_ratio"] = table[f"{s}_n"] / table.plain_n
    families = table.groupby("family").agg(
        circuits=("ratio", "size"), wins=("ratio", lambda s: int((s < 1).sum())),
        losses=("ratio", lambda s: int((s > 1).sum())), esp_median=("esp_gain", "median"), **columns,
    ).sort_values(name)
    families.to_csv(args.out / "families_k20.csv")
    print(families.round(3).to_string())
    sizes = table.groupby(pd.cut(table.num_qubits, [0, 10, 20, 30, 40]), observed=True).agg(
        circuits=("ratio", "size"), **columns)
    sizes.to_csv(args.out / "sizes_k20.csv")
    print(sizes.round(3).to_string())
    devices = table.groupby("device").agg(**columns)
    devices.to_csv(args.out / "devices_k20.csv")
    print(devices.round(4).to_string())


if __name__ == "__main__":
    main()

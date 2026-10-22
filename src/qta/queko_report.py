"""Distance to the optimum on QUEKO: level 3, SABRE seeds and SABRE configuration search.

    python -m qta.queko_report --runs ../data/queko --out ../data/queko_report

Per circuit and method: the depth ratio (depth / optimal depth) of a typical run (median
over runs), of the best run, and for the configuration search the best run within the
compile time of all ``sabre`` seeds of that circuit (``fair``). Ratios are summarised by
the geometric mean over circuits of a benchmark set and device, together with the share
of circuits where the optimum (ratio 1) was reached.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def per_circuit(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (circuit, bench_set, device, optimal), r in runs.groupby(
            ["circuit", "set", "device", "optimal_depth"]):
        row = {"circuit": circuit, "set": bench_set, "device": device, "optimal_depth": optimal}
        for method, m in r.groupby("method"):
            m = m.sort_values("run")
            row[f"{method}_median"] = float(m.depth_ratio.median())
            row[f"{method}_best"] = float(m.depth_ratio.min())
            row[f"{method}_swaps_median"] = float(m.n_swaps.median())
            row[f"{method}_time"] = float(m.time_s.sum())
        sabre_time = row.get("sabre_time")
        search = r[r.method == "sabre_search"].sort_values("run")
        if sabre_time is not None and not search.empty:
            within = search[search.time_s.cumsum() <= sabre_time]
            # at least one configuration is always tried
            within = within if not within.empty else search.iloc[:1]
            row["sabre_search_fair"] = float(within.depth_ratio.min())
            row["sabre_search_fair_trials"] = len(within)
        rows.append(row)
    return pd.DataFrame(rows)


def _gmean(x: pd.Series) -> float:
    return float(np.exp(np.log(x).mean()))


def summary(pc: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["level3_median", "sabre_median", "sabre_best", "sabre_search_fair",
                        "sabre_search_best"] if c in pc]
    rows = []
    for (bench_set, device), d in pc.groupby(["set", "device"]):
        row = {"set": bench_set, "device": device, "circuits": len(d)}
        for c in cols:
            row[c] = _gmean(d[c])
            row[f"{c}_optimal_share"] = float((d[c] <= 1 + 1e-12).mean())
        if "sabre_search_time" in d:
            row["search_time_vs_sabre"] = _gmean(d.sabre_search_time / d.sabre_time)
            row["fair_trials_mean"] = float(d.sabre_search_fair_trials.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    runs = pd.read_csv(args.runs / "queko.csv")
    pc = per_circuit(runs)
    pc.to_csv(args.out / "per_circuit.csv", index=False)
    s = summary(pc)
    s.to_csv(args.out / "summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

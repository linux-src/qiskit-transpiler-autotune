"""Comparing level 3 on pruned devices with plain level 3.

    python -m qta.threshold_report --runs results/threshold --control results/dataset \
        --out results/threshold_report

The control is the level 3 run of the same circuits and seeds on the full device
(``tau = inf``). Ratios are geometric means over circuits, above 1 means the pruned
device is better. A *portfolio* splits the same number of seeds over several taus
and keeps the best run, so it costs as much as the control when a pruned run takes
as long as a plain one (checked by ``time_ratio``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

INF = float("inf")


def load(runs_dir: Path, control_dir: Path) -> pd.DataFrame:
    runs = pd.read_csv(runs_dir / "threshold.csv")
    control = pd.read_csv(control_dir / "baseline.csv")
    control = control[control.level == 3].drop(columns="level").assign(tau=INF)
    pairs = runs[["device", "circuit"]].drop_duplicates()
    control = control.merge(pairs, on=["device", "circuit"])
    return pd.concat([control, runs], ignore_index=True)


def per_circuit(runs: pd.DataFrame) -> pd.DataFrame:
    return (
        runs.groupby(["device", "circuit", "tau"])
        .agg(
            best_log_esp=("log_esp", "max"),
            mean_esp=("esp", "mean"),
            median_esp=("esp", "median"),
            median_n2q=("n_2q", "median"),
            dead=("log_esp", lambda s: int((~np.isfinite(s)).sum())),
            time_s=("time_s", "sum"),
            seeds=("seed", "count"),
        )
        .reset_index()
    )


def _gm_log(diff: pd.Series) -> float:
    diff = diff[np.isfinite(diff)]
    return float(np.exp(diff.mean())) if len(diff) else np.nan


def aggregate(summary: pd.DataFrame, min_esp: float) -> pd.DataFrame:
    control = summary[summary.tau == INF].set_index(["device", "circuit"])
    executable = control.median_esp >= min_esp
    rows = []
    for (device, tau), s in summary[summary.tau != INF].groupby(["device", "tau"]):
        s = s.set_index(["device", "circuit"])
        c = control.loc[s.index]
        ex = executable.loc[s.index]
        gain = s.best_log_esp - c.best_log_esp
        rows.append({
            "device": device,
            "tau": tau,
            "circuits": len(s),
            "executable": int(ex.sum()),
            "best_esp_ratio_all": _gm_log(gain),
            "best_esp_ratio_exec": _gm_log(gain[ex]),
            "mean_esp_ratio_exec": _gm_log(np.log(s.mean_esp[ex] / c.mean_esp[ex])),
            "better_exec": int((gain[ex] > 1e-9).sum()),
            "worse_exec": int((gain[ex] < -1e-9).sum()),
            "n2q_ratio": _gm_log(np.log(c.median_n2q / s.median_n2q.clip(lower=1))),
            "time_ratio": _gm_log(np.log(s.time_s / c.time_s)),
            "dead_runs": int(s.dead.sum()),
            "control_dead_runs": int(c.dead.sum()),
        })
    return pd.DataFrame(rows)


def portfolio(runs: pd.DataFrame, taus: list[float], seeds: int, min_esp: float,
              reps: int = 200, rng_seed: int = 0) -> pd.DataFrame:
    """Best log ESP when ``seeds`` runs are split evenly over ``taus``, against
    the best of ``seeds`` runs of plain level 3. Splits are drawn at random
    from the recorded seeds, ``reps`` times.
    """
    rng = np.random.default_rng(rng_seed)
    per_tau = seeds // len(taus)
    rows = []
    for (device, circuit), r in runs.groupby(["device", "circuit"]):
        control = r[r.tau == INF]
        if len(control) < seeds or not all((r.tau == t).sum() >= per_tau for t in taus):
            continue
        values = {t: r.log_esp[r.tau == t].to_numpy() for t in taus}
        best = np.full(reps, -np.inf)
        for t in taus:
            draws = np.array([rng.choice(values[t], per_tau, replace=False) for _ in range(reps)])
            best = np.maximum(best, draws.max(axis=1))
        rows.append({
            "device": device,
            "circuit": circuit,
            "executable": control.esp.median() >= min_esp,
            "gain": float(np.mean(best) - control.log_esp.max()),
            "oracle_gain": max(values[t].max() for t in taus if t != INF) - control.log_esp.max()
            if any(t != INF for t in taus) else 0.0,
        })
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-esp", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args(argv)

    runs = load(args.runs, args.control)
    summary = per_circuit(runs)
    args.out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out / "per_circuit.csv", index=False)

    agg = aggregate(summary, args.min_esp)
    agg.to_csv(args.out / "aggregate.csv", index=False)
    print(agg.to_string(index=False))

    pruned = sorted(t for t in runs.tau.unique() if t != INF)
    portfolios = {f"inf+{t:g}": [INF, t] for t in pruned}
    portfolios["all"] = [INF, *pruned]
    portfolios["pruned"] = pruned
    rows = []
    for name, taus in portfolios.items():
        p = portfolio(runs, taus, args.seeds, args.min_esp)
        for device, d in p.groupby("device"):
            ex = d[d.executable]
            rows.append({
                "portfolio": name, "device": device, "circuits": len(d), "executable": len(ex),
                "best_esp_ratio_all": float(np.exp(d.gain[np.isfinite(d.gain)].mean())),
                "best_esp_ratio_exec": float(np.exp(ex.gain.mean())),
                "oracle_ratio_exec": float(np.exp(ex.oracle_gain.mean())),
            })
    port = pd.DataFrame(rows)
    port.to_csv(args.out / "portfolio.csv", index=False)
    print(port.to_string(index=False))


if __name__ == "__main__":
    main()

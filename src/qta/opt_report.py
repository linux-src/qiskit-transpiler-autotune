"""How much of the gain from tuning the optimization stage survives seed variance.

    python -m qta.opt_report --runs results/opt_search --control results/dataset \
        --out results/opt_report --objective n2q

For every (device, circuit) pair three comparisons against plain level 3:

* ``naive``: best configuration at one seed against level 3 at that same seed, which is
  how a tuner is usually evaluated when the transpiler seed is fixed;
* ``fair``: best configuration at one seed against the best of as many level 3 seeds
  as fit into the same compile time (taken from the control runs);
* ``seeds_only``: best of all recorded level 3 seeds against level 3 at one seed, the
  gain available without any tuning.

Gains are mean log-ratios (``n2q``: log of the two-qubit count ratio; ``esp``: log ESP
difference), averaged over the recorded seeds of the tuning runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from qta.headroom import OBJECTIVES, expected_best


def _score(runs: pd.DataFrame, objective: str) -> pd.Series:
    return OBJECTIVES[objective](runs)


def compare(runs: pd.DataFrame, control: pd.DataFrame, objective: str) -> pd.DataFrame:
    rows = []
    for (device, circuit), r in runs.groupby(["device", "circuit"]):
        c = control[(control.device == device) & (control.circuit == circuit)]
        if c.empty:
            continue
        c_score = _score(c, objective).to_numpy()
        # Level 3 time from the same session as the tuning runs, so both sides of the
        # budget are measured under the same conditions.
        c_time = r.time_s[r.config == 0].mean()
        r = r.assign(score=_score(r, objective).to_numpy())
        for seed, s in r.groupby("seed"):
            level3 = s[s.config == 0]
            if level3.empty:
                continue
            base = float(level3.score.iloc[0])
            best = float(s.score.max())
            budget = s.time_s.sum()
            k = int(np.clip(round(budget / c_time), 1, len(c_score)))
            rows.append({
                "device": device,
                "circuit": circuit,
                "seed": seed,
                "naive": best - base,
                "fair": best - expected_best(c_score, k),
                "seeds_only": float(c_score.max()) - base,
                "control_seeds": k,
                "budget_capped": round(budget / c_time) > len(c_score),
                "best_config": int(s.config[s.score.idxmax()]),
                "level3_is_best": bool(base >= best - 1e-12),
            })
    return pd.DataFrame(rows)


def aggregate(cmp: pd.DataFrame) -> pd.DataFrame:
    per_pair = cmp.groupby(["device", "circuit"]).agg(
        naive=("naive", "mean"), fair=("fair", "mean"), seeds_only=("seeds_only", "mean"),
        control_seeds=("control_seeds", "mean"), budget_capped=("budget_capped", "any"),
        level3_is_best=("level3_is_best", "mean"),
    ).reset_index()
    rows = []
    for device, d in per_pair.groupby("device"):
        rows.append({
            "device": device,
            "circuits": len(d),
            "naive_ratio": float(np.exp(d.naive.mean())),
            "fair_ratio": float(np.exp(d.fair.mean())),
            "seeds_only_ratio": float(np.exp(d.seeds_only.mean())),
            "naive_wins": int((d.naive > 1e-9).sum()),
            "fair_wins": int((d.fair > 1e-9).sum()),
            "fair_losses": int((d.fair < -1e-9).sum()),
            "mean_control_seeds": float(d.control_seeds.mean()),
            "budget_capped": int(d.budget_capped.sum()),
            "level3_best_share": float(d.level3_is_best.mean()),
        })
    return pd.DataFrame(rows)


def config_ranking(runs: pd.DataFrame, objective: str) -> pd.DataFrame:
    """Mean gain of each configuration over level 3 at the same seed, i.e. whether one
    fixed configuration is a better default than level 3."""
    r = runs.assign(score=_score(runs, objective).to_numpy())
    base = r[r.config == 0].set_index(["device", "circuit", "seed"]).score
    r = r.join(base.rename("base"), on=["device", "circuit", "seed"])
    r["gain"] = r.score - r.base
    r["time_ratio"] = r.time_s / r.join(
        r[r.config == 0].set_index(["device", "circuit", "seed"]).time_s.rename("base_time"),
        on=["device", "circuit", "seed"],
    ).base_time
    out = r.groupby(["device", "config"]).agg(
        gain=("gain", "mean"), wins=("gain", lambda g: int((g > 1e-9).sum())),
        losses=("gain", lambda g: int((g < -1e-9).sum())), time_ratio=("time_ratio", "median"),
    ).reset_index()
    out["ratio"] = np.exp(out.gain)
    return out.sort_values(["device", "gain"], ascending=[True, False])


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--objective", choices=list(OBJECTIVES), default="n2q")
    args = parser.parse_args(argv)

    runs = pd.read_csv(args.runs / "opt_search.csv")
    control = pd.read_csv(args.control / "baseline.csv")
    control = control[control.level == 3]
    args.out.mkdir(parents=True, exist_ok=True)

    cmp = compare(runs, control, args.objective)
    cmp.to_csv(args.out / f"compare_{args.objective}.csv", index=False)
    agg = aggregate(cmp)
    agg.to_csv(args.out / f"aggregate_{args.objective}.csv", index=False)
    print(agg.to_string(index=False))

    ranking = config_ranking(runs, args.objective)
    ranking.to_csv(args.out / f"configs_{args.objective}.csv", index=False)
    print(ranking.groupby("device").head(5).to_string(index=False))


if __name__ == "__main__":
    main()

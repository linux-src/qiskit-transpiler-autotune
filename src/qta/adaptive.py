"""Adaptive allocation of transpiler seeds across a batch of circuits.

    python -m qta.adaptive --runs results/dataset --out results/adaptive --objective n2q

Every (device, circuit) pair has a pool of recorded level 3 runs. A policy repeatedly
asks for one more run of some circuit and receives the next run of that pool in a
random order, so policies are replayed on real data without new transpilations.
All policies get the same compile-time budget: the expected time of giving every
circuit ``k`` seeds. Gain is the mean, over circuits, of the best value obtained
minus the mean value of a single run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from qta.headroom import OBJECTIVES, family


@dataclass
class Pool:
    values: np.ndarray
    times: np.ndarray


@dataclass
class State:
    """Runs a policy has drawn so far for one circuit."""

    pool: Pool
    order: np.ndarray
    taken: int = 0
    seen: list = field(default_factory=list)

    @property
    def exhausted(self) -> bool:
        return self.taken >= len(self.order)

    @property
    def mean_time(self) -> float:
        return float(np.mean(self.pool.times[self.order[: self.taken]])) if self.taken else np.nan

    def draw(self) -> float:
        i = self.order[self.taken]
        self.taken += 1
        self.seen.append(self.pool.values[i])
        return float(self.pool.times[i])


class Replay:
    def __init__(self, pools: list[Pool], rng: np.random.Generator):
        self.states = [State(p, rng.permutation(len(p.values))) for p in pools]
        self.spent = 0.0

    def draw(self, i: int) -> None:
        self.spent += self.states[i].draw()

    def gain(self) -> float:
        return float(np.mean([max(s.seen) - s.pool.values.mean() for s in self.states]))


def uniform(replay: Replay, budget: float, k: int, **_) -> None:
    for _ in range(k):
        for i, s in enumerate(replay.states):
            if not s.exhausted:
                replay.draw(i)


def static(replay: Replay, budget: float, priority: np.ndarray, **_) -> None:
    """Fixed priorities known in advance: one seed each, then greedily by priority / seeds."""
    states = replay.states
    for i in range(len(states)):
        replay.draw(i)
    priority = np.clip(priority, 1e-12, None)
    score = np.array([-np.inf if s.exhausted else priority[i] for i, s in enumerate(states)])
    while replay.spent < budget:
        i = int(np.argmax(score))
        if score[i] == -np.inf:
            break
        replay.draw(i)
        score[i] = -np.inf if states[i].exhausted else priority[i] / states[i].taken


def expected_improvement(seen: list[float], prior_std: float, prior_weight: float) -> float:
    """Expected gain of one more draw over the current best, assuming normal values.

    The sample variance is shrunk towards ``prior_std`` because a circuit is judged
    after very few runs.
    """
    x = np.asarray(seen)
    n = len(x)
    var = (np.var(x, ddof=1) * (n - 1) if n > 1 else 0.0) + prior_weight * prior_std**2
    std = np.sqrt(var / (n - 1 + prior_weight))
    if std <= 0:
        return 0.0
    z = (x.mean() - x.max()) / std
    return float(std * (norm.pdf(z) + z * norm.cdf(z)))


def adaptive_ei(replay: Replay, budget: float, warmup: int = 2, prior_weight: float = 2.0,
                per_second: bool = True, probe_fraction: float = 0.0, **_) -> None:
    """Give every circuit ``warmup`` seeds, then repeatedly the one with the largest
    expected improvement (per second of compile time when ``per_second``).

    With ``warmup=1`` the typical spread cannot be seen yet, so the cheapest
    ``probe_fraction`` of circuits get a second seed to estimate it.
    """
    states = replay.states
    for _ in range(warmup):
        for i, s in enumerate(states):
            if not s.exhausted:
                replay.draw(i)
    if probe_fraction > 0:
        cheapest = np.argsort([s.mean_time for s in states])
        for i in cheapest[: max(2, int(probe_fraction * len(states)))]:
            if len(states[i].seen) < 2 and not states[i].exhausted:
                replay.draw(int(i))
    stds = [np.std(s.seen, ddof=1) for s in states if len(s.seen) > 1]
    prior_std = float(np.mean(stds)) if stds else 0.0

    def score(s: State) -> float:
        if s.exhausted:
            return -np.inf
        ei = expected_improvement(s.seen, prior_std, prior_weight)
        return ei / s.mean_time if per_second else ei

    scores = np.array([score(s) for s in states])
    while replay.spent < budget:
        i = int(np.argmax(scores))
        if scores[i] <= 0:
            break
        replay.draw(i)
        scores[i] = score(states[i])


def successive_halving(replay: Replay, budget: float, warmup: int = 2, **_) -> None:
    """Rounds: everyone alive gets one more seed; after each round drop the half whose
    observed spread is smallest, until one circuit or the budget remains."""
    states = replay.states
    alive = [i for i, s in enumerate(states) if not s.exhausted]
    for _ in range(warmup):
        for i in alive:
            replay.draw(i)
    while replay.spent < budget and alive:
        spread = {i: np.ptp(states[i].seen) / np.sqrt(states[i].taken) for i in alive}
        alive = sorted(alive, key=spread.get, reverse=True)[: max(1, len(alive) // 2)]
        for i in alive:
            if replay.spent >= budget:
                break
            if states[i].exhausted:
                continue
            replay.draw(i)
        alive = [i for i in alive if not states[i].exhausted]
        if len(alive) == 1:
            while replay.spent < budget and not states[alive[0]].exhausted:
                replay.draw(alive[0])
            break


def load_pools(runs: pd.DataFrame, objective: str) -> tuple[pd.DataFrame, list[Pool]]:
    score = OBJECTIVES[objective]
    meta, pools = [], []
    for (device, circuit), r in runs.groupby(["device", "circuit"], sort=False):
        values = score(r).to_numpy(dtype=float)
        meta.append({"device": device, "circuit": circuit, "family": family(circuit),
                     "headroom": values.max() - values.mean(), "n_2q": r.n_2q.median()})
        pools.append(Pool(values, r.time_s.to_numpy(dtype=float)))
    return pd.DataFrame(meta), pools


def evaluate(meta: pd.DataFrame, pools: list[Pool], ks: list[int], reps: int,
             seed: int = 0) -> pd.DataFrame:
    rows = []
    for device, m in meta.groupby("device"):
        dev_pools = [pools[i] for i in m.index]
        mean_times = np.array([p.times.mean() for p in dev_pools])
        policies = {
            "uniform": (uniform, {}),
            "by_size": (static, {"priority": m.n_2q.to_numpy(dtype=float)}),
            "by_cost": (static, {"priority": 1.0 / mean_times}),
            "adaptive_ei": (adaptive_ei, {"warmup": 1, "probe_fraction": 0.25}),
            "adaptive_ei_w2": (adaptive_ei, {"warmup": 2}),
            "successive_halving": (successive_halving, {}),
            "oracle": (static, {"priority": m.headroom.to_numpy() / mean_times}),
        }
        for k in ks:
            budget = k * mean_times.sum()
            for name, (policy, kwargs) in policies.items():
                rng = np.random.default_rng([seed, k])
                gains, spent = [], []
                for _ in range(reps):
                    replay = Replay(dev_pools, rng)
                    policy(replay, budget, k=k, **kwargs)
                    gains.append(replay.gain())
                    spent.append(replay.spent / budget)
                rows.append({
                    "device": device, "seeds_per_circuit": k, "policy": name,
                    "gain_mean": float(np.mean(gains)),
                    "gain_ci95": float(1.96 * np.std(gains, ddof=1) / np.sqrt(reps)),
                    "time_vs_budget": float(np.mean(spent)),
                })
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True, help="directory with baseline.csv")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--objective", choices=list(OBJECTIVES), default="n2q")
    parser.add_argument("--min-esp", type=float, default=0.0)
    parser.add_argument("--k", type=int, nargs="+", default=[2, 3, 4, 6, 8])
    parser.add_argument("--reps", type=int, default=200)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    runs = pd.read_csv(args.runs / "baseline.csv")
    runs = runs[runs.level == 3]
    if args.min_esp > 0:
        median_esp = runs.groupby(["device", "circuit"]).esp.transform("median")
        runs = runs[median_esp >= args.min_esp]

    meta, pools = load_pools(runs, args.objective)
    result = evaluate(meta, pools, args.k, args.reps)
    tag = args.objective + (f"_minesp{args.min_esp:g}" if args.min_esp > 0 else "")
    result.to_csv(args.out / f"adaptive_{tag}.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        for value in ["gain_mean", "time_vs_budget"]:
            print(result.pivot_table(index=["device", "seeds_per_circuit"], columns="policy",
                                     values=value).round(4))
        print("max ci95:", round(result.gain_ci95.max(), 4))


if __name__ == "__main__":
    main()

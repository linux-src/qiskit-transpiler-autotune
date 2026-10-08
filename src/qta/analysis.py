"""Compare random search with the level 3 baseline at an equal compile-time budget.

    python -m qta.analysis --baseline results/baseline --search results/random \
        --out results/report --objective esp

The budget of a (device, circuit) pair is the total time of all level 3 seeds. Within
it, level 3 keeps its best seed and random search keeps the best trial it reached.
Both pick runs by the same objective: fewest two-qubit gates (ties broken by depth)
or highest estimated success probability.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OBJECTIVES = {
    "n2q": {"sort": ["n_2q", "depth"], "ascending": [True, True]},
    "esp": {"sort": ["log_esp", "n_2q"], "ascending": [False, True]},
}
SELECTED = ["o3_best", "rs_budget", "rs_all"]
COLORS = {"o3_best": "#2a78d6", "rs_budget": "#eb6834"}
LABELS = {"o3_best": "O3, лучший из N seed", "rs_budget": "Случайный поиск, тот же бюджет"}
XLABELS = {
    "n2q": "Сокращение двухкубитных вентилей относительно медианы O3, %",
    "esp": "Рост ESP относительно медианы O3, %",
}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def best(runs: pd.DataFrame, objective: str) -> pd.Series | None:
    if runs.empty:
        return None
    spec = OBJECTIVES[objective]
    return runs.sort_values(spec["sort"], ascending=spec["ascending"], kind="stable").iloc[0]


def log_gain(summary: pd.DataFrame, key: str, ref: str, objective: str) -> pd.Series:
    """Log of the improvement factor of ``key`` over ``ref``; positive means better."""
    if objective == "n2q":
        return np.log(summary[f"{ref}_n2q"] / summary[f"{key}_n2q"])
    return summary[f"{key}_log_esp"] - summary[f"{ref}_log_esp"]


def summarize(baseline: pd.DataFrame, search: pd.DataFrame, objective: str = "n2q") -> pd.DataFrame:
    o3 = baseline[baseline.level == 3]
    rows = []
    for (device, circuit), runs in o3.groupby(["device", "circuit"], sort=False):
        trials = search[(search.device == device) & (search.circuit == circuit)].sort_values("trial")
        if trials.empty:
            continue
        budget = runs.time_s.sum()
        within = trials[trials.time_s.cumsum() <= budget]
        row = {
            "device": device,
            "circuit": circuit,
            "num_qubits": runs.num_qubits.iloc[0],
            "o3_seeds": len(runs),
            "o3_median_n2q": runs.n_2q.median(),
            "o3_min_n2q": runs.n_2q.min(),
            "o3_max_n2q": runs.n_2q.max(),
            "o3_median_log_esp": runs.log_esp.median(),
            "o3_median_time_s": runs.time_s.median(),
            "budget_s": budget,
            "rs_budget_trials": len(within),
            "rs_all_trials": len(trials),
            "rs_all_time_s": trials.time_s.sum(),
        }
        for key, pool in zip(SELECTED, [runs, within, trials]):
            chosen = best(pool, objective)
            for col, name in [("n_2q", "n2q"), ("depth", "depth"), ("log_esp", "log_esp")]:
                row[f"{key}_{name}"] = chosen[col] if chosen is not None else np.nan
        row["rs_all_config"] = _config_str(best(trials, objective))
        rows.append(row)
    return pd.DataFrame(rows)


def _config_str(row: pd.Series) -> str:
    keys = ["use_vf2", "max_iterations", "layout_trials", "swap_trials",
            "routing", "heuristic", "routing_trials"]
    return " ".join(f"{k}={row[k]}" for k in keys)


def aggregate(summary: pd.DataFrame, objective: str = "n2q") -> pd.DataFrame:
    """Geometric-mean improvement factors per device (above 1 means better)."""
    out = []
    for device, s in summary.groupby("device"):
        valid = s.dropna(subset=["rs_budget_n2q"])
        vs_best = log_gain(valid, "rs_budget", "o3_best", objective)
        out.append({
            "device": device,
            "objective": objective,
            "circuits": len(s),
            "o3_best_vs_median": _gm(log_gain(s, "o3_best", "o3_median", objective)),
            "rs_budget_vs_median": _gm(log_gain(valid, "rs_budget", "o3_median", objective)),
            "rs_budget_vs_o3_best": _gm(vs_best),
            "rs_all_vs_o3_best": _gm(log_gain(s, "rs_all", "o3_best", objective)),
            "rs_all_time_vs_budget": _gm(np.log(s.rs_all_time_s / s.budget_s)),
            "rs_budget_better": int((vs_best > 1e-12).sum()),
            "rs_budget_equal": int((vs_best.abs() <= 1e-12).sum()),
            "rs_budget_worse": int((vs_best < -1e-12).sum()),
            "rs_budget_missing": int(s.rs_budget_n2q.isna().sum()),
        })
    return pd.DataFrame(out)


def _gm(log_ratio: pd.Series) -> float:
    log_ratio = log_ratio.dropna()
    return float(np.exp(log_ratio.mean())) if len(log_ratio) else np.nan


def _style(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=TEXT, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_reduction(summary: pd.DataFrame, path: Path, objective: str = "n2q") -> None:
    devices = list(summary.device.unique())
    order = summary.groupby("circuit", sort=False).num_qubits.first().index[::-1]
    fig, axes = plt.subplots(1, len(devices), figsize=(5 * len(devices), 0.3 * len(order) + 1.6),
                             sharey=True, squeeze=False)
    h = 0.38
    for ax, device in zip(axes[0], devices):
        s = summary[summary.device == device].set_index("circuit").reindex(order)
        y = np.arange(len(order))
        for i, key in enumerate(["o3_best", "rs_budget"]):
            lg = log_gain(s, key, "o3_median", objective)
            gain = 100 * (1 - np.exp(-lg)) if objective == "n2q" else 100 * np.expm1(lg)
            ax.barh(y + (0.5 - i) * h, gain, height=h - 0.04, color=COLORS[key],
                    label=LABELS[key], edgecolor="white", linewidth=1)
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.set_yticks(y, order)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_title(device, color=TEXT, loc="left", fontsize=11)
        _style(ax)
    fig.supxlabel(XLABELS[objective], color=MUTED, fontsize=10)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, labelcolor=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_seed_spread(baseline: pd.DataFrame, path: Path) -> None:
    o3 = baseline[baseline.level == 3].copy()
    o3["rel"] = o3.n_2q / o3.groupby(["device", "circuit"]).n_2q.transform("median")
    devices = list(o3.device.unique())
    order = list(o3.groupby("circuit", sort=False).num_qubits.first().index[::-1])
    fig, axes = plt.subplots(1, len(devices), figsize=(5 * len(devices), 0.3 * len(order) + 1.4),
                             sharey=True, squeeze=False)
    rng = np.random.default_rng(0)
    for ax, device in zip(axes[0], devices):
        d = o3[o3.device == device]
        for y, circuit in enumerate(order):
            vals = d[d.circuit == circuit].rel.to_numpy()
            ax.scatter(vals, y + rng.uniform(-0.15, 0.15, len(vals)), s=10,
                       color=COLORS["o3_best"], alpha=0.6, linewidths=0)
        ax.axvline(1, color=MUTED, linewidth=0.8)
        ax.set_yticks(range(len(order)), order)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_title(device, color=TEXT, loc="left", fontsize=11)
        _style(ax)
    fig.supxlabel("Число двухкубитных вентилей O3 / медиана по seed", color=MUTED, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figures", type=Path, default=None)
    parser.add_argument("--objective", choices=list(OBJECTIVES), default="n2q")
    args = parser.parse_args(argv)

    baseline = pd.read_csv(args.baseline / "baseline.csv")
    search = pd.read_csv(args.search / "random_search.csv")
    figures = args.figures or args.out
    args.out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    levels = baseline.groupby(["device", "level"]).agg(
        n_2q_median=("n_2q", "median"), esp_median=("esp", "median"),
        time_s_median=("time_s", "median")).round(4)
    summary = summarize(baseline, search, args.objective)
    agg = aggregate(summary, args.objective)

    tag = args.objective
    levels.to_csv(args.out / "levels.csv")
    summary.to_csv(args.out / f"summary_{tag}.csv", index=False)
    agg.to_csv(args.out / f"aggregate_{tag}.csv", index=False)
    plot_reduction(summary, figures / f"rs_vs_o3_{tag}.png", tag)
    plot_seed_spread(baseline, figures / "o3_seed_spread.png")

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(agg.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

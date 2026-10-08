"""Compare random search with the level 3 baseline at an equal compile-time budget.

    python -m qta.analysis --baseline results/baseline --search results/random --out results/report

The budget of a (device, circuit) pair is the total time of all level 3 seeds. Within
it, level 3 keeps its best seed and random search keeps the best trial it reached.
Runs are ranked by two-qubit gate count, ties broken by depth.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RANK = ["n_2q", "depth"]
COLORS = {"o3_best": "#2a78d6", "rs_budget": "#eb6834"}
LABELS = {"o3_best": "O3, лучший из N seed", "rs_budget": "Случайный поиск, тот же бюджет"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def best(runs: pd.DataFrame) -> pd.Series | None:
    if runs.empty:
        return None
    return runs.sort_values(RANK, kind="stable").iloc[0]


def summarize(baseline: pd.DataFrame, search: pd.DataFrame) -> pd.DataFrame:
    o3 = baseline[baseline.level == 3]
    rows = []
    for (device, circuit), runs in o3.groupby(["device", "circuit"], sort=False):
        trials = search[(search.device == device) & (search.circuit == circuit)].sort_values("trial")
        if trials.empty:
            continue
        budget = runs.time_s.sum()
        within = trials[trials.time_s.cumsum() <= budget]
        o3_best, rs_budget, rs_all = best(runs), best(within), best(trials)
        rows.append({
            "device": device,
            "circuit": circuit,
            "num_qubits": runs.num_qubits.iloc[0],
            "o3_seeds": len(runs),
            "o3_median_n2q": runs.n_2q.median(),
            "o3_min_n2q": runs.n_2q.min(),
            "o3_max_n2q": runs.n_2q.max(),
            "o3_median_esp": runs.esp.median(),
            "o3_median_time_s": runs.time_s.median(),
            "budget_s": budget,
            "o3_best_n2q": o3_best.n_2q,
            "o3_best_depth": o3_best.depth,
            "o3_best_esp": o3_best.esp,
            "rs_budget_trials": len(within),
            "rs_budget_n2q": rs_budget.n_2q if rs_budget is not None else np.nan,
            "rs_budget_depth": rs_budget.depth if rs_budget is not None else np.nan,
            "rs_budget_esp": rs_budget.esp if rs_budget is not None else np.nan,
            "rs_all_trials": len(trials),
            "rs_all_time_s": trials.time_s.sum(),
            "rs_all_n2q": rs_all.n_2q,
            "rs_all_esp": rs_all.esp,
            "rs_all_config": _config_str(rs_all),
        })
    return pd.DataFrame(rows)


def _config_str(row: pd.Series) -> str:
    keys = ["use_vf2", "max_iterations", "layout_trials", "swap_trials",
            "routing", "heuristic", "routing_trials"]
    return " ".join(f"{k}={row[k]}" for k in keys)


def aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    out = []
    for device, s in summary.groupby("device"):
        valid = s.dropna(subset=["rs_budget_n2q"])
        out.append({
            "device": device,
            "circuits": len(s),
            "o3_best_vs_median": _gm(s.o3_best_n2q / s.o3_median_n2q),
            "rs_budget_vs_median": _gm(valid.rs_budget_n2q / valid.o3_median_n2q),
            "rs_budget_vs_o3_best": _gm(valid.rs_budget_n2q / valid.o3_best_n2q),
            "rs_all_vs_o3_best": _gm(s.rs_all_n2q / s.o3_best_n2q),
            "rs_all_time_vs_budget": _gm(s.rs_all_time_s / s.budget_s),
            "rs_budget_better": int((valid.rs_budget_n2q < valid.o3_best_n2q).sum()),
            "rs_budget_equal": int((valid.rs_budget_n2q == valid.o3_best_n2q).sum()),
            "rs_budget_worse": int((valid.rs_budget_n2q > valid.o3_best_n2q).sum()),
            "rs_budget_missing": int(s.rs_budget_n2q.isna().sum()),
        })
    return pd.DataFrame(out)


def _gm(ratio: pd.Series) -> float:
    ratio = ratio.dropna()
    return float(np.exp(np.log(ratio).mean())) if len(ratio) else np.nan


def _style(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=TEXT, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_reduction(summary: pd.DataFrame, path: Path) -> None:
    devices = list(summary.device.unique())
    order = summary.groupby("circuit", sort=False).num_qubits.first().index[::-1]
    fig, axes = plt.subplots(1, len(devices), figsize=(5 * len(devices), 0.3 * len(order) + 1.6),
                             sharey=True, squeeze=False)
    h = 0.38
    for ax, device in zip(axes[0], devices):
        s = summary[summary.device == device].set_index("circuit").reindex(order)
        y = np.arange(len(order))
        for i, key in enumerate(["o3_best", "rs_budget"]):
            gain = 100 * (1 - s[f"{key}_n2q"] / s.o3_median_n2q)
            ax.barh(y + (0.5 - i) * h, gain, height=h - 0.04, color=COLORS[key],
                    label=LABELS[key], edgecolor="white", linewidth=1)
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.set_yticks(y, order)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_title(device, color=TEXT, loc="left", fontsize=11)
        _style(ax)
    fig.supxlabel("Сокращение двухкубитных вентилей относительно медианы O3, %",
                  color=MUTED, fontsize=10)
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
    args = parser.parse_args(argv)

    baseline = pd.read_csv(args.baseline / "baseline.csv")
    search = pd.read_csv(args.search / "random_search.csv")
    figures = args.figures or args.out
    args.out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    levels = baseline.groupby(["device", "level"]).agg(
        n_2q_median=("n_2q", "median"), esp_median=("esp", "median"),
        time_s_median=("time_s", "median")).round(4)
    summary = summarize(baseline, search)
    agg = aggregate(summary)

    levels.to_csv(args.out / "levels.csv")
    summary.to_csv(args.out / "summary.csv", index=False)
    agg.to_csv(args.out / "aggregate.csv", index=False)
    plot_reduction(summary, figures / "rs_vs_o3.png")
    plot_seed_spread(baseline, figures / "o3_seed_spread.png")

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(agg.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

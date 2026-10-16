"""Figures for the thesis text, drawn from the recorded experiment data.

    python -m qta.figures --data ../data --out ../figures/thesis

Each figure is a PNG for print: one axis, thin marks, series told apart by marker and
line style as well as colour so they survive greyscale printing. Labels are in Russian
because the thesis is.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from qta.headroom import OBJECTIVES, expected_best  # noqa: E402

# First four slots of the validated categorical palette, in their fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
LINESTYLES = ["-", "--", "-.", ":"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DEVICE_NAMES = {"sherbrooke": "Sherbrooke", "torino": "Torino"}


def _style(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def _percent(v: float) -> str:
    return "0%" if abs(v) < 0.05 else f"{v:+.1f}%"


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(path)


def best_of_k(dataset: pd.DataFrame, max_k: int = 20) -> pd.DataFrame:
    """Mean reduction of two-qubit gates of the best of k level 3 seeds against one seed."""
    score = OBJECTIVES["n2q"]
    rows = []
    for (device, _), r in dataset.groupby(["device", "circuit"]):
        values = score(r).to_numpy()
        base = values.mean()
        for k in range(1, max_k + 1):
            rows.append({"device": device, "k": k, "gain": expected_best(values, k) - base})
    out = pd.DataFrame(rows).groupby(["device", "k"]).gain.mean().reset_index()
    out["reduction_pct"] = (1 - np.exp(-out.gain)) * 100
    return out


def figure_best_of_k(dataset: pd.DataFrame, path: Path) -> None:
    curve = best_of_k(dataset)
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    for i, (device, d) in enumerate(curve.groupby("device")):
        ax.plot(d.k, d.reduction_pct, color=SERIES[i], marker=MARKERS[i], markersize=4,
                linestyle=LINESTYLES[i], linewidth=1.5, label=DEVICE_NAMES[device])
        last = d.iloc[-1]
        ax.annotate(f"{last.reduction_pct:.1f}%", (last.k, last.reduction_pct),
                    textcoords="offset points", xytext=(6, -3), color=INK, fontsize=9)
    _style(ax, "Число запусков O3 с разными seed, k",
           "Сокращение двухкубитных вентилей, %")
    ax.set_xticks([1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    _save(fig, path)


def figure_allocation(adaptive: pd.DataFrame, device: str, path: Path) -> None:
    policies = [("uniform", "поровну"), ("by_cost", "по стоимости"),
                ("adaptive_ei", "адаптивная (EI)"), ("oracle", "оракул")]
    d = adaptive[adaptive.device == device]
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    ends = []
    for i, (policy, label) in enumerate(policies):
        p = d[d.policy == policy].sort_values("seeds_per_circuit")
        pct = (1 - np.exp(-p.gain_mean)) * 100
        ax.plot(p.seeds_per_circuit, pct, color=SERIES[i], marker=MARKERS[i], markersize=4,
                linestyle=LINESTYLES[i], linewidth=1.5, label=label)
        ends.append([p.seeds_per_circuit.iloc[-1], pct.iloc[-1], label])
    # The yellow slot is below 3:1 contrast on white, so every line gets a direct label;
    # labels of lines that end close together are pushed apart vertically.
    ends.sort(key=lambda e: e[1])
    gap = 0.06 * max(e[1] for e in ends)
    placed = []
    for x_end, y_end, label in ends:
        y = y_end if not placed else max(y_end, placed[-1] + gap)
        placed.append(y)
        ax.text(x_end * 1.04, y, label, color=INK, fontsize=8, va="center")
    ax.set_xticks(sorted(d.seeds_per_circuit.unique()))
    ax.set_xlim(right=d.seeds_per_circuit.max() * 1.35)
    _style(ax, "Бюджет: время k запусков O3 на схему, k",
           "Сокращение двухкубитных вентилей, %")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    _save(fig, path)


def figure_threshold_by_size(per_circuit: pd.DataFrame, dataset: pd.DataFrame, tau: float,
                             path: Path) -> None:
    inf = per_circuit[np.isinf(per_circuit.tau)].set_index(["device", "circuit"])
    t = per_circuit[per_circuit.tau == tau].set_index(["device", "circuit"])
    size = dataset.groupby(["device", "circuit"]).n_2q.median()
    gain = (t.best_log_esp - inf.best_log_esp).rename("gain").to_frame().join(size)
    gain = gain[np.isfinite(gain.gain)]
    bins = [0, 50, 200, 1000, np.inf]
    labels = ["< 50", "50–200", "200–1000", "> 1000"]
    gain["bucket"] = pd.cut(gain.n_2q, bins, labels=labels)
    med = gain.groupby(["device", "bucket"], observed=False).gain.median().unstack(0)
    pct = (np.exp(med) - 1) * 100

    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    x = np.arange(len(labels))
    width = 0.36
    for i, device in enumerate(pct.columns):
        ax.bar(x + (i - 0.5) * (width + 0.02), pct[device].to_numpy(), width,
               color=SERIES[i], edgecolor="white", linewidth=1,
               hatch=["", "///"][i], label=DEVICE_NAMES[device])
    for i, device in enumerate(pct.columns):
        for xi, v in zip(x, pct[device].to_numpy()):
            if np.isfinite(v):
                ax.annotate(_percent(v), (xi + (i - 0.5) * (width + 0.02), v), textcoords="offset points",
                            xytext=(0, 3 if v >= 0 else -11), ha="center", color=INK, fontsize=8)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.margins(y=0.12)
    ax.set_xticks(x, labels)
    _style(ax, "Медианное число двухкубитных вентилей схемы у O3",
           f"Медианное изменение ESP, % (τ = {tau:g})")
    ax.legend(frameon=False)
    _save(fig, path)


def figure_tuning_vs_seeds(aggregate: pd.DataFrame, objective: str, path: Path) -> None:
    """Gain of tuning the optimization stage, judged naively and fairly, next to seeds alone."""
    columns = [("naive_ratio", "настройка,\nтот же seed"), ("fair_ratio", "настройка против\nлучшего из N seed"),
               ("seeds_only_ratio", "только перебор\nseed O3")]
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    x = np.arange(len(columns))
    width = 0.36
    for i, (_, row) in enumerate(aggregate.iterrows()):
        values = [(row[c] - 1) * 100 for c, _ in columns]
        bars = ax.bar(x + (i - 0.5) * (width + 0.02), values, width, color=SERIES[i], edgecolor="white",
                      linewidth=1, hatch=["", "///"][i], label=DEVICE_NAMES[row.device])
        for bar, v in zip(bars, values):
            ax.annotate(_percent(v), (bar.get_x() + bar.get_width() / 2, v), textcoords="offset points",
                        xytext=(0, 3 if v >= 0 else -11), ha="center", color=INK, fontsize=8)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(x, [label for _, label in columns])
    metric = "по числу 2q-вентилей" if objective == "n2q" else "по ESP"
    ax.margins(y=0.12)
    _style(ax, "", f"Выигрыш над O3 {metric}, %")
    ax.legend(frameon=False)
    _save(fig, path)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "legend.fontsize": 9})
    dataset = pd.read_csv(args.data / "dataset" / "baseline.csv")
    dataset = dataset[dataset.level == 3]
    figure_best_of_k(dataset, args.out / "best_of_k_seeds.png")

    adaptive = pd.read_csv(args.data / "adaptive" / "adaptive_n2q.csv")
    for device in adaptive.device.unique():
        figure_allocation(adaptive, device, args.out / f"seed_allocation_{device}.png")

    per_circuit = args.data / "threshold_report" / "per_circuit.csv"
    if per_circuit.exists():
        figure_threshold_by_size(pd.read_csv(per_circuit), dataset, 1.0,
                                 args.out / "threshold_gain_by_size.png")

    for objective in ("n2q", "esp"):
        aggregate = args.data / "opt_report" / f"aggregate_{objective}.csv"
        if aggregate.exists():
            figure_tuning_vs_seeds(pd.read_csv(aggregate), objective,
                                   args.out / f"optimization_tuning_vs_seeds_{objective}.png")


if __name__ == "__main__":
    main()

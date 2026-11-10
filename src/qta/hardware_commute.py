"""Run circuits compiled with and without commutation-aware SABRE on an IBM QPU.

    python -m qta.hardware_commute compile --backend ibm_marrakesh --out results/hw --circuits-dir results/circuits
    QTA_SABRE_COMMUTE=1 python -m qta.hardware_commute compile --backend ibm_marrakesh --out results/hw --circuits-dir results/circuits
    python -m qta.hardware_commute plan --out results/hw
    python -m qta.hardware_commute send --out results/hw
    python -m qta.hardware_commute collect --out results/hw

Each instance is one block of seeds. Plain takes the best ESP among the block's plain
compilations; the portfolio alternates the commuting and the plain mode within the compile
time of that same block and also takes the best ESP. Every chosen circuit is submitted several
times, in shuffled order, so that drift of the device does not favour one of the strategies.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import qpy
from qiskit.quantum_info import hellinger_fidelity
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

from qta.circuits import get_device, load_suite
from qta.esp_check import ideal_distribution, with_measurements
from qta.hardware import qpu_seconds
from qta.metrics import measure
from qta.runner import write_environment

MODES = ("off", "on")


def mode_name() -> str:
    """Directory of a compile run, from the switches of the patched SABRE core."""
    absorb = bool(os.environ.get("QTA_SABRE_ABSORB"))
    if os.environ.get("QTA_SABRE_COMMUTE"):
        return "commute_absorb" if absorb else "on"
    return "absorb" if absorb else "off"


def get_backend(name: str, dry_run: bool):
    return get_device(name) if dry_run else QiskitRuntimeService().backend(name)


def compile_mode(args) -> None:
    mode = mode_name()
    backend = get_backend(args.backend, args.dry_run)
    suite = [(c.rsplit("_", 1)[0], int(c.rsplit("_", 1)[1])) for c in args.circuits] if args.circuits \
        else [("qaoa", size) for size in args.sizes]
    circuits = load_suite(args.circuits_dir, suite)
    rows, compiled = [], []
    for cid, qc in circuits.items():
        measured = with_measurements(qc)
        for seed in range(args.instances * args.block):
            start = time.perf_counter()
            out = generate_preset_pass_manager(3, backend=backend, seed_transpiler=seed).run(measured)
            elapsed = time.perf_counter() - start
            m = measure(out, backend.target)
            rows.append({"index": len(compiled), "circuit": cid, "instance": seed // args.block, "seed": seed,
                         "n_2q": m.n_2q, "depth": m.depth, "log_esp": m.log_esp, "time_s": elapsed})
            compiled.append(out)
        print(mode, cid, flush=True)
    directory = args.out / mode
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(directory / "runs.csv", index=False)
    with (directory / "compiled.qpy").open("wb") as f:
        qpy.dump(compiled, f)
    calibration = None if args.dry_run else str(backend.properties().last_update_date)
    write_environment(directory, experiment="hardware_commute", backend=backend.name, args=vars(args),
                      calibration=calibration)


def best_within_budget(runs: list[tuple[str, pd.Series]], budget: float) -> tuple[str, pd.Series]:
    """Best ESP among the runs that fit the budget; the first run always fits."""
    spent, chosen = 0.0, None
    for mode, run in runs:
        if chosen is not None and spent + run.time_s > budget:
            break
        spent += run.time_s
        if chosen is None or run.log_esp > chosen[1].log_esp:
            chosen = (mode, run)
    return chosen


def choose(off: pd.DataFrame, on: pd.DataFrame, portfolio: dict[str, pd.DataFrame] | None = None
           ) -> dict[str, tuple[str, pd.Series]]:
    """Best plain run and best portfolio run within the compile time of the plain block.

    The portfolio cycles through its modes in the given order; by default it alternates the
    commuting (`on`) and the plain mode.
    """
    portfolio = portfolio or {"on": on, "off": off}
    off = off.sort_values("seed")
    plain_runs = [("off", r) for _, r in off.iterrows()]
    budget = off.time_s.sum()
    rows = {mode: [r for _, r in frame.sort_values("seed").iterrows()] for mode, frame in portfolio.items()}
    cycled = [(mode, rows[mode][i]) for i in range(min(map(len, rows.values()))) for mode in rows]
    return {"plain": best_within_budget(plain_runs, budget), "portfolio": best_within_budget(cycled, budget)}


def load_compiled(out: Path, modes=MODES) -> dict[str, list]:
    compiled = {}
    for mode in modes:
        with (out / mode / "compiled.qpy").open("rb") as f:
            compiled[mode] = qpy.load(f)
    return compiled


def plan(args) -> None:
    off = pd.read_csv(args.out / "off" / "runs.csv")
    modes = {mode: pd.read_csv(args.out / mode / "runs.csv") for mode in args.portfolio}
    compiled = load_compiled(args.out, ("off", *args.portfolio))
    if args.min_esp:
        best = off.groupby("circuit").log_esp.max()
        off = off[off.circuit.isin(best[best >= np.log(args.min_esp)].index)]
    rows, submitted = [], []
    for (circuit, instance), block in off.groupby(["circuit", "instance"]):
        same = {mode: frame[(frame.circuit == circuit) & (frame.instance == instance)] for mode, frame in modes.items()}
        for strategy, (mode, run) in choose(block, None, same).items():
            for repeat in range(args.repeats):
                rows.append({"circuit": circuit, "instance": instance, "strategy": strategy, "repeat": repeat,
                             "mode": mode, "seed": int(run.seed), "n_2q": int(run.n_2q), "depth": int(run.depth),
                             "log_esp": run.log_esp})
                submitted.append(compiled[mode][int(run["index"])])
    order = np.random.default_rng(args.shuffle_seed).permutation(len(rows))
    table = pd.DataFrame(rows).iloc[order].reset_index(drop=True)
    table.insert(0, "pub", range(len(table)))
    shuffled = [submitted[i] for i in order]

    ideal = {}
    for cid, qc in load_suite(args.circuits_dir, [(c.rsplit("_", 1)[0], int(c.rsplit("_", 1)[1]))
                                                  for c in table.circuit.unique()]).items():
        ideal[cid] = ideal_distribution(with_measurements(qc))
    table.to_csv(args.out / "plan.csv", index=False)
    (args.out / "ideal.json").write_text(json.dumps(ideal))
    with (args.out / "submitted.qpy").open("wb") as f:
        qpy.dump(shuffled, f)
    (args.out / "shots.json").write_text(json.dumps({"shots": args.shots}))
    view = table[table.repeat == 0].groupby(["circuit", "strategy"]).agg(
        n_2q=("n_2q", "mean"), esp=("log_esp", lambda s: float(np.exp(s).mean())), patched=("mode", lambda s: int((s != "off").sum())))
    print(view.round(3).to_string())
    print(f"{len(table)} pubs, {args.shots} shots each")


def send(args) -> None:
    env = json.loads((args.out / "off" / "environment.json").read_text())
    dry_run = env["args"]["dry_run"]
    backend = get_backend(env["args"]["backend"], dry_run)
    shots = json.loads((args.out / "shots.json").read_text())["shots"]
    with (args.out / "submitted.qpy").open("rb") as f:
        compiled = qpy.load(f)
    print(f"~{qpu_seconds(compiled, backend, shots):.0f} s of shots")
    job = SamplerV2(mode=backend).run(compiled, shots=shots)
    (args.out / "job.json").write_text(json.dumps(
        {"job_id": job.job_id(), "backend": backend.name, "shots": shots, "dry_run": dry_run,
         "submitted": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=2))
    print("job", job.job_id())
    if dry_run:
        evaluate(args.out, job.result())


def evaluate(out: Path, result) -> None:
    table = pd.read_csv(out / "plan.csv")
    with (out / "submitted.qpy").open("rb") as f:
        compiled = qpy.load(f)
    ideal = json.loads((out / "ideal.json").read_text())
    fidelity = []
    for row in table.itertuples():
        counts = getattr(result[row.pub].data, compiled[row.pub].cregs[0].name).get_counts()
        fidelity.append(hellinger_fidelity(ideal[row.circuit], counts))
    table.assign(fidelity=fidelity).to_csv(out / "runs.csv", index=False)
    report(out)


def paired(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per circuit and instance: the mean of the repeats for both strategies."""
    wide = runs.groupby(["circuit", "instance", "strategy"]).agg(
        fidelity=("fidelity", "mean"), n_2q=("n_2q", "first"), log_esp=("log_esp", "first"),
        mode=("mode", "first")).unstack("strategy")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    return wide.reset_index()


def bootstrap_interval(values: np.ndarray, rng, draws: int = 10000) -> tuple[float, float]:
    means = rng.choice(values, (draws, len(values))).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def report(out: Path) -> None:
    table = paired(pd.read_csv(out / "runs.csv"))
    table["gain"] = table.fidelity_portfolio / table.fidelity_plain - 1
    table["n2q_ratio"] = table.n_2q_portfolio / table.n_2q_plain
    table["esp_ratio"] = np.exp(table.log_esp_portfolio - table.log_esp_plain)
    table.to_csv(out / "paired.csv", index=False)
    rng = np.random.default_rng(0)
    groups = [("all", table), *((c, g) for c, g in table.groupby("circuit"))]
    rows = []
    for name, group in groups:
        low, high = bootstrap_interval(group.gain.to_numpy(), rng)
        rows.append({"circuit": name, "instances": len(group), "n2q_plain": group.n_2q_plain.mean(),
                     "n2q_portfolio": group.n_2q_portfolio.mean(), "n2q_ratio": np.exp(np.log(group.n2q_ratio).mean()),
                     "esp_ratio": np.exp(np.log(group.esp_ratio).mean()),
                     "f_plain": group.fidelity_plain.mean(), "f_portfolio": group.fidelity_portfolio.mean(),
                     "gain_percent": 100 * group.gain.mean(), "gain_low": 100 * low, "gain_high": 100 * high,
                     "better": int((group.gain > 0).sum()), "worse": int((group.gain < 0).sum())})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.round(3).to_string(index=False))


def collect(args) -> None:
    meta = json.loads((args.out / "job.json").read_text())
    job = QiskitRuntimeService().job(meta["job_id"])
    print(f"job {meta['job_id']}: {job.status()}")
    if job.status() != "DONE":
        return
    (args.out / "usage.json").write_text(json.dumps(job.metrics(), indent=2, default=str))
    evaluate(args.out, job.result())


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("compile")
    c.add_argument("--backend", required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--circuits-dir", type=Path, required=True)
    c.add_argument("--sizes", type=int, nargs="+", default=[6, 8, 10, 12, 16])
    c.add_argument("--circuits", nargs="+", help="circuit ids such as qpeexact_8; replaces --sizes")
    c.add_argument("--instances", type=int, default=3)
    c.add_argument("--block", type=int, default=20)
    c.add_argument("--dry-run", action="store_true")
    c.set_defaults(func=compile_mode)
    p = sub.add_parser("plan")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--circuits-dir", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--shots", type=int, default=4000)
    p.add_argument("--min-esp", type=float, default=0.0)
    p.add_argument("--shuffle-seed", type=int, default=0)
    p.add_argument("--portfolio", nargs="+", default=["on", "off"],
                   help="modes the portfolio cycles through, as compiled into OUT/<mode>")
    p.set_defaults(func=plan)
    for name, func in (("send", send), ("collect", collect)):
        s = sub.add_parser(name)
        s.add_argument("--out", type=Path, required=True)
        s.set_defaults(func=func)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

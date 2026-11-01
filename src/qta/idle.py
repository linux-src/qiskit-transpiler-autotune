"""ESP with a penalty for idle time, checked on a recorded hardware run.

    python -m qta.idle --run results/hw/run1

Gate-and-readout ESP ignores that a qubit decoheres while it waits for others. Here the
transpiled circuit is scheduled as late as possible with gate lengths from the saved
calibration, and every active qubit's idle time ``t`` (from its first operation to the end
of the circuit, minus the time it is busy) multiplies ESP by ``exp(-t / T)``, with ``T``
being the qubit's T2 (or T1 for comparison).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from qiskit import QuantumCircuit, qpy
from qiskit.transpiler import InstructionDurations, PassManager
from qiskit.transpiler.passes import ALAPScheduleAnalysis

from qta.esp_check import rank_agreement


def load_calibration(path: Path) -> tuple[InstructionDurations, dict[int, dict[str, float]]]:
    """Gate lengths and per-qubit T1/T2 (seconds) from ``BackendProperties.to_dict()``."""
    props = json.loads(Path(path).read_text())
    durations = []
    for gate in props["gates"]:
        for p in gate["parameters"]:
            if p["name"] == "gate_length":
                durations.append((gate["gate"], tuple(gate["qubits"]), p["value"] * 1e-9, "s"))
    coherence = {}
    for q, params in enumerate(props["qubits"]):
        values = {p["name"]: p["value"] * 1e-6 for p in params if p["name"] in ("T1", "T2")}
        coherence[q] = values
    return InstructionDurations(durations), coherence


def idle_times(circuit: QuantumCircuit, durations: InstructionDurations) -> dict[int, float]:
    """Idle seconds of every qubit that has at least one operation, under ALAP scheduling."""
    pm = PassManager([ALAPScheduleAnalysis(durations)])
    pm.run(circuit)
    start_times = pm.property_set["node_start_time"]
    dt = durations.dt or 1.0
    first, busy, end = {}, {}, 0.0
    for node, start in start_times.items():
        if node.name in ("barrier", "delay"):
            continue
        length = durations.get(node.op, [circuit.find_bit(q).index for q in node.qargs], unit="s")
        start_s = start * dt if durations.dt else start
        end = max(end, start_s + length)
        for q in node.qargs:
            i = circuit.find_bit(q).index
            first[i] = min(first.get(i, math.inf), start_s)
            busy[i] = busy.get(i, 0.0) + length
    return {q: max(end - first[q] - busy[q], 0.0) for q in first}


def idle_log_factor(idle: dict[int, float], coherence: dict[int, dict[str, float]],
                    which: str) -> float:
    return -sum(t / coherence[q][which] for q, t in idle.items())


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="directory written by qta.hardware")
    args = parser.parse_args(argv)

    durations, coherence = load_calibration(args.run / "calibration.json")
    runs = pd.read_csv(args.run / "runs.csv")
    with (args.run / "submitted.qpy").open("rb") as f:
        compiled = qpy.load(f)

    rows = []
    for row in runs.itertuples():
        idle = idle_times(compiled[row.pub], durations)
        rows.append({"pub": row.pub, "idle_us": sum(idle.values()) * 1e6,
                     "log_idle_t1": idle_log_factor(idle, coherence, "T1"),
                     "log_idle_t2": idle_log_factor(idle, coherence, "T2")})
    runs = runs.merge(pd.DataFrame(rows), on="pub")
    runs["esp_t1"] = (runs.log_esp + runs.log_idle_t1).map(math.exp)
    runs["esp_t2"] = (runs.log_esp + runs.log_idle_t2).map(math.exp)
    runs.to_csv(args.run / "runs_idle.csv", index=False)

    summary = []
    for metric in ("esp", "esp_t1", "esp_t2"):
        agreement = rank_agreement(runs.assign(esp=runs[metric]))
        agreement.insert(0, "metric", metric)
        summary.append(agreement)
    summary = pd.concat(summary)
    summary.to_csv(args.run / "rank_agreement_idle.csv", index=False)
    print(runs[["circuit", "seed", "depth", "idle_us", "esp", "esp_t1", "esp_t2", "fidelity"]]
          .to_string(index=False))
    print(summary.pivot(index="circuit", columns="metric", values="spearman").to_string())
    print(summary.groupby("metric")[["spearman", "fidelity_of_esp_best"]].agg(["mean", "median"])
          .to_string())


if __name__ == "__main__":
    main()

"""Why pruning dead couplers costs two-qubit gates: fewer edges or a split device?

    python -m qta.prune_diagnosis --out results/prune_diagnosis --circuits-dir results/circuits

Level 3 on four versions of each device, metrics always on the full device:

* ``control``: the full device;
* ``tau1_disconnected``: couplers with error 1 removed, isolated qubits kept;
* ``tau1_largest_component``: the same couplers removed and only the largest connected
  component kept, so the coupling map stays connected;
* ``random_good_edges_removed``: as many working couplers removed at random as there are
  dead ones, keeping the device connected, with the dead couplers left in place.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rustworkx as rx
from qiskit import QuantumCircuit
from qiskit.transpiler import Target, generate_preset_pass_manager

from qta.circuits import DEVICES, SUITES, get_device, load_suite
from qta.metrics import measure
from qta.noise import prune_target
from qta.runner import write_environment


def _couplers(target: Target) -> dict[frozenset, float]:
    return {
        frozenset(q): p.error
        for name in target.operation_names
        for q, p in target[name].items()
        if q is not None and len(q) == 2
    }


def sub_target(target: Target, physical: list[int], drop: set[frozenset]) -> Target:
    """Target on ``physical`` qubits (renumbered from 0) without the couplers in ``drop``."""
    index = {p: i for i, p in enumerate(physical)}
    props_q = target.qubit_properties
    sub = Target(
        num_qubits=len(physical), dt=target.dt, granularity=target.granularity,
        min_length=target.min_length, pulse_alignment=target.pulse_alignment,
        acquire_alignment=target.acquire_alignment,
        qubit_properties=[props_q[p] for p in physical] if props_q else None,
    )
    for name in target.operation_names:
        props = target[name]
        if None in props:
            sub.add_instruction(target.operation_from_name(name), name=name)
            continue
        kept = {
            tuple(index[q] for q in qargs): p
            for qargs, p in props.items()
            if all(q in index for q in qargs) and frozenset(qargs) not in drop
        }
        sub.add_instruction(target.operation_from_name(name), kept, name=name)
    return sub


def to_device(circuit: QuantumCircuit, physical: list[int], num_qubits: int) -> QuantumCircuit:
    """Place a circuit compiled for a sub-target back onto the full device."""
    full = QuantumCircuit(num_qubits, circuit.num_clbits)
    for inst in circuit.data:
        full.append(
            inst.operation,
            [physical[circuit.find_bit(q).index] for q in inst.qubits],
            [full.clbits[circuit.find_bit(c).index] for c in inst.clbits],
        )
    return full


def _graph(n: int, edges) -> rx.PyGraph:
    g = rx.PyGraph()
    g.add_nodes_from(range(n))
    g.add_edges_from_no_data([tuple(e) for e in edges])
    return g


def variants(target: Target, rng: np.random.Generator) -> dict[str, tuple[Target, list[int]]]:
    n = target.num_qubits
    couplers = _couplers(target)
    dead = {c for c, e in couplers.items() if e is not None and e >= 1.0}
    edges = set(couplers)
    largest = sorted(max(rx.connected_components(_graph(n, edges - dead)), key=len))
    good = sorted(tuple(sorted(e)) for e in edges - dead)
    while True:
        drop = {frozenset(good[i]) for i in rng.choice(len(good), len(dead), replace=False)}
        if rx.is_connected(_graph(n, edges - drop)):
            break
    everything = list(range(n))
    return {
        "control": (target, everything),
        "tau1_disconnected": (prune_target(target, 1.0), everything),
        "tau1_largest_component": (sub_target(target, largest, dead), largest),
        "random_good_edges_removed": (sub_target(target, everything, drop), everything),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--circuits-dir", type=Path, default=None)
    parser.add_argument("--suite", choices=list(SUITES), default="main")
    parser.add_argument("--devices", nargs="+", default=list(DEVICES))
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args(argv)

    circuits = load_suite(args.circuits_dir or args.out / "circuits", SUITES[args.suite])
    write_environment(args.out, experiment="prune_diagnosis", args=vars(args))
    for device_name in args.devices:
        full = get_device(device_name).target
        rows = []
        versions = variants(full, np.random.default_rng(0))
        for cid, qc in circuits.items():
            for name, (target, physical) in versions.items():
                for seed in range(args.seeds):
                    out = generate_preset_pass_manager(3, target=target, seed_transpiler=seed).run(qc)
                    if len(physical) < full.num_qubits:
                        out = to_device(out, physical, full.num_qubits)
                    m = measure(out, full)
                    rows.append({"circuit": cid, "variant": name, "seed": seed,
                                 "n_2q": m.n_2q, "log_esp": m.log_esp})
        runs = pd.DataFrame(rows)
        runs.to_csv(args.out / f"prune_diag_{device_name}.csv", index=False)
        print(summary(runs).assign(device=device_name).to_string(index=False), flush=True)


def summary(runs: pd.DataFrame) -> pd.DataFrame:
    best = runs.groupby(["circuit", "variant"]).agg(
        n2q=("n_2q", "min"), n2q_median=("n_2q", "median"), log_esp=("log_esp", "max")
    ).reset_index()
    control = best[best.variant == "control"].set_index("circuit")
    rows = []
    for variant, b in best.groupby("variant", sort=False):
        b = b.set_index("circuit")
        diff = (b.log_esp - control.log_esp)[np.isfinite(b.log_esp) & np.isfinite(control.log_esp)]
        rows.append({
            "variant": variant,
            "best_n2q_ratio": float(np.exp(np.log(control.n2q / b.n2q).mean())),
            "median_n2q_ratio": float(np.exp(np.log(control.n2q_median / b.n2q_median).mean())),
            "best_esp_ratio": float(np.exp(diff.mean())),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()

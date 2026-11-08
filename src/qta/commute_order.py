"""Reorder commuting gates before routing and compare with plain seed search.

    python -m qta.commute_order --out results/commute --circuits-dir results/circuits

Gates commute when on every shared qubit both act along the same axis (Z, X or Y). That is
a sufficient condition, so any order respecting the dependencies below is equivalent to the
original circuit. SABRE itself keeps the written order, so a different written order is a
cheap way to see whether the freedom is worth anything before changing the router.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager

from qta.circuits import DATASET, get_device, load_suite
from qta.metrics import measure
from qta.runner import write_environment

AXIS_1Q = {
    "z": "Z", "s": "Z", "sdg": "Z", "t": "Z", "tdg": "Z", "rz": "Z", "p": "Z", "u1": "Z",
    "x": "X", "sx": "X", "sxdg": "X", "rx": "X",
    "y": "Y", "ry": "Y",
}
AXIS_2Q = {
    "cz": "ZZ", "cp": "ZZ", "crz": "ZZ", "cu1": "ZZ", "rzz": "ZZ", "cs": "ZZ", "csdg": "ZZ",
    "cx": "ZX", "csx": "ZX", "crx": "ZX",
    "cy": "ZY", "cry": "ZY",
    "rxx": "XX", "ryy": "YY",
}
FAMILIES = ("qaoa", "graphstate", "bv", "dj", "vqe_two_local", "modular_adder", "randomcircuit",
            "qft", "qpeexact", "cdkm_ripple_carry_adder")


def wire_axes(qc: QuantumCircuit, inst) -> dict:
    """Axis of the instruction on each wire; None means it commutes with nothing there."""
    name = inst.operation.name
    axes: dict = {bit: None for bit in (*inst.qubits, *inst.clbits)}
    if inst.clbits or getattr(inst.operation, "condition", None) is not None:
        return axes
    if len(inst.qubits) == 1 and name in AXIS_1Q:
        axes[inst.qubits[0]] = AXIS_1Q[name]
    elif len(inst.qubits) == 2 and name in AXIS_2Q:
        axes[inst.qubits[0]], axes[inst.qubits[1]] = AXIS_2Q[name]
    return axes


def levels(qc: QuantumCircuit) -> tuple[list[int], list[set[int]]]:
    """ASAP level of every instruction in the dependency order that ignores commuting pairs."""
    state: dict = {}  # wire -> (axis, current same-axis group, group before it)
    preds: list[set[int]] = []
    for i, inst in enumerate(qc.data):
        need: set[int] = set()
        for bit, axis in wire_axes(qc, inst).items():
            cur_axis, group, before = state.get(bit, (None, [], []))
            if axis is not None and axis == cur_axis:
                need.update(before)
                group.append(i)
            else:
                need.update(group)
                state[bit] = (axis, [i], group)
        preds.append(need)
    level = [0] * len(preds)
    for i, need in enumerate(preds):
        level[i] = 1 + max((level[j] for j in need), default=-1)
    return level, preds


def reorder(qc: QuantumCircuit, seed: int) -> QuantumCircuit:
    """Equivalent circuit written level by level, in random order inside a level."""
    level, _ = levels(qc)
    rng = np.random.default_rng(seed)
    order = sorted(range(len(qc.data)), key=lambda i: (level[i], rng.random()))
    out = qc.copy_empty_like()
    for i in order:
        out.append(qc.data[i])
    return out


def compile_row(qc, backend, seed, method, cid, device) -> dict:
    start = time.perf_counter()
    out = generate_preset_pass_manager(3, backend=backend, seed_transpiler=seed).run(qc)
    elapsed = time.perf_counter() - start
    m = measure(out, backend.target)
    return {"device": device, "circuit": cid, "num_qubits": qc.num_qubits, "method": method, "seed": seed,
            "n_2q": m.n_2q, "depth": m.depth, "log_esp": m.log_esp, "time_s": elapsed}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--circuits-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--max-qubits", type=int, default=20)
    args = parser.parse_args(argv)

    suite = [(name, size) for name, size in DATASET if size <= args.max_qubits and name in FAMILIES]
    circuits = load_suite(args.circuits_dir, suite)
    rows = []
    for device in ("torino", "sherbrooke"):
        backend = get_device(device)
        for cid, qc in circuits.items():
            for seed in range(args.seeds):
                rows.append(compile_row(qc, backend, seed, "seeds", cid, device))
                rows.append(compile_row(reorder(qc, seed), backend, seed, "reorder", cid, device))
            print(device, cid, flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / "runs.csv", index=False)
    write_environment(args.out, experiment="commute_order", args=vars(args))


if __name__ == "__main__":
    main()

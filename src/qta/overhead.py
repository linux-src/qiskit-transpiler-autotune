"""How much room is left for routing and for noise-aware placement on real circuits.

    python -m qta.overhead --out results/overhead --circuits-dir results/circuits

Routing: two-qubit count of level 3 on the device against level 3 with all-to-all
connectivity and the same basis, which contains no routing at all.
Noise: mean error of the two-qubit gates a compilation actually uses against the
error quantiles of the device, for circuits small enough to fit on good couplers.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager

from qta.circuits import DATASET, get_device, load_suite
from qta.runner import write_environment

DEVICES = ("sherbrooke", "torino")
TWO_QUBIT = {"sherbrooke": "ecr", "torino": "cz"}


def count_2q(qc: QuantumCircuit) -> int:
    return sum(1 for inst in qc.data if inst.operation.num_qubits == 2)


def two_qubit_errors(backend, gate: str) -> dict[tuple[int, int], float]:
    return {q: p.error for q, p in backend.target[gate].items() if p is not None and p.error is not None}


def used_errors(qc: QuantumCircuit, errors: dict[tuple[int, int], float]) -> list[float]:
    used = []
    for inst in qc.data:
        if inst.operation.num_qubits == 2:
            pair = tuple(qc.find_bit(q).index for q in inst.qubits)
            used.append(errors[pair])
    return used


def one_circuit(task: tuple) -> list[dict]:
    device, cid, qc, seeds = task
    backend = get_device(device)
    gate = TWO_QUBIT[device]
    basis = ["rz", "sx", "x", gate]
    free = min(count_2q(generate_preset_pass_manager(3, basis_gates=basis, seed_transpiler=s).run(qc))
               for s in range(seeds))
    errors = two_qubit_errors(backend, gate)
    rows = []
    for seed in range(seeds):
        out = generate_preset_pass_manager(3, backend=backend, seed_transpiler=seed).run(qc)
        used = used_errors(out, errors)
        rows.append({"device": device, "circuit": cid, "num_qubits": qc.num_qubits, "seed": seed,
                     "n_2q_free": free, "n_2q": len(used),
                     "mean_error_used": float(np.mean(used)) if used else np.nan})
    return rows


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--circuits-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    circuits = load_suite(args.circuits_dir, DATASET)
    tasks = [(device, cid, qc, args.seeds) for device in DEVICES for cid, qc in circuits.items()]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = [row for chunk in pool.map(one_circuit, tasks) for row in chunk]
    runs = pd.DataFrame(rows)

    args.out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.out / "runs.csv", index=False)
    quantiles = []
    for device in DEVICES:
        values = np.array(list(two_qubit_errors(get_device(device), TWO_QUBIT[device]).values()))
        values = values[values < 1.0]  # dead couplers would hide the working range
        quantiles.append({"device": device, **{f"p{q}": float(np.percentile(values, q)) for q in (10, 25, 50, 75, 90)}})
    pd.DataFrame(quantiles).to_csv(args.out / "device_errors.csv", index=False)
    write_environment(args.out, experiment="overhead", args=vars(args))


if __name__ == "__main__":
    main()

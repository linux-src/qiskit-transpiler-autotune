"""Check that circuits compiled by the commuting router still compute the same thing.

    python -m qta.commute_equivalence --out results/router --circuits-dir results/circuits

Each compiled circuit is restricted to its active qubits and simulated exactly; its output
distribution is compared with the ideal one of the source circuit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, hellinger_fidelity

from qta.circuits import DATASET, get_device, load_suite
from qta.commute_router import build_pass_manager
from qta.esp_check import ideal_distribution, with_measurements
from qta.metrics import measure

CIRCUITS = [("qaoa", 6), ("qaoa", 8), ("qaoa", 10), ("modular_adder", 8), ("modular_adder", 10), ("qft", 8),
            ("qftentangled", 8), ("rg_qft_multiplier", 8), ("vqe_two_local", 6), ("bv", 10)]


def output_distribution(out: QuantumCircuit) -> dict[str, float]:
    index = lambda bit: out.find_bit(bit).index
    active = sorted({index(q) for inst in out.data if inst.operation.name != "barrier" for q in inst.qubits})
    position = {q: i for i, q in enumerate(active)}
    state = QuantumCircuit(len(active))
    measured: dict[int, int] = {}
    for inst in out.data:
        name = inst.operation.name
        if name == "measure":
            measured[index(inst.clbits[0])] = position[index(inst.qubits[0])]
        elif name != "barrier":
            state.append(inst.operation, [position[index(q)] for q in inst.qubits])
    probabilities = np.abs(Statevector(state).data) ** 2
    result: dict[str, float] = {}
    for basis in np.nonzero(probabilities > 1e-12)[0]:
        key = "".join(str((int(basis) >> measured[c]) & 1) for c in range(len(measured) - 1, -1, -1))
        result[key] = result.get(key, 0.0) + float(probabilities[basis])
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--circuits-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args(argv)

    circuits = load_suite(args.circuits_dir, [c for c in CIRCUITS if c in DATASET])
    rows = []
    for device in ("torino", "sherbrooke"):
        backend = get_device(device)
        for cid, qc in circuits.items():
            measured = with_measurements(qc)
            ideal = ideal_distribution(measured)
            for seed in range(args.seeds):
                for commute in (True, False):
                    out = build_pass_manager(backend, seed, commute).run(measured)
                    rows.append({"device": device, "circuit": cid, "seed": seed, "commute": commute,
                                 "n_2q": measure(out, backend.target).n_2q,
                                 "fidelity": hellinger_fidelity(ideal, output_distribution(out))})
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / "equivalence.csv", index=False)
    print(f"{len(rows)} compilations, lowest fidelity {min(r['fidelity'] for r in rows)}")


if __name__ == "__main__":
    main()

"""Level 3 with the commutation-aware SABRE core of the patched Qiskit build.

    QTA_SABRE_COMMUTE=1 python -m qta.sabre_commute run --out results/sabre/on --circuits-dir results/circuits
    python -m qta.sabre_commute run --out results/sabre/off --circuits-dir results/circuits
    QTA_SABRE_COMMUTE=1 python -m qta.sabre_commute check --circuits-dir results/circuits

The switch is read by the Rust code, so it only has an effect on the patched build and the
two modes must run in separate processes. Runs measure compile time: do not run anything else
at the same time.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from qiskit.quantum_info import hellinger_fidelity
from qiskit.transpiler import generate_preset_pass_manager

from qta.circuits import DATASET, SUITES, get_device, load_suite
from qta.commute_equivalence import CIRCUITS, output_distribution
from qta.esp_check import ideal_distribution, with_measurements
from qta.metrics import measure
from qta.runner import write_environment

DEVICES = ("torino", "sherbrooke")


def run(args) -> None:
    suite = [(name, size) for name, size in SUITES[args.suite] if size <= args.max_qubits]
    circuits = load_suite(args.circuits_dir, suite)
    rows = []
    for device in DEVICES:
        backend = get_device(device)
        for cid, qc in circuits.items():
            for seed in range(args.seeds):
                start = time.perf_counter()
                out = generate_preset_pass_manager(3, backend=backend, seed_transpiler=seed).run(qc)
                elapsed = time.perf_counter() - start
                m = measure(out, backend.target)
                rows.append({"device": device, "circuit": cid, "num_qubits": qc.num_qubits, "seed": seed,
                             "n_2q": m.n_2q, "depth": m.depth, "log_esp": m.log_esp, "time_s": elapsed})
            print(device, cid, flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / "runs.csv", index=False)
    write_environment(args.out, experiment="sabre_commute", args=vars(args))


def check(args) -> None:
    circuits = load_suite(args.circuits_dir, [c for c in CIRCUITS if c in DATASET])
    lowest = 1.0
    count = 0
    for device in DEVICES:
        backend = get_device(device)
        for cid, qc in circuits.items():
            measured = with_measurements(qc)
            ideal = ideal_distribution(measured)
            for seed in range(args.seeds):
                out = generate_preset_pass_manager(3, backend=backend, seed_transpiler=seed).run(measured)
                fidelity = hellinger_fidelity(ideal, output_distribution(out))
                lowest = min(lowest, fidelity)
                count += 1
    print(f"{count} compilations, lowest fidelity {lowest}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--circuits-dir", type=Path, required=True)
    r.add_argument("--seeds", type=int, default=20)
    r.add_argument("--max-qubits", type=int, default=40)
    r.add_argument("--suite", choices=SUITES, default="dataset")
    r.set_defaults(func=run)
    c = sub.add_parser("check")
    c.add_argument("--circuits-dir", type=Path, required=True)
    c.add_argument("--seeds", type=int, default=3)
    c.set_defaults(func=check)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

"""Baseline: preset optimization levels over several transpiler seeds.

    python -m qta.baseline --out results/baseline --seeds 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

from qiskit.transpiler import generate_preset_pass_manager

from qta.circuits import DEVICES, load_suite, get_device
from qta.runner import CsvLog, evaluate, write_environment

METRICS = ["n_2q", "depth", "depth_2q", "size", "log_esp", "esp", "time_s"]
FIELDS = ["device", "circuit", "num_qubits", "level", "seed", *METRICS]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--circuits-dir", type=Path, default=None)
    parser.add_argument("--devices", nargs="+", default=list(DEVICES))
    parser.add_argument("--levels", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args(argv)

    circuits = load_suite(args.circuits_dir or args.out / "circuits")
    write_environment(args.out, experiment="baseline", args=vars(args))
    log = CsvLog(args.out / "baseline.csv", FIELDS, key=["device", "circuit", "level", "seed"])

    for device_name in args.devices:
        backend = get_device(device_name)
        for cid, qc in circuits.items():
            for level in args.levels:
                for seed in range(args.seeds):
                    if log.has(device=device_name, circuit=cid, level=level, seed=seed):
                        continue
                    pm = generate_preset_pass_manager(level, backend=backend, seed_transpiler=seed)
                    row = evaluate(qc, pm, backend.target)
                    log.write({"device": device_name, "circuit": cid,
                               "num_qubits": qc.num_qubits, "level": level, "seed": seed, **row})
            print(f"{device_name} {cid} done", flush=True)


if __name__ == "__main__":
    main()

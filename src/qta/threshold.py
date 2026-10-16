"""Level 3 on devices with the couplers of error at least tau removed.

    python -m qta.threshold --suite dataset --out results/threshold --taus 1.0 0.05 0.02

Metrics are always computed against the full device, so ESP of a pruned run is
directly comparable with plain level 3.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from qiskit.transpiler import generate_preset_pass_manager

from qta.baseline import METRICS
from qta.circuits import DEVICES, SUITES, get_device, load_suite
from qta.noise import prune_target
from qta.runner import CsvLog, evaluate, write_environment

FIELDS = ["device", "circuit", "num_qubits", "tau", "seed", *METRICS]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--circuits-dir", type=Path, default=None)
    parser.add_argument("--suite", choices=list(SUITES), default="main")
    parser.add_argument("--devices", nargs="+", default=list(DEVICES))
    parser.add_argument("--taus", nargs="+", type=float, default=[1.0, 0.05, 0.02])
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args(argv)

    circuits = load_suite(args.circuits_dir or args.out / "circuits", SUITES[args.suite])
    write_environment(args.out, experiment="threshold", args=vars(args))
    log = CsvLog(args.out / "threshold.csv", FIELDS, key=["device", "circuit", "tau", "seed"])

    for device_name in args.devices:
        backend = get_device(device_name)
        for tau in args.taus:
            target = prune_target(backend.target, tau)
            for cid, qc in circuits.items():
                for seed in range(args.seeds):
                    if log.has(device=device_name, circuit=cid, tau=tau, seed=seed):
                        continue
                    pm = generate_preset_pass_manager(3, target=target, seed_transpiler=seed)
                    row = evaluate(qc, pm, backend.target)
                    log.write({"device": device_name, "circuit": cid, "num_qubits": qc.num_qubits,
                               "tau": tau, "seed": seed, **row})
            print(f"{device_name} tau={tau} done", flush=True)


if __name__ == "__main__":
    main()

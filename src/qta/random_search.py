"""Random search over layout and routing configurations.

    python -m qta.random_search --out results/random --trials 40
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import numpy as np

from qta.baseline import METRICS
from qta.circuits import DEVICES, get_device, load_suite
from qta.config import TuningConfig, build_pass_manager
from qta.runner import CsvLog, evaluate, write_environment
from qta.search_space import sample

CONFIG_FIELDS = [f.name for f in fields(TuningConfig)]
FIELDS = ["device", "circuit", "num_qubits", "trial", "seed", *CONFIG_FIELDS, *METRICS]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--circuits-dir", type=Path, default=None)
    parser.add_argument("--devices", nargs="+", default=list(DEVICES))
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0, help="seed of the search itself")
    args = parser.parse_args(argv)

    circuits = load_suite(args.circuits_dir or args.out / "circuits")
    write_environment(args.out, experiment="random_search", args=vars(args))
    log = CsvLog(args.out / "random_search.csv", FIELDS, key=["device", "circuit", "trial"])

    for d, device_name in enumerate(args.devices):
        backend = get_device(device_name)
        for c, (cid, qc) in enumerate(circuits.items()):
            # Each (device, circuit) pair gets its own stream so resumed runs draw the same configs.
            rng = np.random.default_rng([args.seed, d, c])
            for trial in range(args.trials):
                config = sample(rng)
                seed = int(rng.integers(2**31))
                if log.has(device=device_name, circuit=cid, trial=trial):
                    continue
                row = evaluate(qc, build_pass_manager(config, backend, seed), backend.target)
                log.write({"device": device_name, "circuit": cid, "num_qubits": qc.num_qubits,
                           "trial": trial, "seed": seed, **config.as_dict(), **row})
            print(f"{device_name} {cid} done", flush=True)


if __name__ == "__main__":
    main()

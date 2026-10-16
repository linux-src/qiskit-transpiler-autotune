"""Level 3 with sampled optimization stage configurations.

    python -m qta.opt_search --suite dataset --out results/opt_search --configs 24 --seeds 3

Configuration 0 is plain level 3; the rest are drawn once from ``qta.optimization.sample``
and shared by all circuits, so a per-circuit choice among them can be compared with
plain level 3 on the same seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qta.baseline import METRICS
from qta.circuits import DEVICES, SUITES, get_device, load_suite
from qta.optimization import O3_OPTIMIZATION, build_pass_manager, sample
from qta.runner import CsvLog, evaluate, write_environment

FIELDS = ["device", "circuit", "num_qubits", "config", "seed", *METRICS]


def configurations(n: int, rng_seed: int = 0) -> list:
    rng = np.random.default_rng(rng_seed)
    configs = [O3_OPTIMIZATION]
    while len(configs) < n:
        c = sample(rng)
        if c not in configs:
            configs.append(c)
    return configs


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--circuits-dir", type=Path, default=None)
    parser.add_argument("--suite", choices=list(SUITES), default="main")
    parser.add_argument("--devices", nargs="+", default=list(DEVICES))
    parser.add_argument("--configs", type=int, default=24)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args(argv)

    circuits = load_suite(args.circuits_dir or args.out / "circuits", SUITES[args.suite])
    configs = configurations(args.configs)
    write_environment(args.out, experiment="opt_search", args=vars(args))
    (args.out / "configs.json").write_text(json.dumps([c.as_dict() for c in configs], indent=2))
    log = CsvLog(args.out / "opt_search.csv", FIELDS, key=["device", "circuit", "config", "seed"])

    for device_name in args.devices:
        backend = get_device(device_name)
        for cid, qc in circuits.items():
            for seed in range(args.seeds):
                for i, config in enumerate(configs):
                    if log.has(device=device_name, circuit=cid, config=i, seed=seed):
                        continue
                    pm = build_pass_manager(config, backend, seed)
                    row = evaluate(qc, pm, backend.target)
                    log.write({"device": device_name, "circuit": cid, "num_qubits": qc.num_qubits,
                               "config": i, "seed": seed, **row})
            print(f"{device_name} {cid} done", flush=True)


if __name__ == "__main__":
    main()

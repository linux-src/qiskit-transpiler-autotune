"""Level 3 with sampled optimization stage configurations.

    python -m qta.opt_search --suite dataset --out results/opt_search --configs 24 --seeds 3

Configuration 0 is plain level 3; the rest are drawn once from ``qta.optimization.sample``
and shared by all circuits, so a per-circuit choice among them can be compared with
plain level 3 on the same seeds. With ``--space sequence`` the rest are random sequences
of optimization passes (``qta.optimization.sample_sequence``) instead of on/off flags.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from qiskit.transpiler.exceptions import TranspilerError

from qta.baseline import METRICS
from qta.circuits import DEVICES, SUITES, get_device, load_suite
from qta.optimization import O3_OPTIMIZATION, build_pass_manager, sample, sample_sequence
from qta.runner import CsvLog, evaluate, write_environment

FIELDS = ["device", "circuit", "num_qubits", "config", "seed", *METRICS, "error"]


SPACES = {"flags": sample, "sequence": sample_sequence}


def configurations(n: int, rng_seed: int = 0, space: str = "flags") -> list:
    rng = np.random.default_rng(rng_seed)
    configs = [O3_OPTIMIZATION]
    while len(configs) < n:
        c = SPACES[space](rng)
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
    parser.add_argument("--space", choices=list(SPACES), default="flags")
    args = parser.parse_args(argv)

    circuits = load_suite(args.circuits_dir or args.out / "circuits", SUITES[args.suite])
    configs = configurations(args.configs, space=args.space)
    write_environment(args.out, experiment="opt_search", args=vars(args))
    (args.out / "configs.json").write_text(json.dumps([c.as_dict() for c in configs], indent=2))
    log = CsvLog(args.out / "opt_search.csv", FIELDS, key=["device", "circuit", "config", "seed"])

    for device_name in args.devices:
        backend = get_device(device_name)
        for seed in range(args.seeds):
            for i, config in enumerate(configs):
                # Building the pass manager costs more than running it on a small circuit,
                # and reusing it gives identical output, so build it once per configuration.
                pm = build_pass_manager(config, backend, seed)
                for cid, qc in circuits.items():
                    if log.has(device=device_name, circuit=cid, config=i, seed=seed):
                        continue
                    try:
                        row = {**evaluate(qc, pm, backend.target), "error": ""}
                    except TranspilerError as exc:
                        row = {"error": str(exc)}
                    log.write({"device": device_name, "circuit": cid, "num_qubits": qc.num_qubits,
                               "config": i, "seed": seed, **row})
                print(f"{device_name} seed={seed} config={i} done", flush=True)


if __name__ == "__main__":
    main()

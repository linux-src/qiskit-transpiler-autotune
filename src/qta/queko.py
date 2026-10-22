"""Layout and routing on QUEKO circuits, whose optimal depth is known.

    python -m qta.queko --repo ../data/queko_src/repo --out ../data/queko --seeds 20 --trials 20

QUEKO (Tan & Cong, IEEE TC 2021) builds circuits that fit a device without any SWAP at
a known depth. Only the layout and routing stages run, so the depth of the result
against the optimum measures layout and routing alone.

VF2Layout finds the exact embedding of these circuits, so level 3 is also run without
it (``sabre``: SabreLayout and routing alone), and the random search over
``qta.search_space`` keeps VF2 off. Level 3 and ``sabre`` run with ``--seeds``
transpiler seeds, the search draws ``--trials`` configurations.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import re
import time
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap, StagedPassManager, Target
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qta.config import O3_DEFAULT, TuningConfig, _layout_stage, _routing_stage
from qta.runner import CsvLog, write_environment
from qta.search_space import sample

# QUEKO file prefix -> name of the device graph in the repository's CONNECTION.py
DEVICES = {"16QBT": "Aspen-4", "20QBT": "Tokyo", "53QBT": "Rochester", "54QBT": "Sycamore"}
_NAME = re.compile(r"(?P<device>\d+QBT)_(?P<depth>\d+)CYC_(?P<density>[A-Z]+)_(?P<serial>\d+)")

FIELDS = ["circuit", "set", "device", "optimal_depth", "method", "run", "seed", "n_swaps",
          "depth", "depth_ratio", "time_s", "config"]


def load_connections(repo: Path) -> dict[str, list[tuple[int, int]]]:
    """Edge lists from CONNECTION.py, read as a literal: the module itself needs cirq."""
    tree = ast.parse((Path(repo) / "CONNECTION.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "INDEX_CONNECTION_LIST" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("INDEX_CONNECTION_LIST not found")


def device_target(edges: list[tuple[int, int]]) -> Target:
    coupling = CouplingMap(edges)
    coupling.make_symmetric()
    return Target.from_configuration(basis_gates=["cx", "x", "swap"], coupling_map=coupling)


def select(repo: Path, bntf: bool = True, bss_serials: int = 3) -> list[tuple[str, Path]]:
    """All BNTF circuits and the first ``bss_serials`` circuits of every BSS setting."""
    chosen = []
    if bntf:
        chosen += [("BNTF", p) for p in sorted((Path(repo) / "BNTF").glob("*.qasm"))]
    for p in sorted((Path(repo) / "BSS").glob("*.qasm")):
        if int(_NAME.match(p.stem)["serial"]) < bss_serials:
            chosen.append(("BSS", p))
    return chosen


def layout_and_routing(config: TuningConfig, target: Target, seed: int) -> StagedPassManager:
    """Only the layout and routing stages of level 3, with ``config`` applied."""
    pm = generate_preset_pass_manager(optimization_level=3, target=target, seed_transpiler=seed)
    routing = _routing_stage(config, target, seed) if config.routing == "separate" else pm.routing
    return StagedPassManager(stages=["layout", "routing"],
                             layout=_layout_stage(config, target, seed), routing=routing)


def measure(qc: QuantumCircuit, pm: StagedPassManager, optimal_depth: int) -> dict:
    start = time.perf_counter()
    out = pm.run(qc)
    elapsed = time.perf_counter() - start
    n_swaps = out.count_ops().get("swap", 0)
    # a SWAP is three CNOTs, as in the QUEKO paper
    depth = out.decompose(gates_to_decompose=["swap"]).depth() if n_swaps else out.depth()
    return {"n_swaps": n_swaps, "depth": depth, "depth_ratio": depth / optimal_depth,
            "time_s": elapsed}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="clone of QUEKO-benchmark")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--bss-serials", type=int, default=3)
    args = parser.parse_args(argv)

    connections = load_connections(args.repo)
    targets = {prefix: device_target(connections[name]) for prefix, name in DEVICES.items()}
    write_environment(args.out, experiment="queko", args=vars(args))
    log = CsvLog(args.out / "queko.csv", FIELDS, key=["circuit", "method", "run"])

    for c, (bench_set, path) in enumerate(select(args.repo, bss_serials=args.bss_serials)):
        meta = _NAME.match(path.stem)
        qc = QuantumCircuit.from_qasm_file(str(path))
        target = targets[meta["device"]]
        optimal = int(meta["depth"])
        rng = np.random.default_rng([0, c])
        sabre = dataclasses.replace(O3_DEFAULT, use_vf2=False)
        runs = [("level3", i, O3_DEFAULT, i) for i in range(args.seeds)]
        runs += [("sabre", i, sabre, i) for i in range(args.seeds)]
        runs += [("sabre_search", i, dataclasses.replace(sample(rng), use_vf2=False),
                  int(rng.integers(2**31))) for i in range(args.trials)]
        for method, run, config, seed in runs:
            if log.has(circuit=path.stem, method=method, run=run):
                continue
            row = measure(qc, layout_and_routing(config, target, seed), optimal)
            log.write({"circuit": path.stem, "set": bench_set, "device": DEVICES[meta["device"]],
                       "optimal_depth": optimal, "method": method, "run": run, "seed": seed,
                       "config": config.as_dict(), **row})
        print(f"{path.stem} done", flush=True)


if __name__ == "__main__":
    main()

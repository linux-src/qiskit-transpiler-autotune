"""Does ESP rank compilations of a circuit the way a noisy simulation does?

    python -m qta.esp_check --out results/esp_check --circuits-dir results/circuits --max-qubits 10

Each small circuit is compiled several ways (level 3 with several seeds and levels 0-2),
and every compilation gets its ESP and its Hellinger fidelity to the ideal output
distribution under the Aer noise model built from the same device snapshot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, hellinger_fidelity
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from scipy.stats import spearmanr

from qta.circuits import DATASET, DEVICES, get_device, load_suite
from qta.metrics import measure
from qta.runner import CsvLog, write_environment

FIELDS = ["device", "circuit", "num_qubits", "compilation", "n_2q", "depth", "log_esp", "esp",
          "fidelity", "shots"]


def with_measurements(qc: QuantumCircuit) -> QuantumCircuit:
    if any(inst.name == "measure" for inst in qc.data):
        return qc
    out = qc.copy()
    out.measure_all()
    return out


def ideal_distribution(qc: QuantumCircuit) -> dict[str, float]:
    """Exact output distribution over the classical bits, keyed like Aer counts."""
    measured = {qc.find_bit(inst.clbits[0]).index: qc.find_bit(inst.qubits[0]).index
                for inst in qc.data if inst.name == "measure"}
    state = Statevector(qc.remove_final_measurements(inplace=False))
    qargs = [measured[c] for c in range(qc.num_clbits)]
    return state.probabilities_dict(qargs=qargs)


def compilations(seeds: int) -> list[tuple[str, int, int]]:
    return [(f"o3_seed{s}", 3, s) for s in range(seeds)] + [(f"o{lvl}", lvl, 0) for lvl in (0, 1, 2)]


def rank_agreement(runs: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation between ESP and simulated fidelity within each circuit."""
    rows = []
    for (device, circuit), r in runs.groupby(["device", "circuit"]):
        if r.esp.nunique() < 3 or r.fidelity.nunique() < 3:
            continue
        rho = spearmanr(r.esp, r.fidelity).statistic
        best_by_esp = r.fidelity[r.esp.idxmax()]
        rows.append({"device": device, "circuit": circuit, "spearman": rho,
                     "fidelity_of_esp_best": best_by_esp, "best_fidelity": r.fidelity.max(),
                     "median_fidelity": r.fidelity.median()})
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--circuits-dir", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", default=list(DEVICES))
    parser.add_argument("--max-qubits", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--shots", type=int, default=4000)
    args = parser.parse_args(argv)

    suite = [(name, size) for name, size in DATASET if size <= args.max_qubits]
    circuits = load_suite(args.circuits_dir, suite)
    write_environment(args.out, experiment="esp_check", args=vars(args))
    log = CsvLog(args.out / "esp_check.csv", FIELDS, key=["device", "circuit", "compilation"])

    for device_name in args.devices:
        backend = get_device(device_name)
        simulator = AerSimulator.from_backend(backend, seed_simulator=1)
        for cid, qc in circuits.items():
            qc = with_measurements(qc)
            ideal = ideal_distribution(qc)
            for label, level, seed in compilations(args.seeds):
                if log.has(device=device_name, circuit=cid, compilation=label):
                    continue
                out = generate_preset_pass_manager(level, backend=backend, seed_transpiler=seed).run(qc)
                m = measure(out, backend.target)
                counts = simulator.run(out, shots=args.shots).result().get_counts()
                log.write({"device": device_name, "circuit": cid, "num_qubits": qc.num_qubits,
                           "compilation": label, "n_2q": m.n_2q, "depth": m.depth,
                           "log_esp": m.log_esp, "esp": m.esp,
                           "fidelity": hellinger_fidelity(ideal, counts), "shots": args.shots})
            print(f"{device_name} {cid} done", flush=True)

    runs = pd.read_csv(args.out / "esp_check.csv")
    agreement = rank_agreement(runs)
    agreement.to_csv(args.out / "rank_agreement.csv", index=False)
    summary = agreement.groupby("device").agg(
        circuits=("spearman", "size"), spearman_mean=("spearman", "mean"),
        spearman_median=("spearman", "median"),
        fidelity_of_esp_best=("fidelity_of_esp_best", "mean"),
        best_fidelity=("best_fidelity", "mean"), median_fidelity=("median_fidelity", "mean"),
    )
    pooled = runs.groupby("device").apply(
        lambda r: spearmanr(r.esp, r.fidelity).statistic, include_groups=False
    ).rename("spearman_pooled")
    summary = summary.join(pooled)
    summary.to_csv(args.out / "summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()

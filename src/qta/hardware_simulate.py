"""Replay the circuits of a hardware job on the Aer noise model of the same backend.

    python -m qta.hardware_simulate --out results/hw

Reads `plan.csv`, `submitted.qpy` and `ideal.json` written by `qta.hardware_commute plan` and
the hardware fidelities in `runs.csv`, simulates every submitted circuit with the noise model
built from the current calibration of the backend, and compares the gain of the portfolio over
plain seeds in simulation and on hardware. If the two disagree for a circuit, the cause is not
in the gate and readout errors of the calibration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from qiskit import qpy
from qiskit.quantum_info import hellinger_fidelity
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService

from qta.hardware_commute import paired
from qta.runner import write_environment


def gains(runs: pd.DataFrame) -> pd.Series:
    table = paired(runs)
    table["gain"] = table.fidelity_portfolio / table.fidelity_plain - 1
    return table.groupby("circuit").gain.mean()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=20000)
    args = parser.parse_args(argv)

    plan = pd.read_csv(args.out / "plan.csv")
    hardware = pd.read_csv(args.out / "runs.csv")
    ideal = json.loads((args.out / "ideal.json").read_text())
    with (args.out / "submitted.qpy").open("rb") as f:
        compiled = qpy.load(f)
    name = json.loads((args.out / "job.json").read_text())["backend"]
    backend = QiskitRuntimeService().backend(name)
    simulator = AerSimulator.from_backend(backend, seed_simulator=1)

    first = plan[plan.repeat == 0]
    fidelity = {}
    for row in first.itertuples():
        circuit = compiled[row.pub]
        counts = simulator.run(circuit, shots=args.shots).result().get_counts()
        fidelity[(row.circuit, row.instance, row.strategy)] = hellinger_fidelity(ideal[row.circuit], counts)
    simulated = first.assign(fidelity=[fidelity[(r.circuit, r.instance, r.strategy)] for r in first.itertuples()])

    directory = args.out / "simulation"
    directory.mkdir(exist_ok=True)
    simulated.to_csv(directory / "runs.csv", index=False)
    table = pd.DataFrame({"hardware_gain": gains(hardware), "simulated_gain": gains(simulated)})
    levels = simulated.groupby(["circuit", "strategy"]).fidelity.mean().unstack()
    table["simulated_plain"], table["simulated_portfolio"] = levels.plain, levels.portfolio
    table.to_csv(directory / "summary.csv")
    write_environment(directory, experiment="hardware_simulate", backend=name, args=vars(args),
                      calibration=str(backend.properties().last_update_date))
    print((table * [100, 100, 1, 1]).round(3).to_string())
    print("Spearman, gain on hardware vs in simulation:",
          round(table.hardware_gain.corr(table.simulated_gain, method="spearman"), 2))


if __name__ == "__main__":
    main()

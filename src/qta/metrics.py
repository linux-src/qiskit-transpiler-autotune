"""Quality metrics of a transpiled circuit."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from qiskit import QuantumCircuit
from qiskit.transpiler import Target

_IGNORED = {"barrier", "delay"}


@dataclass(frozen=True)
class CircuitMetrics:
    n_2q: int
    depth: int
    depth_2q: int
    size: int
    log_esp: float

    @property
    def esp(self) -> float:
        return math.exp(self.log_esp)

    def as_dict(self) -> dict:
        return {**asdict(self), "esp": self.esp}


def log_esp(circuit: QuantumCircuit, target: Target) -> float:
    """Logarithm of the estimated success probability.

    ESP is the product of (1 - error) over all gates and measurements, with error
    rates taken from the target calibration. Instructions without calibration data
    are treated as error-free.
    """
    qubit_index = {q: i for i, q in enumerate(circuit.qubits)}
    total = 0.0
    for inst in circuit.data:
        name = inst.operation.name
        if name in _IGNORED:
            continue
        qargs = tuple(qubit_index[q] for q in inst.qubits)
        props = target[name].get(qargs) if name in target else None
        error = props.error if props is not None and props.error is not None else 0.0
        if error >= 1.0:
            return -math.inf
        total += math.log1p(-error)
    return total


def measure(circuit: QuantumCircuit, target: Target) -> CircuitMetrics:
    def is_2q(inst):
        return inst.operation.num_qubits == 2 and inst.operation.name not in _IGNORED

    return CircuitMetrics(
        n_2q=sum(1 for inst in circuit.data if is_2q(inst)),
        depth=circuit.depth(lambda inst: inst.operation.name not in _IGNORED),
        depth_2q=circuit.depth(is_2q),
        size=sum(1 for inst in circuit.data if inst.operation.name not in _IGNORED),
        log_esp=log_esp(circuit, target),
    )

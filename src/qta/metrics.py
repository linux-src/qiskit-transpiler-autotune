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
        name = inst.name
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
    # Reads only names and bit indices: touching ``inst.operation`` builds a Python
    # gate object per instruction and dominates the cost on large circuits.
    bit_index = {b: i for i, b in enumerate(circuit.qubits + circuit.clbits)}
    level = [0] * len(bit_index)
    level_2q = [0] * len(bit_index)
    n_2q = size = 0
    for inst in circuit.data:
        if inst.name in _IGNORED:
            continue
        size += 1
        bits = [bit_index[b] for b in inst.qubits + inst.clbits]
        is_2q = len(inst.qubits) == 2
        n_2q += is_2q
        new = max(level[b] for b in bits) + 1
        new_2q = max(level_2q[b] for b in bits) + is_2q
        for b in bits:
            level[b] = new
            level_2q[b] = new_2q
    return CircuitMetrics(
        n_2q=n_2q,
        depth=max(level, default=0),
        depth_2q=max(level_2q, default=0),
        size=size,
        log_esp=log_esp(circuit, target),
    )

import math

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate, Measure, XGate
from qiskit.transpiler import InstructionProperties, Target

from qta.metrics import log_esp, measure


@pytest.fixture
def target():
    t = Target(num_qubits=3)
    t.add_instruction(XGate(), {(i,): InstructionProperties(error=0.01) for i in range(3)})
    t.add_instruction(
        CXGate(),
        {(0, 1): InstructionProperties(error=0.1), (1, 2): InstructionProperties(error=0.2)},
    )
    t.add_instruction(Measure(), {(i,): InstructionProperties(error=0.05) for i in range(3)})
    return t


def test_esp_is_product_of_fidelities(target):
    qc = QuantumCircuit(3, 1)
    qc.x(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.barrier()
    qc.measure(2, 0)
    expected = 0.99 * 0.9 * 0.8 * 0.95
    assert math.isclose(math.exp(log_esp(qc, target)), expected)


def test_counts_and_depth(target):
    qc = QuantumCircuit(3)
    qc.x(0)
    qc.x(1)
    qc.cx(0, 1)
    qc.barrier()
    qc.cx(1, 2)
    qc.x(0)
    m = measure(qc, target)
    assert m.n_2q == 2
    assert m.depth_2q == 2
    assert m.depth == 3
    assert m.size == 5

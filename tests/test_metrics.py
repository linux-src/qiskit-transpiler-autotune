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


def test_depth_matches_qiskit_on_transpiled_circuits():
    from qiskit.circuit.random import random_circuit
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime.fake_provider import FakeTorino

    backend = FakeTorino()
    pm = generate_preset_pass_manager(1, backend=backend, seed_transpiler=0)
    for seed in range(5):
        qc = random_circuit(6, 8, max_operands=2, measure=True, seed=seed)
        qc.barrier()
        out = pm.run(qc)
        m = measure(out, backend.target)
        def keep(i):
            return i.operation.name not in ("barrier", "delay")
        assert m.depth == out.depth(keep)
        assert m.depth_2q == out.depth(lambda i: keep(i) and i.operation.num_qubits == 2)
        assert m.n_2q == sum(1 for i in out.data if keep(i) and i.operation.num_qubits == 2)

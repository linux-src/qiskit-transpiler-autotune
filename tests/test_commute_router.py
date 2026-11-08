import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap

from qta.commute_router import Problem, build_pass_manager


def random_circuit(rng, qubits=5, size=40) -> QuantumCircuit:
    qc = QuantumCircuit(qubits)
    for _ in range(size):
        a, b = (int(x) for x in rng.choice(qubits, 2, replace=False))
        kind = rng.integers(5)
        if kind == 0:
            qc.h(a)
        elif kind == 1:
            qc.cx(a, b)
        elif kind == 2:
            qc.cz(a, b)
        elif kind == 3:
            qc.cp(float(rng.random()), a, b)
        else:
            qc.rzz(float(rng.random()), a, b)
    return qc


@pytest.mark.parametrize("commute", [False, True])
def test_compiled_circuit_keeps_the_unitary(commute):
    backend = GenericBackendV2(5, coupling_map=CouplingMap.from_line(5), seed=3)
    rng = np.random.default_rng(2)
    for seed in range(8):
        qc = random_circuit(rng)
        out = build_pass_manager(backend, seed, commute).run(qc)
        assert Operator.from_circuit(out).equiv(Operator(qc))


def test_commuting_gates_share_the_front():
    gates = [("cz", (0, 1), ()), ("cz", (0, 2), ()), ("cz", (0, 3), ())]
    assert Problem(gates, 4, commute=True).in_degree == [0, 0, 0]
    assert Problem(gates, 4, commute=False).in_degree == [0, 1, 1]


def test_hadamard_separates_commuting_gates():
    gates = [("cz", (0, 1), ()), ("h", (0,), ()), ("cz", (0, 2), ())]
    assert Problem(gates, 3, commute=True).in_degree == [0, 1, 1]

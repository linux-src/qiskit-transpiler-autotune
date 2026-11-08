import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from qta.commute_order import levels, reorder

GATES_1Q = ("h", "x", "z", "s", "t", "sx", "y")
GATES_2Q = ("cx", "cz", "cy")


def random_circuit(rng, qubits=5, size=60) -> QuantumCircuit:
    qc = QuantumCircuit(qubits)
    for _ in range(size):
        if rng.random() < 0.3:
            getattr(qc, rng.choice(GATES_1Q))(int(rng.integers(qubits)))
        elif rng.random() < 0.2:
            a, b = rng.choice(qubits, 2, replace=False)
            angle = float(rng.random())
            getattr(qc, rng.choice(("cp", "rzz", "rxx", "crz", "crx")))(angle, int(a), int(b))
        else:
            a, b = rng.choice(qubits, 2, replace=False)
            getattr(qc, rng.choice(GATES_2Q))(int(a), int(b))
    return qc


def test_reorder_keeps_the_unitary():
    rng = np.random.default_rng(1)
    for case in range(40):
        qc = random_circuit(rng)
        assert Operator(reorder(qc, case)).equiv(Operator(qc))


def test_star_of_cnots_with_common_target_is_one_level():
    qc = QuantumCircuit(4)
    for control in range(3):
        qc.cx(control, 3)
    assert levels(qc)[0] == [0, 0, 0]


def test_non_commuting_gates_stay_ordered():
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.h(1)
    qc.cx(0, 1)
    assert levels(qc)[0] == [0, 1, 2]

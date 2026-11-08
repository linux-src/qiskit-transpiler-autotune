import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap, generate_preset_pass_manager


def random_circuit(rng, qubits=5, size=50) -> QuantumCircuit:
    qc = QuantumCircuit(qubits)
    for _ in range(size):
        a, b = (int(x) for x in rng.choice(qubits, 2, replace=False))
        kind = rng.integers(7)
        if kind == 0:
            qc.h(a)
        elif kind == 1:
            qc.cx(a, b)
        elif kind == 2:
            qc.cz(a, b)
        elif kind == 3:
            qc.cp(float(rng.random()), a, b)
        elif kind == 4:
            qc.rzz(float(rng.random()), a, b)
        elif kind == 5:
            qc.rz(float(rng.random()), a)
        else:
            qc.rx(float(rng.random()), a)
    return qc


# On the unpatched build the variable has no effect and both cases are plain level 3
@pytest.mark.parametrize("commute", [False, True])
def test_level3_keeps_the_unitary(commute, monkeypatch):
    if commute:
        monkeypatch.setenv("QTA_SABRE_COMMUTE", "1")
    else:
        monkeypatch.delenv("QTA_SABRE_COMMUTE", raising=False)
    backend = GenericBackendV2(5, coupling_map=CouplingMap.from_line(5), seed=3)
    rng = np.random.default_rng(5)
    for seed in range(25):
        qc = random_circuit(rng)
        out = generate_preset_pass_manager(3, backend=backend, seed_transpiler=seed).run(qc)
        assert Operator.from_circuit(out).equiv(Operator(qc))

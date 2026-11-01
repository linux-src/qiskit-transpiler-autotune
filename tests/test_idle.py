import math

from qiskit import QuantumCircuit
from qiskit.transpiler import InstructionDurations

from qta.idle import idle_log_factor, idle_times


def test_idle_time_of_waiting_qubit():
    qc = QuantumCircuit(2)
    qc.x(1)
    qc.x(0)
    qc.cz(0, 1)
    for _ in range(3):
        qc.x(0)
    durations = InstructionDurations([("x", None, 10e-9, "s"), ("cz", None, 50e-9, "s")])
    idle = idle_times(qc, durations)
    assert math.isclose(idle[0], 0.0, abs_tol=1e-15)
    assert math.isclose(idle[1], 30e-9)


def test_idle_log_factor():
    coherence = {0: {"T1": 100e-6, "T2": 50e-6}}
    assert math.isclose(idle_log_factor({0: 10e-6}, coherence, "T2"), -0.2)

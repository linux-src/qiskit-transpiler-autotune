import pandas as pd
import pytest
from qiskit import QuantumCircuit

from qta.commute_select import commuting_share, depth_ratio, fit, selected, strategy_ratio


def circuit(*gates) -> QuantumCircuit:
    qc = QuantumCircuit(3)
    for name, *qubits in gates:
        getattr(qc, name)(*qubits)
    return qc


def test_commuting_gates_collapse_the_depth():
    qc = circuit(("cz", 0, 1), ("cz", 1, 2))
    assert depth_ratio(qc) == 0.5
    assert commuting_share(qc) == 1.0


def test_different_axes_keep_the_order():
    qc = circuit(("cx", 0, 1), ("cx", 1, 2))
    assert depth_ratio(qc) == 1.0
    assert commuting_share(qc) == 0.0


def test_hadamard_breaks_a_group():
    qc = circuit(("cz", 0, 1), ("h", 1), ("cz", 1, 2))
    assert depth_ratio(qc) == 1.0
    assert commuting_share(qc) == 0.0


def test_circuit_without_two_qubit_gates():
    qc = circuit(("h", 0))
    assert depth_ratio(qc) == 1.0
    assert commuting_share(qc) == 0.0


def test_unselected_circuit_keeps_the_plain_result():
    table = pd.DataFrame({"depth_ratio": [0.1, 0.9], "portfolio_n": [8, 12], "plain_n": [10, 10]})
    mask = selected(table, "depth_ratio", 0.5)
    assert mask.tolist() == [True, False]
    assert strategy_ratio(table, mask).tolist() == [0.8, 1.0]


def test_fit_separates_helped_from_hurt():
    table = pd.DataFrame({"depth_ratio": [0.1, 0.2, 0.8, 0.9], "portfolio_n": [8, 8, 12, 12], "plain_n": [10] * 4})
    assert fit({2: table}, "depth_ratio") == pytest.approx(0.5)

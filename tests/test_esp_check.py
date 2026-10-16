import math

import pandas as pd
from qiskit import QuantumCircuit
from qiskit.synthesis import synth_qft_full

from qta.esp_check import ideal_distribution, rank_agreement, with_measurements


def test_ideal_distribution_follows_measurement_order():
    qc = QuantumCircuit(3, 3)
    qc.x(0)
    qc.measure(0, 2)
    qc.measure(1, 0)
    qc.measure(2, 1)
    assert ideal_distribution(qc) == {"100": 1.0}


def test_ideal_distribution_of_ghz():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    dist = ideal_distribution(with_measurements(qc))
    assert set(dist) == {"000", "111"}
    assert math.isclose(dist["000"], 0.5)


def test_with_measurements_keeps_existing_ones():
    qc = synth_qft_full(3)
    qc.measure_all()
    assert with_measurements(qc) is qc


def test_rank_agreement_perfect_order():
    runs = pd.DataFrame({
        "device": "d", "circuit": "c",
        "esp": [0.1, 0.2, 0.3, 0.4], "fidelity": [0.5, 0.6, 0.7, 0.9],
    })
    row = rank_agreement(runs).iloc[0]
    assert math.isclose(row.spearman, 1.0)
    assert row.fidelity_of_esp_best == 0.9

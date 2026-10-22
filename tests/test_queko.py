from qiskit import QuantumCircuit

from qta.config import O3_DEFAULT
from qta.queko import device_target, layout_and_routing, measure


def test_circuit_that_fits_needs_no_swaps():
    # a path device and a circuit whose interactions form the same path: optimum is reachable
    target = device_target([(0, 1), (1, 2), (2, 3)])
    qc = QuantumCircuit(4)
    qc.cx(2, 0)
    qc.cx(0, 3)
    qc.cx(3, 1)
    row = measure(qc, layout_and_routing(O3_DEFAULT, target, seed=1), optimal_depth=3)
    assert row["n_swaps"] == 0
    assert row["depth_ratio"] == 1.0


def test_swaps_count_as_three_cnots():
    # a triangle of interactions cannot be embedded in a path
    target = device_target([(0, 1), (1, 2)])
    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 0)
    row = measure(qc, layout_and_routing(O3_DEFAULT, target, seed=1), optimal_depth=3)
    assert row["n_swaps"] >= 1
    assert row["depth"] >= 3 + 2 * row["n_swaps"]


def test_report_fair_search_stays_within_sabre_time():
    import pandas as pd

    from qta.queko_report import per_circuit, summary

    rows = []
    for run, (ratio, t) in enumerate([(2.0, 1.0), (1.5, 1.0)]):
        rows.append({"circuit": "c", "set": "BNTF", "device": "d", "optimal_depth": 10,
                     "method": "sabre", "run": run, "depth_ratio": ratio, "n_swaps": 1, "time_s": t})
    for run, (ratio, t) in enumerate([(1.4, 1.5), (1.0, 1.0), (1.1, 0.1)]):
        rows.append({"circuit": "c", "set": "BNTF", "device": "d", "optimal_depth": 10,
                     "method": "sabre_search", "run": run, "depth_ratio": ratio, "n_swaps": 1,
                     "time_s": t})
    pc = per_circuit(pd.DataFrame(rows))
    assert pc.sabre_best.iloc[0] == 1.5
    # sabre seeds took 2 s, so only the first search trial (1.5 s) fits
    assert pc.sabre_search_fair.iloc[0] == 1.4
    assert pc.sabre_search_best.iloc[0] == 1.0
    assert summary(pc).sabre_search_best_optimal_share.iloc[0] == 1.0

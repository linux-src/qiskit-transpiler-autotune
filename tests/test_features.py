from qiskit import QuantumCircuit

from qta.features import extract, interaction_graph


def test_interaction_graph_counts_pairs():
    qc = QuantumCircuit(4)
    qc.cx(0, 1)
    qc.cx(1, 0)
    qc.cz(2, 3)
    qc.h(3)
    g = interaction_graph(qc)
    assert sorted(g.weighted_edge_list()) == [(0, 1, 2), (2, 3, 1)]


def test_features_of_star_and_line():
    star = QuantumCircuit(5)
    for t in range(1, 5):
        star.cx(0, t)
    f = extract(star)
    assert f["n_2q"] == 4
    assert f["ig_max_degree"] == 4
    assert f["ig_frac_degree_gt3"] == 0.2
    assert f["ig_diameter"] == 2
    assert f["ig_is_tree_like"] == 1.0

    line = QuantumCircuit(4)
    line.ccx(0, 1, 2)
    line.cx(2, 3)
    f = extract(line)
    assert f["n_2q"] == 7
    assert f["ig_components"] == 1
    assert f["ig_transitivity"] > 0

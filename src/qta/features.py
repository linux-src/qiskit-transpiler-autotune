"""Device-independent features of a logical circuit."""

from __future__ import annotations

import numpy as np
import rustworkx as rx
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager

_IGNORED = {"barrier", "delay", "measure", "reset"}
_UNROLL = generate_preset_pass_manager(0, basis_gates=["cx", "u"])


def interaction_graph(circuit: QuantumCircuit) -> rx.PyGraph:
    """Undirected graph of qubits; edge weight is the number of two-qubit gates on the pair."""
    index = {q: i for i, q in enumerate(circuit.qubits)}
    graph = rx.PyGraph()
    graph.add_nodes_from(range(circuit.num_qubits))
    edges: dict[tuple[int, int], int] = {}
    for inst in circuit.data:
        if inst.name in _IGNORED or len(inst.qubits) != 2:
            continue
        a, b = sorted(index[q] for q in inst.qubits)
        edges[a, b] = edges.get((a, b), 0) + 1
    graph.add_edges_from([(a, b, w) for (a, b), w in edges.items()])
    return graph


def extract(circuit: QuantumCircuit) -> dict[str, float]:
    qc = _UNROLL.run(circuit)
    n = qc.num_qubits
    ops = [inst for inst in qc.data if inst.name not in _IGNORED]
    n_2q = sum(1 for inst in ops if len(inst.qubits) == 2)

    graph = interaction_graph(qc)
    degrees = np.array([graph.degree(i) for i in graph.node_indices()], dtype=float)
    weights = np.array(graph.edges(), dtype=float)
    active = degrees > 0
    components = rx.connected_components(graph)
    largest = max(components, key=len) if components else set()
    sub = graph.subgraph(sorted(largest))
    dist = rx.distance_matrix(sub) if sub.num_nodes() > 1 else np.zeros((1, 1))
    max_edges = n * (n - 1) / 2

    return {
        "num_qubits": n,
        "n_ops": len(ops),
        "n_2q": n_2q,
        "frac_2q": n_2q / len(ops) if ops else 0.0,
        "n_2q_per_qubit": n_2q / n,
        "depth": qc.depth(lambda i: i.name not in _IGNORED),
        "depth_2q": qc.depth(lambda i: i.name not in _IGNORED and len(i.qubits) == 2),
        "ig_edges": graph.num_edges(),
        "ig_density": graph.num_edges() / max_edges if max_edges else 0.0,
        "ig_max_degree": degrees.max(initial=0),
        "ig_mean_degree": degrees[active].mean() if active.any() else 0.0,
        "ig_std_degree": degrees[active].std() if active.any() else 0.0,
        "ig_frac_degree_gt3": float((degrees > 3).mean()),
        "ig_components": sum(1 for c in components if len(c) > 1),
        "ig_diameter": float(dist.max()),
        "ig_transitivity": rx.transitivity(graph) if graph.num_edges() else 0.0,
        "ig_weight_cv": weights.std() / weights.mean() if weights.size else 0.0,
        "ig_is_tree_like": float(graph.num_edges() <= active.sum() - 1),
    }

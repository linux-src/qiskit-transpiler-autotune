"""SABRE-style router in Python that may use the freedom of commuting gates.

    python -m qta.commute_router --out results/router --circuits-dir results/circuits

With ``commute=False`` a gate waits for every earlier gate on its wires, as in SABRE.
With ``commute=True`` it waits only for gates it may not commute with, so the router picks
the order from the current placement of qubits. Everything else, including the swap score,
is shared, which isolates the effect. This is a prototype for measuring that effect:
speed is not comparable with Qiskit's Rust implementation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit.library import SwapGate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.passmanager.flow_controllers import ConditionalController
from qiskit.transpiler import AnalysisPass, Layout, PassManager, TransformationPass
from qiskit.transpiler.passes import SetLayout, VF2Layout
from qiskit.transpiler.passes.layout.vf2_layout import VF2LayoutStopReason
from qiskit.transpiler.preset_passmanagers import common, generate_preset_pass_manager

from qta.circuits import DATASET, get_device, load_suite
from qta.commute_order import AXIS_1Q, AXIS_2Q
from qta.metrics import measure
from qta.runner import write_environment

LOOKAHEAD_WEIGHT = 0.5
LOOKAHEAD_SIZE = 20
DECAY_STEP = 0.001
DECAY_RESET = 5


def node_axes(name: str, qubits: tuple, clbits: tuple) -> list:
    axes = [None] * len(qubits)
    if clbits:
        return axes
    if len(qubits) == 1 and name in AXIS_1Q:
        axes[0] = AXIS_1Q[name]
    elif len(qubits) == 2 and name in AXIS_2Q:
        axes = list(AXIS_2Q[name])
    return axes


class Problem:
    """Gates as indices: for each one, its qubits and the gates that must wait for it."""

    def __init__(self, gates: list[tuple], num_qubits: int, commute: bool):
        self.qubits = [g[1] for g in gates]
        self.num_qubits = num_qubits
        self.is_2q = [len(g[1]) == 2 and not g[2] and g[0] != "barrier" for g in gates]
        self.succ: list[list[int]] = [[] for _ in gates]
        self.in_degree = [0] * len(gates)
        state: dict = {}  # wire -> (axis, current same-axis group, group before it)
        for i, (name, qubits, clbits) in enumerate(gates):
            axes = node_axes(name, qubits, clbits) if commute else [None] * len(qubits)
            wires = [(("q", q), a) for q, a in zip(qubits, axes)] + [(("c", c), None) for c in clbits]
            need: set[int] = set()
            for wire, axis in wires:
                cur_axis, group, before = state.get(wire, (None, [], []))
                if axis is not None and axis == cur_axis:
                    need.update(before)
                    group.append(i)
                else:
                    need.update(group)
                    state[wire] = (axis, [i], group)
            for j in need:
                self.succ[j].append(i)
            self.in_degree[i] = len(need)


class Device:
    def __init__(self, target):
        coupling = target.build_coupling_map()
        coupling.make_symmetric()
        self.dist = np.asarray(coupling.distance_matrix)
        self.num_qubits = coupling.size()
        self.neighbors: list[list[int]] = [[] for _ in range(self.num_qubits)]
        for a, b in coupling.get_edges():
            self.neighbors[a].append(b)
        self.dist_list = self.dist.tolist()


def route(problem: Problem, device: Device, l2p: list[int], rng) -> tuple[list, list[int], int]:
    """Greedy swap insertion. Returns output events, final placement and the number of swaps."""
    l2p = list(l2p)
    p2l = [-1] * device.num_qubits
    for logical, physical in enumerate(l2p):
        p2l[physical] = logical
    in_degree = list(problem.in_degree)
    is_2q, gate_qubits, succ = problem.is_2q, problem.qubits, problem.succ
    dist, neighbors = device.dist_list, device.neighbors
    ready = [i for i, d in enumerate(in_degree) if d == 0]
    front: dict[int, tuple[int, int]] = {}
    front_on: list[set[int]] = [set() for _ in range(problem.num_qubits)]
    events: list = []
    decay = [1.0] * device.num_qubits
    swaps = since_reset = stuck = 0

    def finish(gate: int) -> None:
        events.append(("gate", gate, [l2p[q] for q in gate_qubits[gate]]))
        for s in succ[gate]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                ready.append(s)

    def advance() -> bool:
        progress = False
        while True:
            while ready:
                gate = ready.pop()
                if is_2q[gate]:
                    front[gate] = gate_qubits[gate]
                    front_on[gate_qubits[gate][0]].add(gate)
                    front_on[gate_qubits[gate][1]].add(gate)
                else:
                    finish(gate)
            done = [g for g, (a, b) in front.items() if dist[l2p[a]][l2p[b]] == 1]
            if not done:
                return progress
            for g in done:
                a, b = front.pop(g)
                front_on[a].discard(g)
                front_on[b].discard(g)
                finish(g)
            progress = True

    def lookahead() -> tuple[list[tuple[int, int]], list[set[int]]]:
        seen: set[int] = set()
        pairs: list[tuple[int, int]] = []
        queue = list(front)
        while queue and len(pairs) < LOOKAHEAD_SIZE:
            next_queue = []
            for gate in queue:
                for s in succ[gate]:
                    if s not in seen:
                        seen.add(s)
                        next_queue.append(s)
                        if is_2q[s] and len(pairs) < LOOKAHEAD_SIZE:
                            pairs.append(gate_qubits[s])
            queue = next_queue
        on: list[set[int]] = [set() for _ in range(problem.num_qubits)]
        for idx, (a, b) in enumerate(pairs):
            on[a].add(idx)
            on[b].add(idx)
        return pairs, on

    def apply_swap(p: int, q: int) -> None:
        nonlocal swaps
        a, b = p2l[p], p2l[q]
        p2l[p], p2l[q] = b, a
        if a >= 0:
            l2p[a] = q
        if b >= 0:
            l2p[b] = p
        events.append(("swap", p, q))
        swaps += 1

    advance()
    ext, ext_on = lookahead()
    while front:
        # Without a way out the greedy score can oscillate forever
        if stuck > 10 * max(1, int(device.dist.max())):
            a, b = min(front.values(), key=lambda g: dist[l2p[g[0]]][l2p[g[1]]])
            while dist[l2p[a]][l2p[b]] > 1:
                here = l2p[a]
                step = min(neighbors[here], key=lambda n: dist[n][l2p[b]])
                apply_swap(here, step)
            stuck = 0
            decay = [1.0] * device.num_qubits
            if advance():
                ext, ext_on = lookahead()
            continue

        n_front, n_ext = len(front), max(len(ext), 1)
        front_sum = sum(dist[l2p[a]][l2p[b]] for a, b in front.values())
        ext_sum = sum(dist[l2p[a]][l2p[b]] for a, b in ext)
        candidates = set()
        for a, b in front.values():
            for p in (l2p[a], l2p[b]):
                for n in neighbors[p]:
                    candidates.add((min(p, n), max(p, n)))

        best, best_score = [], float("inf")
        for p, q in sorted(candidates):
            a, b = p2l[p], p2l[q]
            # new position of a logical qubit if this swap were applied
            place = lambda x: q if x == a else p if x == b else l2p[x]
            df = de = 0.0
            for touched, on, pairs in ((front_on, front_on, None), (ext_on, ext_on, ext)):
                changed = set()
                if a >= 0:
                    changed |= on[a]
                if b >= 0:
                    changed |= on[b]
                delta = 0
                for g in changed:
                    x, y = gate_qubits[g] if pairs is None else pairs[g]
                    delta += dist[place(x)][place(y)] - dist[l2p[x]][l2p[y]]
                if pairs is None:
                    df = delta
                else:
                    de = delta
            score = max(decay[p], decay[q]) * ((front_sum + df) / n_front
                                               + LOOKAHEAD_WEIGHT * (ext_sum + de) / n_ext)
            if score < best_score - 1e-9:
                best, best_score = [(p, q)], score
            elif abs(score - best_score) <= 1e-9:
                best.append((p, q))
        p, q = best[int(rng.integers(len(best)))]
        apply_swap(p, q)
        decay[p] += DECAY_STEP
        decay[q] += DECAY_STEP
        since_reset += 1
        stuck += 1
        if since_reset >= DECAY_RESET:
            decay = [1.0] * device.num_qubits
            since_reset = 0
        if advance():
            stuck = 0
            decay = [1.0] * device.num_qubits
            ext, ext_on = lookahead()
    return events, l2p, swaps


def gates_of(dag: DAGCircuit) -> tuple[list, list[tuple]]:
    index = {q: i for i, q in enumerate(dag.qubits)}
    clindex = {c: i for i, c in enumerate(dag.clbits)}
    nodes = list(dag.topological_op_nodes())
    return nodes, [(n.op.name, tuple(index[q] for q in n.qargs), tuple(clindex[c] for c in n.cargs))
                   for n in nodes]


def random_start(device: Device, width: int, rng) -> list[int]:
    """A compact random region: scattered qubits would only slow the search down."""
    root = int(rng.integers(device.num_qubits))
    nearest = np.argsort(device.dist[root] + rng.random(device.num_qubits) * 0.5)[:width]
    return [int(p) for p in rng.permutation(nearest)]


def search_layout(gates, width, device, commute, trials, iterations, rng) -> list[int]:
    forward = Problem(gates, width, commute)
    backward = Problem(list(reversed(gates)), width, commute)
    best, best_swaps = None, None
    for _ in range(trials):
        layout = random_start(device, width, rng)
        for _ in range(iterations):
            _, layout, _ = route(forward, device, layout, rng)
            _, layout, _ = route(backward, device, layout, rng)
        _, _, swaps = route(forward, device, layout, rng)
        if best_swaps is None or swaps < best_swaps:
            best, best_swaps = layout, swaps
    return best


class CommuteLayout(AnalysisPass):
    def __init__(self, target, commute: bool, seed: int, trials: int = 4, iterations: int = 3):
        super().__init__()
        self.device = Device(target)
        self.commute, self.seed, self.trials, self.iterations = commute, seed, trials, iterations

    def run(self, dag):
        _, gates = gates_of(dag)
        rng = np.random.default_rng(self.seed)
        layout = search_layout(gates, dag.num_qubits(), self.device, self.commute,
                               self.trials, self.iterations, rng)
        self.property_set["layout"] = Layout({q: layout[i] for i, q in enumerate(dag.qubits)})


class CommuteRoute(TransformationPass):
    def __init__(self, target, commute: bool, seed: int, trials: int = 8):
        super().__init__()
        self.device = Device(target)
        self.commute, self.seed, self.trials = commute, seed, trials

    def run(self, dag):
        nodes, gates = gates_of(dag)
        problem = Problem(gates, dag.num_qubits(), self.commute)
        rng = np.random.default_rng(self.seed)
        start = list(range(dag.num_qubits()))
        best = min((route(problem, self.device, start, rng) for _ in range(self.trials)),
                   key=lambda result: result[2])
        events, final, _ = best
        out = dag.copy_empty_like()
        swap = SwapGate()
        for event in events:
            if event[0] == "swap":
                out.apply_operation_back(swap, (out.qubits[event[1]], out.qubits[event[2]]), (), check=False)
            else:
                node = nodes[event[1]]
                out.apply_operation_back(node.op, tuple(out.qubits[p] for p in event[2]), node.cargs, check=False)
        self.property_set["final_layout"] = Layout({q: final[i] for i, q in enumerate(dag.qubits)})
        return out


def _no_perfect_layout(property_set) -> bool:
    if property_set["layout"] is None:
        return True
    reason = property_set["VF2Layout_stop_reason"]
    return reason is not None and reason is not VF2LayoutStopReason.SOLUTION_FOUND


def build_pass_manager(backend, seed: int, commute: bool) -> PassManager:
    """Level 3 with layout and routing replaced; a perfect VF2 layout is kept, as in O3."""
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=seed)
    target = backend.target
    layout = PassManager()
    layout.append(SetLayout(None))
    layout.append(ConditionalController(VF2Layout(seed=-1, call_limit=(30_000_000, 100_000), target=target),
                                        condition=lambda ps: not ps["layout"]))
    layout.append(ConditionalController(CommuteLayout(target, commute, seed), condition=_no_perfect_layout))
    layout.append(common.generate_embed_passmanager(target).to_flow_controller())
    pm.layout = layout
    pm.routing = common.generate_routing_passmanager(
        CommuteRoute(target, commute, seed), target, coupling_map=target.build_coupling_map(),
        vf2_call_limit=30_000_000, vf2_max_trials=100_000, seed_transpiler=-1,
        use_barrier_before_measurement=True)
    return pm


def one_task(task) -> list[dict]:
    device, cid, qc, seeds = task
    backend = get_device(device)
    rows = []
    for seed in range(seeds):
        for commute in (False, True):
            out = build_pass_manager(backend, seed, commute).run(qc)
            m = measure(out, backend.target)
            rows.append({"device": device, "circuit": cid, "num_qubits": qc.num_qubits,
                         "commute": commute, "seed": seed, "n_2q": m.n_2q, "depth": m.depth,
                         "log_esp": m.log_esp})
    return rows


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--circuits-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--max-qubits", type=int, default=20)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)
    suite = [(name, size) for name, size in DATASET if size <= args.max_qubits]
    circuits = load_suite(args.circuits_dir, suite)
    tasks = [(device, cid, qc, args.seeds) for device in ("torino", "sherbrooke") for cid, qc in circuits.items()]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = [row for chunk in pool.map(one_task, tasks) for row in chunk]
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / "runs.csv", index=False)
    write_environment(args.out, experiment="commute_router", args=vars(args))


if __name__ == "__main__":
    main()

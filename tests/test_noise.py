import math

import pytest
from qiskit.synthesis import synth_qft_full
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke, FakeTorino

from qta.noise import prune_target

BACKENDS = [FakeSherbrooke(), FakeTorino()]


def _two_qubit(target):
    return {
        q: p.error
        for name in target.operation_names
        for q, p in target[name].items()
        if q is not None and len(q) == 2
    }


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.name)
def test_infinite_tau_keeps_level3_output(backend):
    qc = synth_qft_full(8)
    qc.measure_all()
    expected = generate_preset_pass_manager(3, backend=backend, seed_transpiler=5).run(qc)
    pruned = prune_target(backend.target, math.inf)
    assert generate_preset_pass_manager(3, target=pruned, seed_transpiler=5).run(qc) == expected


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.name)
@pytest.mark.parametrize("tau", [1.0, 0.02])
def test_pruned_couplers_are_gone_in_both_directions(backend, tau):
    full = _two_qubit(backend.target)
    kept = _two_qubit(prune_target(backend.target, tau))
    bad = {frozenset(q) for q, e in full.items() if e >= tau}
    assert bad
    assert all(e < tau for e in kept.values())
    assert not any(frozenset(q) in bad for q in kept)
    assert set(kept) == {q for q in full if frozenset(q) not in bad}


def test_routed_circuit_avoids_pruned_couplers():
    backend = FakeSherbrooke()
    qc = synth_qft_full(16)
    qc.measure_all()
    out = generate_preset_pass_manager(
        3, target=prune_target(backend.target, 0.02), seed_transpiler=1
    ).run(qc)
    full = _two_qubit(backend.target)
    for inst in out.data:
        if len(inst.qubits) == 2 and inst.name != "barrier":
            qargs = tuple(out.find_bit(q).index for q in inst.qubits)
            assert full[qargs] < 0.02

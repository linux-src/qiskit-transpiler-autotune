import numpy as np
import pytest
from qiskit.circuit.library import efficient_su2
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Operator
from qiskit.synthesis import synth_qft_full
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke, FakeTorino

from qta.optimization import (
    O3_OPTIMIZATION, SEQUENCE_PASSES, OptimizationConfig, PassSequence, build_pass_manager, sample,
    sample_sequence,
)
from qta.opt_search import configurations

BACKENDS = [FakeSherbrooke(), FakeTorino()]


def circuits():
    qft = synth_qft_full(8)
    qft.measure_all()
    su2 = efficient_su2(10, entanglement="full", reps=1)
    su2 = su2.assign_parameters([0.1] * su2.num_parameters)
    return [qft, su2]


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.name)
@pytest.mark.parametrize("seed", [1, 7])
def test_default_config_matches_level3(backend, seed):
    for qc in circuits():
        expected = generate_preset_pass_manager(3, backend=backend, seed_transpiler=seed).run(qc)
        assert build_pass_manager(O3_OPTIMIZATION, backend, seed).run(qc) == expected


def test_sampled_configs_preserve_unitary_and_basis():
    backend = GenericBackendV2(5, coupling_map=[[0, 1], [1, 2], [2, 3], [3, 4]], seed=11)
    qc = synth_qft_full(5)
    rng = np.random.default_rng(3)
    configs = [sample(rng) for _ in range(6)] + [
        OptimizationConfig(peephole=False, remove_identity=False, optimize_1q=False,
                           commutative_cancellation=False, loop="once", vf2_post_layout=False),
        OptimizationConfig(inverse_cancellation=True, consolidate=True, loop="fixed_point"),
    ]
    basis = set(backend.target.operation_names)
    for config in configs:
        out = build_pass_manager(config, backend, seed=2).run(qc)
        assert {inst.name for inst in out.data} <= basis
        assert Operator.from_circuit(out).equiv(Operator(qc))


def test_configurations_start_with_level3_and_are_unique():
    configs = configurations(12)
    assert configs[0] == O3_OPTIMIZATION
    assert len(set(configs)) == 12


def test_fixed_point_loop_terminates_when_passes_oscillate():
    backend = FakeSherbrooke()
    # Resynthesis alone keeps rewriting the same blocks without the size settling.
    config = OptimizationConfig(peephole=False, remove_identity=False, optimize_1q=False,
                                commutative_cancellation=False, consolidate=True, loop="fixed_point")
    qc = synth_qft_full(6)
    qc.measure_all()
    out = build_pass_manager(config, backend, seed=0).run(qc)
    assert {inst.name for inst in out.data} - {"barrier"} <= set(backend.target.operation_names)


def test_pass_sequences_preserve_unitary_and_basis():
    backend = GenericBackendV2(5, coupling_map=[[0, 1], [1, 2], [2, 3], [3, 4]], seed=11)
    qc = synth_qft_full(5)
    rng = np.random.default_rng(5)
    sequences = [sample_sequence(rng) for _ in range(6)] + [PassSequence(tuple(SEQUENCE_PASSES))]
    basis = set(backend.target.operation_names)
    for sequence in sequences:
        out = build_pass_manager(sequence, backend, seed=2).run(qc)
        assert {inst.name for inst in out.data} <= basis
        assert Operator.from_circuit(out).equiv(Operator(qc))


def test_sequence_configurations_start_with_level3():
    configs = configurations(5, space="sequence")
    assert configs[0] == O3_OPTIMIZATION
    assert all(isinstance(c, PassSequence) for c in configs[1:])

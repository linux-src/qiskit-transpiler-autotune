import pytest
from qiskit.circuit.library import efficient_su2
from qiskit.synthesis import synth_qft_full
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke, FakeTorino

from qta.config import O3_DEFAULT, TuningConfig, build_pass_manager

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
        assert build_pass_manager(O3_DEFAULT, backend, seed).run(qc) == expected


@pytest.mark.parametrize("routing", ["joint", "separate"])
@pytest.mark.parametrize("heuristic", ["basic", "lookahead", "decay"])
def test_tuned_output_fits_device(routing, heuristic):
    backend = FakeSherbrooke()
    config = TuningConfig(
        use_vf2=False, max_iterations=1, layout_trials=2, swap_trials=2,
        routing=routing, heuristic=heuristic, routing_trials=2,
    )
    for qc in circuits():
        out = build_pass_manager(config, backend, seed=3).run(qc)
        for inst in out.data:
            if inst.operation.name == "barrier":
                continue
            qargs = tuple(out.find_bit(q).index for q in inst.qubits)
            assert backend.target.instruction_supported(inst.operation.name, qargs)

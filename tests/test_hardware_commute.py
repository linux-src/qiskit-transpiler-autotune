import pandas as pd
import pytest

from qta.hardware_commute import choose


def block(times, esps):
    return pd.DataFrame({"seed": range(len(times)), "time_s": times, "log_esp": esps})


def test_portfolio_stays_within_the_plain_budget():
    off = block([1.0, 1.0], [-3.0, -2.0])
    on = block([1.5, 1.5], [-2.5, -0.5])
    result = choose(off, on)
    assert result["plain"][1].log_esp == -2.0
    # on0 (1.5) fits, off0 would reach 2.5 > 2.0, so the better on1 is never reached
    assert result["portfolio"][0] == "on"
    assert result["portfolio"][1].log_esp == -2.5


def test_portfolio_can_pick_the_plain_mode():
    off = block([2.0, 2.0], [-1.0, -3.0])
    on = block([1.0, 1.0], [-2.0, -2.5])
    assert choose(off, on)["portfolio"][0] == "off"


def test_portfolio_cycles_through_the_given_modes():
    off = block([1.0, 1.0, 1.0], [-3.0, -3.0, -3.0])
    absorb = block([0.9, 0.9, 0.9], [-2.0, -2.5, -1.0])
    both = block([1.0, 1.0, 1.0], [-1.5, -2.5, -0.1])
    result = choose(off, None, {"absorb": absorb, "commute_absorb": both})
    # absorb0 (0.9) + both0 (1.9) fit the 3.0 budget, absorb1 would reach 2.8 and fits, both1 would not
    assert result["portfolio"][0] == "commute_absorb"
    assert result["portfolio"][1].log_esp == -1.5


def test_mode_name_follows_the_switches(monkeypatch):
    from qta.hardware_commute import mode_name
    for commute, absorb, name in ((None, None, "off"), ("1", None, "on"), (None, "1", "absorb"), ("1", "1", "commute_absorb")):
        for var, value in (("QTA_SABRE_COMMUTE", commute), ("QTA_SABRE_ABSORB", absorb)):
            if value:
                monkeypatch.setenv(var, value)
            else:
                monkeypatch.delenv(var, raising=False)
        assert mode_name() == name


def test_region_target_and_place_keep_the_circuit_on_the_chosen_qubits():
    from qiskit import QuantumCircuit
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit.quantum_info import Statevector
    from qiskit.transpiler import CouplingMap, generate_preset_pass_manager

    from qta.hardware_commute import place, region_target, used_qubits

    backend = GenericBackendV2(9, coupling_map=CouplingMap.from_grid(3, 3), seed=1)
    qubits = [4, 5, 8, 7]
    target = region_target(backend.target, qubits)
    assert target.num_qubits == 4
    assert set(target.build_coupling_map().get_edges()) == {(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (0, 3), (3, 0)}

    qc = QuantumCircuit(4)
    qc.h(0)
    for a in range(3):
        qc.cx(0, a + 1)
    small = generate_preset_pass_manager(3, target=target, seed_transpiler=0).run(qc)
    full = place(small, qubits, 9)
    assert used_qubits(full) == sorted(qubits)
    probabilities = Statevector(full).probabilities_dict(qargs=sorted(qubits))
    large = {key: value for key, value in probabilities.items() if value > 1e-9}
    assert large == pytest.approx({"0000": 0.5, "1111": 0.5})


def test_exclusions_use_flatness_equal_counts_and_duplicates():
    from qta.hardware_commute import exclusions, perplexity

    def pair(plain, portfolio):
        return {"plain": ("off", pd.Series({"n_2q": plain})), "portfolio": ("absorb", pd.Series({"n_2q": portfolio}))}

    peaked, flat = {"00": 1.0}, {f"{i:02b}": 0.25 for i in range(4)}
    assert perplexity(flat) == 4 and perplexity(peaked) == 1
    chosen = {("a_4", 0): pair(10, 8), ("b_4", 0): pair(10, 8), ("c_4", 0): pair(7, 7), ("d_4", 0): pair(9, 6)}
    ideal = {"a_4": peaked, "b_4": peaked, "c_4": peaked, "d_4": flat}
    result = exclusions(chosen, ideal, max_perplexity=2, drop_equal=True)
    assert set(result) == {"b_4", "c_4", "d_4"}
    assert "a_4" in result["b_4"]

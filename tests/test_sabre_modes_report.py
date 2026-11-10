import pandas as pd
import pytest

from qta.sabre_modes_report import bootstrap_interval, compare, interleave, summary, within_budget


def frame(n_2q, time_s=1.0):
    return pd.DataFrame({"device": "d", "circuit": "c_4", "num_qubits": 4, "seed": range(len(n_2q)),
                         "n_2q": n_2q, "log_esp": [-0.01 * n for n in n_2q], "time_s": time_s})


def test_within_budget_always_runs_once_and_stops_before_overrun():
    runs = [(3.0, 10, -1.0), (1.0, 8, -0.8), (1.0, 5, -0.5)]
    assert within_budget(runs, 1.0) == (10, -1.0, 1)
    assert within_budget(runs, 4.5) == (8, -0.8, 2)


def test_interleave_cycles_modes_in_order():
    assert interleave({"a": [1, 2], "b": [10, 20]}, ("a", "b")) == [1, 10, 2, 20]


def test_slower_mode_gets_fewer_runs_in_the_same_budget():
    runs = {"off": frame([10, 9, 12, 11]), "absorb": frame([8, 9, 7, 6], time_s=2.0)}
    table = compare(runs, {"absorb": ("absorb",)}, k=4)
    row = table.iloc[0]
    assert (row.plain_n, row.absorb_n, row.absorb_runs) == (9, 8, 2)
    result = summary(table, {"absorb": ("absorb",)}, 4)[0]
    assert result["n2q"] == pytest.approx(8 / 9)
    assert (result["wins"], result["losses"]) == (1, 0)


def test_bootstrap_interval_contains_the_mean_of_constant_ratios():
    low, high = bootstrap_interval([0.9] * 10)
    assert low == pytest.approx(0.9) and high == pytest.approx(0.9)

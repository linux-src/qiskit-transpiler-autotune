import pandas as pd

from qta.analysis import aggregate, summarize


def test_random_search_limited_by_level3_budget():
    baseline = pd.DataFrame({
        "device": "d", "circuit": "c", "num_qubits": 4, "level": 3,
        "seed": [0, 1, 2], "n_2q": [12, 10, 11], "depth": [5, 5, 5],
        "log_esp": [-0.7, -0.5, -0.6], "time_s": [1.0, 1.0, 1.0],
    })
    search = pd.DataFrame({
        "device": "d", "circuit": "c", "trial": [0, 1, 2, 3],
        "n_2q": [11, 9, 9, 7], "depth": [5, 6, 4, 4], "log_esp": [-0.7, -0.4, -0.4, -0.2],
        "time_s": [1.0, 1.0, 0.5, 5.0],
        "use_vf2": True, "max_iterations": 1, "layout_trials": 1, "swap_trials": 1,
        "routing": "joint", "heuristic": "decay", "routing_trials": 1,
    })
    row = summarize(baseline, search).iloc[0]
    assert row.budget_s == 3.0
    assert row.o3_best_n2q == 10
    assert row.rs_budget_trials == 3
    assert row.rs_budget_n2q == 9
    assert row.rs_budget_depth == 4
    assert row.rs_all_n2q == 7


def test_esp_objective_picks_highest_esp():
    import numpy as np

    baseline = pd.DataFrame({
        "device": "d", "circuit": "c", "num_qubits": 4, "level": 3,
        "seed": [0, 1], "n_2q": [10, 12], "depth": [5, 5],
        "log_esp": np.log([0.5, 0.6]), "time_s": [1.0, 1.0],
    })
    search = pd.DataFrame({
        "device": "d", "circuit": "c", "trial": [0, 1],
        "n_2q": [9, 11], "depth": [5, 5], "log_esp": np.log([0.55, 0.66]),
        "time_s": [0.5, 0.5],
        "use_vf2": True, "max_iterations": 1, "layout_trials": 1, "swap_trials": 1,
        "routing": "joint", "heuristic": "decay", "routing_trials": 1,
    })
    s = summarize(baseline, search, "esp")
    row = s.iloc[0]
    assert row.o3_best_n2q == 12
    assert row.rs_budget_n2q == 11
    agg = aggregate(s, "esp").iloc[0]
    assert np.isclose(agg.rs_budget_vs_o3_best, 1.1)

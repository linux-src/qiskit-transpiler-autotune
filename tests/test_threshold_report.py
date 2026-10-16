import math

import numpy as np
import pandas as pd

from qta.threshold_report import INF, aggregate, per_circuit, portfolio


def _runs(values: dict[float, list[float]], circuit="c", device="d") -> pd.DataFrame:
    rows = []
    for tau, logs in values.items():
        for seed, v in enumerate(logs):
            rows.append({"device": device, "circuit": circuit, "tau": tau, "seed": seed,
                         "log_esp": v, "esp": math.exp(v), "n_2q": 10, "time_s": 1.0})
    return pd.DataFrame(rows)


def test_aggregate_ratio_of_best_runs():
    runs = _runs({INF: [-1.0, -2.0, -np.inf, -1.5], 1.0: [-0.5, -2.0, -3.0, -1.0]})
    agg = aggregate(per_circuit(runs), min_esp=1e-3).iloc[0]
    assert math.isclose(agg.best_esp_ratio_exec, math.exp(0.5))
    assert agg.control_dead_runs == 1 and agg.dead_runs == 0
    assert agg.better_exec == 1 and math.isclose(agg.time_ratio, 1.0)


def test_portfolio_of_identical_taus_matches_control():
    logs = [-1.0, -1.2, -0.8, -1.5]
    runs = _runs({INF: logs, 1.0: logs})
    p = portfolio(runs, [INF], seeds=4, min_esp=1e-3)
    assert math.isclose(p.gain.iloc[0], 0.0)
    split = portfolio(runs, [INF, 1.0], seeds=4, min_esp=1e-3)
    assert split.gain.iloc[0] <= 0.0

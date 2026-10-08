import itertools

import numpy as np

from qta.headroom import allocate, expected_best


def test_expected_best_matches_enumeration():
    values = np.array([3.0, -1.0, 2.5, 7.0, 0.0])
    for k in range(1, 6):
        draws = [max(c) for c in itertools.combinations(values, k)]
        assert np.isclose(expected_best(values, k), np.mean(draws))


def test_allocate_respects_budget_and_cap():
    seeds = allocate(np.array([10.0, 1.0, 0.0]), total=12, cap=8)
    assert seeds.sum() == 12
    assert seeds.min() >= 1
    assert seeds.max() <= 8
    assert seeds[0] == 8
    assert seeds[1] > seeds[2]

import numpy as np

from qta.search_space import HEURISTICS, MAX_ITERATIONS, ROUTING, TRIALS, sample


def test_samples_stay_in_bounds_and_are_reproducible():
    assert sample(np.random.default_rng(5)) == sample(np.random.default_rng(5))

    rng = np.random.default_rng(0)
    configs = [sample(rng) for _ in range(500)]
    for c in configs:
        assert MAX_ITERATIONS[0] <= c.max_iterations <= MAX_ITERATIONS[1]
        for t in (c.layout_trials, c.swap_trials, c.routing_trials):
            assert TRIALS[0] <= t <= TRIALS[1]
        assert c.routing in ROUTING
        assert c.heuristic in HEURISTICS
    assert {c.routing for c in configs} == set(ROUTING)
    assert {c.heuristic for c in configs} == set(HEURISTICS)
    assert {c.use_vf2 for c in configs} == {True, False}

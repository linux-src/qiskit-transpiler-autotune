import numpy as np

from qta.adaptive import (
    Pool, Replay, adaptive_ei, by_cost_observed, expected_improvement, static, successive_halving, uniform,
)


def pools():
    flat = [Pool(np.zeros(10), np.ones(10)) for _ in range(3)]
    spread = Pool(np.arange(10, dtype=float), np.ones(10))
    return [*flat, spread]


def test_expected_improvement_grows_with_spread():
    assert expected_improvement([1.0, 1.0], 0.0, 0.0) == 0.0
    assert expected_improvement([0.0, 2.0], 0.0, 0.0) > expected_improvement([0.0, 0.2], 0.0, 0.0)
    # with few runs the prior keeps a circuit alive even if its first runs happened to agree
    assert expected_improvement([1.0, 1.0], 1.0, 2.0) > 0


def test_policies_spend_budget_where_spread_is():
    budget = 14.0
    # a strong prior keeps circuits whose first runs agreed attractive, so it is weakened here
    for policy, kwargs in [(adaptive_ei, {"prior_weight": 0.1}), (successive_halving, {}),
                           (static, {"priority": np.array([0.0, 0.0, 0.0, 1.0])})]:
        replay = Replay(pools(), np.random.default_rng(0))
        # first two runs of the spread circuit already show its spread
        replay.states[-1].order = np.array([0, 9, 1, 8, 2, 7, 3, 6, 4, 5])
        policy(replay, budget, **kwargs)
        *flat, spread = replay.states
        assert spread.taken > max(s.taken for s in flat), policy.__name__
        assert replay.spent <= budget + 1


def test_uniform_and_gain():
    replay = Replay(pools(), np.random.default_rng(0))
    uniform(replay, 0.0, k=10)
    assert [s.taken for s in replay.states] == [10, 10, 10, 10]
    assert np.isclose(replay.gain(), (9 - 4.5) / 4)


def test_by_cost_observed_prefers_cheap_circuits():
    cheap = Pool(np.arange(10, dtype=float), np.full(10, 0.1))
    dear = Pool(np.arange(10, dtype=float), np.full(10, 1.0))
    replay = Replay([cheap, dear], np.random.default_rng(0))
    by_cost_observed(replay, budget=2.0)
    assert replay.states[0].taken > replay.states[1].taken
    assert replay.spent <= 2.0 + 1.0


def test_weighted_gain():
    replay = Replay(pools(), np.random.default_rng(0))
    uniform(replay, 0.0, k=10)
    assert np.isclose(replay.gain(np.array([0.0, 0.0, 0.0, 1.0])), 4.5)
    assert np.isclose(replay.gain(np.ones(4)), replay.gain())

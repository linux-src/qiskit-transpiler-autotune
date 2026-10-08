"""Search space of layout and routing configurations."""

from __future__ import annotations

import math

import numpy as np

from qta.config import TuningConfig

MAX_ITERATIONS = (1, 8)
TRIALS = (1, 128)
ROUTING = ("joint", "separate")
HEURISTICS = ("basic", "lookahead", "decay")


def _log_int(rng: np.random.Generator, low: int, high: int) -> int:
    return int(round(math.exp(rng.uniform(math.log(low), math.log(high)))))


def sample(rng: np.random.Generator) -> TuningConfig:
    """Draw a configuration uniformly; trial counts are drawn on a log scale."""
    return TuningConfig(
        use_vf2=bool(rng.integers(2)),
        max_iterations=int(rng.integers(MAX_ITERATIONS[0], MAX_ITERATIONS[1] + 1)),
        layout_trials=_log_int(rng, *TRIALS),
        swap_trials=_log_int(rng, *TRIALS),
        routing=str(rng.choice(ROUTING)),
        heuristic=str(rng.choice(HEURISTICS)),
        routing_trials=_log_int(rng, *TRIALS),
    )

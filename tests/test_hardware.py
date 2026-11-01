import pandas as pd

from qta.hardware import spread_seeds


def test_spread_seeds_covers_esp_range_and_fewest_2q():
    seeds = pd.DataFrame({
        "seed": range(6),
        "esp": [0.5, 0.1, 0.9, 0.3, 0.7, 0.6],
        "n_2q": [10, 12, 11, 9, 13, 14],
    })
    picks = spread_seeds(seeds, 3)
    assert picks[:3] == [1, 0, 2]
    assert picks[-1] == 3


def test_spread_seeds_drops_duplicates():
    seeds = pd.DataFrame({"seed": [0, 1], "esp": [0.2, 0.4], "n_2q": [5, 9]})
    assert spread_seeds(seeds, 3) == [0, 1]

import pandas as pd

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

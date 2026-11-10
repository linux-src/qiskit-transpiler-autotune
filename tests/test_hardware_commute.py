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

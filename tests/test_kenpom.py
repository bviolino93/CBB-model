from datetime import date

import numpy as np
import pandas as pd
import pytest

import kenpom


def espn_games():
    d1, d2 = date(2026, 11, 20), date(2026, 11, 21)
    rows = [
        (d1, "UConn Huskies", "Kansas State Wildcats", False, 80, 70),
        (d1, "Kansas Jayhawks", "Pennsylvania Quakers", False, 90, 60),
        (d1, "NC State Wolfpack", "Miami Hurricanes", True, 75, 77),
        (d2, "Pennsylvania Quakers", "UConn Huskies", False, 61, 72),
        (d2, "Kansas State Wildcats", "Kansas Jayhawks", False, 68, 71),
    ]
    g = pd.DataFrame(rows, columns=["game_date", "home_team", "away_team",
                                    "neutral", "home_score", "away_score"])
    g["season"] = 2027
    g["game_id"] = [str(i) for i in range(len(g))]
    return g


def fan_rows(d):
    if d == date(2026, 11, 20):
        return [
            {"Home": "Connecticut", "Visitor": "Kansas St.", "HomePred": 78,
             "VisitorPred": 70, "HomeWP": 76, "PredTempo": 68,
             "HomeRank": 5, "VisitorRank": 50},
            {"Home": "Kansas", "Visitor": "Penn", "HomePred": 85,
             "VisitorPred": 62, "HomeWP": 97, "PredTempo": 70,
             "HomeRank": 3, "VisitorRank": 200},
            # neutral site, listed the other way round from ESPN
            {"Home": "Miami FL", "Visitor": "N.C. State", "HomePred": 74,
             "VisitorPred": 72, "HomeWP": 57, "PredTempo": 66,
             "HomeRank": 40, "VisitorRank": 60},
        ]
    return [
        {"Home": "Penn", "Visitor": "Connecticut", "HomePred": 63,
         "VisitorPred": 75, "HomeWP": 13, "PredTempo": 67,
         "HomeRank": 200, "VisitorRank": 5},
        {"Home": "Kansas St.", "Visitor": "Kansas", "HomePred": 69,
         "VisitorPred": 72, "HomeWP": 40, "PredTempo": 69,
         "HomeRank": 50, "VisitorRank": 3},
    ]


def arc_rows(d):
    teams = {"Connecticut": 25, "Kansas St.": 10, "Kansas": 28, "Penn": -5,
             "Miami FL": 12, "N.C. State": 11}
    return [{"TeamName": t, "AdjEM": em, "AdjOE": 110 + em / 2,
             "AdjDE": 110 - em / 2, "AdjTempo": 68} for t, em in teams.items()]


@pytest.fixture
def fake_api(monkeypatch, tmp_path):
    monkeypatch.setattr(kenpom, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(kenpom, "fanmatch", lambda d: pd.DataFrame(fan_rows(d)))
    monkeypatch.setattr(kenpom, "archive", lambda d: pd.DataFrame(arc_rows(d)))
    monkeypatch.setattr(kenpom, "preseason",
                        lambda s: pd.DataFrame(arc_rows(None)))


def test_name_sim_handles_kenpom_spellings():
    assert kenpom.name_sim("Connecticut", "UConn Huskies") == 1.0
    assert kenpom.name_sim("N.C. State", "NC State Wolfpack") == 1.0
    assert kenpom.name_sim("Miami FL", "Miami Hurricanes") == 1.0
    assert kenpom.name_sim("Kansas St.", "Kansas State Wildcats") == 1.0


def test_name_map_uses_opponent_for_odd_names():
    g = espn_games()
    fan = pd.concat([pd.DataFrame(fan_rows(d)).assign(game_date=d)
                     for d in sorted(set(g["game_date"]))])
    m = kenpom.name_map(fan, g)
    assert m["Kansas"] == "Kansas Jayhawks"
    assert m["Kansas St."] == "Kansas State Wildcats"
    assert m["Penn"] == "Pennsylvania Quakers"      # pass 2, via the opponent
    assert m["Miami FL"] == "Miami Hurricanes"


def test_load_kenpom_joins_and_flips_neutral(fake_api):
    g = kenpom.load_kenpom(espn_games(), [2027], verbose=False)
    assert g["kp_margin"].notna().all()
    assert g.loc[0, "kp_margin"] == 8 and g.loc[0, "kp_total"] == 148
    # ESPN has NC State at home; KenPom listed Miami first: flipped
    assert g.loc[2, "kp_margin"] == -2
    assert g.loc[2, "kp_wp"] == pytest.approx(0.43)
    assert g.loc[0, "kp_h_em"] == 25 and g.loc[0, "kp_a_em"] == 10
    assert g.loc[1, "kp_a_pre_em"] == -5


def test_no_key_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(kenpom, "CACHE_DIR", str(tmp_path))
    for n in kenpom.KEY_NAMES:
        monkeypatch.delenv(n, raising=False)
    with pytest.raises(RuntimeError):
        kenpom.call("fanmatch", d="2026-11-20")


def test_rows_envelope():
    assert kenpom._rows([{"a": 1}]) == [{"a": 1}]
    assert kenpom._rows({"data": [{"a": 1}]}) == [{"a": 1}]
    assert kenpom._rows({"error": "x"}) == []


def test_blend_fits_on_earlier_seasons_only():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    import kenpom_backtest as kb
    rng = np.random.default_rng(0)
    n = 3000
    fr = pd.DataFrame({"season": np.repeat([2021, 2022, 2023], n // 3)})
    true = rng.normal(0, 10, n)
    fr["kp_margin"] = true + rng.normal(0, 3, n)
    fr["model_margin"] = true + rng.normal(0, 4, n)
    fr["kp_total"] = 140 + rng.normal(0, 8, n)
    fr["model_total"] = fr["kp_total"] + rng.normal(0, 3, n)
    for c in ("kp_h_em", "kp_a_em", "kp_h_pre_em", "kp_a_pre_em"):
        fr[c] = rng.normal(0, 10, n)
    for c in ("kp_h_oe", "kp_a_oe", "kp_h_de", "kp_a_de"):
        fr[c] = 105 + rng.normal(0, 5, n)
    fr["kp_h_t"] = fr["kp_a_t"] = 68.0
    fr["min_n"] = 10
    fr["neutral"] = False
    fr["rest_diff"] = 0
    fr["home_score"] = 70 + true / 2 + rng.normal(0, 8, n)
    fr["away_score"] = fr["home_score"] - true - rng.normal(0, 8, n)
    out = kb.blend(fr, verbose=False)
    assert out.loc[out["season"] < 2023, "blend_margin"].isna().all()
    m = out[out["season"] == 2023]
    act = m["home_score"] - m["away_score"]
    assert np.abs(m["blend_margin"] - act).mean() <= np.abs(m["kp_margin"] - act).mean() + 0.1

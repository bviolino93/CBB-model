import json
import os

import numpy as np
import pandas as pd

import cbb_data

FIX = os.path.join(os.path.dirname(__file__), "fixtures",
                   "scoreboard_20251115.json")


def rows():
    with open(FIX) as fh:
        return pd.DataFrame(cbb_data.parse_scoreboard(json.load(fh)))


def test_parse_spread_sign_names_and_conferences():
    d = rows().set_index("game_id")
    assert d.loc["401700001", "spread_line"] == 6.5      # home Duke -6.5
    assert d.loc["401700002", "spread_line"] == -3.0     # away UConn -3
    assert d.loc["401700003", "spread_line"] == 0.0
    assert d.loc["401700002", "neutral"]
    assert d.loc["401700002", "home_conf"] == "7"
    assert d.loc["401700001", "total_line"] == 151.5
    assert np.isnan(d.loc["401700003", "total_line"])    # 250 is not a total
    assert np.isnan(d.loc["401700003", "home_score"])     # not played
    assert str(d.loc["401700002", "game_date"]) == "2025-11-15"
    assert d.loc["401700001", "season"] == 2026


def test_current_season():
    from datetime import date
    assert cbb_data.current_season(date(2026, 11, 10)) == 2027
    assert cbb_data.current_season(date(2027, 3, 20)) == 2027


def test_non_d1_pooled():
    g = pd.DataFrame({
        "season": 2026,
        "home_team": ["A"] * 12 + ["B"],
        "away_team": ["B"] * 12 + ["Tiny"],
    })
    out = cbb_data.mark_non_d1(g)
    assert (out["away_team"].iloc[-1]) == cbb_data.NON_D1
    assert out["vs_non_d1"].sum() == 1
    assert (out["home_team"] == "A").sum() == 12


def test_names_match_across_feeds():
    n = cbb_data.norm_name
    assert n("San José State Spartans") == n("San Jose St Spartans")
    assert n("Hawai'i Rainbow Warriors") == n("Hawaii Rainbow Warriors")
    assert n("St. John's Red Storm") == n("St Johns Red Storm")


def test_match_events_by_time_and_names():
    g = rows()
    events = [
        {"id": "e1", "commence_time": "2025-11-16T00:00:00Z",
         "home_team": "Duke Blue Devils", "away_team": "North Carolina Tar Heels"},
        # neutral site, listed the other way round, 10 minutes off
        {"id": "e2", "commence_time": "2025-11-16T02:40:00Z",
         "home_team": "UConn Huskies", "away_team": "Oregon Ducks"},
        {"id": "e3", "commence_time": "2025-11-16T17:00:00Z",
         "home_team": "San Jose St Spartans", "away_team": "Kansas Jayhawks"},
    ]
    m = cbb_data.match_events(g, events)
    assert m == {"e1": "401700001", "e2": "401700002"}


def test_schedule_features():
    days = ["2025-11-04", "2025-11-07", "2025-11-08"]
    g = pd.DataFrame({
        "game_id": ["1", "2", "3"], "season": 2026,
        "game_date": pd.to_datetime(days).date,
        "commence": pd.to_datetime(days, utc=True),
        "home_team": ["A", "B", "A"], "away_team": ["B", "C", "C"],
    })
    f = cbb_data.schedule_features(g).set_index("game_id")
    assert f.loc["3", "home_rest"] == 4
    assert f.loc["3", "away_rest"] == 1
    assert f.loc["3", "min_n"] == 1


def test_schedule_features_survive_a_team_listed_twice():
    days = ["2025-11-04", "2025-11-07"]
    g = pd.DataFrame({
        "game_id": ["1", "2"], "season": 2026,
        "game_date": pd.to_datetime(days).date,
        "commence": pd.to_datetime(days, utc=True),
        "home_team": ["A", "TBD"], "away_team": ["B", "TBD"],
    })
    f = cbb_data.schedule_features(g).set_index("game_id")
    assert f.loc["2", "home_n"] == 0

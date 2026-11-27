import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "tools"))
import breakdown as bd  # noqa: E402


def test_a_card():
    assert bd.a_card("low", 7.0, 5.0)          # bet home, home favoured
    assert not bd.a_card("low", 7.0, -3.0)     # bet home dog
    assert not bd.a_card("high", 7.0, 5.0)
    assert not bd.a_card("mixed", -5.0, -5.0)  # gap under 6
    assert bd.a_card("mixed", -8.0, -5.0)      # bet away favourite
    assert not bd.a_card("low", 9.0, 16.0)     # line 15+


def test_team_form():
    t = pd.Timestamp("2026-11-05T00:00Z")
    g = pd.DataFrame([
        dict(game_id="1", season=2026, commence=t, home_team="A", away_team="B",
             home_score=70, away_score=60, spread_line=5.0, total_line=120.0,
             completed=True),
        dict(game_id="2", season=2027, commence=t + pd.Timedelta(days=200),
             home_team="C", away_team="A", home_score=80, away_score=75,
             spread_line=-2.0, total_line=160.0, completed=True)])
    f = bd.team_form(g, "A", t + pd.Timedelta(days=300))
    assert f["n"] == 2 and f["w"] == 1 and f["this_season"] == 1
    assert f["cover"] == 1 and f["no_cover"] == 1
    assert f["over"] == 1 and f["under"] == 1
    assert f["last"] == ["W +10", "L -5"]


def test_notes_and_markdown():
    b = {"home": "Idaho Vandals", "away": "Cal Poly Mustangs", "neutral": False,
         "tier": "low", "tip_et": "9:00 PM",
         "line": {"home_margin": 1.5, "spread": -1.5, "total": 160.0,
                  "provider": "ESPN BET"},
         "projection": {"margin": 9.0, "total": 150.0, "home_pts": 79.5,
                        "away_pts": 70.5},
         "matchup": {"pace": 70.0, "home_ortg": 110.0, "away_ortg": 100.0,
                     "apm_diff": 4.0, "same_diff": 0.1, "new_diff": 0.0,
                     "four_factors": {}},
         "form": {"home": {"n": 0}, "away": {"n": 0}},
         "schedule": {"rest_diff": 3.0}, "a_card": True}
    b["notes"] = bd.notes(b)
    assert any("7.5 pts better for Idaho" in n for n in b["notes"])
    assert any("10.0 under" in n for n in b["notes"])
    assert any("A rule" in n for n in b["notes"])
    md = bd.to_markdown("2026-11-18", [b])
    assert "Idaho Vandals 80, Cal Poly Mustangs 70" in md or "Idaho Vandals 79" in md


def test_assess_tired_side_loses_points_and_confidence():
    b = {"home": "A", "away": "B", "tier": "low", "a_card": False,
         "line": {"home_margin": 3.0, "spread": -3.0, "total": 140.0},
         "projection": {"margin": 4.0, "total": 141.0},
         "form": {"home": {"n": 0}, "away": {"n": 0}},
         "schedule": {"home_rest": 1.0, "away_rest": 3.0, "rest_diff": -2.0}}
    c = bd.assess(b)
    assert c["adj_margin"] == 2.5 and c["spread"]["side"] == "B"
    assert c["total"]["side"] == "Under" and "A played yesterday" in c["total"]["why"]
    b["schedule"] = {"home_rest": 3.0, "away_rest": 1.0, "rest_diff": 2.0}
    c = bd.assess(b)
    assert c["spread"]["side"] == "A" and "rest edge" in c["spread"]["why"]


def test_assess_caps_big_line_underdogs():
    b = {"home": "Big", "away": "Small", "tier": "mixed", "a_card": False,
         "line": {"home_margin": 30.0, "spread": -30.0, "total": 150.0},
         "projection": {"margin": 18.0, "total": 150.5},
         "form": {"home": {"n": 0}, "away": {"n": 0}}, "schedule": {}}
    c = bd.assess(b)
    assert c["spread"]["side"] == "Small" and c["spread"]["confidence"] == 1


def test_assess_big_gap_caps_confidence_and_rest_not_counted():
    # 11-19 Houston Christian: projection +15.6 vs line +3.5, A rule, rest edge -> was 5/5
    b = {"home": "HCU", "away": "GSU", "tier": "low", "a_card": True,
         "line": {"home_margin": 3.5, "spread": -3.5, "total": 150.5},
         "projection": {"margin": 15.6, "total": 157.4},
         "form": {"home": {"n": 5, "margin": 8.0}, "away": {"n": 5, "margin": -2.0}},
         "schedule": {"home_rest": 4.0, "away_rest": 2.0, "rest_diff": 2.0}}
    sp = bd.assess(b)["spread"]
    assert sp["side"] == "HCU" and sp["confidence"] == 2
    assert "from the line" in sp["why"] and "rest edge (not counted)" in sp["why"]


def test_kenpom_is_context_only():
    # 11-27 Virginia @ Xavier (neutral): line Xavier +6.5, KenPom Virginia by 8
    b = {"home": "Xavier Musketeers", "away": "Virginia Cavaliers", "tier": "high",
         "a_card": False, "neutral": True, "tip_et": "11:00 AM",
         "line": {"home_margin": -6.5, "spread": 6.5, "total": 156.5},
         "projection": {"margin": -5.0, "total": 159.8, "home_pts": 77.4,
                        "away_pts": 82.4},
         "matchup": {"four_factors": {}},
         "form": {"home": {"n": 0}, "away": {"n": 0}}, "schedule": {},
         "kenpom": bd.kenpom_block({"kp_margin": -8.0, "kp_total": 158.0,
                                    "kp_wp": 0.24, "kp_home_rank": 49,
                                    "kp_away_rank": 8, "kp_h_oe": 110.0,
                                    "kp_h_de": 100.0, "kp_a_oe": 118.0,
                                    "kp_a_de": 95.0, "kp_h_t": 74.0,
                                    "kp_a_t": 69.5})}
    without = dict(b, kenpom={})
    c, c0 = bd.assess(b), bd.assess(without)
    assert c["spread"]["side"] == "Xavier Musketeers"
    assert c["spread"]["confidence"] == c0["spread"]["confidence"]
    assert "KenPom Virginia Cavaliers by 8 (#49 vs #8) disagrees" in c["spread"]["why"]
    assert "KenPom total 158 (agrees)" in c["total"]["why"]
    b["call"], b["notes"] = c, []
    md = bd.to_markdown("2026-11-27", [b])
    assert "**KenPom:** Xavier Musketeers 75, Virginia Cavaliers 83" in md

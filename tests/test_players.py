import pandas as pd

from cbb_players import attach_games, parse_box, trim

SUMMARY = {"boxscore": {"players": [
    {"team": {"id": "150", "abbreviation": "DUKE"}, "statistics": [{
        "labels": ["MIN", "FG", "3PT", "FT", "OREB", "DREB", "REB", "AST",
                   "STL", "BLK", "TO", "PF", "PTS"],
        "athletes": [
            {"athlete": {"id": "1", "displayName": "A One"}, "starter": True,
             "stats": ["34", "7-12", "2-5", "3-4", "1", "5", "6", "3", "1",
                       "0", "2", "2", "19"]},
            {"athlete": {"id": "2", "displayName": "B Two"}, "starter": False,
             "didNotPlay": True, "stats": []}]}]},
    {"team": {"id": "153", "abbreviation": "UNC"}, "statistics": [{
        "labels": ["MIN", "FG", "3PT", "FT", "OREB", "DREB", "REB", "AST",
                   "STL", "BLK", "TO", "PF", "PTS"],
        "athletes": [
            {"athlete": {"id": "3", "displayName": "C Three"}, "starter": True,
             "stats": ["40", "5-15", "1-6", "0-0", "2", "3", "5", "6", "2",
                       "1", "4", "3", "11"]}]}]}],
    "teams": []}, "plays": [{"x": 1}]}


def test_parse_trimmed_box():
    t = trim(SUMMARY)
    assert "plays" not in t
    rows = parse_box(t, "99")
    assert len(rows) == 3
    a = rows[0]
    assert (a["fgm"], a["fga"], a["fg3m"], a["fg3a"], a["ftm"], a["fta"]) == \
        (7, 12, 2, 5, 3, 4)
    assert a["min"] == 34 and a["pts"] == 19 and a["tov"] == 2 and a["played"]
    assert not rows[1]["played"] and rows[1]["min"] == 0


def test_attach_by_team_id():
    p = pd.DataFrame(parse_box(trim(SUMMARY), "99"))
    g = pd.DataFrame([{"game_id": "99", "season": 2026,
                       "game_date": pd.Timestamp("2025-12-01").date(),
                       "commence": pd.Timestamp("2025-12-01 23:00", tz="UTC"),
                       "home_team": "Duke Blue Devils",
                       "away_team": "North Carolina Tar Heels",
                       "home_id": "150", "away_id": "153"}])
    b = attach_games(p, g)
    assert set(b["team"]) == {"Duke Blue Devils", "North Carolina Tar Heels"}
    one = b[b["player_id"] == "3"].iloc[0]
    assert one["home"] == 0 and one["opp"] == "Duke Blue Devils"

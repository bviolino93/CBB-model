"""cbb_predict on a small made-up league: features are walk-forward, the
live roster never uses a returning scratch, and player impact finds the
good players."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")
pytest.importorskip("sklearn")

import cbb_predict as cp  # noqa: E402
from cbb_ratings import RatingConfig  # noqa: E402


def league(seasons=(2021, 2022), teams=12, per_team=9, seed=0):
    rng = np.random.default_rng(seed)
    names = [f"T{i}" for i in range(teams)]
    skill = {}
    roster = {}
    pid = 0
    for t in names:
        roster[t] = []
        for _ in range(per_team):
            skill[str(pid)] = rng.normal(0, 3)
            roster[t].append(str(pid))
            pid += 1
    games, box = [], []
    gid = 0
    for s in seasons:
        day = pd.Timestamp(f"{s - 1}-11-05")
        for rnd in range(30):
            order = rng.permutation(names)
            for i in range(0, teams, 2):
                h, a = order[i], order[i + 1]
                gid += 1
                tip = day + pd.Timedelta(days=2 * rnd, hours=19)
                lines = {}
                for t in (h, a):
                    play = [p for p in roster[t] if rng.random() > 0.1][:8]
                    mins = np.full(len(play), 200.0 / len(play))
                    lines[t] = (play, mins)
                str_h = sum(skill[p] * m / 40 for p, m in zip(*lines[h]))
                str_a = sum(skill[p] * m / 40 for p, m in zip(*lines[a]))
                margin = 3 + str_h - str_a + rng.normal(0, 8)
                hs = 70 + margin / 2 + rng.normal(0, 3)
                aws = 70 - margin / 2 + rng.normal(0, 3)
                games.append({"game_id": str(gid), "season": s,
                              "season_type": "REG",
                              "game_date": tip.date(),
                              "commence": tip.tz_localize("UTC"),
                              "home_team": h, "away_team": a,
                              "home_id": h, "away_id": a,
                              "home_conf": "1", "away_conf": "1",
                              "home_score": round(hs), "away_score": round(aws),
                              "spread_line": np.nan, "total_line": np.nan,
                              "neutral": False, "completed": True,
                              "vs_non_d1": False})
                for t, pts in ((h, hs), (a, aws)):
                    play, mins = lines[t]
                    for p, m in zip(play, mins):
                        fga = 7.0
                        box.append({"game_id": str(gid), "team_id": t,
                                    "player_id": p, "player": p,
                                    "starter": True, "played": True,
                                    "min": m, "pts": pts / len(play),
                                    "fgm": 3.0, "fga": fga, "fg3m": 1.0,
                                    "fg3a": 3.0, "ftm": 2.0, "fta": 3.0,
                                    "oreb": 1.0, "dreb": 3.0, "reb": 4.0,
                                    "ast": 1.5 + skill[p] / 3, "stl": 0.5,
                                    "blk": 0.3, "tov": 1.2, "pf": 2.0})
    return pd.DataFrame(games), pd.DataFrame(box), skill


@pytest.fixture(scope="module")
def built():
    from cbb_players import attach_games
    g, b, skill = league()
    b = attach_games(b, g)
    cfg = cp.PredictConfig(ratings=RatingConfig(min_games=50),
                           min_prior_games=60)
    h = cp.build(g, b, cfg, first_season=2021)
    return g, b, h, skill


def test_team_box_possessions(built):
    _, b, _, _ = built
    tb = cp.team_box(b)
    assert len(tb) > 0
    assert tb["poss"].between(40, 100).all()
    assert np.allclose(tb["pace"], tb["poss"], atol=1e-6)   # 200 minutes


def test_features_exist(built):
    _, _, h, _ = built
    late = h[h["season"] == 2022]
    for c in ("eff_margin", "apm_diff_h", "apm_diff_l", "box_diff_l",
              "same_diff_l", "ff_efg_m"):
        assert late[c].notna().mean() > 0.9, c


def test_live_roster_skips_returning_scratch():
    g, b, _ = league(seasons=(2021,), teams=4)
    from cbb_players import attach_games
    b = attach_games(b, g)
    cfg = cp.PredictConfig(roster_games=1)
    av = cp.availability(g, b, cfg)
    tgi = cp.team_game_index(g)
    # Live roster for game n is drawn from game n-1 (plus debuts), so a
    # player who played tonight but not last game and not debuting must
    # be absent.
    q = cp.played_rows(b).merge(tgi, on=["game_id", "team", "season"])
    t = q[q["team"] == "T0"]
    sets = t.groupby("n")["player_id"].apply(set)
    live = av["l"].merge(tgi, on=["game_id", "team"])
    live = live[live["team"] == "T0"].groupby("n")["player_id"].apply(set)
    first = t.groupby("player_id")["n"].min()
    for n in range(2, len(sets)):
        tonight_only = sets[n] - sets[n - 1]
        returning = {p for p in tonight_only if first[p] < n}
        assert not (returning & live.get(n, set()))


def test_apm_finds_skill(built):
    _, _, h, _ = built
    d = h[(h["season"] == 2022) & h["apm_diff_h"].notna()]
    m = d["home_score"] - d["away_score"]
    assert np.corrcoef(d["apm_diff_h"], m)[0, 1] > 0.2


def test_walk_forward_no_peek(built):
    g, b, h, _ = built
    # Changing a late game's score must not move any earlier feature.
    g2 = g.copy()
    last = g2["commence"].max()
    g2.loc[g2["commence"] == last, "home_score"] += 40
    from cbb_players import attach_games
    cfg = cp.PredictConfig(ratings=RatingConfig(min_games=50),
                           min_prior_games=60)
    h2 = cp.build(g2, b, cfg, first_season=2021)
    early = h["commence"] < last
    for c in ("model_margin", "eff_margin", "apm_diff_l", "box_diff_l"):
        a, z = h.loc[early, c].to_numpy(), h2.loc[early, c].to_numpy()
        assert np.allclose(np.nan_to_num(a), np.nan_to_num(z)), c


def test_models_fit_and_predict(built):
    _, _, h, _ = built
    tr = h[h["season"] == 2021]
    te = h[h["season"] == 2022]
    m = cp.fit_models(pd.concat([tr] * 10), trees=False)  # >= 1000 rows
    if not m.margin:
        pytest.skip("too few games")
    p = cp.predict_frame(m, te, "ridge")
    assert p["pred_margin"].notna().all()

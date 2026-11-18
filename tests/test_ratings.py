import numpy as np
import pandas as pd

from cbb_ratings import RatingConfig, walk_forward
import backtest


def _league(seed=0, days=60, strength=None):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(30)]
    strength = strength or {t: (i - 15) * 0.8 for i, t in enumerate(teams)}
    rows = []
    for d in range(days):
        day = pd.Timestamp("2025-11-05") + pd.Timedelta(days=d)
        order = rng.permutation(teams)
        for j in range(0, 30, 2):
            h, a = order[j], order[j + 1]
            m = 3 + strength[h] - strength[a] + rng.normal(0, 8)
            tot = 145 + rng.normal(0, 12)
            rows.append({"game_id": f"{d}-{j}", "season": 2026,
                         "season_type": "REG", "game_date": day.date(),
                         "commence": day.tz_localize("UTC"), "home_team": h,
                         "away_team": a, "home_score": (tot + m) / 2,
                         "away_score": (tot - m) / 2, "neutral": False,
                         "completed": True, "spread_line": 3 + strength[h] - strength[a],
                         "total_line": 145.0})
    return pd.DataFrame(rows), strength


def test_ratings_recover_strength_and_home_court():
    g, s = _league()
    fr = walk_forward(g, RatingConfig(min_games=100, half_life=1000))
    late = fr[fr["game_date"] >= pd.Timestamp("2025-12-25").date()]
    true = 3 + late["home_team"].map(s) - late["away_team"].map(s)
    assert np.corrcoef(late["model_margin"], true)[0, 1] > 0.9
    assert abs((late["model_total"] - 145).mean()) < 2


def test_no_lookahead():
    g, _ = _league()
    a = walk_forward(g, RatingConfig(min_games=100))
    g2 = g.copy()
    last = g2["game_date"] == g2["game_date"].max()
    g2.loc[last, "home_score"] += 50
    b = walk_forward(g2, RatingConfig(min_games=100))
    assert np.allclose(a.loc[last, "model_margin"], b.loc[last, "model_margin"])


def test_bets_and_slope_on_an_exact_close():
    g, _ = _league()
    fr = walk_forward(g, RatingConfig(min_games=100))
    b = backtest.bets(fr, "spreads")
    assert len(b) > 300
    # the close is the true mean here, so the model can't beat it
    sl, t, n = backtest.slope(b["gap"].to_numpy(float), b["resid"].to_numpy(float))
    assert abs(t) < 3

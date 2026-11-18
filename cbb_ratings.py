"""
Team ratings from scores alone, walk-forward: our own line, with no
market input.

Each game day the ratings are refitted on every earlier game, weighted by
recency (half-life in days, with the offseason counted as extra days so
last season fades over the summer), by ridge least squares:

    home_margin  = hca * (not neutral) + r[home] - r[away]
    total points = mu + t[home] + t[away]

College differs from the NBA in three ways that matter here:

  * 360 teams who mostly play their own conference, so early-season
    ratings lean on last season (offseason_days) and on ridge shrinkage
  * blowouts of 40+ against weak teams say little about the next game, so
    margins are capped at `margin_cap` before fitting
  * rosters turn over heavily (transfers), so last season should fade
    faster than in the NBA; tools/tune_ratings.py tunes how much

Nothing here bets. backtest.py and tools/own_line.py score these
against the closing line.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RatingConfig:
    half_life: float = 60.0       # days; untuned starting value
    offseason_days: float = 120.0  # summer counts as this many extra days
    ridge_margin: float = 2.0
    ridge_total: float = 2.0
    margin_cap: float = 25.0      # fit on margins capped at +/- this
    min_games: int = 500          # need this many prior games to project


def _age(dates, seasons: np.ndarray, on, season: int,
         cfg: RatingConfig) -> np.ndarray:
    days = np.asarray((pd.Timestamp(on) - pd.DatetimeIndex(
        pd.to_datetime(np.asarray(dates)))).days, float)
    return days + cfg.offseason_days * (season - seasons)


def fit(prior: pd.DataFrame, on, season: int, cfg: RatingConfig,
        teams: list[str]):
    """Ratings from games before `on`. Returns (hca, r, mu, t)."""
    idx = {t: i for i, t in enumerate(teams)}
    n, k = len(prior), len(teams)
    age = _age(prior["game_date"], prior["season"].to_numpy(), on, season, cfg)
    w = 0.5 ** (age / cfg.half_life)
    hi = prior["home_team"].map(idx).to_numpy()
    ai = prior["away_team"].map(idx).to_numpy()
    rows = np.arange(n)
    neutral = prior["neutral"].astype(float).to_numpy()

    X = np.zeros((n, k + 1))
    X[:, 0] = 1.0 - neutral
    X[rows, hi + 1] += 1.0
    X[rows, ai + 1] -= 1.0
    y = (prior["home_score"] - prior["away_score"]).to_numpy(float)
    y = np.clip(y, -cfg.margin_cap, cfg.margin_cap)
    P = np.eye(k + 1) * cfg.ridge_margin
    P[0, 0] = 1e-6
    beta = np.linalg.solve(X.T @ (X * w[:, None]) + P, X.T @ (w * y))

    Z = np.abs(X)
    Z[:, 0] = 1.0
    yt = (prior["home_score"] + prior["away_score"]).to_numpy(float)
    Q = np.eye(k + 1) * cfg.ridge_total
    Q[0, 0] = 1e-6
    gam = np.linalg.solve(Z.T @ (Z * w[:, None]) + Q, Z.T @ (w * yt))
    return (float(beta[0]), dict(zip(teams, beta[1:])),
            float(gam[0]), dict(zip(teams, gam[1:])))


def project(todays: pd.DataFrame, hca, r, mu, t) -> tuple[pd.Series, pd.Series]:
    h, a = todays["home_team"], todays["away_team"]
    adv = hca * (1.0 - todays["neutral"].astype(float))
    margin = adv + h.map(r).fillna(0) - a.map(r).fillna(0)
    total = mu + h.map(t).fillna(0) + a.map(t).fillna(0)
    return margin, total


def walk_forward(g: pd.DataFrame, cfg: RatingConfig | None = None,
                 first_season: int | None = None) -> pd.DataFrame:
    """Project every game from games before its date only.

    Adds model_margin and model_total (NaN before `first_season` or with
    fewer than cfg.min_games prior games). Upcoming games get a projection
    too, which is what picks use.
    """
    cfg = cfg or RatingConfig()
    g = g.sort_values("commence").reset_index(drop=True).copy()
    done = g[g["completed"]]
    g["model_margin"] = np.nan
    g["model_total"] = np.nan
    for day, todays in g.groupby("game_date", sort=True):
        season = int(todays["season"].iloc[0])
        if first_season and season < first_season:
            continue
        prior = done[(done["game_date"] < day) & (done["season"] >= season - 1)]
        if len(prior) < cfg.min_games:
            continue
        teams = sorted(set(prior["home_team"]) | set(prior["away_team"]))
        m, tot = project(todays, *fit(prior, day, season, cfg, teams))
        g.loc[todays.index, "model_margin"] = m
        g.loc[todays.index, "model_total"] = tot
    return g

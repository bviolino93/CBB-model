"""
Our own prediction of every college game: home margin and total points,
built from players, possessions and the schedule, with no market input.
The college port of the NBA repo's nba_predict.

The first college line (cbb_ratings) is team ratings from final scores.
This one adds:

players     who is playing and how good each is
            * apm: each player's impact on the margin per 40 minutes, a
              ridge regression of every game's margin on the minute shares
              of the players in it (adjusted plus-minus at game level),
              recency-weighted, refitted every few game days on earlier
              games only. A player keeps his number when he transfers.
            * box: box-score production per 40 minutes (Hollinger game
              score) over his last two seasons, shrunk toward a bench
              player
            * continuity: share of the expected minutes from players who
              last played for this same team, and from players with no
              Division I history (freshmen, players from lower levels)
            A team's number is the minutes-weighted sum over its players,
            each at his expected minutes (his last 10 games played).
possessions offense and defense per 100 possessions and pace (per 40
            minutes), opponent-adjusted; each team's four factors and
            three-point rate, for and against
schedule    rest, games played so far, neutral site, early season

Which players count for a game, two ways, because college has no
reliable injury report and a backtest that knows who played has
hindsight the market didn't have:

  played  ("_h")  the players who actually played. Same as the NBA test.
                  Optimistic: knows about late scratches.
  roster  ("_l")  what a live model can know: every player who played
                  for the team in one of its last 5 games, plus anyone
                  playing his first game of the season for it (rosters
                  are public before tip, who sits isn't). This is the one
                  to bet with.

Everything for a game is built from games before its date only.
tools/model_lab.py fits the models season by season and tests them
against final scores and the closing line.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cbb_data import NON_D1, schedule_features
from cbb_ratings import RatingConfig, walk_forward


@dataclass
class PredictConfig:
    ratings: RatingConfig = field(default_factory=RatingConfig)
    eff_half_life: float = 60.0      # days, possessions ratings
    eff_ridge: float = 5.0
    apm_half_life: float = 180.0     # days, player impact
    apm_ridge: float = 4.0
    apm_seasons: int = 2             # earlier seasons kept in the apm fit
    apm_every: int = 2               # refit apm every N game days
    apm_cap: float = 25.0            # margins capped at this in the fit
    exp_games: int = 10              # expected minutes: last N games played
    box_games: int = 60              # box value over last N games played
    box_prior_min: float = 200.0
    box_prior: float = 6.0           # game score per 40 of a bench player
    new_min: float = 10.0            # expected minutes of an unknown player
    roster_games: int = 5            # live roster: played in last N games
    ff_half_life: float = 8.0        # four factors, in team games
    min_prior_games: int = 1500


# ----------------------------------------------------------------------
# Team box scores, possessions and four factors
# ----------------------------------------------------------------------
STAT = ["pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb",
        "tov", "min"]
FF = ["efg", "tovr", "orbr", "ftr", "par3"]


def team_box(p: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game: totals, possessions, four factors, for
    and against."""
    if p is None or p.empty:
        return pd.DataFrame()
    t = p.groupby(["game_id", "team"], as_index=False)[STAT].sum(min_count=1)
    opp = p.drop_duplicates(["game_id", "team"])[["game_id", "team", "opp"]]
    o = t.rename(columns={c: f"o_{c}" for c in STAT}).rename(
        columns={"team": "opp"})
    m = t.merge(opp, on=["game_id", "team"]).merge(o, on=["game_id", "opp"])
    own = m["fga"] - m["oreb"] + m["tov"] + 0.475 * m["fta"]
    oth = m["o_fga"] - m["o_oreb"] + m["o_tov"] + 0.475 * m["o_fta"]
    m["poss"] = 0.5 * (own + oth)
    # Per 40 minutes (200 player-minutes); a box without minutes counts as
    # a regulation game.
    mins = m["min"].where(m["min"] >= 190, 200.0)
    m["pace"] = m["poss"] * 200.0 / mins
    m["ortg"] = 100.0 * m["pts"] / m["poss"]
    m["efg"] = (m["fgm"] + 0.5 * m["fg3m"]) / m["fga"]
    m["tovr"] = m["tov"] / m["poss"]
    m["orbr"] = m["oreb"] / (m["oreb"] + m["o_dreb"])
    m["ftr"] = m["ftm"] / m["fga"]
    m["par3"] = m["fg3a"] / m["fga"]
    m["o_efg"] = (m["o_fgm"] + 0.5 * m["o_fg3m"]) / m["o_fga"]
    m["o_tovr"] = m["o_tov"] / m["poss"]
    m["o_orbr"] = m["o_oreb"] / (m["o_oreb"] + m["dreb"])
    m["o_ftr"] = m["o_ftm"] / m["o_fga"]
    m["o_par3"] = m["o_fg3a"] / m["o_fga"]
    ok = (m["fga"] > 30) & (m["o_fga"] > 30) & (m["poss"] > 40)
    return m[ok].replace([np.inf, -np.inf], np.nan)


def four_factors(g: pd.DataFrame, tb: pd.DataFrame,
                 half_life: float) -> pd.DataFrame:
    """Each team's recent four factors going into each game (exponentially
    weighted over its earlier games this season and last, for and
    against)."""
    cols = FF + [f"o_{c}" for c in FF]
    if tb.empty:
        return pd.DataFrame(columns=["game_id", "team"])
    base = pd.concat([g[["game_id", "commence", "home_team"]]
                      .rename(columns={"home_team": "team"}),
                      g[["game_id", "commence", "away_team"]]
                      .rename(columns={"away_team": "team"})])
    t = base.merge(tb[["game_id", "team"] + cols], on=["game_id", "team"],
                   how="left")
    t = t.sort_values(["team", "commence"]).reset_index(drop=True)
    ew = (t.groupby("team")[cols]
           .transform(lambda s: s.shift(1).ewm(halflife=half_life,
                                               min_periods=3,
                                               ignore_na=True).mean()
                      .ffill()))
    ew.columns = [f"ff_{c}" for c in cols]
    return pd.concat([t[["game_id", "team"]], ew], axis=1)


# ----------------------------------------------------------------------
# Walk-forward helpers
# ----------------------------------------------------------------------
def _weights(dates, seasons, day, season, half_life, offseason):
    d = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates)))
    age = np.asarray((pd.Timestamp(day) - d).days, float)
    age = age + offseason * (season - np.asarray(seasons, float))
    return 0.5 ** (age / half_life)


def efficiency_ratings(g: pd.DataFrame, tb: pd.DataFrame,
                       cfg: PredictConfig) -> pd.DataFrame:
    """Per game: predicted pace and each side's points per 100
    possessions, from opponent-adjusted ratings fitted on earlier games.

        ortg(team vs opp) = mu + home + off[team] - def[opp]
        pace(game)        = p0 + pace[home] + pace[away]
    """
    from scipy import sparse
    out = pd.DataFrame({"game_id": g["game_id"]})
    for c in ("pace_hat", "home_ortg_hat", "away_ortg_hat"):
        out[c] = np.nan
    if tb.empty:
        return out
    teams = sorted(set(g["home_team"]) | set(g["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    k = len(teams)
    info = g.set_index("game_id")[["home_team", "game_date", "season",
                                   "neutral"]]
    r = tb.merge(info, left_on="game_id", right_index=True)
    r = r[r["poss"].notna() & r["ortg"].notna()]
    r["is_home"] = ((r["team"] == r["home_team"])
                    & ~r["neutral"].astype(bool)).astype(float)
    r = r.sort_values("game_date").reset_index(drop=True)
    n = len(r)
    rows = np.arange(n)
    ti, oi = r["team"].map(idx).to_numpy(), r["opp"].map(idx).to_numpy()
    X = sparse.csr_matrix((np.concatenate([np.ones(n), r["is_home"], np.ones(n),
                                           -np.ones(n)]),
                           (np.concatenate([rows] * 4),
                            np.concatenate([np.zeros(n, int), np.ones(n, int),
                                            2 + ti, 2 + k + oi]))),
                          shape=(n, 2 + 2 * k))
    Z = sparse.csr_matrix((np.concatenate([np.ones(n)] * 3),
                           (np.concatenate([rows] * 3),
                            np.concatenate([np.zeros(n, int), 1 + ti,
                                            1 + oi]))), shape=(n, 1 + k))
    y, yp = r["ortg"].to_numpy(float), r["pace"].to_numpy(float)
    gd = pd.to_datetime(r["game_date"]).to_numpy()
    rs = r["season"].to_numpy()
    P = np.eye(2 + 2 * k) * cfg.eff_ridge
    P[0, 0] = P[1, 1] = 1e-6
    Q = np.eye(1 + k) * cfg.eff_ridge
    Q[0, 0] = 1e-6
    res = {}
    for day, todays in g.groupby("game_date", sort=True):
        season = int(todays["season"].iloc[0])
        m = np.flatnonzero((gd < np.datetime64(pd.Timestamp(day)))
                           & (rs >= season - 1))
        if len(m) < 2 * cfg.min_prior_games:
            continue
        w = _weights(gd[m], rs[m], day, season, cfg.eff_half_life,
                     cfg.ratings.offseason_days)
        Xs, Zs = X[m], Z[m]
        beta = np.linalg.solve((Xs.T @ Xs.multiply(w[:, None])).toarray() + P,
                               Xs.T @ (w * y[m]))
        gam = np.linalg.solve((Zs.T @ Zs.multiply(w[:, None])).toarray() + Q,
                              Zs.T @ (w * yp[m]))
        off, dfn = beta[2:2 + k], beta[2 + k:]
        for row in todays.itertuples():
            hi, ai = idx[row.home_team], idx[row.away_team]
            hm = 0.0 if row.neutral else 1.0
            res[row.game_id] = (gam[0] + gam[1 + hi] + gam[1 + ai],
                                beta[0] + hm * beta[1] + off[hi] - dfn[ai],
                                beta[0] + off[ai] - dfn[hi])
    v = pd.DataFrame.from_dict(res, orient="index",
                               columns=["pace_hat", "home_ortg_hat",
                                        "away_ortg_hat"])
    return out[["game_id"]].merge(v, left_on="game_id", right_index=True,
                                  how="left")


# ----------------------------------------------------------------------
# Players: who counts, expected minutes, box value, impact
# ----------------------------------------------------------------------
def game_score(p: pd.DataFrame) -> pd.Series:
    """Hollinger game score."""
    f = lambda c: p[c].fillna(0.0) if c in p else 0.0  # noqa: E731
    return (f("pts") + 0.4 * f("fgm") - 0.7 * f("fga")
            - 0.4 * (f("fta") - f("ftm")) + 0.7 * f("oreb") + 0.3 * f("dreb")
            + f("stl") + 0.7 * f("ast") + 0.7 * f("blk") - 0.4 * f("pf")
            - f("tov"))


def played_rows(box: pd.DataFrame) -> pd.DataFrame:
    q = box[box["played"].astype(bool)].copy()
    # A box without minutes: a starter ~30, anyone else ~12.
    miss = q["min"].isna()
    q.loc[miss, "min"] = np.where(q.loc[miss, "starter"], 30.0, 12.0)
    return q


def player_history(box: pd.DataFrame, cfg: PredictConfig) -> pd.DataFrame:
    """Per played row, each player's numbers AFTER that game: mean minutes
    over his last cfg.exp_games games, game score and minutes summed over
    his last cfg.box_games, and the team he played for."""
    q = played_rows(box).sort_values(["player_id", "commence"])
    q["gs"] = game_score(q)
    grp = q.groupby("player_id", sort=False)
    q["h_min"] = grp["min"].rolling(cfg.exp_games, min_periods=1).mean() \
        .reset_index(level=0, drop=True)
    q["h_gs"] = grp["gs"].rolling(cfg.box_games, min_periods=1).sum() \
        .reset_index(level=0, drop=True)
    q["h_mins"] = grp["min"].rolling(cfg.box_games, min_periods=1).sum() \
        .reset_index(level=0, drop=True)
    return q[["player_id", "commence", "h_min", "h_gs", "h_mins", "team"]] \
        .rename(columns={"team": "h_team"})


def team_game_index(g: pd.DataFrame) -> pd.DataFrame:
    """game_id, team, season, n: the team's n-th game of the season."""
    t = pd.concat([g[["game_id", "season", "commence", "home_team"]]
                   .rename(columns={"home_team": "team"}),
                   g[["game_id", "season", "commence", "away_team"]]
                   .rename(columns={"away_team": "team"})])
    t = t[t["team"] != NON_D1].sort_values("commence")
    t["n"] = t.groupby(["team", "season"]).cumcount()
    return t[["game_id", "team", "season", "n"]]


def availability(g: pd.DataFrame, box: pd.DataFrame,
                 cfg: PredictConfig) -> dict[str, pd.DataFrame]:
    """Who counts for each team-game, both ways (see the module doc).

    {"h": played, "l": live roster}, each: game_id, team, player_id."""
    q = played_rows(box)
    q = q[q["team"] != NON_D1][["game_id", "team", "season", "player_id"]]
    tgi = team_game_index(g)
    q = q.merge(tgi, on=["game_id", "team", "season"])
    hind = q[["game_id", "team", "player_id"]].drop_duplicates()
    # Live: played in one of the team's last roster_games games ...
    rec = pd.concat([q.assign(n=q["n"] + k)
                     for k in range(1, cfg.roster_games + 1)])
    rec = rec[["team", "season", "n", "player_id"]].merge(
        tgi, on=["team", "season", "n"])
    # ... or debuting for the team this season tonight
    first = q.groupby(["team", "season", "player_id"])["n"].transform("min")
    debut = q[q["n"] == first]
    live = pd.concat([rec[["game_id", "team", "player_id"]],
                      debut[["game_id", "team", "player_id"]]])
    return {"h": hind, "l": live.drop_duplicates()}


def attach_form(av: pd.DataFrame, g: pd.DataFrame, hist: pd.DataFrame,
                cfg: PredictConfig) -> pd.DataFrame:
    """Expected minutes, box value per 40 and history flags going INTO
    each availability row's game (as-of join on the player's last game
    played before tip)."""
    a = av.merge(g[["game_id", "commence", "game_date", "season"]],
                 on="game_id")
    a = a.sort_values("commence")
    h = hist.sort_values("commence")
    a = pd.merge_asof(a, h, on="commence", by="player_id",
                      allow_exact_matches=False, direction="backward")
    known = a["h_min"].notna()
    a["exp_min"] = a["h_min"].fillna(cfg.new_min)
    a["box40"] = 40.0 * (a["h_gs"].fillna(0.0)
                         + cfg.box_prior / 40.0 * cfg.box_prior_min) \
        / (a["h_mins"].fillna(0.0) + cfg.box_prior_min)
    a["same"] = (known & (a["h_team"] == a["team"])).astype(float)
    a["new"] = (~known).astype(float)
    tot = a.groupby(["game_id", "team"])["exp_min"].transform("sum")
    a["share"] = 5.0 * a["exp_min"] / tot.where(tot > 0)
    return a.drop(columns=["h_min", "h_gs", "h_mins", "h_team"])


def apm_strength(g: pd.DataFrame, box: pd.DataFrame, avs: dict,
                 cfg: PredictConfig, first_season: int | None = None):
    """Walk-forward player impact summed over each availability set.

    Every cfg.apm_every game days, a ridge of each completed game's
    (capped) home margin on the players' minute shares (+ for home, - for
    away) and a home-court term, fitted on games before that day; solved
    by conjugate gradients, warm-started from the last fit.

    Returns ({key: Series apm by (game_id, team)}, Series hca by day)."""
    from scipy import sparse
    from scipy.sparse.linalg import cg
    done = g[g["completed"]]
    q = played_rows(box)
    q = q[q["game_id"].isin(done["game_id"])]
    tm = q.groupby(["game_id", "team"])["min"].transform("sum")
    q = q.assign(share=5.0 * q["min"] / tm, sign=np.where(q["home"] == 1,
                                                          1.0, -1.0))
    gids = done["game_id"][done["game_id"].isin(q["game_id"])].tolist()
    gi = {x: i for i, x in enumerate(gids)}
    pids = sorted(set(q["player_id"]) | set().union(
        *[set(a["player_id"]) for a in avs.values()]))
    pi = {x: i for i, x in enumerate(pids)}
    npl = len(pids)
    d = done.set_index("game_id").loc[gids]
    ng = len(gids)
    X = sparse.csr_matrix((np.concatenate([(q["share"] * q["sign"]).to_numpy(),
                                           1.0 - d["neutral"].astype(float)
                                           .to_numpy()]),
                           (np.concatenate([q["game_id"].map(gi).to_numpy(),
                                            np.arange(ng)]),
                            np.concatenate([q["player_id"].map(pi).to_numpy(),
                                            np.full(ng, npl)]))),
                          shape=(ng, npl + 1))
    y = np.clip((d["home_score"] - d["away_score"]).to_numpy(float),
                -cfg.apm_cap, cfg.apm_cap)
    dnp = pd.to_datetime(d["game_date"]).to_numpy()
    seasons = d["season"].to_numpy()
    lam = np.full(npl + 1, cfg.apm_ridge)
    lam[-1] = 1e-3
    L = sparse.diags(lam)

    days = sorted(set().union(*[set(a["game_date"]) for a in avs.values()]))
    if first_season:
        keep = set(g.loc[g["season"] >= first_season, "game_date"])
        days = [x for x in days if x in keep]
    season_of = g.groupby("game_date")["season"].first()
    # Per availability set: row positions of each day, and the player
    # column of each row; filled with that day's ratings as we go.
    sets = {}
    for key, a in avs.items():
        cols = np.array([pi[c] for c in a["player_id"]], dtype=int)
        sets[key] = (a, cols, a.groupby("game_date").indices,
                     np.full(len(a), np.nan))
    beta = np.zeros(npl + 1)
    hca = {}
    fitted, last_season = False, None
    for i, day in enumerate(days):
        season = int(season_of[day])
        if not fitted or i % cfg.apm_every == 0 or season != last_season:
            m = np.flatnonzero((dnp < np.datetime64(pd.Timestamp(day)))
                               & (seasons >= season - cfg.apm_seasons))
            if len(m) < cfg.min_prior_games:
                continue
            w = _weights(dnp[m], seasons[m], day, season, cfg.apm_half_life,
                         cfg.ratings.offseason_days)
            Xs = X[m]
            A = (Xs.T @ Xs.multiply(w[:, None])).tocsr() + L
            beta, _ = cg(A, Xs.T @ (w * y[m]), x0=beta, rtol=1e-6,
                         maxiter=500)
            fitted, last_season = True, season
        hca[day] = float(beta[-1])
        for a, cols, pos, vals in sets.values():
            ix = pos.get(day)
            if ix is not None:
                vals[ix] = beta[cols[ix]]
    out = {}
    for key, (a, cols, pos, vals) in sets.items():
        ok = np.isfinite(vals)
        out[key] = (a[ok].assign(x=a["share"].to_numpy()[ok] * vals[ok])
                    .groupby(["game_id", "team"])["x"].sum())
    return out, pd.Series(hca)


# ----------------------------------------------------------------------
# Everything per game
# ----------------------------------------------------------------------
SCHED_M = ["rest_diff", "early", "neutral_f", "n_diff"]
SCHED_T = ["early", "month", "neutral_f"]
POSS_M = ["eff_margin", "ff_efg_m", "ff_tov_m", "ff_orb_m", "ff_ftr_m",
          "ff_3p_m"]
POSS_T = ["eff_total", "pace_hat", "ff_efg_s", "ff_tov_s", "ff_orb_s",
          "ff_ftr_s", "ff_3p_s"]
PLAY_M = {k: [f"apm_diff_{k}", f"box_diff_{k}", f"same_diff_{k}",
              f"new_diff_{k}"] for k in ("h", "l")}
PLAY_T = {k: [f"apm_sum_{k}", f"box_sum_{k}"] for k in ("h", "l")}
MARGIN_F = ["model_margin"] + SCHED_M + POSS_M + PLAY_M["l"]
TOTAL_F = ["model_total"] + SCHED_T + POSS_T + PLAY_T["l"]


def build(games: pd.DataFrame, box: pd.DataFrame,
          cfg: PredictConfig | None = None, first_season: int | None = None,
          verbose: bool = False) -> pd.DataFrame:
    """Every game in `games` with the features above, walk-forward."""
    import time
    t0 = time.time()

    def say(msg):
        if verbose:
            print(f"  {msg} ({time.time() - t0:.0f}s)", flush=True)

    cfg = cfg or PredictConfig()
    g = games.copy()
    g["game_id"] = g["game_id"].astype(str)
    if "min_n" not in g:
        g = schedule_features(g)
    g["neutral_f"] = g["neutral"].astype(float)
    g["n_diff"] = g["home_n"].fillna(0) - g["away_n"].fillna(0)
    g["early"] = (g["min_n"].fillna(0) < 8).astype(float)
    g["month"] = pd.to_datetime(g["game_date"]).dt.month.map(
        lambda m: (m - 11) % 12)
    say("score ratings")
    g = walk_forward(g, cfg.ratings, first_season)
    if box is None or box.empty:
        return g
    box = box.copy()
    box["game_id"] = box["game_id"].astype(str)
    box["player_id"] = box["player_id"].astype(str)

    say("possessions ratings")
    tb = team_box(box)
    g = g.merge(efficiency_ratings(g, tb, cfg), on="game_id", how="left")
    g["eff_margin"] = g["pace_hat"] * (g["home_ortg_hat"]
                                       - g["away_ortg_hat"]) / 100.0
    g["eff_total"] = g["pace_hat"] * (g["home_ortg_hat"]
                                      + g["away_ortg_hat"]) / 100.0
    ff = four_factors(g, tb, cfg.ff_half_life).set_index(["game_id", "team"])
    for side in ("home", "away"):
        idx = list(zip(g["game_id"], g[f"{side}_team"]))
        for c in ff.columns:
            g[f"{side}_{c}"] = ff[c].reindex(idx).to_numpy()
    for f, lab, sign in (("efg", "efg", 1), ("tovr", "tov", -1),
                         ("orbr", "orb", 1), ("ftr", "ftr", 1),
                         ("par3", "3p", 1)):
        hm = g[f"home_ff_{f}"] + g[f"away_ff_o_{f}"]
        am = g[f"away_ff_{f}"] + g[f"home_ff_o_{f}"]
        g[f"ff_{lab}_m"] = sign * (hm - am)
        g[f"ff_{lab}_s"] = hm + am

    say("player availability and form")
    hist = player_history(box, cfg)
    avs = {k: attach_form(a, g, hist, cfg)
           for k, a in availability(g, box, cfg).items()}
    say("player impact (apm)")
    apm, hca = apm_strength(g, box, avs, cfg, first_season)
    g["apm_hca"] = g["game_date"].map(hca)
    for k, a in avs.items():
        s = (a.assign(bx=a["share"] * a["box40"], sm=a["share"] * a["same"],
                      nw=a["share"] * a["new"])
              .groupby(["game_id", "team"])[["bx", "sm", "nw"]].sum())
        s["apm"] = apm[k]
        for side in ("home", "away"):
            idx = list(zip(g["game_id"], g[f"{side}_team"]))
            for c in ("apm", "bx", "sm", "nw"):
                g[f"{side}_{c}_{k}"] = s[c].reindex(idx).to_numpy()
        g[f"apm_diff_{k}"] = (g[f"home_apm_{k}"] - g[f"away_apm_{k}"]
                              + g["apm_hca"].fillna(0) * (1 - g["neutral_f"]))
        g[f"apm_sum_{k}"] = g[f"home_apm_{k}"] + g[f"away_apm_{k}"]
        g[f"box_diff_{k}"] = g[f"home_bx_{k}"] - g[f"away_bx_{k}"]
        g[f"box_sum_{k}"] = g[f"home_bx_{k}"] + g[f"away_bx_{k}"]
        g[f"same_diff_{k}"] = g[f"home_sm_{k}"] - g[f"away_sm_{k}"]
        g[f"new_diff_{k}"] = g[f"home_nw_{k}"] - g[f"away_nw_{k}"]
    say("features done")
    return g


# ----------------------------------------------------------------------
# Models: ridge on the features, boosted trees on the ridge's misses
# ----------------------------------------------------------------------
@dataclass
class Models:
    margin: dict = field(default_factory=dict)
    total: dict = field(default_factory=dict)
    n: int = 0


def _X(d: pd.DataFrame, cols, fill: dict) -> np.ndarray:
    return np.column_stack([d[c].astype(float).fillna(fill.get(c, 0.0))
                            .to_numpy() if c in d else
                            np.full(len(d), fill.get(c, 0.0)) for c in cols])


def _fit_one(d: pd.DataFrame, y: np.ndarray, cols, trees: bool) -> dict:
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    fill = {c: float(d[c].median()) if c in d and d[c].notna().any() else 0.0
            for c in cols}
    X = _X(d, cols, fill)
    ridge = make_pipeline(StandardScaler(),
                          RidgeCV(alphas=np.logspace(-2, 3, 12)))
    ridge.fit(X, y)
    out = {"ridge": ridge, "cols": list(cols), "fill": fill, "gbm": None}
    if trees:
        from sklearn.ensemble import HistGradientBoostingRegressor
        Xn = np.column_stack([d[c].astype(float).to_numpy() if c in d else
                              np.full(len(d), np.nan) for c in cols])
        gbm = HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.03, max_iter=300,
            min_samples_leaf=200, l2_regularization=1.0, random_state=0)
        gbm.fit(Xn, y - ridge.predict(X))
        out["gbm"] = gbm
    return out


def fit_models(h: pd.DataFrame, trees: bool = True, margin_cols=None,
               total_cols=None) -> Models:
    d = h[h["completed"] & h["model_margin"].notna()]
    if "vs_non_d1" in d:
        d = d[~d["vs_non_d1"].astype(bool)]
    if len(d) < 1000:
        return Models(n=len(d))
    mc = [c for c in (margin_cols or MARGIN_F) if c in d]
    tc = [c for c in (total_cols or TOTAL_F) if c in d]
    ym = (d["home_score"] - d["away_score"]).to_numpy(float)
    yt = (d["home_score"] + d["away_score"]).to_numpy(float)
    return Models(_fit_one(d, ym, mc, trees), _fit_one(d, yt, tc, trees),
                  len(d))


def _pred_one(m: dict, d: pd.DataFrame, kind: str) -> np.ndarray:
    r = m["ridge"].predict(_X(d, m["cols"], m["fill"]))
    if kind == "ridge" or m.get("gbm") is None:
        return r
    Xn = np.column_stack([d[c].astype(float).to_numpy() if c in d else
                          np.full(len(d), np.nan) for c in m["cols"]])
    return r + m["gbm"].predict(Xn)


def predict_frame(models: Models, d: pd.DataFrame, kind: str = "trees"
                  ) -> pd.DataFrame:
    out = pd.DataFrame(index=d.index)
    if not models.margin:
        out["pred_margin"] = d.get("model_margin")
        out["pred_total"] = d.get("model_total")
        return out
    out["pred_margin"] = _pred_one(models.margin, d, kind)
    out["pred_total"] = _pred_one(models.total, d, kind)
    return out


def ridge_weights(m: dict) -> dict:
    """Points per one standard deviation of each input (for reading)."""
    return dict(zip(m["cols"], np.round(m["ridge"][-1].coef_, 2)))

"""
Does the market under- (or over-) react when a regular misses a game?

College basketball has no injury report, but the box scores show who
played. Before each game we know who sat out the team's LAST game, so:

  regular  played in 70%+ of the team's earlier games this season and
           averaged 25+ minutes when he played (older boxes without
           minutes: started 80%+ of the games he played)
  missing  a regular who did not play in the team's previous game (not
           in the box, or listed DNP). He may or may not play tonight;
           we only use what was known before tip.
  new      missing for the first time (played the game before that):
           the freshest news, where the line is most likely to lag

Bets are graded against ESPN's closing line, rules found on 2020-22 and
checked on 2023-26 (kenpom_research.show). Split by tier: high (both
teams high-major), mixed, low (neither), where the market is thinnest.

    python tools/missing_players.py --first 2019
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from cbb_data import NON_D1, load_games, tier  # noqa: E402
from cbb_players import load_box  # noqa: E402
from kenpom_research import BE, SPLIT, show  # noqa: E402

MIN_GAMES = 4          # team games before anyone can be a regular
SHARE = 0.70           # of the team's games played
MINUTES = 25.0
START_SHARE = 0.80


def team_absences(g: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    """Per (game_id, team): regulars missing from the previous game,
    their usual minutes and points, and how many are new absences."""
    p = p[p["played"].astype(bool)]
    played = p.groupby(["game_id", "team"])["player_id"].apply(set).to_dict()
    stats = {(r.game_id, r.team, r.player_id): (r.min, r.pts, r.starter)
             for r in p[["game_id", "team", "player_id", "min", "pts",
                         "starter"]].itertuples(index=False)}
    have = set(p["game_id"].astype(str))
    rows = []
    cols = ["game_id", "season", "commence"]
    tg = pd.concat([g[cols].assign(team=g["home_team"]),
                    g[cols].assign(team=g["away_team"])])
    tg = tg[(tg["team"] != NON_D1)].sort_values(["team", "commence"])
    for (team, season), x in tg.groupby(["team", "season"], sort=False):
        ids = x["game_id"].astype(str).tolist()
        seen: dict = {}          # player -> [games played, minutes, pts, starts, minute games]
        n_box = 0
        prev, prev2 = None, None
        for gid in ids:
            if prev is not None and n_box >= MIN_GAMES and prev in have:
                miss = new = 0
                mins = pts = 0.0
                for pid, (gp, mn, pt, st, mg) in seen.items():
                    if gp < SHARE * n_box:
                        continue
                    avg_min = mn / mg if mg else np.nan
                    if mg >= 0.5 * gp:
                        reg = avg_min >= MINUTES
                    else:
                        reg = st >= START_SHARE * gp
                    if not reg or pid in played.get((prev, team), set()):
                        continue
                    miss += 1
                    mins += avg_min if np.isfinite(avg_min) else 30.0
                    pts += pt / gp
                    if prev2 is not None and pid in played.get((prev2, team), set()):
                        new += 1
                rows.append((gid, team, miss, new, mins, pts))
            if gid in have:
                n_box += 1
                for pid in played.get((gid, team), set()):
                    mn, pt, st = stats[(gid, team, pid)]
                    s = seen.setdefault(pid, [0, 0.0, 0.0, 0, 0])
                    s[0] += 1
                    s[2] += pt if np.isfinite(pt) else 0.0
                    s[3] += int(bool(st))
                    if np.isfinite(mn) and mn > 0:
                        s[1] += mn
                        s[4] += 1
                prev2, prev = prev, gid
    return pd.DataFrame(rows, columns=["game_id", "team", "miss", "new",
                                       "miss_min", "miss_pts"])


def build(g: pd.DataFrame, ab: pd.DataFrame) -> pd.DataFrame:
    f = g[g["completed"] & g["spread_line"].notna()].copy()
    f["game_id"] = f["game_id"].astype(str)
    a = ab.set_index(["game_id", "team"])
    for side in ("home", "away"):
        idx = list(zip(f["game_id"], f[f"{side}_team"]))
        for c in ("miss", "new", "miss_min", "miss_pts"):
            f[f"{side[0]}_{c}"] = a[c].reindex(idx).to_numpy()
    f["actual_m"] = f["home_score"] - f["away_score"]
    f["actual_t"] = f["home_score"] + f["away_score"]
    f["tier"] = tier(f)
    known = f["h_miss"].notna() & f["a_miss"].notna()
    return f[known & (f["season"] >= 2020)].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2019)
    a = ap.parse_args()
    g = load_games(a.first)
    g["game_id"] = g["game_id"].astype(str)
    p = load_box(g)
    print(f"{len(p)} player rows, {p['game_id'].nunique()} games with a box")
    f = build(g, team_absences(g, p))
    print(f"{len(f)} graded games with both teams' previous box. find = "
          f"2020-{SPLIT - 1}, check = {SPLIT}-27. Break-even {BE:.1%}.")
    hm, am = f["h_miss"] > 0, f["a_miss"] > 0
    print(f"  games with a regular missing last game: home {hm.mean():.1%}, "
          f"away {am.mean():.1%}")
    z = pd.Series(0.0, index=f.index)
    hits: list = []

    def side_rules(lab: str, hcond: pd.Series, acond: pd.Series, sub=None):
        only_h, only_a = hcond & ~acond, acond & ~hcond
        if sub is not None:
            only_h, only_a = only_h & sub, only_a & sub
        fade = z.mask(only_h, -1).mask(only_a, 1)
        show(f"{lab}: fade the short-handed team", f, fade, "spread", out=hits)
        show(f"{lab}: back the short-handed team", f, -fade, "spread", out=hits)

    print("\n== spreads: one team had a regular miss its last game ==")
    side_rules("any regular missed last game", hm, am)
    side_rules("NEW absence (played the game before)", f["h_new"] > 0, f["a_new"] > 0)
    side_rules("missed 2+ straight (market has seen it)",
               hm & (f["h_new"] == 0), am & (f["a_new"] == 0))
    side_rules("missing 30+ minutes of regulars", f["h_miss_min"] >= 30, f["a_miss_min"] >= 30)
    side_rules("missing a 15+ ppg scorer", f["h_miss_pts"] >= 15, f["a_miss_pts"] >= 15)
    for t in ("high", "mixed", "low"):
        side_rules(f"[{t}] any regular missed last game", hm, am, f["tier"] == t)
        side_rules(f"[{t}] NEW absence", f["h_new"] > 0, f["a_new"] > 0, f["tier"] == t)

    print("\n== totals: a regular missed last game ==")
    anym = hm | am
    for lab, cond in (("either team missing a regular", anym),
                      ("either team, NEW absence", (f["h_new"] > 0) | (f["a_new"] > 0)),
                      ("missing 15+ ppg between them", (f["h_miss_pts"] + f["a_miss_pts"]) >= 15)):
        show(f"{lab}: under", f, z.mask(cond, -1), "total", out=hits)
        show(f"{lab}: over", f, z.mask(cond, 1), "total", out=hits)
        for t in ("high", "mixed", "low"):
            show(f"[{t}] {lab}: under", f, z.mask(cond & (f["tier"] == t), -1), "total", out=hits)

    print("\n== does the line react? average margin vs the close (team view) ==")
    for lab, cond in (("short-handed team, any", hm & ~am), ("short-handed team, NEW", (f["h_new"] > 0) & ~am)):
        res = (f["actual_m"] - f["spread_line"])[cond]
        resa = -(f["actual_m"] - f["spread_line"])[am & ~hm]
        both = pd.concat([res, resa])
        print(f"  {lab:28s} beats the close by {both.mean():+.2f} pts (n={len(both)})")

    print("\n== held in both periods ==")
    for h in hits:
        print(h)
    if not hits:
        print("  none")


if __name__ == "__main__":
    main()

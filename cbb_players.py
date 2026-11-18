"""
Player box scores for every Division I game, from ESPN's game summaries
(public JSON, no key; works from GitHub Actions, not from the sandbox).

One row per player per game:

    game_id, team_id, player_id, player, starter, played, min, pts, fgm,
    fga, fg3m, fg3a, ftm, fta, oreb, dreb, reb, ast, stl, blk, tov, pf

plus, after attach_games: season, game_date, commence, team, opp, home.

The summary is big (play-by-play, news); only the box block is kept on
disk, trimmed to what parse_box reads, so the cache stays small (about
4 KB a game, 45,000 games for eight seasons). Fetching is time-budgeted
like fill_lines (env CBB_FETCH_MINUTES): a cold cache fills over a run or
two and games not reached yet simply have no box.

Team names come from the games frame by ESPN team id, so non-D1
opponents carry the pooled name NON_D1 there too.

    python cbb_players.py --first 2019     # fill the cache, coverage table
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from cbb_data import CACHE_DIR, NONE, load_games

SUMMARY = ("https://site.api.espn.com/apis/site/v2/sports/basketball/"
           "mens-college-basketball/summary")
COLS = ["game_id", "team_id", "player_id", "player", "starter", "played",
        "min", "pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb",
        "dreb", "reb", "ast", "stl", "blk", "tov", "pf"]
NUM = COLS[6:]


def _num(x) -> float:
    try:
        return float(str(x).replace("+", ""))
    except (TypeError, ValueError):
        return float("nan")


def _made_att(x) -> tuple[float, float]:
    """"7-9" -> (7, 9); anything else NaN."""
    s = str(x or "")
    if "-" not in s:
        return float("nan"), float("nan")
    m, a = s.split("-", 1)
    return _num(m), _num(a)


def _minutes(x) -> float:
    s = str(x)
    if ":" in s:
        m, sec = s.split(":", 1)
        return _num(m) + _num(sec) / 60.0
    return _num(s)


def trim(js: dict) -> dict:
    """Only the box score part of a summary, only the fields parse_box
    reads."""
    teams = []
    for tm in (js.get("boxscore") or {}).get("players", []) or []:
        stats = []
        for st in tm.get("statistics", []) or []:
            stats.append({
                "labels": st.get("labels") or st.get("names") or [],
                "athletes": [{
                    "athlete": {"id": (a.get("athlete") or {}).get("id"),
                                "displayName": (a.get("athlete") or {})
                                .get("displayName", "")},
                    "starter": bool(a.get("starter")),
                    "didNotPlay": bool(a.get("didNotPlay")),
                    "stats": a.get("stats") or []}
                    for a in st.get("athletes", []) or []]})
        teams.append({"team": {"id": (tm.get("team") or {}).get("id")},
                      "statistics": stats})
    return {"boxscore": {"players": teams}}


def parse_box(js: dict, game_id: str) -> list[dict]:
    """Player rows from a (trimmed) summary."""
    rows = []
    for tm in (js.get("boxscore") or {}).get("players", []) or []:
        tid = str((tm.get("team") or {}).get("id", ""))
        for st in tm.get("statistics", []) or []:
            idx = {lab: i for i, lab in enumerate(st.get("labels") or [])}
            for a in st.get("athletes", []) or []:
                ath = a.get("athlete") or {}
                stats = a.get("stats") or []

                def get(lab, stats=stats):
                    i = idx.get(lab)
                    return stats[i] if i is not None and i < len(stats) else None

                dnp = bool(a.get("didNotPlay")) or not stats
                fgm, fga = _made_att(get("FG"))
                fg3m, fg3a = _made_att(get("3PT"))
                ftm, fta = _made_att(get("FT"))
                mins = _minutes(get("MIN"))
                r = {"game_id": str(game_id), "team_id": tid,
                     "player_id": str(ath.get("id", "")),
                     "player": ath.get("displayName", ""),
                     "starter": bool(a.get("starter")),
                     "min": mins, "pts": _num(get("PTS")), "fgm": fgm,
                     "fga": fga, "fg3m": fg3m, "fg3a": fg3a, "ftm": ftm,
                     "fta": fta, "oreb": _num(get("OREB")),
                     "dreb": _num(get("DREB")), "reb": _num(get("REB")),
                     "ast": _num(get("AST")), "stl": _num(get("STL")),
                     "blk": _num(get("BLK")), "tov": _num(get("TO")),
                     "pf": _num(get("PF"))}
                if dnp:
                    r.update(dict.fromkeys(NUM, 0.0))
                # Some older boxes have no minutes column: a player with
                # any line counts as having played.
                if not dnp and not np.isfinite(r["min"]):
                    any_stat = any(np.isfinite(r[c]) and r[c] > 0
                                   for c in ("pts", "fga", "fta", "reb",
                                             "ast", "pf", "tov"))
                    r["played"] = any_stat
                else:
                    r["played"] = (not dnp) and r["min"] > 0
                rows.append(r)
    return rows


def fetch_summary(game_id: str, cache_dir: str = CACHE_DIR) -> dict:
    """A final game's box score, trimmed and cached for good. A 404 is
    cached as NONE; other failures are not cached (next run retries)."""
    d = os.path.join(cache_dir, "box")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{game_id}.json")
    if os.path.exists(path):
        try:
            with open(path) as fh:
                js = json.load(fh)
            if js:
                return js
        except (OSError, ValueError):
            pass
    js = None
    for i in range(3):
        try:
            r = requests.get(SUMMARY, params={"event": game_id}, timeout=30)
            if r.status_code == 200:
                js = trim(r.json())
                break
            if r.status_code == 404:
                js = NONE
                break
        except (requests.RequestException, ValueError):
            pass
        time.sleep(2 ** i)
    if js is None:
        return {}
    with open(path, "w") as fh:
        json.dump(js, fh, separators=(",", ":"))
    return js


def load_box(games: pd.DataFrame, cache_dir: str = CACHE_DIR,
             workers: int = 16, budget_min: float | None = None
             ) -> pd.DataFrame:
    """Player rows for every completed game in `games`."""
    if budget_min is None:
        budget_min = float(os.environ.get("CBB_FETCH_MINUTES", "300"))
    deadline = time.time() + budget_min * 60
    done = games[games["completed"]]
    ids = done["game_id"].astype(str).tolist()
    bdir = os.path.join(cache_dir, "box")

    def one(gid):
        if time.time() > deadline and not os.path.exists(
                os.path.join(bdir, f"{gid}.json")):
            return {}
        return fetch_summary(gid, cache_dir)

    with ThreadPoolExecutor(workers) as ex:
        docs = list(ex.map(one, ids))
    missed = sum(1 for d in docs if not d)
    if missed:
        print(f"  box scores: {missed} games not fetched yet (time budget "
              f"or errors); run again to fill them")
    rows = [r for gid, js in zip(ids, docs) for r in parse_box(js, gid)]
    p = pd.DataFrame(rows, columns=COLS)
    return attach_games(p, done)


def attach_games(p: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Season, date, team name, opponent and home flag on box rows, by
    ESPN team id. Rows whose team id isn't one of the game's two teams
    are dropped."""
    g = games[["game_id", "season", "game_date", "commence", "home_team",
               "away_team", "home_id", "away_id"]].copy()
    g["game_id"] = g["game_id"].astype(str)
    p = p.copy()
    p["game_id"] = p["game_id"].astype(str)
    p = p.merge(g, on="game_id", how="inner")
    home = p["team_id"].astype(str) == p["home_id"].astype(str)
    away = p["team_id"].astype(str) == p["away_id"].astype(str)
    p = p[home | away].copy()
    home = home[p.index]
    p["home"] = home.astype(int)
    p["team"] = np.where(home, p["home_team"], p["away_team"])
    p["opp"] = np.where(home, p["away_team"], p["home_team"])
    p = p[p["player_id"].astype(str) != ""]
    return (p.drop(columns=["home_team", "away_team", "home_id", "away_id"])
             .sort_values(["commence", "game_id"]).reset_index(drop=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2019)
    ap.add_argument("--last", type=int, default=None)
    a = ap.parse_args()
    g = load_games(a.first, a.last)
    p = load_box(g)
    done = g[g["completed"] & ~g["vs_non_d1"]]
    have = set(p["game_id"])
    print(f"{len(p)} player rows, {p['game_id'].nunique()} games with a box")
    cov = done.assign(box=done["game_id"].astype(str).isin(have)).groupby(
        "season")["box"].mean().round(3)
    mins = p.groupby("season")["min"].apply(lambda s: s.notna().mean()).round(3)
    print(pd.DataFrame({"D1 games with box": cov,
                        "rows with minutes": mins}).to_string())


if __name__ == "__main__":
    main()

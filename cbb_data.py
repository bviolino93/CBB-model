"""
Men's college basketball games, scores and a market line per game, from
ESPN's public JSON (no key; works from GitHub Actions).

Same layout as the NBA repo's nba_data.py, with what college needs:

  * the scoreboard only lists featured games unless asked for Division I
    (groups=50) with a high limit; a Saturday has 150+ games
  * Division I teams also play non-D1 schools a few times a year. Those
    opponents are pooled into one team, NON_D1, so their handful of games
    don't each get a rating (see mark_non_d1)
  * neutral sites are common (November events, conference and NCAA
    tournaments), so the flag matters
  * each team's conference comes along, for testing where the market is
    thinnest (low-major games) separately

The frame load_games() returns:

    game_id, season, season_type, game_date, commence, home_team,
    away_team, home_id, away_id, home_conf, away_conf, home_score,
    away_score, spread_line, total_line, neutral, completed

  * season is the year the season ENDS (2027 = the 2026-27 season)
  * spread_line is the expected HOME margin (positive = home favoured)
  * team names are ESPN display names ("Duke Blue Devils"); live odds are
    joined to them by match_events, not by exact name

    python cbb_data.py --first 2019            # fill the cache, coverage table
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import time
import unicodedata
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

CACHE_DIR = os.environ.get("CBB_CACHE", ".cache_cbb")
SCOREBOARD = ("https://site.api.espn.com/apis/site/v2/sports/basketball/"
              "mens-college-basketball/scoreboard")
CORE_ODDS = ("https://sports.core.api.espn.com/v2/sports/basketball/leagues/"
             "mens-college-basketball/events/{id}/competitions/{id}/odds")
SEASON_TYPES = {1: "PRE", 2: "REG", 3: "POST"}
NON_D1 = "NON_D1"

# ESPN conference ids (groups) of the six high-major conferences:
# ACC, Big East, Big Ten, Big 12, Pac-12, SEC. Used only to split reports.
HIGH_MAJOR = {"2", "4", "7", "8", "21", "23"}

COLS = ["game_id", "season", "season_type", "game_date", "commence",
        "home_team", "away_team", "home_id", "away_id", "home_conf",
        "away_conf", "home_score", "away_score", "spread_line", "total_line",
        "neutral", "completed", "line_books"]


def current_season(today: date | None = None) -> int:
    """Season label (end year). A season starts in November."""
    today = today or datetime.now(timezone.utc).date()
    return today.year + 1 if today.month >= 8 else today.year


# ----------------------------------------------------------------------
# Parsing (pure, tested on fixtures)
# ----------------------------------------------------------------------
_DETAILS = re.compile(r"^\s*([A-Z0-9&]{2,6})\s+([-+]?\d+(?:\.\d+)?)\s*$")
MAX_SPREAD = 50.0        # college spreads reach the 40s against cupcakes


def _home_spread(odds: dict, home_abbr: str, away_abbr: str) -> float:
    """Expected home margin from an ESPN odds block, or NaN.

    `details` reads like "DUKE -8.5" or "EVEN". Fallbacks: the home side's
    pointSpread close (newer feeds), then `spread` stated from home.
    """
    det = str(odds.get("details") or "").strip()
    if det.upper() in ("EVEN", "PK", "PICK"):
        return 0.0
    m = _DETAILS.match(det)
    if m:
        abbr, pts = m.group(1), float(m.group(2))
        if abbr == home_abbr:
            return -pts
        if abbr == away_abbr:
            return pts
    try:
        line = odds["pointSpread"]["home"]["close"]["line"]
        return -float(str(line).replace("+", ""))
    except (KeyError, TypeError, ValueError):
        pass
    hto = odds.get("homeTeamOdds") or {}
    fav = hto.get("favorite")
    try:
        sp = abs(float(odds.get("spread")))
    except (TypeError, ValueError):
        return float("nan")
    if fav is True:
        return sp
    if fav is False and (odds.get("awayTeamOdds") or {}).get("favorite"):
        return -sp
    return float("nan")


def _total(odds: dict) -> float:
    try:
        v = float(odds.get("overUnder"))
        if 90 < v < 220:
            return v
    except (TypeError, ValueError):
        pass
    return float("nan")


def parse_scoreboard(js: dict) -> list[dict]:
    """One row per game in a scoreboard response."""
    rows = []
    for ev in js.get("events", []):
        try:
            comp = ev["competitions"][0]
            teams = {c["homeAway"]: c for c in comp["competitors"]}
            h, a = teams["home"], teams["away"]
        except (KeyError, IndexError):
            continue
        status = comp.get("status", ev.get("status", {})).get("type", {})
        done = bool(status.get("completed"))
        odds = (comp.get("odds") or [{}])[0]
        ht, at = h.get("team", {}), a.get("team", {})
        ha, aa = ht.get("abbreviation", ""), at.get("abbreviation", "")
        season = ev.get("season", {})
        commence = pd.to_datetime(ev.get("date"), utc=True)

        def score(c):
            try:
                return float(c.get("score")) if done else float("nan")
            except (TypeError, ValueError):
                return float("nan")

        rows.append({
            "game_id": str(ev.get("id")),
            "season": int(season.get("year", 0)) or None,
            "season_type": SEASON_TYPES.get(int(season.get("type", 2)), "REG"),
            # US calendar date of tip-off (ET), the date bettors use.
            "game_date": (commence - pd.Timedelta(hours=5)).date()
            if pd.notna(commence) else None,
            "commence": commence,
            "home_team": str(ht.get("displayName", "")).strip(),
            "away_team": str(at.get("displayName", "")).strip(),
            "home_id": str(ht.get("id", "")), "away_id": str(at.get("id", "")),
            "home_conf": str(ht.get("conferenceId", "")),
            "away_conf": str(at.get("conferenceId", "")),
            "home_abbr": ha, "away_abbr": aa,
            "home_score": score(h), "away_score": score(a),
            "spread_line": _home_spread(odds, ha, aa),
            "total_line": _total(odds),
            "neutral": bool(comp.get("neutralSite", False)),
            "completed": done,
        })
    return rows


# ----------------------------------------------------------------------
# Fetching with a per-day cache
# ----------------------------------------------------------------------
def _get(params: dict, tries: int = 4) -> dict:
    params = {"groups": 50, "limit": 500, **params}
    for i in range(tries):
        try:
            r = requests.get(SCOREBOARD, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(2 ** i)
    raise RuntimeError(f"ESPN scoreboard failed for {params}")


_CALLED_OFF = re.compile(r"postpon|cancel|suspend|forfeit", re.I)


def day_final(js: dict) -> bool:
    """Every game on the day is over (or called off)."""
    for ev in js.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        st = (comp.get("status") or ev.get("status") or {}).get("type", {})
        if not (st.get("completed") or st.get("state") == "post"
                or _CALLED_OFF.search(str(st.get("name", "")))):
            return False
    return True


def fetch_day(d: date, cache_dir: str = CACHE_DIR) -> list[dict]:
    """A day's Division I games, cached for good once all are final."""
    os.makedirs(os.path.join(cache_dir, "days"), exist_ok=True)
    path = os.path.join(cache_dir, "days", f"{d:%Y%m%d}.json")
    today = datetime.now(timezone.utc).date()
    if os.path.exists(path) and d < today - timedelta(days=1):
        with open(path) as fh:
            js = json.load(fh)
        if day_final(js):
            return parse_scoreboard(js)
    js = _get({"dates": f"{d:%Y%m%d}"})
    with open(path, "w") as fh:
        json.dump(js, fh)
    return parse_scoreboard(js)


def season_days(season: int):
    """Every date that could hold a game: Nov 1 to Apr 10."""
    d, end = date(season - 1, 11, 1), date(season, 4, 10)
    today = datetime.now(timezone.utc).date()
    while d <= min(end, today + timedelta(days=7)):
        yield d
        d += timedelta(days=1)


def load_season(season: int, cache_dir: str = CACHE_DIR,
                pause: float = 0.15) -> pd.DataFrame:
    rows = []
    for d in season_days(season):
        cached = os.path.exists(os.path.join(cache_dir, "days",
                                             f"{d:%Y%m%d}.json"))
        rows.extend(fetch_day(d, cache_dir))
        if not cached:
            time.sleep(pause)
    return pd.DataFrame(rows, columns=COLS + ["home_abbr", "away_abbr"])


def mark_non_d1(g: pd.DataFrame, min_games: int = 10) -> pd.DataFrame:
    """Pool opponents that aren't Division I into one team, NON_D1.

    groups=50 returns every game with a D1 team in it, so a non-D1 school
    shows up only in its few games against D1 sides. A team with fewer
    than `min_games` games in a season is treated as non-D1 that season
    (D1 teams play about 30). This looks at the whole season, but which
    schools are D1 is known before it starts, so nothing leaks.
    """
    g = g.copy()
    n = pd.concat([g[["season", "home_team"]].rename(columns={"home_team": "t"}),
                   g[["season", "away_team"]].rename(columns={"away_team": "t"})]
                  ).value_counts()
    for side in ("home", "away"):
        cnt = [n.get((s, t), 0) for s, t in zip(g["season"], g[f"{side}_team"])]
        small = pd.Series(cnt, index=g.index) < min_games
        g.loc[small, f"{side}_team"] = NON_D1
    g["vs_non_d1"] = (g["home_team"] == NON_D1) | (g["away_team"] == NON_D1)
    return g


# ----------------------------------------------------------------------
# Lines for past games: ESPN's core odds API
# ----------------------------------------------------------------------
SKIP_PROVIDER = re.compile(r"live|numberfire|teamrankings|projection|"
                           r"accuscore|predict", re.I)


def parse_core_odds(js: dict, home_abbr: str, away_abbr: str
                    ) -> tuple[float, float, int]:
    """(home margin line, total, books used): medians across books."""
    spreads, totals = [], []
    for it in js.get("items", []):
        name = str((it.get("provider") or {}).get("name", ""))
        if SKIP_PROVIDER.search(name):
            continue
        sp = _home_spread(it, home_abbr, away_abbr)
        if pd.notna(sp) and abs(sp) <= MAX_SPREAD:
            spreads.append(sp)
        tt = _total(it)
        if pd.notna(tt):
            totals.append(tt)
    med = lambda v: float(pd.Series(v).median()) if v else float("nan")
    return med(spreads), med(totals), max(len(spreads), len(totals))


NONE = {"_none": True}    # cached marker: the API has nothing for this game


def fetch_core_odds(game_id: str, cache_dir: str = CACHE_DIR) -> dict:
    """A game's odds from the core API, cached for good. A 404 is cached as
    NONE; other failures are not cached, so the next run tries again."""
    os.makedirs(os.path.join(cache_dir, "odds"), exist_ok=True)
    path = os.path.join(cache_dir, "odds", f"{game_id}.json")
    if os.path.exists(path):
        with open(path) as fh:
            js = json.load(fh)
        if js:
            return js
    js = None
    for i in range(3):
        try:
            r = requests.get(CORE_ODDS.format(id=game_id), timeout=30)
            if r.status_code == 200:
                js = r.json()
                break
            if r.status_code == 404:
                js = NONE
                break
        except requests.RequestException:
            pass
        time.sleep(2 ** i)
    if js is None:
        return {}
    with open(path, "w") as fh:
        json.dump(js, fh)
    return js


def fill_lines(g: pd.DataFrame, cache_dir: str = CACHE_DIR, workers: int = 16,
               budget_min: float | None = None) -> pd.DataFrame:
    """Fill missing spread/total on completed games from the core API.

    About 5,000 Division I games a season, one request each, cached for
    good. `budget_min` stops fetching new games after that many minutes
    (env CBB_FETCH_MINUTES), so a cold cache fills over a few runs instead
    of timing out; games not reached yet keep NaN lines.
    """
    from concurrent.futures import ThreadPoolExecutor
    if budget_min is None:
        budget_min = float(os.environ.get("CBB_FETCH_MINUTES", "300"))
    need = g[g["completed"] & (g["spread_line"].isna() | g["total_line"].isna())]
    if need.empty:
        return g
    deadline = time.time() + budget_min * 60
    odir = os.path.join(cache_dir, "odds")

    def one(gid):
        if time.time() > deadline and not os.path.exists(
                os.path.join(odir, f"{gid}.json")):
            return {}
        return fetch_core_odds(gid, cache_dir)

    with ThreadPoolExecutor(workers) as ex:
        docs = list(ex.map(one, need["game_id"]))
    g = g.copy()
    for idx, js, ha, aa in zip(need.index, docs, need["home_abbr"],
                               need["away_abbr"]):
        sp, tt, n = parse_core_odds(js, ha, aa)
        if pd.isna(g.at[idx, "spread_line"]):
            g.at[idx, "spread_line"] = sp
        if pd.isna(g.at[idx, "total_line"]):
            g.at[idx, "total_line"] = tt
        g.at[idx, "line_books"] = n
    missed = sum(1 for d in docs if not d)
    if missed:
        print(f"  core odds: {missed} games not fetched yet (time budget); "
              f"run again to fill them")
    return g


def load_games(first: int = 2019, last: int | None = None,
               cache_dir: str = CACHE_DIR, past_lines: bool = True
               ) -> pd.DataFrame:
    """All Division I games first..last (season end years), oldest first,
    non-D1 opponents pooled as NON_D1."""
    last = last or current_season()
    g = pd.concat([load_season(s, cache_dir) for s in range(first, last + 1)],
                  ignore_index=True)
    g = g[g["season_type"] != "PRE"]
    g = g.drop_duplicates("game_id").sort_values("commence")
    if past_lines:
        g = fill_lines(g, cache_dir)
    g = mark_non_d1(g)
    return g.reset_index(drop=True)


# ----------------------------------------------------------------------
# Joining The Odds API's team names to ESPN's
# ----------------------------------------------------------------------
_WORDS = [(r"\bstate\b", "st"), (r"\bsaint\b", "st"), (r"\bst\.", "st"),
          (r"\buniversity\b", ""), (r"\bthe\b", ""), (r"&", " and "),
          (r"\buconn\b", "connecticut"), (r"\bumass\b", "massachusetts"),
          (r"\bole miss\b", "mississippi"), (r"\bpitt\b", "pittsburgh")]


def norm_name(name: str) -> str:
    """Lowercase, no accents or punctuation, common words shortened, so
    "San José State Spartans" and "San Jose St Spartans" agree."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore")
    s = s.decode().lower().replace("'", "").replace("’", "")
    for pat, rep in _WORDS:
        s = re.sub(pat, rep, s)
    s = re.sub(r"[^a-z0-9() ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _sim(a: str, b: str) -> float:
    a, b = norm_name(a), norm_name(b)
    if a == b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def match_events(games: pd.DataFrame, events: list[dict],
                 min_score: float = 0.80, hours: float = 3.0) -> dict:
    """Map each Odds API event id to an ESPN game_id.

    Candidates are ESPN games tipping within `hours` of the event; the
    best pairing by name similarity of both teams wins if both names score
    at least `min_score`. Home and away may be swapped at neutral sites,
    so the swapped pairing is tried too. Unmatched events are left out.
    """
    out = {}
    if games.empty:
        return out
    gc = pd.to_datetime(games["commence"], utc=True)
    for e in events:
        try:
            tip = pd.Timestamp(e["commence_time"]).tz_convert("UTC")
        except (KeyError, ValueError, TypeError):
            continue
        near = games[(gc - tip).abs() <= pd.Timedelta(hours=hours)]
        best, best_s = None, 0.0
        for gid, ht, at in zip(near["game_id"], near["home_team"],
                               near["away_team"]):
            s1 = min(_sim(e.get("home_team", ""), ht), _sim(e.get("away_team", ""), at))
            s2 = min(_sim(e.get("home_team", ""), at), _sim(e.get("away_team", ""), ht))
            s = max(s1, s2)
            if s > best_s:
                best, best_s = gid, s
        if best is not None and best_s >= min_score:
            out[e.get("id")] = best
    return out


# ----------------------------------------------------------------------
# Schedule features
# ----------------------------------------------------------------------
def team_games(g: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game: rest days and game number in the season."""
    cols = ["game_id", "season", "game_date", "commence"]
    home = g[cols].assign(team=g["home_team"], home=1)
    away = g[cols].assign(team=g["away_team"], home=0)
    t = pd.concat([home, away])
    t = t[t["team"] != NON_D1].sort_values(["team", "commence"])
    gd = pd.to_datetime(t["game_date"])
    key = [t["team"], t["season"]]
    t["rest"] = (gd - gd.groupby(key).shift(1)).dt.days.clip(upper=7)
    t["n_prior"] = t.groupby(key).cumcount()
    return t


def schedule_features(g: pd.DataFrame) -> pd.DataFrame:
    """Per game: each side's rest and games played so far this season."""
    t = team_games(g).set_index(["game_id", "team"])
    out = g.copy()
    for side in ("home", "away"):
        idx = list(zip(out["game_id"], out[f"{side}_team"]))
        out[f"{side}_rest"] = t["rest"].reindex(idx).to_numpy()
        out[f"{side}_n"] = t["n_prior"].reindex(idx).to_numpy()
    out["rest_diff"] = (out["home_rest"].fillna(4).clip(upper=5)
                        - out["away_rest"].fillna(4).clip(upper=5))
    out["min_n"] = out[["home_n", "away_n"]].min(axis=1)
    return out


def tier(g: pd.DataFrame) -> pd.Series:
    """'high' when both teams are in a high-major conference, 'mixed' for
    one, 'low' for neither: where the market is deepest to thinnest."""
    h = g["home_conf"].astype(str).isin(HIGH_MAJOR).astype(int)
    a = g["away_conf"].astype(str).isin(HIGH_MAJOR).astype(int)
    return (h + a).map({2: "high", 1: "mixed", 0: "low"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2019)
    ap.add_argument("--last", type=int, default=None)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    g = load_games(a.first, a.last)
    done = g[g["completed"]]
    print(f"{len(g)} games, {len(done)} completed, "
          f"{int(done['vs_non_d1'].sum())} against non-D1 opponents")
    d1 = done[~done["vs_non_d1"]]
    print(d1.groupby("season").agg(
        games=("game_id", "size"),
        teams=("home_team", "nunique"),
        neutral=("neutral", "mean"),
        with_spread=("spread_line", lambda s: s.notna().mean().round(3)),
        with_total=("total_line", lambda s: s.notna().mean().round(3)),
        home_margin=("home_score", lambda s: (
            s - d1.loc[s.index, "away_score"]).mean().round(2)),
    ).round(3).to_string())
    print("\nline coverage by tier:")
    print(d1.assign(tier=tier(d1)).groupby("tier")["spread_line"]
          .apply(lambda s: s.notna().mean()).round(3).to_string())
    if a.out:
        g.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()

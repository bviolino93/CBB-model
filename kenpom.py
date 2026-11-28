"""
KenPom API: dated ratings and game predictions, joined to ESPN's games.

What we use, all of it dated so a backtest only sees what was known on
the morning of a game:

  * fanmatch (d=YYYY-MM-DD): KenPom's own prediction for every game that
    day: HomePred, VisitorPred (points), HomeWP, PredTempo
  * archive (d=YYYY-MM-DD): every team's ratings as of that date: AdjEM,
    AdjOE, AdjDE, AdjTempo
  * archive (preseason=true, y=season): the preseason ratings

The key comes from the repository secret, read from the first of
KENPOM_API_KEY, KENPOM_KEY, KENPOM_TOKEN, KENPOM_API_TOKEN, KENPOM that
is set. Responses are cached in .cache_kenpom/ (past dates for good, today
for an hour). The data is for our own use: it is never committed.

    python kenpom.py --probe            # which secret is set, one request
    python kenpom.py --first 2021       # fill the cache, coverage table
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache

import numpy as np
import pandas as pd
import requests

from cbb_data import norm_name

BASE = "https://kenpom.com/api.php"
CACHE_DIR = os.environ.get("KENPOM_CACHE", ".cache_kenpom")
KEY_NAMES = ("KENPOM_API_KEY", "KENPOM_KEY", "KENPOM_TOKEN",
             "KENPOM_API_TOKEN", "KENPOM")
PAUSE = float(os.environ.get("KENPOM_PAUSE", "1.0"))


def api_key() -> tuple[str, str]:
    """(env name, key) of the first KenPom key that is set, or ('', '')."""
    for n in KEY_NAMES:
        v = os.environ.get(n, "").strip()
        if v:
            return n, v
    return "", ""


def _rows(js) -> list[dict]:
    """The list of records in a response, whatever the envelope."""
    if isinstance(js, list):
        return [r for r in js if isinstance(r, dict)]
    if isinstance(js, dict):
        for v in js.values():
            if isinstance(v, list) and (not v or isinstance(v[0], dict)):
                return v
    return []


_session = requests.Session()


def call(endpoint: str, cache: bool = True, max_age_h: float | None = None,
         tries: int = 5, **params) -> list[dict]:
    """GET one endpoint, cached under .cache_kenpom/<endpoint>/."""
    tag = "_".join(f"{k}={params[k]}" for k in sorted(params)) or "all"
    path = os.path.join(CACHE_DIR, endpoint, re.sub(r"[^\w=.-]", "_", tag) + ".json")
    if cache and os.path.exists(path):
        age_h = (time.time() - os.path.getmtime(path)) / 3600
        if max_age_h is None or age_h < max_age_h:
            with open(path) as fh:
                return _rows(json.load(fh))
    _, key = api_key()
    if not key:
        raise RuntimeError("no KenPom key: set one of " + ", ".join(KEY_NAMES))
    err = ""
    for i in range(tries):
        try:
            r = _session.get(BASE, params={"endpoint": endpoint, **params},
                             headers={"Authorization": f"Bearer {key}"},
                             timeout=60)
        except requests.RequestException as e:
            err = str(e)
            time.sleep(2 ** i)
            continue
        if r.status_code == 200:
            js = r.json()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as fh:
                json.dump(js, fh)
            time.sleep(PAUSE)
            return _rows(js)
        err = f"HTTP {r.status_code}: {r.text[:200]}"
        if r.status_code in (401, 403):
            raise RuntimeError(f"KenPom refused the key ({err})")
        if r.status_code == 404:
            return []
        # 429: KenPom's rate limit. Wait it out, longer each time.
        time.sleep(60 * (i + 1) if r.status_code == 429 else min(60, 3 * 2 ** i))
    raise RuntimeError(f"KenPom {endpoint} {params} failed: {err}")


def _age(d: date) -> float | None:
    """Cache lifetime: past dates for good, today and later for an hour."""
    today = datetime.now(timezone.utc).date()
    return None if d < today - timedelta(days=1) else 1.0


def fanmatch(d: date) -> pd.DataFrame:
    rows = call("fanmatch", max_age_h=_age(d), d=f"{d:%Y-%m-%d}")
    return pd.DataFrame(rows)


def archive(d: date) -> pd.DataFrame:
    rows = call("archive", max_age_h=_age(d), d=f"{d:%Y-%m-%d}")
    return pd.DataFrame(rows)


def current_ratings(season: int) -> pd.DataFrame:
    return pd.DataFrame(call("ratings", max_age_h=1.0, y=season))


def preseason(season: int) -> pd.DataFrame:
    return pd.DataFrame(call("archive", preseason="true", y=season))


SEASON_TABLES = ("four-factors", "pointdist", "misc-stats", "height")


def season_table(endpoint: str, season: int) -> pd.DataFrame:
    """One season-level table (four-factors, pointdist, misc-stats, height).
    These are as of the end of the season (or today for the current one),
    so a backtest may only use them from the season BEFORE a game."""
    cur = season >= datetime.now(timezone.utc).year + (
        datetime.now(timezone.utc).month >= 8)
    df = pd.DataFrame(call(endpoint, max_age_h=12.0 if cur else None, y=season))
    if df.empty:
        return df
    for c in df.columns:
        if c not in ("TeamName", "ConfShort", "DataThrough", "ConfOnly"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ----------------------------------------------------------------------
# KenPom names ("Connecticut", "N.C. State", "Miami FL") to ESPN display
# names ("UConn Huskies", "NC State Wolfpack", "Miami Hurricanes")
# ----------------------------------------------------------------------
_KP_WORDS = [(r"\bn\.?c\.? state\b", "nc st"), (r"\bmiami fl\b", "miami"),
             (r"\bmiami oh\b", "miami (oh)"), (r"\bcal st\.?\b", "cal st"),
             (r"\bst\.$", "st"), (r"\bfla\.\b", "florida"),
             (r"\bunc\b", "unc"), (r"\bmass\.\b", "massachusetts")]


def _kp_norm(name: str) -> str:
    s = str(name).lower()
    for pat, rep in _KP_WORDS:
        s = re.sub(pat, rep, s)
    return norm_name(s)


@lru_cache(maxsize=None)
def name_sim(kp: str, espn: str) -> float:
    """How well a KenPom school name fits an ESPN display name. ESPN's is
    the school plus a mascot, so the KenPom name is compared with each
    leading run of ESPN's words as well as the whole name."""
    import difflib
    a, words = _kp_norm(kp), norm_name(espn).split()
    best = 0.0
    for k in range(1, len(words) + 1):
        b = " ".join(words[:k])
        s = 1.0 if a == b else difflib.SequenceMatcher(None, a, b).ratio()
        best = max(best, s)
    return best


def _vote(votes: dict) -> dict:
    out = {}
    for kp, c in votes.items():
        name, n = c.most_common(1)[0]
        if n >= 2 or len(c) == 1:
            out[kp] = name
    return out


def name_map(kp_games: pd.DataFrame, espn: pd.DataFrame,
             min_pair: float = 0.85) -> dict:
    """{KenPom name: ESPN name} from games on the same date.

    Pass 1: each KenPom game is paired with the ESPN game that day (or the
    day either side) whose two teams both fit well, and each KenPom name
    takes the ESPN name it was paired with most often.
    Pass 2: a name still unmapped ("Penn", "LIU") takes the other slot of
    the ESPN game where its opponent, already mapped, played that day.
    """
    by_day = {d: x for d, x in espn.groupby("game_date")}
    days = lambda d: [by_day[x] for x in (d, d - timedelta(days=1),
                                          d + timedelta(days=1)) if x in by_day]
    votes: dict[str, Counter] = defaultdict(Counter)
    for d, h, v in zip(kp_games["game_date"], kp_games["Home"], kp_games["Visitor"]):
        best, best_s = None, 0.0
        for day in days(d):
            if best_s >= 0.99:
                break
            for ht, at in zip(day["home_team"], day["away_team"]):
                s1 = min(name_sim(h, ht), name_sim(v, at))
                s2 = min(name_sim(h, at), name_sim(v, ht))
                if s1 > best_s:
                    best, best_s = (ht, at), s1
                if s2 > best_s:
                    best, best_s = (at, ht), s2
        if best and best_s >= min_pair:
            votes[h][best[0]] += 1
            votes[v][best[1]] += 1
    out = _vote(votes)
    taken = set(out.values())
    votes2: dict[str, Counter] = defaultdict(Counter)
    for d, h, v in zip(kp_games["game_date"], kp_games["Home"], kp_games["Visitor"]):
        for miss, known in ((h, v), (v, h)):
            if miss in out or known not in out:
                continue
            for day in days(d):
                for ht, at in zip(day["home_team"], day["away_team"]):
                    other = at if ht == out[known] else ht if at == out[known] else None
                    if other and other not in taken:
                        votes2[miss][other] += 1
    out.update(_vote(votes2))
    return out


# ----------------------------------------------------------------------
# Joining to ESPN's games
# ----------------------------------------------------------------------
FAN_COLS = ["HomePred", "VisitorPred", "HomeWP", "PredTempo", "HomeRank",
            "VisitorRank"]
ARC_COLS = ["AdjEM", "AdjOE", "AdjDE", "AdjTempo"]


def _num(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_kenpom(g: pd.DataFrame, seasons, ratings: bool = True,
                budget_min: float | None = None, verbose: bool = True
                ) -> pd.DataFrame:
    """`g` (ESPN games) with KenPom columns added, home perspective:

        kp_margin, kp_total, kp_wp, kp_tempo, kp_home_rank, kp_away_rank
        kp_h_em, kp_a_em, kp_h_oe, kp_a_oe, kp_h_de, kp_a_de, kp_h_t, kp_a_t
        kp_h_pre_em, kp_a_pre_em            (preseason AdjEM)

    Fetches fanmatch and archive for every date in `g` of those seasons
    (time-budgeted by KENPOM_FETCH_MINUTES; dates not reached stay NaN).
    """
    if budget_min is None:
        budget_min = float(os.environ.get("KENPOM_FETCH_MINUTES", "120"))
    deadline = time.time() + budget_min * 60
    g = g.copy()
    for c in ("kp_margin", "kp_total", "kp_wp", "kp_tempo", "kp_home_rank",
              "kp_away_rank"):
        g[c] = np.nan
    g["kp_home_name"] = None
    g["kp_away_name"] = None
    fan, arc, skipped = [], [], 0
    pre: dict[int, pd.DataFrame] = {}
    sel = g[g["season"].isin(list(seasons))]
    for season, sg in sel.groupby("season"):
        try:
            p = _num(preseason(int(season)), ARC_COLS)
            if not p.empty:
                pre[int(season)] = p
        except RuntimeError as e:
            print(f"  kenpom preseason {season}: {e}")
        for d in sorted(set(sg["game_date"])):
            cached = os.path.exists(os.path.join(
                CACHE_DIR, "fanmatch", f"d={d:%Y-%m-%d}.json"))
            if not cached and time.time() > deadline:
                skipped += 1
                continue
            try:
                f = fanmatch(d)
                a = archive(d) if ratings else pd.DataFrame()
                if ratings and a.empty and d >= datetime.now(timezone.utc).date() - timedelta(days=1):
                    # The archive only has past dates: today uses current ratings.
                    a = current_ratings(int(season))
            except RuntimeError as e:
                # Rate limited for good: keep what we have, the next run
                # carries on from the cache.
                print(f"  kenpom: stopped fetching at {d}: {str(e)[:120]}")
                deadline = 0
                skipped += 1
                continue
            if not f.empty:
                fan.append(f.assign(game_date=d, season=int(season)))
            if not a.empty:
                arc.append(a.assign(game_date=d, season=int(season)))
    if skipped and verbose:
        print(f"  kenpom: {skipped} dates not fetched yet (time budget); "
              f"run again to fill them")
    if not fan:
        return g
    fan = _num(pd.concat(fan, ignore_index=True), FAN_COLS)
    if fan["HomeWP"].max(skipna=True) > 1.5:      # a percent, not a share
        fan["HomeWP"] = fan["HomeWP"] / 100
    # One name map per season (KenPom spellings are stable within one).
    maps = {}
    for season, fs in fan.groupby("season"):
        maps[season] = name_map(fs, g[g["season"] == season])
    fan["h"] = [maps[s].get(n) for s, n in zip(fan["season"], fan["Home"])]
    fan["a"] = [maps[s].get(n) for s, n in zip(fan["season"], fan["Visitor"])]
    key = {}
    for i, r in enumerate(fan.itertuples()):
        if r.h and r.a:
            key[(r.game_date, r.h, r.a)] = (i, False)
            key.setdefault((r.game_date, r.a, r.h), (i, True))
    hits = 0
    for idx, d, ht, at in zip(g.index, g["game_date"], g["home_team"],
                              g["away_team"]):
        k = None
        for dd in (d, d - timedelta(days=1), d + timedelta(days=1)):
            k = key.get((dd, ht, at))
            if k:
                break
        if not k:
            continue
        i, swap = k
        r = fan.iloc[i]
        hp, vp = (r["VisitorPred"], r["HomePred"]) if swap else (r["HomePred"], r["VisitorPred"])
        g.at[idx, "kp_margin"] = hp - vp
        g.at[idx, "kp_total"] = hp + vp
        g.at[idx, "kp_wp"] = (1 - r["HomeWP"]) if swap else r["HomeWP"]
        g.at[idx, "kp_tempo"] = r.get("PredTempo", np.nan)
        g.at[idx, "kp_home_rank"] = r["VisitorRank"] if swap else r["HomeRank"]
        g.at[idx, "kp_away_rank"] = r["HomeRank"] if swap else r["VisitorRank"]
        g.at[idx, "kp_home_name"] = r["Visitor"] if swap else r["Home"]
        g.at[idx, "kp_away_name"] = r["Home"] if swap else r["Visitor"]
        hits += 1
    if verbose:
        n = len(sel)
        print(f"  kenpom: {len(fan)} predictions, matched {hits} of {n} "
              f"ESPN games ({hits / max(n, 1):.0%})")
    for side in ("h", "a"):
        for short in ("em", "oe", "de", "t", "pre_em"):
            g[f"kp_{side}_{short}"] = np.nan
    if ratings and arc:
        arc = _num(pd.concat(arc, ignore_index=True), ARC_COLS)
        arc["espn"] = [maps.get(s, {}).get(n) for s, n in
                       zip(arc["season"], arc["TeamName"])]
        arc = arc.dropna(subset=["espn"]).drop_duplicates(["game_date", "espn"])
        ai = arc.set_index(["game_date", "espn"])
        for side, team in (("h", "home_team"), ("a", "away_team")):
            ix = pd.MultiIndex.from_arrays([g["game_date"], g[team]])
            sub = ai.reindex(ix)
            for c, short in (("AdjEM", "em"), ("AdjOE", "oe"), ("AdjDE", "de"),
                             ("AdjTempo", "t")):
                g[f"kp_{side}_{short}"] = sub[c].to_numpy() if c in sub else np.nan
        pre_em = {}
        for s, p in pre.items():
            m = maps.get(s, {})
            for n, em in zip(p["TeamName"], p["AdjEM"]):
                if n in m:
                    pre_em[(s, m[n])] = em
        for side, team in (("h", "home_team"), ("a", "away_team")):
            g[f"kp_{side}_pre_em"] = [pre_em.get((int(s), t), np.nan)
                                      for s, t in zip(g["season"], g[team])]
    return g


def show_day(d: date, team: str = "") -> None:
    """KenPom's prediction next to ESPN's line for one date's games."""
    from cbb_data import fetch_day
    g = pd.DataFrame(fetch_day(d))
    g = g.assign(season=g["season"].fillna(0).astype(int), game_date=d)
    k = load_kenpom(g, sorted(set(g["season"])), budget_min=30)
    if team:
        t = team.lower()
        k = k[k["home_team"].str.lower().str.contains(t)
              | k["away_team"].str.lower().str.contains(t)]
    for r in k.sort_values("commence").itertuples():
        if not np.isfinite(r.kp_margin):
            print(f"{r.away_team} @ {r.home_team}: no KenPom match")
            continue
        hp, ap_ = (r.kp_total + r.kp_margin) / 2, (r.kp_total - r.kp_margin) / 2
        line = ("no line" if not np.isfinite(r.spread_line) else
                f"line {r.home_team} {-r.spread_line:+.1f}, total "
                f"{r.total_line:.1f}")
        print(f"{r.away_team} @ {r.home_team}{' (neutral)' if r.neutral else ''}: "
              f"KenPom {r.home_team} {hp:.0f}-{ap_:.0f} ({r.kp_margin:+.1f}, "
              f"total {r.kp_total:.0f}, win {r.kp_wp:.0%}); ranks "
              f"{r.kp_home_rank:.0f} vs {r.kp_away_rank:.0f}; AdjEM "
              f"{r.kp_h_em:+.1f} vs {r.kp_a_em:+.1f}, tempo "
              f"{r.kp_h_t:.1f}/{r.kp_a_t:.1f}; {line}"
              + (f"; FINAL {r.home_team} {r.home_score:.0f}-{r.away_score:.0f}"
                 if getattr(r, "completed", False) and np.isfinite(r.home_score) else ""))


def show_gaps(d: date, top: int = 15, lines_dir: str = "data/lines") -> None:
    """Biggest gaps between KenPom's number and the best DraftKings /
    FanDuel line, games not started yet. Gap size alone is not a tested
    edge (see tools/kenpom_backtest.py)."""
    from books import best_lines, pick_best
    from cbb_data import fetch_day
    g = pd.DataFrame(fetch_day(d))
    g = g.assign(season=g["season"].fillna(0).astype(int), game_date=d)
    k = load_kenpom(g, sorted(set(g["season"])), ratings=False, budget_min=10)
    now = pd.Timestamp.now(tz="UTC")
    k = k[k["kp_margin"].notna() & (pd.to_datetime(k["commence"], utc=True) > now)]
    bl = best_lines(k, d, lines_dir)
    sp, to = [], []
    for r in k.itertuples():
        rec = bl.get(str(r.game_id))
        tip = pd.to_datetime(r.commence, utc=True).tz_convert("America/New_York")
        tip = f"{tip:%-I:%M %p} ET"
        if not rec:
            continue
        m, t = r.kp_margin, r.kp_total
        for side, team, edge_of in (("home", r.home_team, lambda pt: m + pt),
                                    ("away", r.away_team, lambda pt: -m + pt)):
            b = pick_best(rec, "spread", side)
            if b and edge_of(b[1]) > 0:
                fav = r.home_team if m > 0 else r.away_team
                sp.append((edge_of(b[1]), f"{team} {b[1]:+g} ({b[2]:+d}, {b[0]}): "
                           f"KenPom {fav} by {abs(m):.1f}; {r.away_team} @ "
                           f"{r.home_team}, {tip}"))
        for side, edge_of in (("over", lambda pt: t - pt), ("under", lambda pt: pt - t)):
            b = pick_best(rec, "total", side)
            if b and edge_of(b[1]) > 0:
                to.append((edge_of(b[1]), f"{side.title()} {b[1]:g} ({b[2]:+d}, {b[0]}): "
                           f"KenPom total {t:.1f}; {r.away_team} @ {r.home_team}, {tip}"))
    print(f"{len(k)} games not started with a KenPom number, {len(bl)} with "
          f"DK/FD lines\n\nSPREADS (gap = KenPom margin vs the best line)")
    for e, line in sorted(sp, reverse=True)[:top]:
        print(f"  {e:4.1f}  {line}")
    print("\nTOTALS")
    for e, line in sorted(to, reverse=True)[:top]:
        print(f"  {e:4.1f}  {line}")


def _implied(american: float) -> float:
    a = float(american)
    return 100 / (a + 100) if a > 0 else -a / (-a + 100)


def show_ml(d: date, top: int = 15, lines_dir: str = "data/lines") -> None:
    """KenPom win chance vs DraftKings / FanDuel moneylines (vig removed
    per book, then averaged), games not started. Lists sides where the
    best price is worth more than KenPom's chance says. Not a tested
    edge: there is no college moneyline history to backtest."""
    import glob
    from books import BOOKS, NAMES
    from cbb_data import _sim, fetch_day, match_events
    g = pd.DataFrame(fetch_day(d))
    g = g.assign(season=g["season"].fillna(0).astype(int), game_date=d)
    k = load_kenpom(g, sorted(set(g["season"])), ratings=False, budget_min=10)
    now = pd.Timestamp.now(tz="UTC")
    k = k[k["kp_wp"].notna() & (pd.to_datetime(k["commence"], utc=True) > now)]
    files = sorted(glob.glob(os.path.join(lines_dir, "games_*.csv")))
    if not files or k.empty:
        print("no lines or no KenPom games")
        return
    s = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    s = s[s["book"].isin(BOOKS) & (s["market"] == "h2h")]
    tip = pd.to_datetime(s["commence_time"], utc=True)
    s = s[((tip - pd.Timedelta(hours=5)).dt.date == d)
          & (pd.to_datetime(s["snapshot_utc"], utc=True) < tip)]
    s = s.sort_values("snapshot_utc").groupby(["event_id", "book", "outcome"]).tail(1)
    ev = s.drop_duplicates("event_id")
    gmap = match_events(k, [{"id": r.event_id, "commence_time": r.commence_time,
                             "home_team": r.home_team, "away_team": r.away_team}
                            for r in ev.itertuples()])
    ki = k.set_index(k["game_id"].astype(str))
    rows = []
    for eid, gid in gmap.items():
        if str(gid) not in ki.index:
            continue
        r = ki.loc[str(gid)]
        e = s[s["event_id"] == eid]
        fair, best = {"home": [], "away": []}, {}
        for bk, x in e.groupby("book"):
            if len(x) != 2:
                continue
            side = {}
            for o in x.itertuples():
                side["home" if _sim(o.outcome, r["home_team"]) >= _sim(o.outcome, r["away_team"]) else "away"] = o.price
            if len(side) != 2:
                continue
            ph, pa = _implied(side["home"]), _implied(side["away"])
            fair["home"].append(ph / (ph + pa))
            fair["away"].append(pa / (ph + pa))
            for sd in ("home", "away"):
                if sd not in best or side[sd] > best[sd][1]:
                    best[sd] = (NAMES.get(bk, bk), int(side[sd]))
        if not fair["home"]:
            continue
        t = pd.to_datetime(r["commence"], utc=True).tz_convert("America/New_York")
        for sd, wp in (("home", r["kp_wp"]), ("away", 1 - r["kp_wp"])):
            bk, price = best[sd]
            dec = 1 + (price / 100 if price > 0 else 100 / -price)
            mkt = float(np.mean(fair[sd]))
            rows.append({"team": r[f"{sd}_team"], "price": price, "book": bk,
                         "kp": wp, "mkt": mkt, "ev": wp * dec - 1,
                         "game": f"{r['away_team']} @ {r['home_team']}",
                         "tip": f"{t:%-I:%M %p} ET", "margin": r["kp_margin"]})
    x = pd.DataFrame(rows)
    print(f"{len(gmap)} games with DK/FD moneylines and a KenPom number\n")
    if x.empty:
        return
    for lab, sub in (("UNDERDOGS (+100 or longer)", x[x["price"] >= 100]),
                     ("FAVORITES", x[x["price"] < 100])):
        print(lab + "  (KenPom win % vs market no-vig %, EV per $1 at best price)")
        sub = sub[sub["ev"] > 0].sort_values("ev", ascending=False).head(top)
        for r in sub.itertuples():
            print(f"  {r.team} {r.price:+d} ({r.book}): KenPom {r.kp:.0%} vs market "
                  f"{r.mkt:.0%}, EV {r.ev:+.2f}; KenPom margin {r.margin:+.0f} (home); "
                  f"{r.game}, {r.tip}")
        print()


def show_matchup(teams: str, season: int) -> None:
    """Side-by-side KenPom profile for two teams (comma separated, KenPom
    names or a piece of them): ratings, four factors, shot mix, misc
    stats, height and experience, plus each team's games this season."""
    names = [t.strip().lower() for t in teams.split(",") if t.strip()]
    tabs = {"ratings": current_ratings(season)}
    for ep in SEASON_TABLES:
        try:
            tabs[ep] = season_table(ep, season)
        except Exception as e:                       # noqa: BLE001
            print(f"{ep}: {str(e)[:100]}")
    for ep, t in tabs.items():
        if t.empty or "TeamName" not in t:
            continue
        if "ConfOnly" in t:
            t = t[t["ConfOnly"].astype(str).str.lower() != "true"]
        pick = t[t["TeamName"].str.lower().apply(
            lambda n: any(n == x or n.startswith(x) for x in names))]
        drop = [c for c in pick.columns if c in ("DataThrough", "ConfOnly", "Season")]
        print(f"\n== {ep} ==")
        print(pick.drop(columns=drop).T.to_string(header=False))


def show_form(teams: str, season: int) -> None:
    """Each team's games this season: score, closing line, KenPom's
    pregame margin, and how the team did against both."""
    from cbb_data import fill_lines, load_season
    names = [t.strip().lower() for t in teams.split(",") if t.strip()]
    g = load_season(season)
    g = g[g["completed"].astype(bool) & (g["season_type"] != "PRE")]
    hit = lambda col: g[col].str.lower().apply(lambda n: any(x in n for x in names))
    g = g[hit("home_team") | hit("away_team")]
    g = fill_lines(g)
    g = g.assign(season=season)
    k = load_kenpom(g, [season], ratings=False, budget_min=10, verbose=False)
    for x in names:
        print(f"\n== {x}: games this season ==")
        rows = k[k["home_team"].str.lower().str.contains(x, regex=False)
                 | k["away_team"].str.lower().str.contains(x, regex=False)]
        ats, akp = [], []
        for r in rows.sort_values("commence").itertuples():
            home = x in r.home_team.lower()
            opp = r.away_team if home else r.home_team
            sgn = 1 if home else -1
            m = sgn * (r.home_score - r.away_score)
            line = sgn * r.spread_line if np.isfinite(r.spread_line) else np.nan
            kp = sgn * r.kp_margin if np.isfinite(r.kp_margin) else np.nan
            vs_l = m - line if np.isfinite(line) else np.nan
            vs_k = m - kp if np.isfinite(kp) else np.nan
            tot = r.home_score + r.away_score
            ou = ("" if not np.isfinite(r.total_line) else
                  f" total {tot:.0f} vs {r.total_line:.1f} ({'O' if tot > r.total_line else 'U'})")
            where = "vs" if r.neutral else ("home" if home else "at")
            print(f"  {r.game_date}  {where:4s} {opp:32s} {'W' if m > 0 else 'L'} {m:+4.0f}"
                  f"  line {line:+5.1f} ({vs_l:+5.1f})  KenPom {kp:+5.1f} ({vs_k:+5.1f}){ou}")
            ats.append(vs_l)
            akp.append(vs_k)
        a = pd.Series(ats).dropna()
        b = pd.Series(akp).dropna()
        if len(a):
            print(f"  ATS {int((a > 0).sum())}-{int((a < 0).sum())}, avg vs line {a.mean():+.1f}, "
                  f"last 3 vs line {a.tail(3).mean():+.1f}; avg vs KenPom {b.mean():+.1f}, "
                  f"last 3 vs KenPom {b.tail(3).mean():+.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--first", type=int, default=2021)
    ap.add_argument("--day", default="", help="print KenPom's games for a date")
    ap.add_argument("--team", default="", help="with --day: only games with this text")
    ap.add_argument("--gaps", action="store_true",
                    help="with --day: biggest KenPom vs DK/FD gaps")
    ap.add_argument("--ml", action="store_true",
                    help="with --day: KenPom win chance vs DK/FD moneylines")
    ap.add_argument("--matchup", default="", help="two teams, comma separated")
    ap.add_argument("--season", type=int, default=2027)
    a = ap.parse_args()
    if a.matchup:
        show_matchup(a.matchup, a.season)
        show_form(a.matchup, a.season)
        return
    if a.day and a.ml:
        show_ml(pd.Timestamp(a.day).date())
        return
    if a.day and a.gaps:
        show_gaps(pd.Timestamp(a.day).date())
        return
    if a.day:
        show_day(pd.Timestamp(a.day).date(), a.team)
        return
    if a.probe:
        name, key = api_key()
        print("KenPom key secret:", name or "NONE SET",
              f"(length {len(key)})" if key else "")
        for n in ("KENPOM_EMAIL", "KENPOM_USER", "KENPOM_USERNAME",
                  "KENPOM_PASSWORD"):
            if os.environ.get(n):
                print(f"also set: {n}")
        if not key:
            raise SystemExit("no KenPom key secret found")
        d = datetime.now(timezone.utc).date() - timedelta(days=2)
        rows = call("fanmatch", cache=False, d=f"{d:%Y-%m-%d}")
        print(f"fanmatch {d}: {len(rows)} games")
        if rows:
            print("fields:", sorted(rows[0]))
        old = call("fanmatch", cache=False, d="2019-01-12")
        print(f"fanmatch 2019-01-12: {len(old)} games (history check)")
        if not rows:
            raise SystemExit("KenPom answered but returned no games")
        return
    from cbb_data import load_games
    g = load_games(a.first, past_lines=False)
    k = load_kenpom(g, range(a.first, g["season"].max() + 1))
    for s, x in k.groupby("season"):
        print(s, len(x), f"kenpom line on {x['kp_margin'].notna().mean():.0%}")


if __name__ == "__main__":
    main()

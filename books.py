"""
DraftKings and FanDuel lines for today's games, from the Odds API
snapshots log_lines.py keeps in data/lines/games_<season>.csv.

    best_lines(games, day) -> {game_id: {"spread": {...}, "total": {...}}}

For each game and book the latest snapshot before tip is kept. The
breakdown then quotes our side at whichever of the two books gives the
better number (more points on a spread, the better side of a total),
then the better price.
"""
from __future__ import annotations

import glob
import os

import pandas as pd

from cbb_data import match_events

BOOKS = ("draftkings", "fanduel")
NAMES = {"draftkings": "DraftKings", "fanduel": "FanDuel"}


def load_snapshots(day, lines_dir: str = "data/lines") -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(lines_dir, "games_*.csv")))
    if not files:
        return pd.DataFrame()
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d = d[d["book"].isin(BOOKS) & d["market"].isin(["spreads", "totals"])]
    tip = pd.to_datetime(d["commence_time"], utc=True)
    et = (tip - pd.Timedelta(hours=5)).dt.date
    d = d[(et == pd.Timestamp(day).date())
          & (pd.to_datetime(d["snapshot_utc"], utc=True) < tip)]
    if d.empty:
        return d
    return (d.sort_values("snapshot_utc")
            .groupby(["event_id", "book", "market", "outcome"]).tail(1))


def best_lines(games: pd.DataFrame, day, lines_dir: str = "data/lines") -> dict:
    snap = load_snapshots(day, lines_dir)
    if snap.empty or games.empty:
        return {}
    ev = snap.drop_duplicates("event_id")
    events = [{"id": r.event_id, "commence_time": r.commence_time,
               "home_team": r.home_team, "away_team": r.away_team}
              for r in ev.itertuples()]
    gmap = match_events(games, events)
    gi = games.set_index("game_id")
    out: dict = {}
    for eid, gid in gmap.items():
        e = snap[snap["event_id"] == eid]
        odds_home = e["home_team"].iloc[0]
        swapped = False
        if gid in gi.index:
            from cbb_data import _sim
            g = gi.loc[gid]
            swapped = _sim(odds_home, g["away_team"]) > _sim(odds_home, g["home_team"])
        rec = {"spread": {}, "total": {}}
        for bk, x in e.groupby("book"):
            sp = x[x["market"] == "spreads"]
            h = sp[sp["outcome"] == odds_home]
            a = sp[sp["outcome"] != odds_home]
            if len(h) and len(a):
                hp = (float(h["point"].iloc[0]), int(h["price"].iloc[0]))
                ap = (float(a["point"].iloc[0]), int(a["price"].iloc[0]))
                if swapped:
                    hp, ap = ap, hp
                rec["spread"][bk] = {"home": hp, "away": ap}
            to = x[x["market"] == "totals"]
            o, u = to[to["outcome"] == "Over"], to[to["outcome"] == "Under"]
            if len(o) and len(u):
                rec["total"][bk] = {"over": (float(o["point"].iloc[0]), int(o["price"].iloc[0])),
                                    "under": (float(u["point"].iloc[0]), int(u["price"].iloc[0]))}
        out[str(gid)] = rec
    return out


def pick_best(rec: dict, market: str, side: str):
    """(book, point, price) giving the best number for `side` ('home',
    'away', 'over', 'under'), or None."""
    best = None
    for bk, q in (rec or {}).get(market, {}).items():
        point, price = q[side]
        # more points is better for a spread side and an under; fewer for an over
        key = (-point if side == "over" else point, price)
        if best is None or key > best[0]:
            best = (key, bk, point, price)
    return None if best is None else (NAMES.get(best[1], best[1]), best[2], best[3])

"""
Snapshot college basketball lines from The Odds API into CSVs, so the
model can be scored against numbers you could actually bet and every bet
gets a closing-line value.

Each run appends one row per book x market x outcome with a
`snapshot_utc` stamp (same format as the NBA repo's log_lines.py), for
games that haven't tipped yet.

Credit cost (The Odds API bills markets x regions, whatever the number of
games, so a 150-game Saturday costs the same as a 10-game Tuesday):
    game lines   3 markets x 2 regions (us,eu for Pinnacle)  = 6 a run
    close        2 markets, bookmakers list (bills as 1)       = 2 a run
Two line snapshots a day is ~360 credits a month. College tips run from
noon to midnight ET in waves, so the close is taken once per wave: one
snapshot covers every game tipping in the next `--close` minutes, and the
next is taken only for a game it didn't cover. Roughly 15-25 close
snapshots on a Saturday and 5-10 on a weeknight, so ~1,000 credits a
month in season.

    ODDS_API_KEY=... python log_lines.py [--out-dir data/lines]
    ODDS_API_KEY=... python log_lines.py --close 30
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timedelta, timezone

import requests

from cbb_data import current_season

API = "https://api.the-odds-api.com/v4/sports/basketball_ncaab"
GAME_MARKETS = "spreads,totals,h2h"
CLOSE_BOOKS = ("pinnacle,betonlineag,lowvig,matchbook,betfair_ex_eu,"
               "draftkings,fanduel,betmgm,williamhill_us")
FIELDS = ["snapshot_utc", "event_id", "commence_time", "home_team",
          "away_team", "book", "market", "outcome", "player", "point",
          "price", "last_update"]


def _get(url, params):
    r = requests.get(url, params=params, timeout=30)
    if r.status_code >= 400:
        print(f"  HTTP {r.status_code}: {r.text[:300]}")
    r.raise_for_status()
    left = r.headers.get("x-requests-remaining")
    if left is not None:
        print(f"  credits remaining: {left}")
    return r.json()


def _rows(event, snap):
    for bk in event.get("bookmakers", []):
        for mk in bk.get("markets", []):
            for oc in mk.get("outcomes", []):
                yield {
                    "snapshot_utc": snap,
                    "event_id": event.get("id"),
                    "commence_time": event.get("commence_time"),
                    "home_team": event.get("home_team"),
                    "away_team": event.get("away_team"),
                    "book": bk.get("key"),
                    "market": mk.get("key"),
                    "outcome": oc.get("name"),
                    "player": oc.get("description", ""),
                    "point": oc.get("point"),
                    "price": oc.get("price"),
                    "last_update": mk.get("last_update", bk.get("last_update")),
                }


def _append(path, rows):
    rows = list(rows)
    if not rows:
        return 0
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
    return len(rows)


def pregame(events, now):
    """Events that haven't tipped (the feed also returns live ones)."""
    out = []
    for e in events:
        try:
            tip = datetime.fromisoformat(
                str(e.get("commence_time")).replace("Z", "+00:00"))
        except ValueError:
            continue
        if tip > now:
            out.append(e)
    return out


def todays_games(now) -> list:
    """Today's ESPN Division I games (US date). Free."""
    from zoneinfo import ZoneInfo
    from cbb_data import _get as espn, parse_scoreboard
    day = now.astimezone(ZoneInfo("America/New_York")).date()
    return parse_scoreboard(espn({"dates": f"{day:%Y%m%d}"}))


def tips_soon(now, minutes: float, games: list) -> list:
    end = now + timedelta(minutes=minutes)
    return [r for r in games
            if r["commence"] is not None and now < r["commence"] <= end]


def close_due(now, minutes: float, games: list, state: dict) -> list:
    """Games tipping within `minutes` that no close snapshot has covered:
    the last snapshot was taken before (tip - minutes), or there was none."""
    last = state.get("last_close")
    last = datetime.fromisoformat(last) if last else None
    out = []
    for r in tips_soon(now, minutes, games):
        if last is None or last < r["commence"] - timedelta(minutes=minutes):
            out.append(r)
    return out


def log_close(key: str, out_dir: str, minutes: float, now) -> None:
    path = os.path.join(out_dir, "close_state.json")
    try:
        with open(path) as fh:
            state = json.load(fh)
    except (OSError, ValueError):
        state = {}
    due = close_due(now, minutes, todays_games(now), state)
    if not due:
        print(f"no uncovered tip in the next {minutes:g} minutes; "
              f"no credits spent")
        return
    print(f"{len(due)} games tipping soon, e.g. "
          + ", ".join(f"{r['away_team']} @ {r['home_team']}" for r in due[:4]))
    snap = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    until = (now + timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%SZ")
    events = pregame(_get(f"{API}/odds", {
        "apiKey": key, "bookmakers": CLOSE_BOOKS, "markets": "spreads,totals",
        "oddsFormat": "american", "commenceTimeTo": until}), now)
    n = _append(os.path.join(out_dir, f"close_{current_season(now.date())}.csv"),
                (r for e in events for r in _rows(e, snap)))
    print(f"  {len(events)} pregame events, {n} rows")
    state["last_close"] = now.isoformat()
    with open(path, "w") as fh:
        json.dump(state, fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/lines")
    ap.add_argument("--days-ahead", type=float, default=1.5)
    ap.add_argument("--regions", default="us,eu")
    ap.add_argument("--close", type=float, default=0.0, metavar="MINUTES",
                    help="closing-line mode: log only if a game tips within "
                         "MINUTES (checked on ESPN first)")
    a = ap.parse_args()
    key = (os.environ.get("ODDS_API_KEY") or "").strip()
    if not key:
        raise SystemExit("set ODDS_API_KEY")
    now = datetime.now(timezone.utc)
    if a.close:
        log_close(key, a.out_dir, a.close, now)
        return
    snap = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    until = (now + timedelta(days=a.days_ahead)).strftime("%Y-%m-%dT%H:%M:%SZ")
    events = pregame(_get(f"{API}/odds", {
        "apiKey": key, "regions": a.regions, "markets": GAME_MARKETS,
        "oddsFormat": "american", "commenceTimeTo": until}), now)
    n = _append(os.path.join(a.out_dir, f"games_{current_season(now.date())}.csv"),
                (r for e in events for r in _rows(e, snap)))
    print(f"{len(events)} events, {n} rows")


if __name__ == "__main__":
    main()

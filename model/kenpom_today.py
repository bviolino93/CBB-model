"""
Today's KenPom plays, filtered to the spots that did best this season
(Nov 2026, model/kenpom_day.py). None is a proven edge over past seasons.

  * Totals, KenPom 4+ points off the best DK/FD number (season 55%).
    Marked "with the middle" when it is an over on a total under 145 or
    an under on a total over 155 (season 53-54%); against the middle
    (overs at 155+, unders under 145) did badly and is left out.
  * Spreads, high-major games (both teams in a top-six conference),
    KenPom 2+ points off the best DK/FD number (high-major season 54.5%).
    Gaps of 6+ are left out (season 47%).

Priced at the better of DraftKings / FanDuel in data/lines/.

    python -m model.kenpom_today [--date YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from books import best_lines, pick_best  # noqa: E402
from cbb_data import fetch_day, tier  # noqa: E402
import kenpom  # noqa: E402

ET = ZoneInfo("America/New_York")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="")
    ap.add_argument("--lines-dir", default="data/lines")
    a = ap.parse_args()
    d = date.fromisoformat(a.date) if a.date else datetime.now(ET).date()
    g = pd.DataFrame(fetch_day(d))
    g = g.assign(season=g["season"].fillna(0).astype(int), game_date=d)
    k = kenpom.load_kenpom(g, sorted(set(g["season"])), ratings=False, budget_min=10)
    now = pd.Timestamp.now(tz="UTC")
    k = k[k["kp_margin"].notna() & (pd.to_datetime(k["commence"], utc=True) > now)].copy()
    k["tier"] = tier(k)
    bl = best_lines(k, d, a.lines_dir)
    print(f"# KenPom plays, {d}\n")
    print(f"{len(k)} games not started with a KenPom number, {len(bl)} with DK/FD lines.\n")
    tot, spr = [], []
    for r in k.itertuples():
        rec = bl.get(str(r.game_id))
        if not rec:
            continue
        tip = pd.to_datetime(r.commence, utc=True).tz_convert(ET)
        game = f"{r.away_team} @ {r.home_team}, {tip:%-I:%M %p} ET"
        kh, ka = (r.kp_total + r.kp_margin) / 2, (r.kp_total - r.kp_margin) / 2
        kp = f"KenPom {r.home_team} {kh:.0f}-{ka:.0f}"
        # totals
        for side in ("over", "under"):
            b = pick_best(rec, "total", side)
            if not b:
                continue
            gap = (r.kp_total - b[1]) if side == "over" else (b[1] - r.kp_total)
            if gap < 4:
                continue
            if (side == "over" and b[1] >= 155) or (side == "under" and b[1] < 145):
                continue     # against the middle: did badly this season
            middle = (side == "over" and b[1] < 145) or (side == "under" and b[1] > 155)
            tot.append((gap, f"{side.title()} {b[1]:g} ({b[2]:+d}, {b[0]}) | {game} | "
                        f"{kp}, total {r.kp_total:.1f} | gap {gap:.1f}"
                        f"{' | with the middle' if middle else ''} | {r.tier}-major"))
        # spreads, high-major only
        if r.tier != "high":
            continue
        for side, team, sgn in (("home", r.home_team, 1), ("away", r.away_team, -1)):
            b = pick_best(rec, "spread", side)
            if not b:
                continue
            gap = sgn * r.kp_margin + b[1]
            if 2 <= gap < 6:
                spr.append((gap, f"{team} {b[1]:+g} ({b[2]:+d}, {b[0]}) | {game} | "
                            f"{kp} | gap {gap:.1f}"))
    print("## Totals (KenPom 4+ off the line)\n")
    for gap, s in sorted(tot, reverse=True) or [(0, "none")]:
        print(f"- {s}")
    print("\n## Spreads (high-major, KenPom 2-6 off the line)\n")
    for gap, s in sorted(spr, reverse=True) or [(0, "none")]:
        print(f"- {s}")
    snap = ""
    try:
        import glob
        f = sorted(glob.glob(os.path.join(a.lines_dir, "games_*.csv")))[-1]
        snap = pd.read_csv(f, usecols=["snapshot_utc"])["snapshot_utc"].max()
    except Exception:  # noqa: BLE001
        pass
    print(f"\nLines from the snapshot at {snap} UTC; check the price before betting.")


if __name__ == "__main__":
    main()

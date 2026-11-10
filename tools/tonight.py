"""
Tonight's college spread picks from the player-based trees line.

Fits cbb_predict's trees model (live roster, no market input) on every
completed Division I game from --first on, predicts today's games, and
lists the spreads where our line is --gap+ points off ESPN's pregame
line (the backtested rule: gap 4+ won 53.6% over five held-out seasons
against ESPN's number; totals are not proven and are shown for reference
only).

    python tools/tonight.py --first 2021 --date 2026-11-10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cbb_data  # noqa: E402
import cbb_predict as cp  # noqa: E402
from cbb_data import CACHE_DIR, NON_D1, load_games, tier  # noqa: E402

_mark = cbb_data.mark_non_d1


def mark_non_d1(g: pd.DataFrame, min_games: int = 10) -> pd.DataFrame:
    """mark_non_d1, except in the season under way: there every team has
    played only a few games, so a team counts as D1 if it was D1 last
    season (10+ games) or already has 10+ games this season."""
    cur = g["season"].max()
    out = _mark(g, min_games)
    prev = out[out["season"] == cur - 1]
    d1 = set(prev["home_team"]) | set(prev["away_team"])
    d1.discard(NON_D1)
    k = (g["season"] == cur).to_numpy()
    for side in ("home", "away"):
        t = g.loc[k, f"{side}_team"]
        out.loc[k, f"{side}_team"] = t.where(t.isin(d1)
                                             | (out.loc[k, f"{side}_team"] != NON_D1),
                                             NON_D1)
    out["vs_non_d1"] = (out["home_team"] == NON_D1) | (out["away_team"] == NON_D1)
    return out


cbb_data.mark_non_d1 = mark_non_d1
from cbb_players import load_box  # noqa: E402


def espn_meta(day) -> dict:
    """Per game: odds provider, line text, status, from the cached raw
    scoreboard of that day (and the next, for late tips)."""
    out = {}
    for d in (day, day + timedelta(days=1)):
        path = os.path.join(CACHE_DIR, "days", f"{d:%Y%m%d}.json")
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            js = json.load(fh)
        for ev in js.get("events", []):
            comp = (ev.get("competitions") or [{}])[0]
            odds = (comp.get("odds") or [{}])[0]
            st = comp.get("status", ev.get("status", {})).get("type", {})
            out[str(ev.get("id"))] = {
                "provider": (odds.get("provider") or {}).get("name", ""),
                "details": odds.get("details", ""),
                "state": st.get("state", ""),
                "status": st.get("shortDetail", ""),
            }
    return out


def fmt(x: float) -> str:
    return "PK" if abs(x) < 0.01 else f"{x:+.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2021)
    ap.add_argument("--date", default="")
    ap.add_argument("--gap", type=float, default=4.0)
    a, _ = ap.parse_known_args()
    t0 = time.time()
    day = (pd.Timestamp(a.date).date() if a.date else
           (datetime.now(timezone.utc) - timedelta(hours=5)).date())

    g = load_games(a.first - 1, past_lines=False)
    g = g[g["season_type"].isin(["REG", "POST"])].reset_index(drop=True)
    print(f"loaded {len(g)} games ({time.time() - t0:.0f}s)", flush=True)
    box = load_box(g)
    print(f"box rows {len(box)}, games {box['game_id'].nunique()} "
          f"({time.time() - t0:.0f}s)", flush=True)
    h = cp.build(g, box, cp.PredictConfig(), first_season=a.first,
                 verbose=True)
    train = h[h["completed"] & (h["season"] >= a.first)
              & (pd.to_datetime(h["game_date"]).dt.date < day)]
    m = cp.fit_models(train, trees=True)
    print(f"fitted on {m.n} games ({time.time() - t0:.0f}s)", flush=True)

    te = h[(pd.to_datetime(h["game_date"]).dt.date == day)
           & ~h["completed"]].copy()
    print(f"today {day}: {len(te)} games not final, "
          f"{int(te['vs_non_d1'].sum())} against non-D1")
    te = te[~te["vs_non_d1"].astype(bool)]
    if te.empty:
        print("no D1 games left today")
        return
    p = cp.predict_frame(m, te, "trees")
    te["pred_margin"], te["pred_total"] = p["pred_margin"], p["pred_total"]
    te["tier"] = tier(te)
    meta = espn_meta(day)
    te["provider"] = te["game_id"].map(lambda i: meta.get(i, {}).get("provider", ""))
    te["state"] = te["game_id"].map(lambda i: meta.get(i, {}).get("state", ""))
    te["status"] = te["game_id"].map(lambda i: meta.get(i, {}).get("status", ""))
    te["gap"] = te["pred_margin"] - te["spread_line"]
    te["tgap"] = te["pred_total"] - te["total_line"]
    te["tip_et"] = (pd.to_datetime(te["commence"], utc=True)
                    - pd.Timedelta(hours=5)).dt.strftime("%-I:%M%p")

    def pick(r):
        if r["gap"] > 0:
            return f"{r['home_team']} {fmt(-r['spread_line'])}"
        return f"{r['away_team']} {fmt(r['spread_line'])}"

    lines = te[te["spread_line"].notna() & te["pred_margin"].notna()].copy()
    lines["pick"] = lines.apply(pick, axis=1)
    lines = lines.sort_values("commence")
    print(f"\n{len(te)} D1 games today not final, {len(lines)} with a "
          f"spread line")
    picks = lines[lines["gap"].abs() >= a.gap].copy()
    picks["abs"] = picks["gap"].abs()
    print(f"\nPICKS: spreads where our line is {a.gap:g}+ off "
          f"({len(picks)})")
    for _, r in picks.sort_values("abs", ascending=False).iterrows():
        print(f"  {r['tip_et']:>7s} ET  {r['away_team']} @ {r['home_team']}"
              f"{' (neutral)' if r['neutral'] else ''}: "
              f"BET {r['pick']}  | our home margin {r['pred_margin']:+.1f} vs "
              f"line {r['spread_line']:+.1f} (gap {r['gap']:+.1f}) | "
              f"{r['tier']}, games {r['home_n']:.0f}/{r['away_n']:.0f} | "
              f"{r['provider']} | {r['state']} {r['status']}")
    print("\nALL GAMES (home margin: ours vs line; total: ours vs line)")
    for _, r in lines.iterrows():
        print(f"  {r['tip_et']:>7s}  {r['away_team']} @ {r['home_team']}: "
              f"{r['pred_margin']:+.1f} vs {r['spread_line']:+.1f} "
              f"(gap {r['gap']:+.1f}); total {r['pred_total']:.1f} vs "
              f"{r['total_line']:.1f} | {r['tier']} | {r['state']}")
    lines.to_csv("tonight.csv", index=False)
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

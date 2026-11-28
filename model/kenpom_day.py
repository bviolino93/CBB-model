"""
How KenPom's predictions did against the spread and the total on a date.

For every completed Division I game with a KenPom FanMatch prediction:

  * spread: KenPom's side is the home team when KenPom's home margin is
    above the line (spread_line = expected home margin). It wins when the
    real margin lands on the same side of the line; equal is a push.
  * total: over when KenPom's total is above the line.

Graded against two lines:
  * ESPN: the pregame line on ESPN's scoreboard (core odds API median when
    the scoreboard has none), the line the backtests call the close.
  * DK/FD: the last DraftKings / FanDuel snapshot before tip in
    data/lines/ (games_*.csv and close_*.csv), averaged over the two
    books. Only games we logged.

    python -m model.kenpom_day --date 2026-11-27 [--to 2026-11-27] [--csv out.csv]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile
from datetime import date, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cbb_data import fetch_day, fill_lines, tier  # noqa: E402
import kenpom  # noqa: E402

BREAK_EVEN = 0.5238   # -110 both ways
GAP_BINS = [0, 2, 4, 6, np.inf]
GAP_LABELS = ["0-2", "2-4", "4-6", "6+"]


def load_days(d0: date, d1: date) -> pd.DataFrame:
    rows = []
    d = d0
    while d <= d1:
        rows.extend(fetch_day(d))
        d += timedelta(days=1)
    g = pd.DataFrame(rows)
    if g.empty:
        return g
    g = g[g["completed"]].drop_duplicates("game_id")
    g = g.assign(season=g["season"].fillna(0).astype(int))
    g = fill_lines(g, budget_min=10)
    return g.reset_index(drop=True)


def book_lines(g: pd.DataFrame, d0: date, d1: date,
               lines_dir: str = "data/lines") -> pd.DataFrame:
    """Home spread and total from DK/FD (mean of the two books), last
    snapshot before tip, for the games we logged."""
    from books import best_lines
    files = sorted(glob.glob(os.path.join(lines_dir, "games_*.csv"))
                   + glob.glob(os.path.join(lines_dir, "close_*.csv")))
    out = pd.DataFrame(index=g.index, columns=["dk_spread", "dk_total"], dtype=float)
    if not files:
        return out
    allx = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    with tempfile.TemporaryDirectory() as tmp:
        allx.to_csv(os.path.join(tmp, "games_all.csv"), index=False)
        d = d0
        while d <= d1:
            day = g[g["game_date"] == d]
            bl = best_lines(day, d, tmp) if len(day) else {}
            for idx, gid in zip(day.index, day["game_id"]):
                rec = bl.get(str(gid))
                if not rec:
                    continue
                # Odds API home point is the home handicap: -6.5 = home by 6.5
                hs = [-q["home"][0] for q in rec["spread"].values()]
                ts = [q["over"][0] for q in rec["total"].values()]
                if hs:
                    out.at[idx, "dk_spread"] = float(np.mean(hs))
                if ts:
                    out.at[idx, "dk_total"] = float(np.mean(ts))
            d += timedelta(days=1)
    return out


def grade(pred: pd.Series, line: pd.Series, actual: pd.Series) -> pd.DataFrame:
    """'W' / 'L' / 'P' for a bet on the side of `line` that `pred` is on."""
    ok = pred.notna() & line.notna() & actual.notna() & (pred != line)
    side = np.sign(pred - line)
    res = np.sign(actual - line)
    r = pd.Series(np.where(res == 0, "P", np.where(side == res, "W", "L")),
                  index=pred.index)
    return pd.DataFrame({"result": r.where(ok), "gap": (pred - line).abs().where(ok)})


def record(x: pd.DataFrame) -> str:
    w, l, p = (x["result"] == "W").sum(), (x["result"] == "L").sum(), (x["result"] == "P").sum()
    n = w + l
    pct = f"{w / n:.1%}" if n else "-"
    se = f" (se {np.sqrt(BREAK_EVEN * (1 - BREAK_EVEN) / n):.1%})" if n else ""
    return f"{w}-{l}" + (f"-{p}" if p else "") + f"  {pct}{se}"


def table(df: pd.DataFrame, res: str, gap: str) -> list[str]:
    x = df[[res, gap, "tier"]].rename(columns={res: "result", gap: "gap"})
    x = x[x["result"].notna()]
    lines = [f"| all | {record(x)} |"]
    x = x.assign(bin=pd.cut(x["gap"], GAP_BINS, labels=GAP_LABELS, right=False))
    for b in GAP_LABELS:
        s = x[x["bin"] == b]
        if len(s):
            lines.append(f"| gap {b} | {record(s)} |")
    for t in ("high", "mixed", "low"):
        s = x[x["tier"] == t]
        if len(s):
            lines.append(f"| {t}-major | {record(s)} |")
    return ["| games | record (W-L-P, win %) |", "| --- | --- |"] + lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYY-MM-DD (ET)")
    ap.add_argument("--to", default="", help="last date of a range")
    ap.add_argument("--csv", default="")
    ap.add_argument("--lines-dir", default="data/lines")
    a = ap.parse_args()
    d0 = date.fromisoformat(a.date)
    d1 = date.fromisoformat(a.to) if a.to else d0

    g = load_days(d0, d1)
    if g.empty:
        print("no completed games")
        return
    g = kenpom.load_kenpom(g, sorted(set(g["season"])), ratings=False,
                           budget_min=20)
    g = g.join(book_lines(g, d0, d1, a.lines_dir))
    g["tier"] = tier(g)
    g["margin"] = g["home_score"] - g["away_score"]
    g["total"] = g["home_score"] + g["away_score"]
    k = g[g["kp_margin"].notna()].copy()

    for tag, sp, to in (("espn", "spread_line", "total_line"),
                        ("dk", "dk_spread", "dk_total")):
        s = grade(k["kp_margin"], k[sp], k["margin"])
        t = grade(k["kp_total"], k[to], k["total"])
        k[f"{tag}_ats"], k[f"{tag}_ats_gap"] = s["result"], s["gap"]
        k[f"{tag}_ou"], k[f"{tag}_ou_gap"] = t["result"], t["gap"]

    span = f"{d0}" if d0 == d1 else f"{d0} to {d1}"
    print(f"# KenPom vs the line, {span}\n")
    print(f"{len(g)} completed D1 games, {len(k)} with a KenPom prediction, "
          f"{k['spread_line'].notna().sum()} with an ESPN spread, "
          f"{k['dk_spread'].notna().sum()} with a logged DK/FD spread.\n")
    sw = (np.sign(k["kp_margin"]) == np.sign(k["margin"])).mean()
    print(f"Straight up: KenPom picked the winner in {sw:.1%} of games.")
    both = k[k["spread_line"].notna()]
    if len(both):
        print(f"Margin error (MAE): KenPom {np.abs(both['kp_margin'] - both['margin']).mean():.2f}, "
              f"ESPN line {np.abs(both['spread_line'] - both['margin']).mean():.2f} "
              f"over {len(both)} games.")
    bt = k[k["total_line"].notna()]
    if len(bt):
        print(f"Total error (MAE): KenPom {np.abs(bt['kp_total'] - bt['total']).mean():.2f}, "
              f"ESPN line {np.abs(bt['total_line'] - bt['total']).mean():.2f} "
              f"over {len(bt)} games. KenPom over/under the line: "
              f"{(bt['kp_total'] > bt['total_line']).sum()} overs, "
              f"{(bt['kp_total'] < bt['total_line']).sum()} unders; "
              f"games went {(bt['total'] > bt['total_line']).sum()} over, "
              f"{(bt['total'] < bt['total_line']).sum()} under.")
    print(f"\nBreak-even at -110 is {BREAK_EVEN:.1%}.\n")
    for tag, name in (("espn", "ESPN pregame line"), ("dk", "DraftKings/FanDuel (logged games only)")):
        for mk, label in (("ats", "Spread"), ("ou", "Total")):
            print(f"## {label} vs {name}\n")
            print("\n".join(table(k, f"{tag}_{mk}", f"{tag}_{mk}_gap")) + "\n")

    print("## Game by game (ESPN line)\n")
    print("| game | final | KenPom | line | ATS | total line | O/U |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for r in k.sort_values("commence").itertuples():
        kp_h = (r.kp_total + r.kp_margin) / 2
        kp_a = (r.kp_total - r.kp_margin) / 2
        ln = "-" if pd.isna(r.spread_line) else f"{r.home_team} {-r.spread_line:+.1f}"
        tl = "-" if pd.isna(r.total_line) else f"{r.total_line:.1f}"
        print(f"| {r.away_team} @ {r.home_team} | {r.away_score:.0f}-{r.home_score:.0f} | "
              f"{kp_a:.0f}-{kp_h:.0f} | {ln} | {r.espn_ats if isinstance(r.espn_ats, str) else '-'} | {tl} | {r.espn_ou if isinstance(r.espn_ou, str) else '-'} |")
    if a.csv:
        k.drop(columns=["commence"]).to_csv(a.csv, index=False)


if __name__ == "__main__":
    main()

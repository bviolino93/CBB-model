"""
Market-only patterns, no model: do they hold season by season?

  * underdogs against the spread, by spread size; home vs away
  * totals: over when the line is low, under when it is high (lines set
    too far from the middle)

ESPN pregame lines (the backtests' close), every completed D1 game.

    python -m model.market_patterns --first 2020
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cbb_data import load_games, tier  # noqa: E402

BE = 0.5238


def rec(w: pd.Series) -> str:
    w = w.dropna()
    n = len(w)
    if not n:
        return "-"
    k = int(w.sum())
    return f"{k}-{n - k} {k / n:.1%} (se {np.sqrt(BE * (1 - BE) / n):.1%})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2020)
    a = ap.parse_args()
    g = load_games(a.first)
    g = g[g["completed"] & ~g["vs_non_d1"] & g["spread_line"].notna()].copy()
    g["tier"] = tier(g)
    g["month"] = pd.to_datetime(g["game_date"]).dt.month
    m = g["home_score"] - g["away_score"]
    t = g["home_score"] + g["away_score"]
    cov = np.sign(m - g["spread_line"])
    ok = (cov != 0) & (g["spread_line"] != 0)
    home_dog = g["spread_line"] < 0
    g["dog_w"] = np.where(ok, np.where(home_dog, cov > 0, cov < 0), np.nan)
    g["home_w"] = np.where(cov != 0, cov > 0, np.nan)
    g["home_dog"] = home_dog
    g["size"] = pd.cut(g["spread_line"].abs(), [0, 3, 7, 12, 20, 99], right=False,
                       labels=["<3", "3-7", "7-12", "12-20", "20+"])
    ou = np.sign(t - g["total_line"])
    g["over_w"] = np.where(g["total_line"].notna() & (ou != 0), ou > 0, np.nan)
    # where each season's total line sits: quintile within the season
    g["tq"] = g.groupby("season")["total_line"].transform(
        lambda s: pd.qcut(s, 5, labels=False, duplicates="drop"))
    seasons = sorted(g["season"].unique())

    def by_season(title, mask, col, flip=False):
        print(f"## {title}\n")
        print("| season | record |\n| --- | --- |")
        for s in seasons:
            x = g[mask & (g["season"] == s)][col]
            print(f"| {s} | {rec(1 - x if flip else x)} |")
        x = g[mask][col]
        print(f"| **all** | {rec(1 - x if flip else x)} |\n")

    print(f"# Market patterns, {seasons[0]}-{seasons[-1]} (ESPN pregame lines)\n")
    print(f"{len(g)} completed D1 games with a line. Break-even at -110: {BE:.1%}\n")
    every = pd.Series(True, index=g.index)
    by_season("Every underdog ATS", every, "dog_w")
    by_season("Underdogs getting under 12 points", g["spread_line"].abs() < 12, "dog_w")
    by_season("Home underdogs", home_dog, "dog_w")
    by_season("Home teams ATS", every, "home_w")
    print("## Underdogs ATS by spread size and month (all seasons)\n")
    print("| | record |\n| --- | --- |")
    for s, x in g.groupby("size", observed=True):
        print(f"| spread {s} | {rec(x['dog_w'])} |")
    for mth, lab in ((11, "Nov"), (12, "Dec"), (1, "Jan"), (2, "Feb"), (3, "Mar")):
        print(f"| {lab} | {rec(g[g['month'] == mth]['dog_w'])} |")
    for tr in ("high", "mixed", "low"):
        print(f"| {tr}-major | {rec(g[g['tier'] == tr]['dog_w'])} |")
    print()
    by_season("Over in the lowest fifth of each season's totals", g["tq"] == 0, "over_w")
    by_season("Under in the highest fifth of each season's totals", g["tq"] == 4, "over_w", flip=True)
    print("## Overs by total-line fifth and month (all seasons)\n")
    print("| | over record |\n| --- | --- |")
    for q in range(5):
        print(f"| fifth {q + 1} (low to high) | {rec(g[g['tq'] == q]['over_w'])} |")
    for mth, lab in ((11, "Nov"), (12, "Dec"), (1, "Jan"), (2, "Feb"), (3, "Mar")):
        x = g[g["month"] == mth]
        print(f"| {lab}: low fifth over | {rec(x[x['tq'] == 0]['over_w'])} |")
        print(f"| {lab}: high fifth under | {rec(1 - x[x['tq'] == 4]['over_w'])} |")


if __name__ == "__main__":
    main()

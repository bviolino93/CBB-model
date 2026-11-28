"""
Underdog moneylines: how KenPom did when it picked the market underdog.

The underdog is the side with the longer moneyline in ESPN's pregame odds
(core odds API, median over the books ESPN lists). For each completed D1
game with a KenPom FanMatch prediction:

  * KenPom picks the underdog to win outright when its win chance for the
    underdog is above 50%.
  * "value" dogs: KenPom's win chance is above the no-vig market chance,
    even if under 50%.

Profit is flat 1 unit a bet at the median price.

    python -m model.kenpom_ml --date 2026-11-01 --to 2026-11-27
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cbb_data import SKIP_PROVIDER, fetch_core_odds, tier  # noqa: E402
import kenpom  # noqa: E402
from model.kenpom_day import load_days  # noqa: E402


def _ml(side: dict) -> float:
    """American moneyline from one side's odds block, or NaN."""
    if not isinstance(side, dict):
        return float("nan")
    for path in (("moneyLine",), ("close", "odds"), ("current", "moneyLine", "american"),
                 ("close", "moneyLine", "american")):
        v = side
        for p in path:
            v = v.get(p) if isinstance(v, dict) else None
        try:
            f = float(str(v).replace("+", ""))
            if abs(f) >= 100:
                return f
        except (TypeError, ValueError):
            continue
    return float("nan")


def parse_ml(js: dict) -> tuple[float, float, int]:
    """(home ML, away ML, books): medians across ESPN's providers."""
    h, a = [], []
    for it in (js or {}).get("items", []):
        name = str((it.get("provider") or {}).get("name", ""))
        if SKIP_PROVIDER.search(name):
            continue
        hm = _ml(it.get("homeTeamOdds"))
        am = _ml(it.get("awayTeamOdds"))
        if np.isnan(hm) or np.isnan(am):
            ml = it.get("moneyline") or {}
            hm, am = _ml(ml.get("home")), _ml(ml.get("away"))
        if np.isfinite(hm) and np.isfinite(am):
            h.append(hm)
            a.append(am)
    if not h:
        return float("nan"), float("nan"), 0
    def med(v):
        # A median of -100 and +100 is 0: anything inside (-100, 100) is even money.
        m = float(np.median(v))
        return 100.0 if abs(m) < 100 else m
    return med(h), med(a), len(h)


def implied(ml: float) -> float:
    return 100 / (ml + 100) if ml > 0 else -ml / (-ml + 100)


def payout(ml: float) -> float:
    """Profit on a 1-unit winning bet."""
    return ml / 100 if ml > 0 else 100 / -ml


def summary(x: pd.DataFrame) -> str:
    if x.empty:
        return "| 0 | - | - | - | - |"
    w = int(x["dog_won"].sum())
    n = len(x)
    pl = np.where(x["dog_won"], x["dog_ml"].map(payout), -1.0).sum()
    return (f"| {n} | {w}-{n - w} ({w / n:.1%}) | {x['dog_ml'].median():+.0f} | "
            f"{pl:+.2f} | {pl / n:+.1%} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--to", default="")
    ap.add_argument("--csv", default="")
    a = ap.parse_args()
    d0 = date.fromisoformat(a.date)
    d1 = date.fromisoformat(a.to) if a.to else d0

    g = load_days(d0, d1)
    g = kenpom.load_kenpom(g, sorted(set(g["season"])), ratings=False, budget_min=20)
    g = g[g["kp_wp"].notna()].copy()
    with ThreadPoolExecutor(16) as ex:
        docs = list(ex.map(fetch_core_odds, g["game_id"]))
    ml = [parse_ml(js) for js in docs]
    g["home_ml"] = [m[0] for m in ml]
    g["away_ml"] = [m[1] for m in ml]
    if not np.isfinite(g["home_ml"]).any():
        print("No moneylines found in ESPN's odds. Sample item keys:")
        for js in docs[:3]:
            for it in (js or {}).get("items", [])[:2]:
                print("  ", sorted(it.keys()), sorted((it.get("homeTeamOdds") or {}).keys()))
        return
    g = g[np.isfinite(g["home_ml"]) & np.isfinite(g["away_ml"])
          & (g["home_ml"] != g["away_ml"])].copy()
    g["tier"] = tier(g)
    home_dog = g["home_ml"] > g["away_ml"]
    g["dog"] = np.where(home_dog, g["home_team"], g["away_team"])
    g["dog_ml"] = np.where(home_dog, g["home_ml"], g["away_ml"])
    fav_ml = np.where(home_dog, g["away_ml"], g["home_ml"])
    pd_ = g["dog_ml"].map(implied)
    pf = pd.Series(fav_ml, index=g.index).map(implied)
    g["mkt_dog_wp"] = pd_ / (pd_ + pf)
    g["kp_dog_wp"] = np.where(home_dog, g["kp_wp"], 1 - g["kp_wp"])
    margin = g["home_score"] - g["away_score"]
    g["dog_won"] = np.where(home_dog, margin > 0, margin < 0)

    span = f"{d0}" if d0 == d1 else f"{d0} to {d1}"
    print(f"# KenPom and underdog moneylines, {span}\n")
    print(f"{len(g)} completed D1 games with a KenPom prediction and an ESPN "
          f"moneyline. Underdogs won {g['dog_won'].mean():.1%} outright "
          f"(no-vig market expected {g['mkt_dog_wp'].mean():.1%}).\n")
    hdr = ("| bets | record | median price | profit (units) | ROI |\n"
           "| --- | --- | --- | --- | --- |")
    pick = g[g["kp_dog_wp"] > 0.5]
    value = g[g["kp_dog_wp"] > g["mkt_dog_wp"]]
    print("## KenPom picks the underdog to win (KenPom win chance > 50%)\n")
    print(hdr)
    print(summary(pick) + "\n")
    print("## Every underdog\n")
    print(hdr)
    print(summary(g) + "\n")
    print("## Value underdogs (KenPom win chance above the no-vig market chance)\n")
    print("| edge (KenPom - market) | bets | record | median price | profit | ROI |")
    print("| --- | --- | --- | --- | --- | --- |")
    e = value["kp_dog_wp"] - value["mkt_dog_wp"]
    for lo, hi, lab in ((0, .03, "0-3 pts"), (.03, .06, "3-6 pts"),
                        (.06, .10, "6-10 pts"), (.10, 1, "10+ pts"), (0, 1, "all")):
        print(f"| {lab} " + summary(value[(e > lo) & (e <= hi)]))
    print()
    print("## KenPom-picked underdogs by price\n")
    print("| price | bets | record | median price | profit | ROI |")
    print("| --- | --- | --- | --- | --- | --- |")
    for lo, hi in ((100, 130), (130, 170), (170, 250), (250, 100000)):
        s = pick[(pick["dog_ml"] >= lo) & (pick["dog_ml"] < hi)]
        print(f"| +{lo} to +{hi if hi < 100000 else '...'} " + summary(s))
    print()
    print("## KenPom-picked underdogs by tier\n")
    print("| tier | bets | record | median price | profit | ROI |")
    print("| --- | --- | --- | --- | --- | --- |")
    for t in ("high", "mixed", "low"):
        print(f"| {t}-major " + summary(pick[pick["tier"] == t]))
    print()
    print("## Every KenPom-picked underdog\n")
    print("| date | game | underdog | price | KenPom win | market win | final | result |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in pick.sort_values("commence").itertuples():
        print(f"| {r.game_date} | {r.away_team} @ {r.home_team} | {r.dog} | {r.dog_ml:+.0f} | "
              f"{r.kp_dog_wp:.0%} | {r.mkt_dog_wp:.0%} | {r.away_score:.0f}-{r.home_score:.0f} | "
              f"{'W' if r.dog_won else 'L'} |")
    if a.csv:
        g.drop(columns=["commence"]).to_csv(a.csv, index=False)


if __name__ == "__main__":
    main()

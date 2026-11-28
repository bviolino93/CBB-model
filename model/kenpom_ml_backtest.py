"""
Backtest: back the moneyline underdog when KenPom picks it to win.

Every completed D1 game from --first (season end year) to today with a
KenPom FanMatch prediction (published the morning of the game) and an
ESPN pregame moneyline for both sides (median over ESPN's books). The
underdog is the side with the longer price. A bet is 1 unit on the
underdog at that price whenever KenPom's win chance for it is above
--min-wp (default 50%). Nothing is fitted, so every season is
out-of-sample for the rule itself.

ESPN core odds are fetched per game and cached in .cache_cbb/odds
(time-budgeted by CBB_ML_FETCH_MINUTES; run again to fill the rest).

    python -m model.kenpom_ml_backtest --first 2020
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cbb_data import CACHE_DIR, fetch_core_odds, load_games, tier  # noqa: E402
import kenpom  # noqa: E402
from model.kenpom_ml import implied, parse_ml, payout  # noqa: E402


def fetch_all(ids, budget_min: float) -> list[dict]:
    deadline = time.time() + budget_min * 60
    odir = os.path.join(CACHE_DIR, "odds")

    def one(gid):
        if time.time() > deadline and not os.path.exists(os.path.join(odir, f"{gid}.json")):
            return None
        try:
            return fetch_core_odds(gid)
        except Exception:      # noqa: BLE001 - one bad game must not stop the run
            return None

    with ThreadPoolExecutor(16) as ex:
        return list(ex.map(one, ids))


def dogs(g: pd.DataFrame, docs) -> pd.DataFrame:
    ml = [parse_ml(js) if js else (np.nan, np.nan, 0) for js in docs]
    g = g.assign(home_ml=[m[0] for m in ml], away_ml=[m[1] for m in ml])
    g = g[np.isfinite(g["home_ml"]) & np.isfinite(g["away_ml"])
          & (g["home_ml"] != g["away_ml"])].copy()
    home_dog = g["home_ml"] > g["away_ml"]
    g["dog"] = np.where(home_dog, g["home_team"], g["away_team"])
    g["dog_ml"] = np.where(home_dog, g["home_ml"], g["away_ml"])
    fav = pd.Series(np.where(home_dog, g["away_ml"], g["home_ml"]), index=g.index)
    pdg, pf = g["dog_ml"].map(implied), fav.map(implied)
    g["mkt_dog_wp"] = pdg / (pdg + pf)
    g["kp_dog_wp"] = np.where(home_dog, g["kp_wp"], 1 - g["kp_wp"])
    m = g["home_score"] - g["away_score"]
    g["dog_won"] = np.where(home_dog, m > 0, m < 0)
    g["pl"] = np.where(g["dog_won"], g["dog_ml"].map(payout), -1.0)
    g["tier"] = tier(g)
    g["month"] = pd.to_datetime(g["game_date"]).dt.month
    return g


def row(label: str, x: pd.DataFrame) -> str:
    n = len(x)
    if not n:
        return f"| {label} | 0 | - | - | - | - | - |"
    w = int(x["dog_won"].sum())
    pl = x["pl"].sum()
    # se of ROI from the spread of per-bet results
    se = x["pl"].std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    exp = x["mkt_dog_wp"].mean()
    return (f"| {label} | {n} | {w}-{n - w} ({w / n:.1%}) | {exp:.1%} | "
            f"{x['dog_ml'].median():+.0f} | {pl:+.1f} | {pl / n:+.1%} (se {se:.1%}) |")


HDR = ("| | bets | record | market expected | median price | units | ROI |\n"
       "| --- | --- | --- | --- | --- | --- | --- |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2020)
    ap.add_argument("--min-wp", type=float, default=0.5)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    g = load_games(a.first, past_lines=False)
    g = g[g["completed"] & ~g["vs_non_d1"]]
    g = kenpom.load_kenpom(g, sorted(set(g["season"])), ratings=False)
    g = g[g["kp_wp"].notna()].reset_index(drop=True)
    print(f"{len(g)} completed D1 games with a KenPom prediction")
    docs = fetch_all(g["game_id"], float(os.environ.get("CBB_ML_FETCH_MINUTES", "150")))
    missing = sum(d is None for d in docs)
    if missing:
        print(f"  ESPN odds: {missing} games not fetched yet (time budget); run again")
    d = dogs(g, docs)
    print(f"{len(d)} with a moneyline on both sides\n")
    pick = d[d["kp_dog_wp"] > a.min_wp]

    print(f"# Underdog moneylines when KenPom has the dog above {a.min_wp:.0%}\n")
    print("## By season\n")
    print(HDR)
    for s, x in pick.groupby("season"):
        print(row(str(s), x))
    print(row("**all**", pick))
    print(row("every underdog (baseline)", d))
    print()
    for name, col, groups in (
            ("price", "dog_ml", ((-10000, 100, "pick'em (minus price)"), (100, 130, "+100 to +129"),
                                 (130, 170, "+130 to +169"), (170, 250, "+170 to +249"),
                                 (250, 100000, "+250 and up"))),
            ("KenPom win chance", "kp_dog_wp", ((0.5, 0.55, "50-55%"), (0.55, 0.6, "55-60%"),
                                                (0.6, 1.01, "60%+"))),
    ):
        print(f"## By {name}\n")
        print(HDR)
        for lo, hi, lab in groups:
            print(row(lab, pick[(pick[col] >= lo) & (pick[col] < hi)]))
        print()
    print("## By tier\n")
    print(HDR)
    for t in ("high", "mixed", "low"):
        print(row(f"{t}-major", pick[pick["tier"] == t]))
    print()
    print("## By month\n")
    print(HDR)
    for mth, lab in ((11, "Nov"), (12, "Dec"), (1, "Jan"), (2, "Feb"), (3, "Mar"), (4, "Apr")):
        print(row(lab, pick[pick["month"] == mth]))
    print()
    print("## Neutral site\n")
    print(HDR)
    print(row("neutral", pick[pick["neutral"]]))
    print(row("home/away", pick[~pick["neutral"]]))
    print()
    print("## Other thresholds (all seasons)\n")
    print(HDR)
    for t in (0.45, 0.5, 0.55, 0.6):
        print(row(f"KenPom dog > {t:.0%}", d[d["kp_dog_wp"] > t]))
    if a.out:
        d.drop(columns=["commence"]).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()

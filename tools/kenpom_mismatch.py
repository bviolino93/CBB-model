"""
Does KenPom lean over (and toward the favorite) in mismatches, and is
that lean right? Buckets games by KenPom's projected margin and reports,
against ESPN's closing line: how often KenPom's total is over the line,
how often the game actually went over, how KenPom's over / under leans
did, and favorite / KenPom-side cover rates. Pushes are left out.

    python tools/kenpom_mismatch.py --games kp_games.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

BE = 110 / 210


def pct(won: pd.Series) -> str:
    won = won.dropna()
    n = len(won)
    return f"{won.mean():5.1%} n={n:5d}" if n else "   -  n=    0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="kp_games.csv")
    a = ap.parse_args()
    f = pd.read_csv(a.games, low_memory=False)
    f = f[f["kp_margin"].notna() & f["total_line"].notna() & f["spread_line"].notna()]
    am = f["home_score"] - f["away_score"]
    at = f["home_score"] + f["away_score"]
    f = f.assign(
        absm=f["kp_margin"].abs(),
        tgap=f["kp_total"] - f["total_line"],
        sgap=f["kp_margin"] - f["spread_line"],
        over=np.where(at == f["total_line"], np.nan, (at > f["total_line"]).astype(float)),
        fav_cover=np.where(am == f["spread_line"], np.nan,
                           (np.sign(am - f["spread_line"]) == np.sign(f["spread_line"])).astype(float)),
        home_cover=np.where(am == f["spread_line"], np.nan,
                            (am > f["spread_line"]).astype(float)),
    )
    f.loc[f["spread_line"] == 0, "fav_cover"] = np.nan
    bins = [0, 5, 10, 15, 20, 25, 99]
    f["bucket"] = pd.cut(f["absm"], bins, right=False,
                         labels=["0-5", "5-10", "10-15", "15-20", "20-25", "25+"])
    print(f"{len(f)} games with KenPom and a closing line. Break-even {BE:.1%}.\n")
    print("== totals by KenPom projected margin ==")
    print("  margin  KenPom total over line | games over | KenPom 3+ over: won | "
          "KenPom 3+ under: won | avg KenPom-line")
    for b, x in f.groupby("bucket", observed=True):
        lo = x[x["tgap"] >= 3]
        lu = x[x["tgap"] <= -3]
        print(f"  {b:6s}  {(x['tgap'] > 0).mean():5.1%} (n={len(x):5d})        | "
              f"{pct(x['over'])} | {pct(lo['over'])} | {pct(1 - lu['over'])} | "
              f"{x['tgap'].mean():+.2f}")
    print("\n== spreads by KenPom projected margin ==")
    print("  margin  favorite covers | KenPom side 3+ covers | KenPom likes fav (of 3+) | "
          "KenPom 3+ on fav: won | on dog: won")
    for b, x in f.groupby("bucket", observed=True):
        s = x[x["sgap"].abs() >= 3]
        side_home = np.sign(s["sgap"])
        won = np.where(side_home > 0, s["home_cover"], 1 - s["home_cover"])
        won = pd.Series(won, index=s.index)
        on_fav = np.sign(s["sgap"]) == np.sign(s["spread_line"])
        print(f"  {b:6s}  {pct(x['fav_cover'])} | {pct(won)} | {on_fav.mean():5.1%}"
              f"                   | {pct(won[on_fav])} | {pct(won[~on_fav])}")
    print("\n== by season, games with KenPom margin 15+: over rate / KenPom 3+ over won ==")
    big = f[f["absm"] >= 15]
    for s, x in big.groupby("season"):
        print(f"  {s}: games over {pct(x['over'])}   KenPom 3+ over won "
              f"{pct(x.loc[x['tgap'] >= 3, 'over'])}   fav covers {pct(x['fav_cover'])}")


if __name__ == "__main__":
    main()

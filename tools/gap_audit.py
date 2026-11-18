"""Do big model-vs-line gaps win? Slices the lab's per-game held-out
predictions (lab_games.csv from "Player model lab") by gap size, games
played, line size and side, for the trees spread line.

    python tools/gap_audit.py lab_games.csv
"""
import sys

import numpy as np
import pandas as pd

d = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "lab_games.csv")
d = d[d["spread_line"].notna() & d["trees_margin"].notna()].copy()
d["gap"] = d["trees_margin"] - d["spread_line"]
d["act"] = d["home_score"] - d["away_score"]
d["res"] = np.sign(d["gap"] * (d["act"] - d["spread_line"]))
d["ag"] = d["gap"].abs()
d["al"] = d["spread_line"].abs()
fav_home = d["spread_line"] > 0
bet_home = d["gap"] > 0
d["bet_dog"] = np.where(fav_home, ~bet_home, bet_home) & (d["spread_line"] != 0)
dt = pd.to_datetime(d["game_date"])
d["first_week"] = (dt.dt.month == 11) & (dt.dt.day <= 16)
d["mn"] = d["min_n"].fillna(0)


def rec(x):
    w, l = int((x["res"] > 0).sum()), int((x["res"] < 0).sum())
    n = w + l
    if not n:
        return "    0 bets"
    p = w / n
    return f"{n:5d} bets {p:6.1%} (se {np.sqrt(p * (1 - p) / n):.1%})"


g = d[d["ag"] >= 4]
print(f"held-out games with a line: {len(d)}; gap 4+: {len(g)} {rec(g)}\n")
buckets = [(4, 6), (6, 8), (8, 10), (10, 99)]
print("BY GAP SIZE")
for lo, hi in buckets:
    print(f"  gap {lo}-{hi if hi < 99 else '+'}: {rec(d[(d.ag >= lo) & (d.ag < hi)])}")
for lab, k in [("games played < 4", g.mn < 4), ("games played 4-7", (g.mn >= 4) & (g.mn < 8)),
               ("games played 8+", g.mn >= 8), ("first week (Nov 1-16)", g.first_week)]:
    print(f"\n{lab}: {rec(g[k])}")
    for lo, hi in buckets:
        s = g[k & (g.ag >= lo) & (g.ag < hi)]
        print(f"  gap {lo}-{hi if hi < 99 else '+'}: {rec(s)}")
print("\nBY TIER x GAP")
for t in ("low", "mixed", "high"):
    for lo, hi in buckets:
        s = g[(g.tier == t) & (g.ag >= lo) & (g.ag < hi)]
        print(f"  {t:5s} gap {lo}-{hi if hi < 99 else '+'}: {rec(s)}")
print("\nBY LINE SIZE x SIDE (gap 4+)")
for lab, k in [("|line| < 10", g.al < 10), ("|line| 10-20", (g.al >= 10) & (g.al < 20)),
               ("|line| 20+", g.al >= 20)]:
    print(f"  {lab:12s} bet dog: {rec(g[k & g.bet_dog])}   bet fav: {rec(g[k & ~g.bet_dog])}")
print("\nFIRST WEEK, gap 4+, by tier and side")
fw = g[g.first_week]
for t in ("low", "mixed", "high"):
    s = fw[fw.tier == t]
    print(f"  {t:5s} bet dog: {rec(s[s.bet_dog])}   bet fav: {rec(s[~s.bet_dog])}")
print("\nCANDIDATE RULES")
rules = {
    "current: low tier, gap 4+": (g.tier == "low"),
    "low tier, gap 4-8": (g.tier == "low") & (g.ag < 8),
    "any tier, gap 4-8": g.ag < 8,
    "low tier, gap 4-8, |line| < 15": (g.tier == "low") & (g.ag < 8) & (g.al < 15),
    "any tier, gap 4-8, |line| < 15": (g.ag < 8) & (g.al < 15),
}
for lab, k in rules.items():
    s = g[k]
    by = " ".join(f"{(x.res > 0).sum() / max((x.res != 0).sum(), 1):.0%}"
                  for _, x in s.groupby("season"))
    print(f"  {lab:34s} {rec(s)}  by season {by}")

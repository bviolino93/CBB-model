"""Which college spread picks should get real money? An honest rule search.

Reads the lab's per-game held-out predictions (lab_games.csv from "Player
model lab"; every prediction was made by a model fit on earlier seasons
only). Tries a grid of pick rules, CHOOSES on the early test seasons and
then REPORTS the chosen rules on the later seasons they never saw. Also
shows how many picks a night each rule gives, since one pick every few days
is no use to a daily bettor.

    python tools/rule_search.py lab_games.csv [--split 2025]
"""
import argparse
import itertools

import numpy as np
import pandas as pd

BE = 110 / 210  # break-even at -110


def load(path):
    d = pd.read_csv(path)
    d = d[d["spread_line"].notna() & d["trees_margin"].notna()].copy()
    d["act"] = d["home_score"] - d["away_score"]
    d["gap"] = d["trees_margin"] - d["spread_line"]
    d["ag"] = d["gap"].abs()
    d["res"] = np.sign(d["gap"] * (d["act"] - d["spread_line"]))
    d = d[d["res"] != 0]  # pushes and zero gaps
    d["win"] = (d["res"] > 0).astype(int)
    # A second, simpler model (ridge on players) that must agree on the side.
    r = d["+players live_margin"] - d["spread_line"]
    d["agree"] = np.sign(r) == np.sign(d["gap"])
    d["agree2"] = d["agree"] & (r.abs() >= 2)
    d["r_gap"] = r
    bet_home = d["gap"] > 0
    fav_home = d["spread_line"] > 0
    d["bet_home"] = bet_home
    d["bet_dog"] = np.where(fav_home, ~bet_home, bet_home) & (d["spread_line"] != 0)
    d["al"] = d["spread_line"].abs()
    d["mn"] = d["min_n"].fillna(0)
    d["books"] = d["line_books"].fillna(-1)
    d["month"] = pd.to_datetime(d["game_date"]).dt.month
    return d


FILTERS = {
    "tier": {"any": None, "low": lambda d: d.tier == "low",
             "low+mixed": lambda d: d.tier != "high",
             "high": lambda d: d.tier == "high"},
    "gap": {f"{lo}-{hi if hi < 99 else '+'}": (lambda lo, hi: lambda d: (d.ag >= lo) & (d.ag < hi))(lo, hi)
            for lo, hi in [(3, 99), (4, 99), (5, 99), (6, 99), (8, 99), (4, 8), (3, 6)]},
    "side": {"any": None, "home": lambda d: d.bet_home, "away": lambda d: ~d.bet_home,
             "dog": lambda d: d.bet_dog, "fav": lambda d: ~d.bet_dog},
    "agree": {"any": None, "ridge agrees": lambda d: d.agree,
              "ridge agrees 2+": lambda d: d.agree2},
    "line": {"any": None, "|line|<15": lambda d: d.al < 15},
    "season part": {"any": None, "early(<8 games)": lambda d: d.mn < 8,
                    "Nov-Dec": lambda d: d.month.isin([11, 12])},
}


def rec(x):
    n = len(x)
    if not n:
        return "    0 bets"
    p = x["win"].mean()
    return f"{n:5d} bets {p:6.1%} (se {np.sqrt(p * (1 - p) / n):.1%})"


def per_night(x, all_nights):
    c = x.groupby("game_date").size().reindex(all_nights, fill_value=0)
    return f"avg {c.mean():.1f}/night, {(c >= 2).mean():.0%} of nights 2+, {(c == 0).mean():.0%} with none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="lab_games.csv")
    ap.add_argument("--split", type=int, default=2025,
                    help="first season of the confirmation set")
    ap.add_argument("--min-bets", type=int, default=300)
    a = ap.parse_args()
    d = load(a.path)
    tr, te = d[d.season < a.split], d[d.season >= a.split]
    print(f"games with a line, no push: {len(d)}. Choose on seasons "
          f"{sorted(tr.season.unique())} ({len(tr)}), confirm on "
          f"{sorted(te.season.unique())} ({len(te)}). Break-even {BE:.1%}.\n")
    print(f"baseline every game: choose {tr.win.mean():.1%}, confirm {te.win.mean():.1%}")
    print(f"current rule (low tier, gap 4+): choose {rec(tr[(tr.tier == 'low') & (tr.ag >= 4)])}"
          f" | confirm {rec(te[(te.tier == 'low') & (te.ag >= 4)])}\n")

    keys = list(FILTERS)
    rows = []
    for combo in itertools.product(*[FILTERS[k].items() for k in keys]):
        def mask(x):
            m = np.ones(len(x), bool)
            for _, f in combo:
                if f is not None:
                    m &= f(x).to_numpy()
            return m
        mt, mc = mask(tr), mask(te)
        n = int(mt.sum())
        if n < a.min_bets:
            continue
        p = tr.win.to_numpy()[mt].mean()
        # Rank on a pessimistic win rate: 1.5 standard errors under the mean.
        lb = p - 1.5 * np.sqrt(p * (1 - p) / n)
        q = te.win.to_numpy()[mc]
        rows.append({"rule": ", ".join(f"{k}={v}" for k, (v, _) in zip(keys, combo)
                                       if v != "any"),
                     "n_choose": n, "win_choose": p, "lb": lb,
                     "n_confirm": len(q), "win_confirm": q.mean() if len(q) else np.nan,
                     "combo": combo})
    r = pd.DataFrame(rows).sort_values("lb", ascending=False)
    print(f"{len(r)} rules with {a.min_bets}+ bets in the choose seasons.")
    print(f"All rules, average confirm win rate: {r.win_confirm.mean():.1%}. "
          f"Top 20 by choose-season score: {r.head(20).win_confirm.mean():.1%}. "
          f"Bottom half: {r.tail(len(r) // 2).win_confirm.mean():.1%}.")
    print("(If the top 20 don't beat the average on unseen seasons, the "
          "search found noise.)\n")
    print("TOP 20 RULES (chosen on early seasons, confirmed on later ones)")
    for _, x in r.head(20).iterrows():
        print(f"  {x.win_choose:5.1%} on {x.n_choose:4d} -> {x.win_confirm:5.1%} on "
              f"{x.n_confirm:4d}   {x.rule or 'everything'}")

    print("\nCORRELATION choose vs confirm win rate across all rules: "
          f"{r[['win_choose', 'win_confirm']].corr().iloc[0, 1]:+.2f}")

    print("\nSINGLE FILTERS on top of gap 4+ (all seasons, then per season)")
    g = d[d.ag >= 4]
    for k in keys:
        if k == "gap":
            continue
        for v, f in FILTERS[k].items():
            if f is None:
                continue
            s = g[f(g).to_numpy()]
            by = " ".join(f"{x.win.mean():.0%}" for _, x in s.groupby("season"))
            print(f"  {k:11s} {v:17s} {rec(s)}   by season {by}")

    nights = sorted(d.game_date.unique())
    nov = [n for n in nights if pd.Timestamp(n).month == 11]
    print("\nPICKS PER NIGHT (all test seasons; November only)")
    cand = {
        "current: low tier, gap 4+": (d.tier == "low") & (d.ag >= 4),
        "any tier, gap 4+": d.ag >= 4,
        "gap 4+, ridge agrees": (d.ag >= 4) & d.agree,
        "low+mixed, gap 4+, ridge agrees": (d.tier != "high") & (d.ag >= 4) & d.agree,
        "gap 4+, ridge agrees 2+": (d.ag >= 4) & d.agree2,
        "gap 6+, ridge agrees 2+": (d.ag >= 6) & d.agree2,
    }
    for _, x in r.head(3).iterrows():
        m = np.ones(len(d), bool)
        for _, f in x.combo:
            if f is not None:
                m &= f(d).to_numpy()
        cand[f"top rule: {x.rule}"] = pd.Series(m, index=d.index)
    for lab, k in cand.items():
        s = d[k]
        sn = s[s.month == 11]
        by = " ".join(f"{x.win.mean():.0%}" for _, x in s.groupby("season"))
        print(f"  {lab}\n    {rec(s)}  by season {by}\n    all nights: {per_night(s, nights)}"
              f"\n    November:   {per_night(sn, nov)}  {rec(sn)}")

    print("\nTOP N A NIGHT by gap (only games where the ridge agrees)")
    ag = d[d.agree].sort_values("ag", ascending=False)
    for n in (1, 2, 3, 5, 8):
        top = ag.groupby("game_date").head(n)
        by = " ".join(f"{x.win.mean():.0%}" for _, x in top.groupby("season"))
        print(f"  top {n}: {rec(top)}  by season {by}")

    print("\nTOTALS (trees total vs total line), gap 6+ / 8+, ridge agrees")
    if "trees_total" in d and d["total_line"].notna().any():
        t = d[d.total_line.notna()].copy()
        tg = t.trees_total - t.total_line
        rg = t["+players live_total"] - t.total_line
        ta = t.home_score + t.away_score
        res = np.sign(tg * (ta - t.total_line))
        for thr in (4, 6, 8):
            for lab, k in (("any", tg.abs() >= thr),
                           ("ridge agrees", (tg.abs() >= thr) & (np.sign(rg) == np.sign(tg)))):
                x = res[k & (res != 0)]
                n = len(x)
                p = (x > 0).mean() if n else float("nan")
                print(f"  gap {thr}+ {lab:13s} {n:5d} bets {p:.1%}")


if __name__ == "__main__":
    main()

"""
Other ways to use the data than "bet the KenPom gap". Every rule is found
on 2020-22 and checked on 2023-26 (kenpom_research.show); the classifier
is refitted each season on earlier seasons only. Break-even 52.4%.

  A. team home court: teams that beat (or miss) KenPom at home by more
     than KenPom's flat home edge, measured on earlier seasons only
  B. luck: teams winning more (or less) than KenPom's win chances said,
     earlier games this season only
  C. spots: letdown after an upset, bounce-back after a blowout loss
  D. fade KenPom in games it projects within 5
  E. line movement (opener to close; ESPN openers exist 2023+ only)
  F. a cover classifier: probability of covering, bet only when sure

    python tools/kenpom_alt.py --games kp_games.csv --odds .cache_cbb/odds
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from kenpom_research import BE, SPLIT, add_openers, build, grade, rec, show  # noqa: E402


def long_games(f: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game, team perspective, in time order."""
    rows = []
    for side, sgn in (("home", 1), ("away", -1)):
        rows.append(pd.DataFrame({
            "game_id": f["game_id"], "season": f["season"],
            "t": pd.to_datetime(f["commence"], utc=True), "team": f[f"{side}_team"],
            "side": side, "neutral": f["neutral_f"],
            "margin": sgn * f["actual_m"], "kp": sgn * f["kp_margin"],
            "line": sgn * f["spread_line"],
            "wp": f["kp_wp"] if side == "home" else 1 - f["kp_wp"]}))
    return pd.concat(rows).sort_values(["team", "t"]).reset_index(drop=True)


def add_features(f: pd.DataFrame) -> pd.DataFrame:
    f = f.copy()
    L = long_games(f)
    L["vs_kp"] = L["margin"] - L["kp"]
    L["won"] = (L["margin"] > 0).astype(float)
    # A. home-court residual from EARLIER seasons: home vs_kp minus away vs_kp.
    hca = {}
    for team, x in L[L["neutral"] == 0].groupby("team"):
        for s in x["season"].unique():
            p = x[x["season"] < s]
            h, a = p[p["side"] == "home"]["vs_kp"], p[p["side"] == "away"]["vs_kp"]
            if len(h) >= 25 and len(a) >= 25:
                hca[(team, s)] = (h.mean() - a.mean()) / 2
    f["h_hca_res"] = [hca.get((t, s), np.nan) for t, s in zip(f["home_team"], f["season"])]
    # B. luck: wins minus KenPom win chances, earlier games this season.
    g = L.groupby(["team", "season"])
    L["luck"] = g["won"].transform(lambda s: s.shift(1).expanding().sum()) - \
        g["wp"].transform(lambda s: s.shift(1).expanding().sum())
    L["n_prev"] = g.cumcount()
    L["luck_pg"] = L["luck"] / L["n_prev"].where(L["n_prev"] >= 5)
    # C. last game: upset win as a 7+ underdog, or 20+ loss as a favourite.
    L["prev_upset"] = g.apply(lambda x: ((x["line"] <= -7) & (x["margin"] > 0)).shift(1)
                              ).reset_index(level=[0, 1], drop=True).reindex(L.index)
    L["prev_blowout_loss"] = g.apply(lambda x: ((x["line"] > 0) & (x["margin"] <= -20)).shift(1)
                                     ).reset_index(level=[0, 1], drop=True).reindex(L.index)
    for side in ("home", "away"):
        x = L[L["side"] == side].set_index("game_id")
        for c in ("luck_pg", "prev_upset", "prev_blowout_loss"):
            f[f"{side[0]}_{c}"] = x[c].reindex(f["game_id"]).to_numpy()
    return f


def classifier(f: pd.DataFrame) -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    feats = ["kp_gap", "kp_tgap", "spread_line", "total_line", "kp_h_em", "kp_a_em",
             "kp_h_oe", "kp_a_oe", "kp_h_de", "kp_a_de", "tempo_avg", "tempo_diff",
             "pre_shift", "min_n", "same_conf", "neutral_f", "month",
             "h_luck_pg", "a_luck_pg", "h_hca_res", "h_vs_kp_5", "a_vs_kp_5"]
    feats = [c for c in feats if c in f]
    f = f.copy()
    for c in feats:
        f[c] = pd.to_numeric(f[c], errors="coerce").astype(float)
    print("\n== F. cover classifier (fitted on earlier seasons only) ==")
    for market, act, ln in (("spread", "actual_m", "spread_line"),
                            ("total", "actual_t", "total_line")):
        y = np.sign(f[act] - f[ln])
        p = pd.Series(np.nan, index=f.index)
        for s in sorted(f["season"].unique()):
            tr = (f["season"] < s) & (y != 0) & y.notna()
            te = f["season"] == s
            if s < 2022 or tr.sum() < 3000:
                continue
            m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.03,
                                               max_leaf_nodes=15, min_samples_leaf=300,
                                               l2_regularization=1.0, random_state=0)
            use = [c for c in feats if f.loc[tr, c].nunique() > 1]
            m.fit(f.loc[tr, use], (y[tr] > 0).astype(int))
            p[te] = m.predict_proba(f.loc[te, use])[:, 1]
        for th in (0.53, 0.55, 0.57, 0.60):
            bet = pd.Series(0.0, index=f.index)
            bet[p >= th] = 1
            bet[p <= 1 - th] = -1
            r = grade(f, bet, market)
            for lab, msk in (("2022-24", f["season"] <= 2024), ("2025-27", f["season"] >= 2025)):
                n, w, se = rec(r[msk])
                print(f"  {market} p>={th:.2f} {lab}: {w:6.1%} n={n:5d} (se {se:4.1%})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="kp_games.csv")
    ap.add_argument("--odds", default=".cache_cbb/odds")
    a = ap.parse_args()
    fr = pd.read_csv(a.games, low_memory=False)
    fr = fr[fr["completed"].astype(bool) & fr["spread_line"].notna() & fr["kp_margin"].notna()]
    fr = add_openers(fr, a.odds)
    f = add_features(build(fr))
    print(f"{len(f)} games. find = 2020-{SPLIT - 1}, check = {SPLIT}-26. Break-even {BE:.1%}.")
    hits: list = []
    z = pd.Series(0.0, index=f.index)

    print("\n== A. team home court vs KenPom's flat home edge (earlier seasons) ==")
    home = f["neutral_f"] == 0
    for t in (1.5, 2.5, 3.5):
        show(f"home team beats KenPom at home by {t}+ historically: home",
             f, z.mask(home & (f["h_hca_res"] >= t), 1), "spread", out=hits)
        show(f"home team misses KenPom at home by {t}+ historically: away",
             f, z.mask(home & (f["h_hca_res"] <= -t), -1), "spread", out=hits)

    print("\n== B. luck (wins minus KenPom win chances, per game, this season) ==")
    for t in (0.10, 0.15, 0.20):
        d = f["h_luck_pg"] - f["a_luck_pg"]
        show(f"fade the luckier team (gap {t:.2f}+ per game)", f,
             z.mask(d >= t, -1).mask(d <= -t, 1), "spread", out=hits)
        show(f"back the luckier team (gap {t:.2f}+ per game)", f,
             z.mask(d >= t, 1).mask(d <= -t, -1), "spread", out=hits)

    print("\n== C. spots ==")
    show("fade a team off an upset win as a 7+ dog", f,
         z.mask(f["h_prev_upset"] == True, -1).mask(f["a_prev_upset"] == True, 1),  # noqa: E712
         "spread", out=hits)
    show("back a team off a 20+ loss as a favourite", f,
         z.mask(f["h_prev_blowout_loss"] == True, 1).mask(f["a_prev_blowout_loss"] == True, -1),  # noqa: E712
         "spread", out=hits)

    print("\n== D. fade KenPom where it projects a close game ==")
    close = f["kp_margin"].abs() < 5
    for t in (2, 3, 4):
        show(f"projected within 5, KenPom {t}+ off the line: fade KenPom", f,
             z.mask(close & (f["kp_gap"].abs() >= t), -np.sign(f["kp_gap"])), "spread", out=hits)
        show(f"projected within 5, KenPom {t}+ off the total: fade KenPom", f,
             z.mask(close & (f["kp_tgap"].abs() >= t), -np.sign(f["kp_tgap"])), "total", out=hits)

    print("\n== E. line movement, opener to close (2023+ only, graded at the close) ==")
    mv = f["spread_line"] - f["open_spread"]           # + = moved toward home
    tmv = f["total_line"] - f["open_total"]
    for t in (1.5, 2.5, 3.5):
        show(f"spread moved {t}+: bet WITH the move", f, z.mask(mv.abs() >= t, np.sign(mv)), "spread")
        show(f"spread moved {t}+: bet AGAINST the move", f, z.mask(mv.abs() >= t, -np.sign(mv)), "spread")
        show(f"total moved {t}+: bet WITH the move", f, z.mask(tmv.abs() >= t, np.sign(tmv)), "total")
        show(f"total moved {t}+: bet AGAINST the move", f, z.mask(tmv.abs() >= t, -np.sign(tmv)), "total")
    agree = np.sign(mv) == np.sign(f["kp_margin"] - f["open_spread"])
    show("moved 1.5+ toward KenPom's side, KenPom still 2+ off: KenPom", f,
         z.mask((mv.abs() >= 1.5) & agree & (f["kp_gap"].abs() >= 2), np.sign(f["kp_gap"])), "spread")
    show("moved 1.5+ away from KenPom, KenPom 2+ off: KenPom", f,
         z.mask((mv.abs() >= 1.5) & ~agree & (f["kp_gap"].abs() >= 2), np.sign(f["kp_gap"])), "spread")

    classifier(f)
    print("\n== held in both periods ==")
    for h in hits:
        print(h)
    if not hits:
        print("  none")


if __name__ == "__main__":
    main()

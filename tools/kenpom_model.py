"""
A college basketball model built only on KenPom data, tested against the
closing line on seasons it never saw.

Inputs per game (home minus away where it makes sense), all known on the
morning of the game:

  dated     FanMatch prediction (margin, total, win prob, tempo), the
            archive ratings on that date (AdjEM, AdjOE, AdjDE, tempo, rank)
            and the preseason ratings
  style     last season's final four factors, shot mix (pointdist),
            3-point / FT / block / steal rates (misc-stats), height,
            experience, bench and continuity. Only LAST season's tables,
            because this season's are end-of-season numbers (hindsight).
  roster    this season's height / experience / continuity (variant
            "+roster"). Roster-based, so mostly known in November, but
            KenPom weights them by minutes played: a little hindsight.
  context   neutral site, rest days, month, games played

Models, each refitted per season on earlier seasons only:

  outcome   boosted trees predicting the final margin / total from the
            KenPom inputs alone (no market input), then bet vs the close
  vs_line   boosted trees predicting result minus the closing line, with
            the line as an input (where does the market misprice KenPom?)

Report: MAE vs the close, record by gap, top gaps a night and by season.
Break-even at -110 is 52.4%; pushes are left out.

    python tools/kenpom_model.py --games kp_games.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from kenpom import SEASON_TABLES, season_table  # noqa: E402

BE = 110 / 210

STYLE = {
    "four-factors": ["eFG_Pct", "TO_Pct", "OR_Pct", "FT_Rate", "DeFG_Pct",
                     "DTO_Pct", "DOR_Pct", "DFT_Rate", "AdjOE", "AdjDE",
                     "AdjTempo"],
    "pointdist": ["OffFg3", "OffFt", "DefFg3", "DefFt"],
    "misc-stats": ["FG3Pct", "FTPct", "F3GRate", "OppFG3Pct", "OppF3GRate",
                   "BlockPct", "StlRate", "ARate"],
    "height": ["AvgHgt", "HgtEff", "Exp", "Bench", "Continuity"],
}
ROSTER = ["AvgHgt", "HgtEff", "Exp", "Bench", "Continuity"]


def profiles(seasons) -> dict:
    """{(season, KenPom name): {col: value}} for each table, plus
    roster columns from the same season's height table."""
    prof, roster = {}, {}
    for s in seasons:
        for ep in SEASON_TABLES:
            try:
                t = season_table(ep, s)
            except Exception as e:                 # noqa: BLE001
                print(f"  {ep} {s}: {str(e)[:100]}")
                continue
            if t.empty or "TeamName" not in t:
                print(f"  {ep} {s}: empty")
                continue
            if "ConfOnly" in t:
                t = t[t["ConfOnly"].astype(str).str.lower() != "true"]
            cols = [c for c in STYLE[ep] if c in t]
            for r in t[["TeamName"] + cols].itertuples(index=False):
                d = prof.setdefault((s, r[0]), {})
                d.update({f"{ep[:2]}_{c}": v for c, v in zip(cols, r[1:])})
                if ep == "height":
                    roster[(s, r[0])] = {c: v for c, v in zip(cols, r[1:])
                                         if c in ROSTER}
    return prof, roster


def build(fr: pd.DataFrame, prof: dict, roster: dict) -> tuple[pd.DataFrame, list, list]:
    f = fr.copy()
    f["game_date"] = pd.to_datetime(f["game_date"])
    f["actual_m"] = f["home_score"] - f["away_score"]
    f["actual_t"] = f["home_score"] + f["away_score"]
    f["neutral_f"] = f["neutral"].astype(bool).astype(float)
    f["month"] = f["game_date"].dt.month
    base = ["kp_margin", "kp_total", "kp_wp", "kp_tempo", "kp_home_rank",
            "kp_away_rank", "neutral_f", "month"]
    for side in ("h", "a"):
        f[f"kp_{side}_shift"] = f[f"kp_{side}_em"] - f[f"kp_{side}_pre_em"]
        base += [f"kp_{side}_{c}" for c in ("em", "oe", "de", "t", "pre_em", "shift")]
    for c in ("home_rest", "away_rest", "min_n"):
        if c in f:
            base.append(c)
    style_cols = sorted({k for d in prof.values() for k in d})
    style = []
    for side, name in (("h", "kp_home_name"), ("a", "kp_away_name")):
        keys = list(zip(f["season"].astype(int) - 1, f[name]))
        rows = [prof.get(k, {}) for k in keys]
        for c in style_cols:
            f[f"{side}_prev_{c}"] = [r.get(c, np.nan) for r in rows]
            style.append(f"{side}_prev_{c}")
        keys = list(zip(f["season"].astype(int), f[name]))
        rows = [roster.get(k, {}) for k in keys]
        for c in ROSTER:
            f[f"{side}_now_{c}"] = [r.get(c, np.nan) for r in rows]
    # Matchup terms from last season's styles.
    def pair(a, b, name):
        if a in f and b in f:
            f[name] = f[a] + f[b]
            style.append(name)
    for o, d in (("h", "a"), ("a", "h")):
        pair(f"{o}_prev_fo_OR_Pct", f"{d}_prev_fo_DOR_Pct", f"{o}_oreb_matchup")
        pair(f"{o}_prev_fo_TO_Pct", f"{d}_prev_fo_DTO_Pct", f"{o}_tov_matchup")
        pair(f"{o}_prev_fo_FT_Rate", f"{d}_prev_fo_DFT_Rate", f"{o}_ftr_matchup")
        pair(f"{o}_prev_mi_F3GRate", f"{d}_prev_mi_OppF3GRate", f"{o}_3pa_matchup")
    roster_cols = [f"{s}_now_{c}" for s in ("h", "a") for c in ROSTER]
    cover = f[[c for c in style if "prev_fo" in c]].notna().any(axis=1).mean()
    rc = f[roster_cols].notna().any(axis=1).mean()
    print(f"  last-season style found for {cover:.0%} of games, this-season "
          f"roster for {rc:.0%}")
    return f, base + style, roster_cols


def walk(f: pd.DataFrame, feats: list, target: str, first_test: int):
    from sklearn.ensemble import HistGradientBoostingRegressor
    pred = pd.Series(np.nan, index=f.index)
    y = f[target]
    for s in sorted(f["season"].unique()):
        if s < first_test:
            continue
        tr = (f["season"] < s) & y.notna()
        te = f["season"] == s
        if tr.sum() < 3000 or not te.any():
            continue
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04,
                                          max_leaf_nodes=15, min_samples_leaf=200,
                                          l2_regularization=1.0, random_state=0)
        m.fit(f.loc[tr, feats], y[tr])
        pred[te] = m.predict(f.loc[te, feats])
    return pred


def record(won: pd.Series) -> str:
    won = won.dropna()
    n = len(won)
    if not n:
        return "n=0"
    p = won.mean()
    se = np.sqrt(BE * (1 - BE) / n)
    flag = "  <- beats break-even by 2 se" if p - BE > 2 * se else ""
    return f"{int(won.sum())}-{n - int(won.sum())} {p:.1%} (n={n}){flag}"


def grade(f, side, market):
    act, ln = ("actual_m", "spread_line") if market == "spread" else ("actual_t", "total_line")
    res = np.sign(f[act] - f[ln])
    won = (res == side).astype(float)
    return won.where((side != 0) & (res != 0) & f[ln].notna() & f[act].notna())


def evaluate(f: pd.DataFrame, label: str, pm: pd.Series, pt: pd.Series,
             as_edge: bool) -> None:
    """pm/pt: predicted margin and total (or, when as_edge, predicted
    result minus the line)."""
    print(f"\n== {label} ==")
    for market, p, ln, act, gaps in (
            ("spread", pm, "spread_line", "actual_m", (1, 2, 3, 4, 6)),
            ("total", pt, "total_line", "actual_t", (2, 3, 4, 6, 8))):
        ok = p.notna() & f[ln].notna()
        if not ok.any():
            print(f"  {market}: no predictions")
            continue
        gap = p if as_edge else p - f[ln]
        if not as_edge:
            mae = (p - f[act]).abs()[ok].mean()
            mae_c = (f[ln] - f[act]).abs()[ok].mean()
            print(f"  {market} MAE {mae:.2f} vs close {mae_c:.2f} (n={int(ok.sum())})")
        side = np.sign(gap).where(ok, 0)
        won = grade(f, side, market)
        for g in gaps:
            print(f"  {market} gap {g}+   {record(won[gap.abs() >= g])}")
        rank = gap.abs().where(ok).groupby(f["game_date"]).rank(ascending=False)
        print(f"  {market} top 3 a night  {record(won[rank <= 3])}")
        g0 = gaps[2]
        per = "  ".join(f"{s}: {won[(f['season'] == s) & (gap.abs() >= g0)].mean():.0%}"
                        f"/{int(won[(f['season'] == s) & (gap.abs() >= g0)].notna().sum())}"
                        for s in sorted(f.loc[ok, "season"].unique()))
        print(f"  {market} gap {g0}+ by season: {per}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="kp_games.csv")
    ap.add_argument("--first-test", type=int, default=2022)
    a = ap.parse_args()
    fr = pd.read_csv(a.games, low_memory=False)
    fr = fr[fr["kp_margin"].notna() & fr["kp_home_name"].notna()]
    seasons = range(int(fr["season"].min()) - 1, int(fr["season"].max()) + 1)
    prof, roster = profiles(seasons)
    f, feats, rcols = build(fr, prof, roster)
    print(f"{len(f)} games, {len(feats)} KenPom inputs; tested on "
          f"{a.first_test}+ only, each season fitted on earlier ones. "
          f"Break-even {BE:.1%}.")

    test = f["season"] >= a.first_test
    evaluate(f[test], "KenPom FanMatch as published (for comparison)",
             f.loc[test, "kp_margin"], f.loc[test, "kp_total"], as_edge=False)
    for tag, cols in (("", feats), (" +roster", feats + rcols)):
        pm = walk(f, cols, "actual_m", a.first_test)
        pt = walk(f, cols, "actual_t", a.first_test)
        evaluate(f[test], f"outcome model{tag}: KenPom inputs only, no market",
                 pm[test], pt[test], as_edge=False)
        g = f.assign(em=f["actual_m"] - f["spread_line"],
                     et=f["actual_t"] - f["total_line"])
        ctx = cols + ["spread_line", "total_line"]
        g["sl_gap"] = g["kp_margin"] - g["spread_line"]
        g["tl_gap"] = g["kp_total"] - g["total_line"]
        ctx += ["sl_gap", "tl_gap"]
        em = walk(g, ctx, "em", a.first_test)
        et = walk(g, ctx, "et", a.first_test)
        evaluate(f[test], f"vs_line model{tag}: KenPom inputs + the line",
                 em[test], et[test], as_edge=True)


if __name__ == "__main__":
    main()

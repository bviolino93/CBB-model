"""
Is there anything in the KenPom data (and ours) that beats the line?

Reads the game frame tools/kenpom_backtest.py writes (kp_games.csv: ESPN
games with KenPom's dated prediction and ratings, our score ratings and
the pregame line), adds the opening line from ESPN's cached odds, and
runs four things:

  1. audit: hindsight check on the archived ratings, how often the line
     moves, pushes, and the break-even
  2. rule screen: each idea a bettor might try, scored separately on
     2020-22 (find) and 2023-26 (check). Only an idea that wins in BOTH
     periods counts; with ~40 ideas, one or two will look good by chance
  3. KenPom vs the OPENING line: the market is softest at the open
  4. boosted trees on everything (no market input except the line level),
     predicting result minus line, fitted on earlier seasons only

    python tools/kenpom_research.py --games kp_games.csv --odds .cache_cbb/odds
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

BE = 110 / 210
SPLIT = 2023


# ----------------------------------------------------------------------
# Opening lines from ESPN's core odds (cached per game by cbb_data)
# ----------------------------------------------------------------------
_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _num(x):
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, dict):
        for k in ("american", "alternateDisplayValue", "displayValue", "value"):
            if k in x:
                v = _num(x[k])
                if v is not None:
                    return v
        return None
    if isinstance(x, str):
        s = x.strip().lower().replace("pk", "0").replace("even", "0")
        m = _NUM.search(s.lstrip("ou"))
        return float(m.group()) if m else None
    return None


def open_lines(js: dict) -> tuple[float, float]:
    """(open home margin, open total): medians across books, from each
    item's homeTeamOdds.open.pointSpread and open.total."""
    sp, tt = [], []
    for it in js.get("items", []) if isinstance(js, dict) else []:
        name = str((it.get("provider") or {}).get("name", "")).lower()
        if re.search(r"live|numberfire|teamrankings|projection|accuscore|predict", name):
            continue
        h = ((it.get("homeTeamOdds") or {}).get("open") or {}).get("pointSpread")
        v = _num(h)
        if v is not None and abs(v) <= 50:
            sp.append(-v)                       # book spread -> home margin
        t = ((it.get("open") or {}).get("total"))
        v = _num(t)
        if v is not None and 90 < v < 220:
            tt.append(v)
    med = lambda v: float(np.median(v)) if v else np.nan
    return med(sp), med(tt)


def add_openers(fr: pd.DataFrame, odds_dir: str) -> pd.DataFrame:
    fr = fr.copy()
    o_sp, o_tt, shown = [], [], False
    for gid in fr["game_id"].astype(str):
        p = os.path.join(odds_dir, f"{gid}.json")
        js = {}
        if os.path.exists(p):
            try:
                with open(p) as fh:
                    js = json.load(fh)
            except ValueError:
                js = {}
        if not shown and js.get("items"):
            it = js["items"][0]
            print("  sample odds item keys:", sorted(it)[:25])
            print("  homeTeamOdds keys:", sorted((it.get("homeTeamOdds") or {}))[:20])
            print("  open:", json.dumps(it.get("open"))[:300])
            print("  homeTeamOdds.open:", json.dumps((it.get("homeTeamOdds") or {}).get("open"))[:300])
            shown = True
        a, b = open_lines(js)
        o_sp.append(a)
        o_tt.append(b)
    fr["open_spread"], fr["open_total"] = o_sp, o_tt
    # Junk guard: an opener 12+ pts from the close is a data error.
    fr.loc[(fr["open_spread"] - fr["spread_line"]).abs() > 12, "open_spread"] = np.nan
    fr.loc[(fr["open_total"] - fr["total_line"]).abs() > 20, "open_total"] = np.nan
    return fr


# ----------------------------------------------------------------------
# Features a bettor would look at (all known before tip)
# ----------------------------------------------------------------------
def team_rolling(fr: pd.DataFrame) -> pd.DataFrame:
    """Each team's last-3 and last-5 margin against KenPom's expectation
    and against the line, from games BEFORE this one (same season)."""
    rows = []
    for side, other, sgn in (("home", "away", 1), ("away", "home", -1)):
        d = pd.DataFrame({
            "game_id": fr["game_id"], "season": fr["season"],
            "commence": pd.to_datetime(fr["commence"], utc=True),
            "team": fr[f"{side}_team"],
            "vs_kp": sgn * ((fr["home_score"] - fr["away_score"]) - fr["kp_margin"]),
            "vs_line": sgn * ((fr["home_score"] - fr["away_score"]) - fr["spread_line"]),
            "tot_vs_kp": (fr["home_score"] + fr["away_score"]) - fr["kp_total"],
            "tot_vs_line": (fr["home_score"] + fr["away_score"]) - fr["total_line"],
            "side": side})
        rows.append(d)
    t = pd.concat(rows).sort_values(["team", "season", "commence"])
    g = t.groupby(["team", "season"])
    for c in ("vs_kp", "vs_line", "tot_vs_kp", "tot_vs_line"):
        for n in (3, 5):
            t[f"{c}_{n}"] = g[c].transform(lambda s: s.shift(1).rolling(n, min_periods=n).mean())
    t["streak_over"] = g["tot_vs_line"].transform(
        lambda s: (s.shift(1) > 0).astype(float).rolling(3, min_periods=3).sum())
    t["streak_cover"] = g["vs_line"].transform(
        lambda s: (s.shift(1) > 0).astype(float).rolling(3, min_periods=3).sum())
    out = fr.copy()
    for side in ("home", "away"):
        x = t[t["side"] == side].set_index("game_id")
        for c in [c for c in t.columns if c.endswith(("_3", "_5")) or c.startswith("streak")]:
            out[f"{side[0]}_{c}"] = x[c].reindex(out["game_id"]).to_numpy()
    return out


def build(fr: pd.DataFrame) -> pd.DataFrame:
    f = team_rolling(fr)
    f["actual_m"] = f["home_score"] - f["away_score"]
    f["actual_t"] = f["home_score"] + f["away_score"]
    f["kp_gap"] = f["kp_margin"] - f["spread_line"]
    f["kp_tgap"] = f["kp_total"] - f["total_line"]
    f["own_gap"] = f["model_margin"] - f["spread_line"]
    f["own_tgap"] = f["model_total"] - f["total_line"]
    f["tempo_avg"] = (f["kp_h_t"] + f["kp_a_t"]) / 2
    f["tempo_diff"] = (f["kp_h_t"] - f["kp_a_t"]).abs()
    # Points per 100 each side should score vs this defence, league ~ 105.
    f["h_exp_eff"] = f["kp_h_oe"] + f["kp_a_de"]
    f["a_exp_eff"] = f["kp_a_oe"] + f["kp_h_de"]
    f["eff_sum"] = f["h_exp_eff"] + f["a_exp_eff"]
    f["pre_shift"] = ((f["kp_h_em"] - f["kp_h_pre_em"]) - (f["kp_a_em"] - f["kp_a_pre_em"]))
    f["min_n"] = f["min_n"].fillna(0)
    f["same_conf"] = (f["home_conf"].astype(str) == f["away_conf"].astype(str)).astype(float)
    f["neutral_f"] = f["neutral"].astype(float)
    f["home_fav"] = (f["spread_line"] > 0).astype(float)
    f["abs_line"] = f["spread_line"].abs()
    f["month"] = pd.to_datetime(f["game_date"]).dt.month
    return f


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------
def grade(f: pd.DataFrame, bet: pd.Series, market: str, line: str | None = None) -> pd.Series:
    """+1 / -1 / 0 for betting `bet` (+1 home or over, -1 away or under)."""
    if market == "spread":
        ln = f[line or "spread_line"]
        r = np.sign(f["actual_m"] - ln) * bet
    else:
        ln = f[line or "total_line"]
        r = np.sign(f["actual_t"] - ln) * bet
    return r.where((bet != 0) & ln.notna(), np.nan)


def rec(r: pd.Series) -> tuple[int, float, float]:
    r = r.dropna()
    r = r[r != 0]
    n = len(r)
    if n == 0:
        return 0, np.nan, np.nan
    w = float((r > 0).mean())
    return n, w, float(np.sqrt(w * (1 - w) / n))


def show(label: str, f: pd.DataFrame, bet: pd.Series, market: str,
         line: str | None = None, out: list | None = None) -> None:
    r = grade(f, bet, market, line)
    a, b = r[f["season"] < SPLIT], r[f["season"] >= SPLIT]
    na, wa, _ = rec(a)
    nb, wb, sb = rec(b)
    ok = (na >= 80 and nb >= 80 and wa > BE and wb > BE)
    strong = ok and (wb - 2 * sb > BE)
    tag = "HOLDS + beats BE by 2se" if strong else ("holds both periods" if ok else "")
    line_ = (f"  {label:58s} find {wa:6.1%} n={na:5d} | check {wb:6.1%} n={nb:5d}"
             f" (se {sb:4.1%})  {tag}")
    print(line_)
    if out is not None and ok:
        out.append(line_)


def audit(f: pd.DataFrame) -> None:
    print("\n== 1. audit ==")
    # Hindsight: archive ratings on a game's date must not already include
    # that game. If they did, a team's rating change INTO a game would track
    # that game's surprise as strongly as the change OUT of it.
    rows = []
    for side, sgn in (("home", 1), ("away", -1)):
        rows.append(pd.DataFrame({
            "team": f[f"{side}_team"], "season": f["season"],
            "commence": pd.to_datetime(f["commence"], utc=True),
            "em": f[f"kp_{side[0]}_em"],
            "surprise": sgn * (f["actual_m"] - f["kp_margin"])}))
    t = pd.concat(rows).dropna().sort_values(["team", "season", "commence"])
    g = t.groupby(["team", "season"])
    t["into"] = t["em"] - g["em"].shift(1)
    t["out"] = g["em"].shift(-1) - t["em"]
    ok = t[["into", "out", "surprise"]].dropna()
    print(f"  corr(this game's surprise, rating change INTO the game) "
          f"{ok['surprise'].corr(ok['into']):+.3f}  <- ~0 means no hindsight")
    print(f"  corr(this game's surprise, rating change AFTER the game) "
          f"{ok['surprise'].corr(ok['out']):+.3f}  <- should be clearly positive")
    act = f["actual_m"]
    for c, lab in (("kp_margin", "KenPom"), ("spread_line", "ESPN pregame line"),
                   ("open_spread", "ESPN opening line")):
        m = f[c].notna()
        print(f"  margin MAE {lab:18s} {np.abs(f.loc[m, c] - act[m]).mean():.2f}  (n={int(m.sum())})")
    m = f["open_spread"].notna()
    mv = (f.loc[m, "spread_line"] - f.loc[m, "open_spread"])
    print(f"  opening line available on {m.mean():.0%} of games; line moved 1+ pt on "
          f"{(mv.abs() >= 1).mean():.0%}")
    toward = np.sign(mv) == np.sign(f.loc[m, "kp_margin"] - f.loc[m, "open_spread"])
    print(f"  when KenPom disagrees with the opener, the line moves toward KenPom "
          f"{toward[(f.loc[m, 'kp_margin'] - f.loc[m, 'open_spread']).abs() >= 2].mean():.0%}"
          " of the time (2+ pt disagreements)")
    pushes = (f["actual_m"] == f["spread_line"]).mean()
    print(f"  spread pushes {pushes:.1%} (left out of win %); break-even at -110 = {BE:.2%}")


def screen(f: pd.DataFrame) -> list:
    print(f"\n== 2. rule screen: find = 2020-{SPLIT - 1}, check = {SPLIT}-26 ==")
    hits: list = []
    s = lambda x: np.sign(x).fillna(0)
    gap, tg = f["kp_gap"], f["kp_tgap"]
    early = f["min_n"] < 5
    # spreads
    for g in (3, 5):
        show(f"KenPom spread gap {g}+, first 4 games of the season", f[early],
             s(gap[early]).where(gap[early].abs() >= g, 0), "spread", out=hits)
    show("KenPom gap 3+ in conference games", f[f["same_conf"] == 1],
         s(gap).where((gap.abs() >= 3) & (f["same_conf"] == 1), 0)[f["same_conf"] == 1],
         "spread", out=hits)
    show("KenPom gap 3+, non-conference", f[f["same_conf"] == 0],
         s(gap).where((gap.abs() >= 3) & (f["same_conf"] == 0), 0)[f["same_conf"] == 0],
         "spread", out=hits)
    big = f["pre_shift"].abs() >= 6
    show("team far from its preseason rating: back the preseason view",
         f, s(-f["pre_shift"]).where(big & (f["min_n"] < 12), 0), "spread", out=hits)
    show("team far from its preseason rating: back the current view",
         f, s(f["pre_shift"]).where(big & (f["min_n"] < 12), 0), "spread", out=hits)
    for n in (3, 5):
        hot = f[f"h_vs_kp_{n}"] - f[f"a_vs_kp_{n}"]
        show(f"fade the team that beat KenPom by more over last {n} (8+ pts)",
             f, s(-hot).where(hot.abs() >= 8, 0), "spread", out=hits)
        show(f"back the team that beat KenPom by more over last {n} (8+ pts)",
             f, s(hot).where(hot.abs() >= 8, 0), "spread", out=hits)
        cov = f[f"h_vs_line_{n}"] - f[f"a_vs_line_{n}"]
        show(f"fade the team covering by more over last {n} (8+ pts)",
             f, s(-cov).where(cov.abs() >= 8, 0), "spread", out=hits)
    show("home underdog of 1-7 (home side)", f,
         pd.Series(np.where((f["spread_line"] < 0) & (f["spread_line"] > -7.5)
                            & (f["neutral_f"] == 0), 1, 0), index=f.index), "spread", out=hits)
    show("big favourite 20+ (fav side)", f,
         pd.Series(np.where(f["abs_line"] >= 20, np.sign(f["spread_line"]), 0), index=f.index),
         "spread", out=hits)
    show("big favourite 20+ (dog side)", f,
         pd.Series(np.where(f["abs_line"] >= 20, -np.sign(f["spread_line"]), 0), index=f.index),
         "spread", out=hits)
    b2b = (f["home_rest"] <= 1) & (f["away_rest"] >= 2)
    show("home team on a back-to-back vs rested team (away side)", f,
         pd.Series(np.where(b2b, -1, 0), index=f.index), "spread", out=hits)
    b2b = (f["away_rest"] <= 1) & (f["home_rest"] >= 2)
    show("away team on a back-to-back vs rested team (home side)", f,
         pd.Series(np.where(b2b, 1, 0), index=f.index), "spread", out=hits)
    agree = (s(gap) == s(f["own_gap"])) & (gap.abs() >= 2) & (f["own_gap"].abs() >= 4)
    show("KenPom 2+ and our ratings 4+ on the same side", f, s(gap).where(agree, 0), "spread", out=hits)
    # totals
    for g in (4, 6):
        show(f"KenPom total gap {g}+, first 4 games", f[early],
             s(tg[early]).where(tg[early].abs() >= g, 0), "total", out=hits)
    fast, slow = f["tempo_avg"] >= 71, f["tempo_avg"] <= 64.5
    show("both teams fast (avg tempo 71+): over", f, pd.Series(np.where(fast, 1, 0), index=f.index), "total", out=hits)
    show("both teams fast (avg tempo 71+): under", f, pd.Series(np.where(fast, -1, 0), index=f.index), "total", out=hits)
    show("both teams slow (avg tempo <=64.5): under", f, pd.Series(np.where(slow, -1, 0), index=f.index), "total", out=hits)
    show("both teams slow (avg tempo <=64.5): over", f, pd.Series(np.where(slow, 1, 0), index=f.index), "total", out=hits)
    mism = f["tempo_diff"] >= 6
    show("big tempo mismatch (6+): under", f, pd.Series(np.where(mism, -1, 0), index=f.index), "total", out=hits)
    show("big tempo mismatch (6+): over", f, pd.Series(np.where(mism, 1, 0), index=f.index), "total", out=hits)
    exp_t = f["eff_sum"] / 2 * f["tempo_avg"] / 100 * 2 / 2.1   # rough points from O vs D
    eg = exp_t - f["total_line"]
    show("offence-vs-defence points 6+ off the total", f, s(eg).where(eg.abs() >= 6, 0), "total", out=hits)
    for n in (3, 5):
        ov = f[f"h_tot_vs_line_{n}"] + f[f"a_tot_vs_line_{n}"]
        show(f"both teams' games over the line by 10+ avg last {n}: under", f,
             pd.Series(np.where(ov >= 20, -1, 0), index=f.index), "total", out=hits)
        show(f"both teams' games under the line by 10+ avg last {n}: over", f,
             pd.Series(np.where(ov <= -20, 1, 0), index=f.index), "total", out=hits)
    st = (f["h_streak_over"] >= 3) | (f["a_streak_over"] >= 3)
    show("either team over 3 straight: under (the football rule)", f,
         pd.Series(np.where(st, -1, 0), index=f.index), "total", out=hits)
    st = (f["h_streak_over"] == 0) | (f["a_streak_over"] == 0)
    show("either team under 3 straight: over", f,
         pd.Series(np.where(st, 1, 0), index=f.index), "total", out=hits)
    for lo, hi, lab in ((0, 130, "total under 130"), (160, 999, "total 160+")):
        m = (f["total_line"] >= lo) & (f["total_line"] < hi)
        show(f"{lab}: over", f, pd.Series(np.where(m, 1, 0), index=f.index), "total", out=hits)
        show(f"{lab}: under", f, pd.Series(np.where(m, -1, 0), index=f.index), "total", out=hits)
    show("conference games: KenPom total gap 4+", f[f["same_conf"] == 1],
         s(tg).where((tg.abs() >= 4) & (f["same_conf"] == 1), 0)[f["same_conf"] == 1], "total", out=hits)
    show("March (tournaments): KenPom total gap 4+", f[f["month"] == 3],
         s(tg).where((tg.abs() >= 4) & (f["month"] == 3), 0)[f["month"] == 3], "total", out=hits)
    return hits


def vs_open(f: pd.DataFrame, hits: list) -> None:
    print("\n== 3. KenPom vs the OPENING line (graded against the opener and the close) ==")
    m = f["open_spread"].notna()
    og = f["kp_margin"] - f["open_spread"]
    for g in (2, 3, 4, 6):
        bet = np.sign(og).where(m & (og.abs() >= g), 0).fillna(0)
        show(f"spread: KenPom {g}+ off the opener, graded vs opener", f, bet, "spread", "open_spread", hits)
        show(f"spread: KenPom {g}+ off the opener, graded vs close", f, bet, "spread", None, hits)
    m = f["open_total"].notna()
    ot = f["kp_total"] - f["open_total"]
    for g in (3, 4, 6):
        bet = np.sign(ot).where(m & (ot.abs() >= g), 0).fillna(0)
        show(f"total: KenPom {g}+ off the opener, graded vs opener", f, bet, "total", "open_total", hits)
        show(f"total: KenPom {g}+ off the opener, graded vs close", f, bet, "total", None, hits)


FEATS = ["kp_gap", "kp_tgap", "own_gap", "own_tgap", "spread_line", "total_line",
         "kp_h_em", "kp_a_em", "kp_h_oe", "kp_a_oe", "kp_h_de", "kp_a_de",
         "kp_h_t", "kp_a_t", "tempo_avg", "tempo_diff", "pre_shift", "min_n",
         "same_conf", "neutral_f", "home_rest", "away_rest", "month",
         "h_vs_kp_5", "a_vs_kp_5", "h_vs_line_5", "a_vs_line_5",
         "h_tot_vs_line_5", "a_tot_vs_line_5", "h_streak_over", "a_streak_over"]


def trees(f: pd.DataFrame, hits: list) -> None:
    from sklearn.ensemble import HistGradientBoostingRegressor
    print("\n== 4. boosted trees on everything: predict result minus line, "
          "fitted on earlier seasons only ==")
    for market, act, ln in (("spread", "actual_m", "spread_line"),
                            ("total", "actual_t", "total_line")):
        f = f.copy()
        f["pred"] = np.nan
        y = f[act] - f[ln]
        for s in sorted(f["season"].unique()):
            if s < 2022:
                continue
            tr, te = f["season"] < s, f["season"] == s
            ok = tr & y.notna() & f[ln].notna()
            if ok.sum() < 3000:
                continue
            m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.04,
                                              max_leaf_nodes=15, min_samples_leaf=200,
                                              l2_regularization=1.0, random_state=0)
            m.fit(f.loc[ok, FEATS], y[ok])
            f.loc[te, "pred"] = m.predict(f.loc[te, FEATS])
        p = f["pred"]
        for g in ((0.75, 1.0, 1.5, 2.0) if market == "spread" else (1.0, 1.5, 2.0, 3.0)):
            bet = np.sign(p).where(p.abs() >= g, 0).fillna(0)
            r = grade(f, bet, market)
            for lab, part in (("2022-24", r[(f["season"] >= 2022) & (f["season"] < 2025)]),
                              ("2025-26", r[f["season"] >= 2025])):
                n, w, se = rec(part)
                print(f"  {market}: predicted edge {g}+ pts, {lab}: {w:6.1%} n={n:5d} (se {se:4.1%})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="kp_games.csv")
    ap.add_argument("--odds", default=".cache_cbb/odds")
    a = ap.parse_args()
    fr = pd.read_csv(a.games, low_memory=False)
    fr = fr[fr["kp_margin"].notna() & fr["spread_line"].notna()]
    fr = add_openers(fr, a.odds)
    f = build(fr)
    print(f"{len(f)} games 2020-26 with KenPom and a line; opener on "
          f"{f['open_spread'].notna().mean():.0%}")
    audit(f)
    hits = screen(f)
    vs_open(f, hits)
    trees(f, hits)
    print("\n== ideas that won in BOTH periods ==")
    print("\n".join(hits) if hits else "  none")


if __name__ == "__main__":
    main()

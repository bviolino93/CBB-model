"""
How accurately do we predict the margin and the total, and which extra
inputs make KenPom's prediction more accurate?

Start from KenPom's FanMatch margin and total (known the morning of the
game). Then, one group of inputs at a time, boosted trees learn the part
of the result KenPom misses (result minus KenPom), refitted each season
on earlier seasons only, and are scored on seasons they never saw.
Every input is known before tip.

  missing    regulars who missed the team's previous game (box scores)
  form       last 3 / 5 games vs KenPom and vs the line, cover streaks
  rest       days of rest, games played, neutral site, month
  matchup    season-to-date four factors from box scores, offence vs the
             other side's defence: turnovers vs forced turnovers,
             offensive boards vs defensive boards, FT rate, 3-point
             share; plus last season's KenPom height and experience
  pace       both tempos, the gap between them
  shooting   season-to-date 3P% and opponents' 3P% vs league average,
             with attempts (hot or cold shooting tends to regress)
  all        every group together

Scored per group: mean absolute error (MAE) of the margin and total,
straight-up winner %, and side % against the closing line (pick the
side the prediction is on; break-even 52.4%), with the closing line's
own MAE for reference.

    python tools/kenpom_accuracy.py --games kp_games.csv --box-cache .cache_box
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
from cbb_players import load_box  # noqa: E402
from kenpom_research import build as kp_build  # noqa: E402
from missing_players import team_absences  # noqa: E402

BE = 110 / 210
FIRST_TEST = 2023


# ----------------------------------------------------------------------
# Season-to-date team stats from box scores (games BEFORE this one)
# ----------------------------------------------------------------------
def team_box(p: pd.DataFrame) -> pd.DataFrame:
    s = ["fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "tov", "pts"]
    t = p.groupby(["game_id", "team"], as_index=False)[s].sum()
    o = t.rename(columns={c: f"o_{c}" for c in s}).rename(columns={"team": "opp"})
    t = t.merge(o, on="game_id")
    return t[t["team"] != t["opp"]]


def to_date(g: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    tb = team_box(p)
    meta = pd.concat([g[["game_id", "season", "commence"]].assign(team=g["home_team"]),
                      g[["game_id", "season", "commence"]].assign(team=g["away_team"])])
    meta["game_id"] = meta["game_id"].astype(str)
    tb["game_id"] = tb["game_id"].astype(str)
    t = meta.merge(tb, on=["game_id", "team"], how="inner")
    t = t.sort_values(["team", "season", "commence"])
    num = [c for c in tb.columns if c not in ("game_id", "team", "opp")]
    c = t.groupby(["team", "season"])[num].cumsum() - t[num]
    n = t.groupby(["team", "season"]).cumcount()
    poss = c["fga"] - c["oreb"] + c["tov"] + 0.475 * c["fta"]
    oposs = c["o_fga"] - c["o_oreb"] + c["o_tov"] + 0.475 * c["o_fta"]
    out = pd.DataFrame({"game_id": t["game_id"], "team": t["team"], "n_box": n})
    out["efg"] = (c["fgm"] + 0.5 * c["fg3m"]) / c["fga"]
    out["to"] = c["tov"] / poss
    out["orb"] = c["oreb"] / (c["oreb"] + c["o_dreb"])
    out["ftr"] = c["fta"] / c["fga"]
    out["tpa"] = c["fg3a"] / c["fga"]
    out["tp"] = c["fg3m"] / c["fg3a"]
    out["tp_att"] = c["fg3a"]
    out["d_efg"] = (c["o_fgm"] + 0.5 * c["o_fg3m"]) / c["o_fga"]
    out["d_to"] = c["o_tov"] / oposs
    out["d_orb"] = c["o_oreb"] / (c["o_oreb"] + c["dreb"])
    out["d_ftr"] = c["o_fta"] / c["o_fga"]
    out["d_tpa"] = c["o_fg3a"] / c["o_fga"]
    out["d_tp"] = c["o_fg3m"] / c["o_fg3a"]
    out["d_tp_att"] = c["o_fg3a"]
    out = out.replace([np.inf, -np.inf], np.nan)
    out.loc[out["n_box"] < 3, [x for x in out.columns if x not in ("game_id", "team", "n_box")]] = np.nan
    return out


def add_inputs(f: pd.DataFrame, g: pd.DataFrame, p: pd.DataFrame, groups: dict) -> pd.DataFrame:
    f = f.copy()
    f["game_id"] = f["game_id"].astype(str)
    # missing
    ab = team_absences(g, p).set_index(["game_id", "team"])
    miss = []
    for side in ("home", "away"):
        idx = list(zip(f["game_id"], f[f"{side}_team"]))
        for c in ("miss", "new", "miss_min", "miss_pts"):
            f[f"{side[0]}_{c}"] = ab[c].reindex(idx).to_numpy()
            miss.append(f"{side[0]}_{c}")
    f["miss_min_diff"] = f["h_miss_min"].fillna(0) - f["a_miss_min"].fillna(0)
    f["miss_pts_diff"] = f["h_miss_pts"].fillna(0) - f["a_miss_pts"].fillna(0)
    groups["missing"] = miss + ["miss_min_diff", "miss_pts_diff"]
    # matchup + shooting from box to date
    td = to_date(g, p).set_index(["game_id", "team"])
    for side in ("home", "away"):
        idx = list(zip(f["game_id"], f[f"{side}_team"]))
        for c in td.columns:
            f[f"{side[0]}_td_{c}"] = td[c].reindex(idx).to_numpy()
    lg = {c: np.nanmean(td[c]) for c in ("tp", "d_tp")}
    mt = []
    for o, d in (("h", "a"), ("a", "h")):
        for x, y in (("to", "d_to"), ("orb", "d_orb"), ("ftr", "d_ftr"),
                     ("tpa", "d_tpa"), ("efg", "d_efg")):
            name = f"{o}_{x}_vs_{d}_{y}"
            f[name] = f[f"{o}_td_{x}"] + f[f"{d}_td_{y}"]
            mt.append(name)
    groups["matchup"] = mt + groups.get("style_prev", [])
    sh = []
    for s in ("h", "a"):
        f[f"{s}_tp_luck"] = f[f"{s}_td_tp"] - lg["tp"]
        f[f"{s}_dtp_luck"] = f[f"{s}_td_d_tp"] - lg["d_tp"]
        sh += [f"{s}_tp_luck", f"{s}_dtp_luck", f"{s}_td_tp_att", f"{s}_td_d_tp_att"]
    groups["shooting"] = sh
    return f


def walk(f: pd.DataFrame, feats: list, target: str) -> pd.Series:
    from sklearn.ensemble import HistGradientBoostingRegressor
    pred = pd.Series(np.nan, index=f.index)
    y = f[target]
    for s in sorted(f["season"].unique()):
        if s < FIRST_TEST:
            continue
        tr = (f["season"] < s) & y.notna()
        te = f["season"] == s
        if tr.sum() < 3000 or not te.any():
            continue
        X = f[feats].astype(float)
        m = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.04,
                                          max_leaf_nodes=15, min_samples_leaf=300,
                                          l2_regularization=1.0, random_state=0)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def score(f: pd.DataFrame, pm: pd.Series, pt: pd.Series) -> dict:
    ok = pm.notna() & f["spread_line"].notna() & pt.notna() & f["total_line"].notna()
    x = f[ok]
    pm, pt = pm[ok], pt[ok]
    out = {"n": int(ok.sum()),
           "mae_m": (pm - x["actual_m"]).abs().mean(),
           "mae_t": (pt - x["actual_t"]).abs().mean()}
    su = (np.sign(pm) == np.sign(x["actual_m"]))[(x["actual_m"] != 0) & (pm != 0)]
    out["winner"] = su.mean()
    side = np.sign(pm - x["spread_line"])
    res = np.sign(x["actual_m"] - x["spread_line"])
    v = (side * res)[(side != 0) & (res != 0)]
    out["ats"] = (v > 0).mean()
    big = (pm - x["spread_line"]).abs() >= 3
    vb = (side * res)[(side != 0) & (res != 0) & big]
    out["ats3"], out["n3"] = (vb > 0).mean(), len(vb)
    side = np.sign(pt - x["total_line"])
    res = np.sign(x["actual_t"] - x["total_line"])
    v = (side * res)[(side != 0) & (res != 0)]
    out["ou"] = (v > 0).mean()
    return out


def line(label: str, r: dict, ref: dict | None = None) -> str:
    d = ""
    if ref:
        d = (f"  (margin {r['mae_m'] - ref['mae_m']:+.3f}, total "
             f"{r['mae_t'] - ref['mae_t']:+.3f})")
    return (f"  {label:22s} MAE margin {r['mae_m']:.3f} total {r['mae_t']:.3f}"
            f" | winner {r['winner']:.1%} | ATS {r['ats']:.1%}"
            f" (3+ off: {r['ats3']:.1%} n={r['n3']}) | O/U {r['ou']:.1%}"
            f" | n={r['n']}{d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="kp_games.csv")
    ap.add_argument("--box-cache", default=".cache_box")
    a = ap.parse_args()
    from cbb_data import load_games
    fr = pd.read_csv(a.games, low_memory=False)
    fr = fr[fr["completed"].astype(bool) & fr["spread_line"].notna()
            & fr["kp_margin"].notna() & fr["kp_total"].notna()]
    f = kp_build(fr)
    g = load_games(int(f["season"].min()) - 1)
    g["game_id"] = g["game_id"].astype(str)
    p = load_box(g, cache_dir=a.box_cache)
    print(f"box scores: {p['game_id'].nunique()} games")

    groups: dict = {}
    try:
        from kenpom_model import build as km_build, profiles
        seasons = range(int(fr["season"].min()) - 1, int(fr["season"].max()) + 1)
        prof, roster = profiles(seasons)
        fk, _, _ = km_build(fr.assign(game_id=fr["game_id"]), prof, roster)
        keep = [c for c in fk.columns if c.startswith(("h_prev_", "a_prev_"))
                and any(k in c for k in ("he_", "Hgt", "Exp", "Continuity", "Bench"))]
        fk = fk.set_index(fk["game_id"].astype(str))[keep]
        f["game_id"] = f["game_id"].astype(str)
        for c in keep:
            f[c] = fk[c].reindex(f["game_id"]).to_numpy()
        groups["style_prev"] = keep
    except Exception as e:                 # noqa: BLE001
        print(f"  last-season height/experience not loaded: {str(e)[:120]}")

    f = add_inputs(f, g, p, groups)
    f["em_m"] = f["actual_m"] - f["kp_margin"]
    f["em_t"] = f["actual_t"] - f["kp_total"]
    base = ["kp_margin", "kp_total", "kp_wp", "kp_tempo", "kp_h_em", "kp_a_em",
            "kp_h_oe", "kp_a_oe", "kp_h_de", "kp_a_de", "kp_h_pre_em", "kp_a_pre_em"]
    groups["form"] = [c for c in f.columns
                      if (c[:2] in ("h_", "a_") and c.endswith(("_3", "_5")))
                      or c.startswith(("h_streak", "a_streak"))]
    groups["rest"] = [c for c in ("home_rest", "away_rest", "min_n", "neutral_f", "month") if c in f]
    groups["pace"] = ["kp_h_t", "kp_a_t", "tempo_diff", "tempo_avg"]
    groups["all"] = sorted({c for k in ("missing", "form", "rest", "matchup", "pace", "shooting")
                            for c in groups.get(k, [])})

    test = f["season"] >= FIRST_TEST
    t = f[test]
    print(f"{int(test.sum())} test games ({FIRST_TEST}+), each season fitted on "
          f"earlier seasons only. Break-even ATS {BE:.1%}.\n")
    cov = {k: f.loc[test, v].notna().any(axis=1).mean() for k, v in groups.items() if v}
    print("input coverage in test games: " + ", ".join(f"{k} {v:.0%}" for k, v in cov.items()))
    print("\n== accuracy on games the models never saw ==")
    print(line("closing line", score(t, t["spread_line"], t["total_line"])))
    ref = score(t, t["kp_margin"], t["kp_total"])
    print(line("KenPom as published", ref))
    rm = walk(f, base, "em_m")
    rt = walk(f, base, "em_t")
    kb = score(t, (f["kp_margin"] + rm)[test], (f["kp_total"] + rt)[test])
    print(line("KenPom recalibrated", kb, ref))
    for k in ("missing", "form", "rest", "matchup", "pace", "shooting", "all"):
        cols = groups.get(k, [])
        if not cols:
            print(f"  + {k}: no inputs")
            continue
        rm = walk(f, base + cols, "em_m")
        rt = walk(f, base + cols, "em_t")
        r = score(t, (f["kp_margin"] + rm)[test], (f["kp_total"] + rt)[test])
        print(line(f"+ {k}", r, kb))
    print("\n(the bracket is the change in MAE vs 'KenPom recalibrated'; "
          "negative = more accurate)")
    # The size of a real gain: what 0.05 points of MAE is worth.
    print("\nmargin MAE by season, KenPom vs line: " + "  ".join(
        f"{s}: {(t.loc[t['season'] == s, 'kp_margin'] - t.loc[t['season'] == s, 'actual_m']).abs().mean():.2f}"
        f"/{(t.loc[t['season'] == s, 'spread_line'] - t.loc[t['season'] == s, 'actual_m']).abs().mean():.2f}"
        for s in sorted(t["season"].unique())))


if __name__ == "__main__":
    main()

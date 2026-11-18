"""
Which inputs predict college games best, judged on seasons the models
never saw, two ways:

  1. accuracy against the final score: mean absolute error of the home
     margin and the total, next to the closing line's own error
  2. against the closing line: does the model know anything the line
     doesn't? Slope of (final - close) on (model - close) with its t-stat
     (0 = nothing beyond the line), and the record betting our side at
     each size of disagreement and for the top N a night (52.4% breaks
     even at -110), overall and where the market is thin (low-major,
     early season, November)

Each test season is predicted by models fitted on every earlier season;
the features are walk-forward too. Variants, each adding to the one
before:

  ratings          final-score team ratings alone (the current model)
  +possessions     schedule, pace and efficiency ratings, four factors
  +players live    player impact, box value and roster continuity over
                   the live roster (what we could know before tip)
  +players played  the same over the players who actually played
                   (hindsight: knows late scratches; NOT bettable)
  players only     player inputs and schedule, no team ratings
  trees            the live set, ridge plus boosted trees on its misses
  blend            trees and the close, weight fitted on the season
                   before (not our own line; shows what the line adds)

Division I games only (games against non-D1 schools left out).

    python tools/model_lab.py --first 2019 --test 5
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cbb_predict as cp  # noqa: E402
from cbb_data import load_games, tier  # noqa: E402
from cbb_players import load_box  # noqa: E402

BREAK_EVEN = 110 / 210
GAPS = {"spreads": (0, 1, 2, 3, 4, 6), "totals": (0, 2, 3, 4, 6, 8)}
TOPS = (1, 2, 3, 5)

R_M, R_T = ["model_margin"], ["model_total"]
POSS_M = R_M + cp.SCHED_M + cp.POSS_M
POSS_T = R_T + cp.SCHED_T + cp.POSS_T
VARIANTS = {
    "ratings": (R_M, R_T, False),
    "+possessions": (POSS_M, POSS_T, False),
    "+players live": (POSS_M + cp.PLAY_M["l"], POSS_T + cp.PLAY_T["l"], False),
    "+players played": (POSS_M + cp.PLAY_M["h"], POSS_T + cp.PLAY_T["h"],
                        False),
    "players only": (cp.PLAY_M["l"] + cp.SCHED_M,
                     cp.PLAY_T["l"] + cp.SCHED_T, False),
    "trees": (cp.MARGIN_F, cp.TOTAL_F, True),
    "trees no players": (POSS_M, POSS_T, True),
}


def rec(res: np.ndarray, seasons: np.ndarray, nights: int | None = None
        ) -> str:
    keep = res != 0
    r, s = res[keep], seasons[keep]
    n = len(r)
    if not n:
        return "no bets"
    w = (r > 0).mean()
    se = np.sqrt(w * (1 - w) / n)
    flag = ("ABOVE break-even" if w - 2 * se > BREAK_EVEN else
            "below break-even" if w + 2 * se < BREAK_EVEN else "unproven")
    per = " ".join(f"{(r[s == x] > 0).mean():.0%}" for x in sorted(set(s)))
    rate = f" {n / nights:5.1f}/night" if nights else ""
    return (f"{n:6d} bets{rate}  win {w:.1%} (se {se:.1%}) {flag:16s} "
            f"by season {per}")


def info(pred, line, actual):
    """Slope and t of (actual - line) on (pred - line)."""
    x, y = pred - line, actual - line
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 50 or x.std() == 0:
        return float("nan"), float("nan")
    xc = x - x.mean()
    b = (xc @ (y - y.mean())) / (xc @ xc)
    resid = y - y.mean() - b * xc
    se = np.sqrt(resid @ resid / (len(x) - 2) / (xc @ xc))
    return b, b / se


def blend(tr, te, p):
    """Weights on our line and the close, fitted on the season before the
    test season with a model fitted on the seasons before that."""
    last = tr[tr["season"] == tr["season"].max()]
    last = last[last["spread_line"].notna() & last["total_line"].notna()]
    m2 = cp.fit_models(tr[tr["season"] < tr["season"].max()], trees=True)
    if not m2.margin or len(last) < 500:
        return pd.DataFrame({"pred_margin": te["spread_line"],
                             "pred_total": te["total_line"]}, index=te.index), None
    pl = cp.predict_frame(m2, last)
    bw = {}
    for k, line, y in (("margin", "spread_line",
                        last["home_score"] - last["away_score"]),
                       ("total", "total_line",
                        last["home_score"] + last["away_score"])):
        X = np.column_stack([pl[f"pred_{k}"], last[line], np.ones(len(last))])
        bw[k] = np.linalg.lstsq(X, y.to_numpy(float), rcond=None)[0]
    out = pd.DataFrame({
        "pred_margin": bw["margin"][0] * p["pred_margin"]
        + bw["margin"][1] * te["spread_line"] + bw["margin"][2],
        "pred_total": bw["total"][0] * p["pred_total"]
        + bw["total"][1] * te["total_line"] + bw["total"][2]}, index=te.index)
    return out, (bw["margin"][0], bw["total"][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2019)
    ap.add_argument("--last", type=int, default=None)
    ap.add_argument("--test", type=int, default=5)
    ap.add_argument("--save", default="")
    ap.add_argument("--load", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--min-train", dest="min_train", type=int, default=8000)
    a, _ = ap.parse_known_args()
    t0 = time.time()
    cfg = cp.PredictConfig()
    if a.load:
        h = pd.read_pickle(a.load)
    else:
        # One warm-up season before the first projected one: last season's
        # players and ratings carry into November.
        g = load_games(a.first - 1, a.last)
        g = g[g["season_type"].isin(["REG", "POST"])].reset_index(drop=True)
        print(f"loaded {len(g)} games ({time.time() - t0:.0f}s)", flush=True)
        box = load_box(g)
        print(f"box rows {len(box)}, games with a box "
              f"{box['game_id'].nunique()} ({time.time() - t0:.0f}s)",
              flush=True)
        h = cp.build(g, box, cfg, first_season=a.first, verbose=True)
        print(f"features built ({time.time() - t0:.0f}s)", flush=True)
        if a.save:
            h.to_pickle(a.save)
    h = h[h["completed"] & ~h["vs_non_d1"].astype(bool)
          & (h["season"] >= a.first) & h["model_margin"].notna()].copy()
    cov = h.groupby("season").agg(
        games=("game_id", "size"),
        line=("spread_line", lambda s: s.notna().mean()),
        eff=("eff_margin", lambda s: s.notna().mean()),
        apm=("apm_diff_l", lambda s: s.notna().mean()),
        box=("box_diff_l", lambda s: s.notna().mean()))
    print("\ncoverage by season (D1 games)\n" + cov.round(3).to_string())
    seasons = sorted(h["season"].unique())
    tests = [s for s in seasons[-a.test:] if (h["season"] < s).sum() >= a.min_train]
    print(f"\ntest seasons {tests}; each fitted on seasons {seasons[0]}.."
          "the one before")

    preds = {k: [] for k in list(VARIANTS) + ["blend", "close"]}
    frames, last_models = [], None
    for s in tests:
        tr, te = h[h["season"] < s], h[h["season"] == s].copy()
        te = te[te["spread_line"].notna() & te["total_line"].notna()]
        frames.append(te)
        for name, (mc, tc, trees) in VARIANTS.items():
            m = cp.fit_models(tr, trees=trees, margin_cols=mc, total_cols=tc)
            p = cp.predict_frame(m, te, "trees" if trees else "ridge")
            preds[name].append(p)
            if name == "trees":
                last_models = m
                bp, w = blend(tr, te, p)
                preds["blend"].append(bp)
                if w:
                    print(f"  {s}: blend weight on our line {w[0]:.2f} "
                          f"(spread), {w[1]:.2f} (total)")
        preds["close"].append(pd.DataFrame({"pred_margin": te["spread_line"],
                                            "pred_total": te["total_line"]},
                                           index=te.index))
        print(f"  {s} done ({time.time() - t0:.0f}s)", flush=True)
    te = pd.concat(frames)
    te["tier"] = tier(te)
    nights = te["game_date"].nunique()
    am = (te["home_score"] - te["away_score"]).to_numpy(float)
    at = (te["home_score"] + te["away_score"]).to_numpy(float)
    sm = te["spread_line"].to_numpy(float)
    st = te["total_line"].to_numpy(float)
    ss = te["season"].to_numpy()
    print(f"\n{len(te)} test games, {nights} nights. Break-even at -110: "
          f"{BREAK_EVEN:.1%}")

    print("\nACCURACY vs final score (lower is better), and information "
          "beyond the closing line")
    print(f"  {'model':16s} {'margin MAE':>10s} {'total MAE':>10s}  "
          f"{'spread slope':>12s} {'t':>6s}  {'total slope':>11s} {'t':>6s}")
    P = {}
    for name, parts in preds.items():
        p = pd.concat(parts).reindex(te.index)
        P[name] = p
        pm, pt = p["pred_margin"].to_numpy(float), p["pred_total"].to_numpy(float)
        bm, tm = info(pm, sm, am)
        bt, tt = info(pt, st, at)
        print(f"  {name:16s} {np.nanmean(np.abs(pm - am)):10.2f} "
              f"{np.nanmean(np.abs(pt - at)):10.2f}  {bm:12.3f} {tm:6.1f}  "
              f"{bt:11.3f} {tt:6.1f}")
    print("  (slope 1 = our disagreement with the line is fully right, "
          "0 = worthless; |t| > 2 is real)")

    print("\nMAE by season (margin / total)")
    for name in ("close", "ratings", "+possessions", "+players live",
                 "+players played", "trees"):
        p = P[name]
        row = []
        for s in tests:
            k = ss == s
            row.append(f"{s}: {np.nanmean(np.abs(p['pred_margin'].to_numpy()[k] - am[k])):.2f}"
                       f"/{np.nanmean(np.abs(p['pred_total'].to_numpy()[k] - at[k])):.2f}")
        print(f"  {name:16s} " + "  ".join(row))

    gd = te["game_date"].to_numpy()
    for name in ("ratings", "+possessions", "+players live",
                 "+players played", "players only", "trees", "blend"):
        p = P[name]
        print(f"\n{name.upper()} against the closing line")
        for market, pred, line, act in (
                ("spreads", p["pred_margin"].to_numpy(float), sm, am),
                ("totals", p["pred_total"].to_numpy(float), st, at)):
            gap = pred - line
            res = np.sign(gap * (act - line))
            print(f"  {market}: mean |gap| {np.nanmean(np.abs(gap)):.2f}")
            for t in GAPS[market]:
                k = np.abs(gap) >= max(t, 1e-9)
                print(f"    gap {t}+          {rec(res[k], ss[k], nights)}")
            d = pd.DataFrame({"night": gd, "a": np.abs(gap), "res": res,
                              "s": ss})
            d = d[np.isfinite(d["a"])].sort_values("a", ascending=False)
            for n in TOPS:
                top = d.groupby("night").head(n)
                print(f"    top {n} a night  "
                      f"{rec(top['res'].to_numpy(), top['s'].to_numpy(), nights)}")

    # Where the market is thinnest. The old line's one near-signal was
    # November spreads at 4+ off the close.
    month = pd.to_datetime(te["game_date"]).dt.month.to_numpy()
    early = (te["min_n"].fillna(0) < 8).to_numpy()
    tiers = te["tier"].to_numpy()
    parts = {"tier=high": tiers == "high", "tier=mixed": tiers == "mixed",
             "tier=low": tiers == "low", "early (<8 games)": early,
             "later": ~early, "November": month == 11,
             "Dec": month == 12, "Jan-Feb": np.isin(month, (1, 2)),
             "March+": np.isin(month, (3, 4))}
    # Is the edge real or an artefact of the line? By spread size, how
    # many books ESPN's number came from, and which side we bet.
    absl = np.abs(sm)
    books = te["line_books"].fillna(-1).to_numpy() if "line_books" in te \
        else np.full(len(te), -1.0)
    for name in ("trees", "trees no players", "+players live"):
        pred = P[name]["pred_margin"].to_numpy(float)
        gap = pred - sm
        res = np.sign(gap * (am - sm))
        fav_home = sm > 0
        bet_home = gap > 0
        bet_fav = np.where(fav_home, bet_home, ~bet_home) & (sm != 0)
        cuts = {"|line| < 5": absl < 5, "|line| 5-10": (absl >= 5) & (absl < 10),
                "|line| 10-20": (absl >= 10) & (absl < 20),
                "|line| 20+": absl >= 20,
                "books: scoreboard": books < 0, "books 1-2": (books >= 0) & (books < 3),
                "books 3+": books >= 3,
                "bet favourite": bet_fav, "bet underdog": ~bet_fav & (sm != 0),
                "bet home": bet_home, "bet away": ~bet_home}
        print(f"\n{name.upper()} spreads, gap 4+, by line size / source / side")
        for lab, k in cuts.items():
            b, t = info(pred[k], sm[k], am[k])
            kk = k & (np.abs(gap) >= 4)
            print(f"  {lab:18s} n={k.sum():5d} slope {b:+.3f} (t {t:+.1f})  "
                  f"{rec(res[kk], ss[kk])}")

    for name in ("ratings", "+players live", "trees", "+players played"):
        p = P[name]
        print(f"\n{name.upper()} by slice (slope vs close; record at gap)")
        for market, pred, line, act, gp in (
                ("spreads", p["pred_margin"].to_numpy(float), sm, am, 4),
                ("totals", p["pred_total"].to_numpy(float), st, at, 6)):
            gap = pred - line
            res = np.sign(gap * (act - line))
            for lab, k in parts.items():
                b, t = info(pred[k], line[k], act[k])
                kk = k & (np.abs(gap) >= gp)
                print(f"  {market:7s} {lab:17s} n={k.sum():5d} slope {b:+.3f} "
                      f"(t {t:+.1f})  gap {gp}+: {rec(res[kk], ss[kk])}")

    if last_models is not None and last_models.margin:
        print("\nLatest fitted ridge weights (points per 1 sd of input)")
        print("  margin:", cp.ridge_weights(last_models.margin))
        print("  total: ", cp.ridge_weights(last_models.total))
    if a.out:
        cols = ["game_id", "season", "game_date", "home_team", "away_team",
                "home_score", "away_score", "spread_line", "total_line",
                "tier", "min_n", "line_books"]
        o = te[cols].copy()
        for name in ("ratings", "+players live", "trees", "trees no players"):
            o[f"{name}_margin"] = P[name]["pred_margin"]
            o[f"{name}_total"] = P[name]["pred_total"]
        o.to_csv(a.out, index=False)
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

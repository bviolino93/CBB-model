"""
Does KenPom beat the closing line, alone or blended with our own numbers?

Every number here was known on the morning of the game:

  * KenPom's FanMatch prediction for the game (score for each side)
  * KenPom's ratings archived on that date (AdjEM, AdjOE, AdjDE, tempo)
    and the preseason ratings
  * our own score-based ratings (cbb_ratings.walk_forward, no market input)

Lines tested:

  kenpom   KenPom's FanMatch line as published
  own      our score-based ratings line
  blend    a ridge fit of the final margin / total on the KenPom and own
           numbers, refitted each season on earlier seasons only (the
           target is the real result, never the market)
  agree    the blend, bet only when KenPom and our own line are both on
           the same side of the close

For each, against ESPN's pregame (closing) line: slope, MAE, record by
gap size and top N gaps a night, plus slices by market depth, season
stage and month (backtest.py's reports). Break-even at -110 is 52.4%.

    python tools/kenpom_backtest.py --first 2019 [--out kp_games.csv]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from backtest import BREAK_EVEN, bets, market_report, rec, slices  # noqa: E402
from cbb_data import load_games, schedule_features, tier  # noqa: E402
from cbb_ratings import walk_forward  # noqa: E402
from kenpom import load_kenpom  # noqa: E402

RIDGE = 5.0


def features(fr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Home-perspective design matrices for margin and total."""
    early = (fr["min_n"] < 8).astype(float)
    home = (~fr["neutral"].astype(bool)).astype(float)
    em = fr["kp_h_em"] - fr["kp_a_em"]
    pre = fr["kp_h_pre_em"] - fr["kp_a_pre_em"]
    t = (fr["kp_h_t"] + fr["kp_a_t"]) / 2
    rest = fr["rest_diff"] if "rest_diff" in fr else 0.0
    xm = pd.DataFrame({
        "kp": fr["kp_margin"],
        "own": fr["model_margin"],
        "em_poss": em * t / 100,
        "pre_poss": (pre - em) * t / 100,      # preseason view vs current
        "kp_early": fr["kp_margin"] * early,
        "own_early": fr["model_margin"] * early,
        "home": home,
        "rest": rest,
    })
    xt = pd.DataFrame({
        "kp": fr["kp_total"],
        "own": fr["model_total"],
        "eff": (fr["kp_h_oe"] + fr["kp_a_oe"] + fr["kp_h_de"] + fr["kp_a_de"])
        / 2 * t / 100,
        "tempo": t,
        "kp_early": fr["kp_total"] * early,
        "own_early": fr["model_total"] * early,
    })
    return xm, xt


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = RIDGE):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    Z = (X - mu) / sd
    A = Z.T @ Z + lam * np.eye(Z.shape[1])
    b = np.linalg.solve(A, Z.T @ (y - y.mean()))
    return mu, sd, b, y.mean()


def ridge_pred(m, X):
    mu, sd, b, c = m
    return ((X - mu) / sd) @ b + c


def blend(fr: pd.DataFrame, min_train: int = 2, verbose: bool = True
          ) -> pd.DataFrame:
    """blend_margin / blend_total: each season predicted by a ridge fitted
    on all earlier seasons (needs `min_train` of them)."""
    fr = fr.copy()
    xm, xt = features(fr)
    ym = (fr["home_score"] - fr["away_score"]).to_numpy(float)
    yt = (fr["home_score"] + fr["away_score"]).to_numpy(float)
    fr["blend_margin"] = np.nan
    fr["blend_total"] = np.nan
    seasons = sorted(fr["season"].unique())
    for s in seasons[min_train:]:
        tr = (fr["season"] < s).to_numpy()
        te = (fr["season"] == s).to_numpy()
        for X, y, col in ((xm, ym, "blend_margin"), (xt, yt, "blend_total")):
            Xa = X.to_numpy(float)
            ok_tr = tr & np.isfinite(Xa).all(1) & np.isfinite(y)
            ok_te = te & np.isfinite(Xa).all(1)
            if ok_tr.sum() < 500 or not ok_te.any():
                continue
            m = ridge_fit(Xa[ok_tr], y[ok_tr])
            fr.loc[ok_te, col] = ridge_pred(m, Xa[ok_te])
            if verbose and s == seasons[-1]:
                coefs = ", ".join(f"{c} {b / sd:+.2f}" for c, b, sd in
                                  zip(X.columns, m[2], m[1]))
                print(f"  {col} weights for {s} (per unit): {coefs}")
    return fr


def report(fr: pd.DataFrame, label: str, margin: str, total: str) -> None:
    x = fr.assign(model_margin=fr[margin], model_total=fr[total])
    x = x[x["model_margin"].notna() | x["model_total"].notna()]
    print(f"\n== {label} ({x['season'].min()}-{x['season'].max()}, "
          f"{len(x)} games) ==")
    for market in ("spreads", "totals"):
        print("\n".join(market_report(x, market)))
    print("\n".join(slices(x)))


def agree_report(fr: pd.DataFrame) -> None:
    """The blend, only where KenPom and our own line both sit on the same
    side of the close as the blend."""
    print("\n== agree: blend bet only when KenPom and own line both agree ==")
    for market, m, k, o, c, gaps in (
            ("spreads", "blend_margin", "kp_margin", "model_margin",
             "spread_line", (1, 2, 3, 4)),
            ("totals", "blend_total", "kp_total", "model_total",
             "total_line", (2, 3, 4, 6))):
        x = fr.assign(model_margin=fr["blend_margin"], model_total=fr["blend_total"])
        b = bets(x, market)
        side = np.sign(fr.loc[b.index, m] - fr.loc[b.index, c])
        ok = ((np.sign(fr.loc[b.index, k] - fr.loc[b.index, c]) == side)
              & (np.sign(fr.loc[b.index, o] - fr.loc[b.index, c]) == side))
        b = b[ok]
        nights = b["night"].nunique()
        for gp in gaps:
            print(f"  {market} gap {gp}+  {rec(b[b['gap'].abs() >= gp], nights)}")


def by_season(fr: pd.DataFrame) -> None:
    print("\n== MAE by season: KenPom / own / blend / close ==")
    act_m = fr["home_score"] - fr["away_score"]
    act_t = fr["home_score"] + fr["away_score"]
    for s, x in fr.groupby("season"):
        ok = x[["kp_margin", "spread_line"]].notna().all(axis=1)
        okt = x[["kp_total", "total_line"]].notna().all(axis=1)
        if not ok.any():
            continue
        e = lambda col, act, msk: np.abs(x[col] - act[x.index])[msk].mean()
        print(f"  {s}: margin {e('kp_margin', act_m, ok):.2f} / "
              f"{e('model_margin', act_m, ok):.2f} / "
              f"{e('blend_margin', act_m, ok):.2f} / "
              f"{e('spread_line', act_m, ok):.2f}   total "
              f"{e('kp_total', act_t, okt):.2f} / {e('model_total', act_t, okt):.2f}"
              f" / {e('blend_total', act_t, okt):.2f} / "
              f"{e('total_line', act_t, okt):.2f}   n={int(ok.sum())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2019)
    ap.add_argument("--last", type=int, default=None)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    g = schedule_features(load_games(a.first - 1, a.last))
    fr = walk_forward(g, first_season=a.first)
    seasons = range(a.first, int(fr["season"].max()) + 1)
    fr = load_kenpom(fr, seasons)
    fr = fr[fr["completed"] & ~fr["vs_non_d1"] & (fr["season"] >= a.first)
            & fr["kp_margin"].notna()]
    cover = fr[["spread_line", "total_line"]].notna().mean()
    print(f"{len(fr)} completed D1 games with a KenPom line; market lines on "
          f"{cover['spread_line']:.0%} / {cover['total_line']:.0%}. "
          f"Break-even at -110 is {BREAK_EVEN:.1%}.")
    fr = blend(fr)
    by_season(fr)
    report(fr, "kenpom (FanMatch as published)", "kp_margin", "kp_total")
    report(fr, "own score ratings", "model_margin", "model_total")
    held = fr[fr["blend_margin"].notna()]
    report(held, "blend (fitted on earlier seasons only)", "blend_margin",
           "blend_total")
    agree_report(held)
    if a.out:
        fr.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()

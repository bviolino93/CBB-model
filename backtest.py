"""
Walk-forward backtest against the closing line: does our own line beat it?

For every Division I game (non-D1 opponents left out), the ratings are
fitted on earlier games only. Then, split into tuning seasons and
held-out seasons:

  * slope of (result - close) on (model - close), with its t-stat. Zero
    means the close already knows everything the ratings know; a positive,
    significant slope is the share of our disagreement worth betting
  * the record betting our side when we disagree by a gap or more, and
    betting the N biggest gaps each night, against the close at -110
    (52.4% breaks even), with a standard error
  * the same split by market depth (both teams high-major, one, neither),
    by how far into the season (either team under 8 games), and by month,
    because college markets are thinnest in low-major games and early
    November, which is where an edge from our own numbers is likeliest

The closing line here is ESPN's pregame number (a median of the books it
carries). That is close to the real close, but not exact; clv on logged
lines is the stricter test once data/lines/ has a season in it.

    python backtest.py --first 2019 --split 2024 [--out games.csv]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from cbb_data import load_games, schedule_features, tier
from cbb_ratings import RatingConfig, walk_forward

BREAK_EVEN = 110 / 210
SPREAD_GAPS = (2, 3, 4, 5, 6, 8)
TOTAL_GAPS = (3, 4, 6, 8, 10)
TOPS = (1, 2, 3, 5)


def slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    """OLS slope of y on x (with intercept) and its t-stat."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 30 or x.std() == 0:
        return float("nan"), float("nan"), int(len(x))
    X = np.column_stack([np.ones_like(x), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    s2 = res @ res / (len(x) - 2)
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    return float(b[1]), float(b[1] / se), int(len(x))


def bets(fr: pd.DataFrame, market: str) -> pd.DataFrame:
    """One row per game: our gap to the close and the result betting our
    side (+1 win, -1 loss, 0 push)."""
    if market == "spreads":
        model, close = fr["model_margin"], fr["spread_line"]
        actual = fr["home_score"] - fr["away_score"]
    else:
        model, close = fr["model_total"], fr["total_line"]
        actual = fr["home_score"] + fr["away_score"]
    gap = model - close
    out = pd.DataFrame({"season": fr["season"], "night": fr["game_date"],
                        "gap": gap, "res": np.sign(gap * (actual - close)),
                        "resid": actual - close})
    return out[np.isfinite(out["gap"]) & np.isfinite(out["resid"])]


def rec(b: pd.DataFrame, nights: int | None = None) -> str:
    b = b[b["res"] != 0]
    n = len(b)
    if n == 0:
        return "no bets"
    w = (b["res"] > 0).mean()
    se = np.sqrt(w * (1 - w) / n)
    flag = ("ABOVE break-even" if w - 2 * se > BREAK_EVEN else
            "below break-even" if w + 2 * se < BREAK_EVEN else "unproven")
    per = ", ".join(f"{(x['res'] > 0).mean():.0%}"
                    for _, x in b.groupby("season"))
    rate = f"  {n / nights:.1f}/night" if nights else ""
    return (f"{n:5d} bets  win {w:.1%} (se {se:.1%})  {flag}{rate}"
            f"  by season {per}")


def market_report(fr: pd.DataFrame, market: str) -> list[str]:
    b = bets(fr, market)
    if b.empty:
        return [f"  {market}: no games with a line"]
    nights = b["night"].nunique()
    sl, t, n = slope(b["gap"].to_numpy(float), b["resid"].to_numpy(float))
    model = fr["model_margin" if market == "spreads" else "model_total"]
    close = fr["spread_line" if market == "spreads" else "total_line"]
    actual = (fr["home_score"] - fr["away_score"] if market == "spreads"
              else fr["home_score"] + fr["away_score"])
    ok = model.notna() & close.notna() & actual.notna()
    out = [f"  {market.upper()}: {n} games, slope {sl:+.3f} (t {t:+.2f}), "
           f"MAE model {np.abs(model - actual)[ok].mean():.2f} vs close "
           f"{np.abs(close - actual)[ok].mean():.2f}, mean |gap| "
           f"{b['gap'].abs().mean():.2f}"]
    for gap in (SPREAD_GAPS if market == "spreads" else TOTAL_GAPS):
        out.append(f"    gap {gap:>2}+       {rec(b[b['gap'].abs() >= gap], nights)}")
    r = b.assign(a=b["gap"].abs()).sort_values("a", ascending=False)
    for k in TOPS:
        out.append(f"    top {k} a night {rec(r.groupby('night').head(k), nights)}")
    return out


def slices(fr: pd.DataFrame) -> list[str]:
    """Slope and the gap rule by market depth, season stage and month."""
    out = ["  -- where the market might be thin --"]
    fr = fr.assign(tier=tier(fr), stage=np.where(fr["min_n"] < 8, "early", "later"),
                   month=pd.to_datetime(fr["game_date"]).dt.month)
    for col, keys in (("tier", ("high", "mixed", "low")),
                      ("stage", ("early", "later")),
                      ("month", (11, 12, 1, 2, 3))):
        for k in keys:
            part = fr[fr[col] == k]
            for market, gap in (("spreads", 4), ("totals", 6)):
                b = bets(part, market)
                sl, t, n = slope(b["gap"].to_numpy(float),
                                 b["resid"].to_numpy(float))
                out.append(f"    {col}={k!s:6s} {market:7s} n={n:5d} slope "
                           f"{sl:+.3f} (t {t:+.2f})  gap {gap}+: "
                           f"{rec(b[b['gap'].abs() >= gap])}")
    return out


def blend_report(tune: pd.DataFrame, held: pd.DataFrame) -> list[str]:
    """Close + k * (model - close), k fitted on tuning seasons only: bet the
    held-out games where the blend is off the close by 1+ point (spreads)
    or 2+ (totals). Only worth anything if the tuning slope is positive."""
    out = ["  -- blended line, weight fitted on the tuning seasons --"]
    for market, gaps in (("spreads", (1, 1.5, 2)), ("totals", (1.5, 2, 3))):
        bt, bh = bets(tune, market), bets(held, market)
        k, t, _ = slope(bt["gap"].to_numpy(float), bt["resid"].to_numpy(float))
        if not np.isfinite(k) or k <= 0:
            out.append(f"    {market}: tuning slope {k:+.3f}, nothing to blend")
            continue
        bh = bh.assign(gap=bh["gap"] * k)
        for gp in gaps:
            out.append(f"    {market} weight {k:.2f}, blended gap {gp}+: "
                       f"{rec(bh[bh['gap'].abs() >= gp], bh['night'].nunique())}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2019)
    ap.add_argument("--last", type=int, default=None)
    ap.add_argument("--split", type=int, default=2024)
    ap.add_argument("--half-life", type=float, default=RatingConfig.half_life)
    ap.add_argument("--offseason", type=float, default=RatingConfig.offseason_days)
    ap.add_argument("--cap", type=float, default=RatingConfig.margin_cap)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    g = schedule_features(load_games(a.first, a.last))
    cfg = RatingConfig(half_life=a.half_life, offseason_days=a.offseason,
                       margin_cap=a.cap)
    fr = walk_forward(g, cfg, first_season=a.first + 1)
    fr = fr[fr["completed"] & ~fr["vs_non_d1"] & fr["model_margin"].notna()]
    cover = fr[["spread_line", "total_line"]].notna().mean()
    print(f"{len(fr)} completed D1 games with a projection; lines on "
          f"{cover['spread_line']:.0%} (spread) / {cover['total_line']:.0%} "
          f"(total). Break-even at -110 is {BREAK_EVEN:.1%}.")
    tune, held = fr[fr["season"] < a.split], fr[fr["season"] >= a.split]
    for lab, part in ((f"{a.first + 1}-{a.split - 1} (tuning)", tune),
                      (f"{a.split}-now (held out)", held)):
        print(f"\n== {lab} ==")
        for market in ("spreads", "totals"):
            print("\n".join(market_report(part, market)))
        print("\n".join(slices(part)))
    print()
    print("\n".join(blend_report(tune, held)))
    if a.out:
        fr.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()

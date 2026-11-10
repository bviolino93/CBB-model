"""
Tune the ratings on the tuning seasons only: half-life, how much of last
season carries over (offseason_days) and the blowout cap.

Scored by MAE of the home margin and total against final scores, overall
and early (either team under 8 games in), and by the slope of
(result - close) on (model - close), the number that matters for betting.

    python tools/tune_ratings.py --first 2019 --split 2024
"""
import argparse
import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtest import slope  # noqa: E402
from cbb_data import load_games, schedule_features  # noqa: E402
from cbb_ratings import RatingConfig, walk_forward  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2019)
    ap.add_argument("--split", type=int, default=2024)
    a = ap.parse_args()
    g = schedule_features(load_games(a.first, a.split - 1))
    rows = []
    for hl, off, cap in itertools.product((30.0, 60.0, 120.0),
                                          (60.0, 120.0, 240.0), (20.0, 30.0)):
        t0 = time.time()
        fr = walk_forward(g, RatingConfig(half_life=hl, offseason_days=off,
                                          margin_cap=cap), a.first + 1)
        fr = fr[fr["completed"] & ~fr["vs_non_d1"] & fr["model_margin"].notna()]
        m = fr["home_score"] - fr["away_score"]
        p = fr["home_score"] + fr["away_score"]
        e = fr["min_n"] < 8
        sm, tm, _ = slope((fr["model_margin"] - fr["spread_line"]).to_numpy(float),
                          (m - fr["spread_line"]).to_numpy(float))
        st, tt, _ = slope((fr["model_total"] - fr["total_line"]).to_numpy(float),
                          (p - fr["total_line"]).to_numpy(float))
        em, et = (fr["model_margin"] - m).abs(), (fr["model_total"] - p).abs()
        rows.append((hl, off, cap, em.mean(), em[e].mean(), et.mean(),
                     et[e].mean(), sm, tm, st, tt))
        print(f"hl {hl:4.0f} off {off:4.0f} cap {cap:3.0f}: margin MAE "
              f"{em.mean():.3f} (early {em[e].mean():.3f})  total MAE "
              f"{et.mean():.3f} (early {et[e].mean():.3f})  slope vs close "
              f"spr {sm:+.3f} (t {tm:+.1f}) tot {st:+.3f} (t {tt:+.1f})  "
              f"[{time.time() - t0:.0f}s]", flush=True)
    close_m = (fr["spread_line"] - m).abs().mean()
    close_t = (fr["total_line"] - p).abs().mean()
    best = min(rows, key=lambda r: r[3])
    print(f"\nclose MAE: margin {close_m:.3f}, total {close_t:.3f}")
    print(f"best margin MAE: half_life {best[0]:.0f}, offseason {best[1]:.0f}, "
          f"cap {best[2]:.0f}")
    best = min(rows, key=lambda r: r[5])
    print(f"best total MAE: half_life {best[0]:.0f}, offseason {best[1]:.0f}, "
          f"cap {best[2]:.0f}")
    if not np.isfinite(best[7]):
        print("(no closing lines in range for the slope)")


if __name__ == "__main__":
    main()

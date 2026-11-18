"""
Game-by-game breakdown for one college basketball day: what a bettor
would look at for each Division I game, in one sheet, so the picks can
be made game by game.

For every game it lists the line (ESPN's pregame number), our projected
score (cbb_predict trees model, live roster, no market input), matchup
(possessions ratings, pace, four-factor edges, player impact, roster
continuity), recent form (record, margin, covers, overs), rest and games
played, conference tier, and whether the one backtested rule (the A
card in tools/tonight.py) fires. It writes breakdown/cbb/<date>.md and
.json and makes no picks itself.

    python tools/breakdown.py [--date 2026-11-18] [--first 2021]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import tonight  # noqa: E402  (patches cbb_data.mark_non_d1 for a young season)
import cbb_predict as cp  # noqa: E402
from cbb_data import load_games, tier  # noqa: E402
from cbb_players import load_box  # noqa: E402

K = "l"      # live-roster player features (no hindsight on who played)


def team_form(g: pd.DataFrame, team: str, before, last: int = 10) -> dict:
    """Last `last` completed games of `team` before `before` (this season
    or last): record, margin, points, covers and overs vs the line."""
    d = g[g["completed"] & (pd.to_datetime(g["commence"], utc=True)
                            < pd.Timestamp(before))
          & ((g["home_team"] == team) | (g["away_team"] == team))]
    d = d.sort_values("commence").tail(last)
    if d.empty:
        return {"n": 0}
    home = (d["home_team"] == team).to_numpy()
    hm = (d["home_score"] - d["away_score"]).to_numpy(float)
    m = np.where(home, hm, -hm)
    pf = np.where(home, d["home_score"], d["away_score"]).astype(float)
    pa = np.where(home, d["away_score"], d["home_score"]).astype(float)
    sl = d["spread_line"].to_numpy(float)
    vs = np.where(home, hm - sl, sl - hm)
    tot = (d["home_score"] + d["away_score"]).to_numpy(float)
    tl = d["total_line"].to_numpy(float)
    ok, okt = np.isfinite(vs), np.isfinite(tl)
    this = d["season"].iloc[-1]
    return {"n": int(len(d)), "w": int((m > 0).sum()), "l": int((m < 0).sum()),
            "this_season": int((d["season"] == this).sum()),
            "margin": round(float(m.mean()), 1),
            "pf": round(float(pf.mean()), 1), "pa": round(float(pa.mean()), 1),
            "cover": int((vs[ok] > 0).sum()), "no_cover": int((vs[ok] < 0).sum()),
            "vs_line": round(float(vs[ok].mean()), 1) if ok.any() else None,
            "over": int((tot[okt] > tl[okt]).sum()),
            "under": int((tot[okt] < tl[okt]).sum()),
            "last": [f"{'W' if x > 0 else 'L'} {x:+.0f}" for x in m[-5:]]}


def _f(x, nd=1):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if np.isfinite(x) else None


def a_card(tier_: str, gap: float, line_home: float) -> bool:
    """tools/tonight.py's A rule: low/mixed tier, gap 6+, bet the
    favourite, |line| under 15 (56.6% on unseen seasons). gap and
    line_home are expected home margins (+ = home)."""
    if tier_ == "high" or gap is None or line_home is None or abs(gap) < 6:
        return False
    bet_home = gap > 0
    fav = (bet_home and line_home > 0) or (not bet_home and line_home < 0)
    return fav and abs(line_home) < 15


def notes(b: dict) -> list[str]:
    out = []
    hs, as_ = b["home"], b["away"]
    s = b["schedule"]
    rd = s.get("rest_diff")
    if rd is not None and abs(rd) >= 2:
        out.append(f"{hs if rd > 0 else as_} has {abs(rd):.0f} more days of rest")
    for side, name in (("home", hs), ("away", as_)):
        if (s.get(f"{side}_rest") or 9) <= 1:
            out.append(f"{name} played yesterday")
        f = b["form"][side]
        if f.get("n", 0) >= 5 and (f["cover"] >= f["n"] - 1
                                   or f["no_cover"] >= f["n"] - 1):
            out.append(f"{name} {f['cover']}-{f['no_cover']} against the spread "
                       f"last {f['n']}")
        if f.get("this_season", 0) <= 2:
            out.append(f"{name}: only {f.get('this_season', 0)} games this "
                       "season, ratings lean on last year")
    m = b["matchup"]
    if m.get("new_diff") is not None and abs(m["new_diff"]) >= 0.25:
        more = hs if m["new_diff"] > 0 else as_
        out.append(f"{more} has much more new talent in its rotation")
    p, ln = b["projection"], b["line"]
    if p.get("margin") is not None and ln.get("home_margin") is not None:
        gap = p["margin"] - ln["home_margin"]
        if abs(gap) >= 4:
            out.append(f"Our projection is {abs(gap):.1f} pts better for "
                       f"{hs if gap > 0 else as_} than the line")
    if p.get("total") is not None and ln.get("total") is not None:
        tg = p["total"] - ln["total"]
        if abs(tg) >= 5:
            out.append(f"Our total is {abs(tg):.1f} {'over' if tg > 0 else 'under'}"
                       " the line")
    if b.get("a_card"):
        out.append("Backtested A rule fires (low/mixed tier, 6+ gap, "
                   "favourite, line under 15: 56.6% on unseen seasons)")
    return out


def to_markdown(day, games: list[dict]) -> str:
    out = [f"# College basketball breakdown {day}", "",
           "Line: home team's spread as books quote it (negative = home "
           "favoured), ESPN pregame. Projection: our model, no market input.",
           ""]
    for b in games:
        hs, as_ = b["home"], b["away"]
        ln, p, m, s = b["line"], b["projection"], b["matchup"], b["schedule"]
        out.append(f"## {as_} @ {hs}{' (neutral)' if b['neutral'] else ''} "
                   f"({b['tip_et']} ET, {b['tier']} tier)")
        if ln.get("spread") is not None:
            out.append(f"**Line:** {hs} {ln['spread']:+.1f}, total "
                       + (f"{ln['total']:.1f}" if ln.get("total") is not None
                          else "none") + f" ({ln.get('provider') or 'ESPN'})")
        else:
            out.append("**Line:** none yet")
        if p.get("margin") is not None:
            out.append(f"**Projection:** {hs} {p['home_pts']:.0f}, {as_} "
                       f"{p['away_pts']:.0f} ({hs} {p['margin']:+.1f}, total "
                       f"{p['total']:.1f})")
        mm = []
        if m.get("pace") is not None:
            mm.append(f"pace {m['pace']:.1f}, offense per 100 {hs} "
                      f"{m.get('home_ortg')} / {as_} {m.get('away_ortg')}")
        if m.get("apm_diff") is not None:
            mm.append(f"player impact {hs} {m['apm_diff']:+.1f}")
        if m.get("same_diff") is not None:
            mm.append(f"returning-minutes edge {hs} {m['same_diff']:+.2f}")
        ff = [f"{k} {v:+.3f}" for k, v in m.get("four_factors", {}).items()
              if v is not None]
        if ff:
            mm.append("four factors for " + hs + ": " + ", ".join(ff))
        if mm:
            out.append("**Matchup:** " + "; ".join(mm))
        for side, name in (("home", hs), ("away", as_)):
            f = b["form"][side]
            r = s.get(f"{side}_rest")
            rest = "" if r is None else f", rest {r:.0f}d"
            if not f.get("n"):
                out.append(f"- {name}: no games yet{rest}")
                continue
            vl = f.get("vs_line")
            out.append(f"- {name}: {f['w']}-{f['l']} last {f['n']} "
                       f"({f['this_season']} this season), margin "
                       f"{f['margin']:+.1f}, scores {f['pf']:.0f} allows "
                       f"{f['pa']:.0f}; ATS {f['cover']}-{f['no_cover']}"
                       + (f" ({vl:+.1f} vs line)" if vl is not None else "")
                       + f"; O/U {f['over']}-{f['under']}; last 5 "
                       + ", ".join(f["last"]) + rest)
        if b["notes"]:
            out.append("**Notes:** " + "; ".join(b["notes"]))
        out.append("")
    return "\n".join(out)


def build_day(day, first: int = 2021) -> list[dict]:
    t0 = time.time()
    g = load_games(first - 1, past_lines=False)
    g = g[g["season_type"].isin(["REG", "POST"])].reset_index(drop=True)
    g["game_id"] = g["game_id"].astype(str)
    box = load_box(g)
    print(f"{len(g)} games, box rows {len(box)} ({time.time() - t0:.0f}s)",
          flush=True)
    h = cp.build(g, box, cp.PredictConfig(), first_season=first, verbose=True)
    train = h[h["completed"] & (h["season"] >= first)
              & (pd.to_datetime(h["game_date"]).dt.date < day)]
    m = cp.fit_models(train, trees=True)
    print(f"fitted on {m.n} games ({time.time() - t0:.0f}s)", flush=True)
    te = h[(pd.to_datetime(h["game_date"]).dt.date == day) & ~h["completed"]]
    te = te[~te["vs_non_d1"].astype(bool)].copy()
    if te.empty:
        return []
    pr = cp.predict_frame(m, te, "trees")
    te["pred_margin"], te["pred_total"] = pr["pred_margin"], pr["pred_total"]
    te["tier"] = tier(te)
    meta = tonight.espn_meta(day)
    out = []
    for r in te.sort_values("commence").to_dict("records"):
        gid = r["game_id"]
        pm, pt = _f(r["pred_margin"]), _f(r["pred_total"])
        lh = _f(r["spread_line"])                  # expected home margin
        gap = None if pm is None or lh is None else pm - lh
        b = {"game_id": gid, "home": r["home_team"], "away": r["away_team"],
             "neutral": bool(r["neutral"]), "tier": r["tier"],
             "commence": str(r["commence"]),
             "tip_et": (pd.Timestamp(r["commence"]) - pd.Timedelta(hours=5)
                        ).strftime("%-I:%M %p"),
             "line": {"home_margin": lh, "spread": None if lh is None else -lh,
                      "total": _f(r["total_line"]),
                      "provider": meta.get(gid, {}).get("provider", "")},
             "projection": {"margin": pm, "total": pt,
                            "home_pts": None if pm is None or pt is None
                            else round((pt + pm) / 2, 1),
                            "away_pts": None if pm is None or pt is None
                            else round((pt - pm) / 2, 1)},
             "matchup": {"pace": _f(r.get("pace_hat")),
                         "home_ortg": _f(r.get("home_ortg_hat")),
                         "away_ortg": _f(r.get("away_ortg_hat")),
                         "apm_diff": _f(r.get(f"apm_diff_{K}")),
                         "box_diff": _f(r.get(f"box_diff_{K}")),
                         "same_diff": _f(r.get(f"same_diff_{K}"), 2),
                         "new_diff": _f(r.get(f"new_diff_{K}"), 2),
                         "four_factors": {k: _f(r.get(f"ff_{k}_m"), 3)
                                          for k in ("efg", "tov", "orb",
                                                    "ftr", "3p")}},
             "form": {"home": team_form(g, r["home_team"], r["commence"]),
                      "away": team_form(g, r["away_team"], r["commence"])},
             "schedule": {k: _f(r.get(k)) for k in
                          ("home_rest", "away_rest", "rest_diff", "home_n",
                           "away_n")},
             "a_card": a_card(r["tier"], gap, lh)}
        b["notes"] = notes(b)
        out.append(b)
    print(f"breakdown done: {len(out)} games ({time.time() - t0:.0f}s)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="")
    ap.add_argument("--first", type=int, default=2021)
    ap.add_argument("--out", default="breakdown")
    a, _ = ap.parse_known_args()
    day = (pd.Timestamp(a.date).date() if a.date else
           (datetime.now(timezone.utc) - timedelta(hours=5)).date())
    games = build_day(day, a.first)
    os.makedirs(os.path.join(a.out, "cbb"), exist_ok=True)
    md = to_markdown(day, games)
    with open(os.path.join(a.out, "cbb", f"{day}.md"), "w") as fh:
        fh.write(md)
    with open(os.path.join(a.out, "cbb", f"{day}.json"), "w") as fh:
        json.dump({"date": str(day), "games": games}, fh, indent=1, default=str)
    print(md)


if __name__ == "__main__":
    main()

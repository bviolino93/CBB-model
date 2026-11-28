"""
Audit of the college basketball models for leaks and silent errors.

Five checks, each printed with numbers:

  1. leakage      (a) KenPom archive dated d: does it already include
                      games played on d?  (b) FanMatch: built from the
                      ratings of that morning, or later ones?  (c)
                      truncation test: rebuild every feature with the
                      data cut off at a date and compare with the full
                      build (any difference = the feature saw the future)
  2. neutral      ESPN's neutral flag against the venue: home teams away
                  from their own arena with the flag off (wrongly given
                  home court) and the reverse; how far off each model is
                  in those games
  3. merges       rows lost in every join: ESPN -> KenPom FanMatch and
                  archive, non-D1 pooling (backtest and live), team
                  renames, box scores, missing lines, Odds API -> ESPN
  4. overfitting  in-sample vs out-of-sample RMSE, log-loss of the cover
                  probability and ROI at -110, per test season
  5. calibration  predicted cover probability vs actual cover rate by
                  bucket, out of sample; and the breakdown's 1-5
                  confidence

Reads kp_games.csv (tools/kenpom_backtest.py --out) and the ESPN /
KenPom / box caches.

    python tools/audit.py --games kp_games.csv --box-cache .cache_box
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import traceback
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import cbb_data  # noqa: E402
import cbb_predict as cp  # noqa: E402
from cbb_data import CACHE_DIR, NON_D1, load_games, tier  # noqa: E402
from cbb_players import load_box  # noqa: E402

BE = 110 / 210
T0 = time.time()


def say(msg=""):
    print(msg, flush=True)


def head(t):
    say(f"\n{'=' * 78}\n{t}  ({time.time() - T0:.0f}s)\n{'=' * 78}")


def roi(win: np.ndarray) -> float:
    """ROI per unit risked at -110 (win pays 1, loss costs 1.1)."""
    win = win[np.isfinite(win)]
    if not len(win):
        return np.nan
    return (win.sum() * 1.0 - (1 - win).sum() * 1.1) / (len(win) * 1.1)


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


# ----------------------------------------------------------------------
# 1a/1b. KenPom archive and FanMatch timing
# ----------------------------------------------------------------------
def load_archive_panel() -> pd.DataFrame:
    rows = []
    for p in glob.glob(os.path.join(".cache_kenpom", "archive", "d=*.json")):
        d = os.path.basename(p)[2:12]
        try:
            with open(p) as fh:
                js = json.load(fh)
        except ValueError:
            continue
        recs = js if isinstance(js, list) else next(
            (v for v in js.values() if isinstance(v, list)), [])
        for r in recs:
            if isinstance(r, dict) and "TeamName" in r:
                rows.append((pd.Timestamp(d), r["TeamName"],
                             pd.to_numeric(r.get("AdjEM"), errors="coerce"),
                             pd.to_numeric(r.get("AdjTempo"), errors="coerce")))
    return pd.DataFrame(rows, columns=["date", "team", "em", "tempo"])


def kenpom_timing(kp: pd.DataFrame):
    head("1a. KenPom archive dated d: before or after that day's games?")
    raw = load_archive_panel()
    arc = raw.copy()
    say(f"archive snapshots: {arc['date'].nunique()} dates, {len(arc)} rows")
    if arc.empty:
        return
    arc = arc.sort_values(["team", "date"])
    arc["next_date"] = arc.groupby("team")["date"].shift(-1)
    arc["delta"] = arc.groupby("team")["em"].shift(-1) - arc["em"]
    arc = arc[(arc["next_date"] - arc["date"]).dt.days == 1]
    played = set()
    for c in ("kp_home_name", "kp_away_name"):
        x = kp[["game_date", c]].dropna()
        played |= set(zip(pd.to_datetime(x["game_date"]), x[c]))
    on_d = np.array([(d, t) in played for d, t in zip(arc["date"], arc["team"])])
    on_d1 = np.array([(n, t) in played for n, t in zip(arc["next_date"], arc["team"])])
    ad = arc["delta"].abs().to_numpy()
    for lab, k in (("played on d only", on_d & ~on_d1),
                   ("played on d+1 only", on_d1 & ~on_d),
                   ("played neither day", ~on_d & ~on_d1)):
        say(f"  |AdjEM(d+1) - AdjEM(d)| for teams that {lab:20s}: "
            f"{np.nanmean(ad[k]):.3f}  (n={k.sum()})")
    say("  -> if archive(d) is the pre-game rating for d, the change is big for"
        " 'played on d' and small for 'played on d+1'. The reverse = leak.")

    head("1b. FanMatch prediction: made from the ratings of that morning?")
    full = raw.drop_duplicates(["date", "team"]).set_index(["date", "team"])
    last = raw.sort_values("date").groupby("team").tail(1).set_index("team")
    x = kp[kp["kp_home_name"].notna() & kp["kp_away_name"].notna()].copy()
    x["d"] = pd.to_datetime(x["game_date"])
    home = 1 - x["neutral"].astype(bool).astype(float)

    def implied(day_shift):
        dd = x["d"] + pd.Timedelta(days=day_shift)
        h = full.reindex(pd.MultiIndex.from_arrays([dd, x["kp_home_name"]]))
        a = full.reindex(pd.MultiIndex.from_arrays([dd, x["kp_away_name"]]))
        em = h["em"].to_numpy() - a["em"].to_numpy()
        tp = (h["tempo"].to_numpy() + a["tempo"].to_numpy()) / 2
        return em * tp / 100

    res = {}
    for lab, v in (("archive day before (d-1)", implied(-1)),
                   ("archive on game day (d)", implied(0)),
                   ("archive day after (d+1)", implied(1)),
                   ("archive day +7", implied(7))):
        ok = np.isfinite(v) & x["kp_margin"].notna().to_numpy()
        X = np.column_stack([v[ok], home.to_numpy()[ok], np.ones(ok.sum())])
        b = np.linalg.lstsq(X, x["kp_margin"].to_numpy()[ok], rcond=None)[0]
        err = np.abs(X @ b - x["kp_margin"].to_numpy()[ok]).mean()
        res[lab] = err
        say(f"  FanMatch margin vs margin implied by {lab:26s}: MAE {err:.3f} "
            f"(fitted home court {b[1]:+.2f}, n={ok.sum()})")
    # Same per season against the final archive of that season.
    x["season"] = x["season"].astype(int)
    lh = last.reindex(x["kp_home_name"])
    la = last.reindex(x["kp_away_name"])
    v = ((lh["em"].to_numpy() - la["em"].to_numpy())
         * (lh["tempo"].to_numpy() + la["tempo"].to_numpy()) / 200)
    cur = x["season"] == x["season"].max()
    ok = np.isfinite(v) & cur.to_numpy()
    if ok.sum() > 100:
        X = np.column_stack([v[ok], home.to_numpy()[ok], np.ones(ok.sum())])
        b = np.linalg.lstsq(X, x["kp_margin"].to_numpy()[ok], rcond=None)[0]
        say(f"  (current season vs latest ratings: MAE "
            f"{np.abs(X @ b - x['kp_margin'].to_numpy()[ok]).mean():.3f})")
    say("  -> smallest at d (or d-1) = FanMatch is point-in-time. Smallest at "
        "d+1/+7 = it was rebuilt with later ratings (leak).")
    # Does FanMatch beat the close? A leak would make it beat the market.
    act = x["home_score"] - x["away_score"]
    ok = x["spread_line"].notna()
    say(f"  margin MAE FanMatch {np.abs(x.loc[ok, 'kp_margin'] - act[ok]).mean():.2f} "
        f"vs close {np.abs(x.loc[ok, 'spread_line'] - act[ok]).mean():.2f} "
        f"(n={ok.sum()}); a leaked line would beat the close")


# ----------------------------------------------------------------------
# 1c. Truncation test: features rebuilt with the future cut off
# ----------------------------------------------------------------------
def truncate(g: pd.DataFrame, box: pd.DataFrame, day):
    gt = g[pd.to_datetime(g["game_date"]).dt.date <= day].copy()
    k = pd.to_datetime(gt["game_date"]).dt.date == day
    gt.loc[k, ["home_score", "away_score"]] = np.nan
    gt.loc[k, "completed"] = False
    bt = box[pd.to_datetime(box["game_date"]).dt.date < day]
    return gt, bt


def compare(full: pd.DataFrame, cut: pd.DataFrame, cols, ids) -> pd.DataFrame:
    a = full.set_index("game_id").reindex(ids)
    b = cut.set_index("game_id").reindex(ids)
    out = []
    for c in cols:
        if c not in a or c not in b:
            continue
        x, y = a[c].astype(float).to_numpy(), b[c].astype(float).to_numpy()
        both = np.isfinite(x) & np.isfinite(y)
        nan_diff = np.isfinite(x) != np.isfinite(y)
        diff = both & (np.abs(x - y) > 1e-6)
        out.append((c, int(diff.sum() + nan_diff.sum()), len(ids),
                    float(np.abs(x - y)[both].mean()) if both.any() else 0.0,
                    float(np.abs(x - y)[both].max()) if both.any() else 0.0))
    return pd.DataFrame(out, columns=["feature", "games_differ", "games",
                                      "mean_abs_diff", "max_abs_diff"])


def truncation_trees(g, box, h_full, days, models):
    head("1c. Truncation test, player/trees model (cbb_predict): features "
         "rebuilt with everything from the test day on removed")
    cols = sorted(set(cp.MARGIN_F + cp.TOTAL_F))
    all_rows, pred_rows = [], []
    for day in days:
        gt, bt = truncate(g, box, day)
        ht = cp.build(gt, bt, cp.PredictConfig(), first_season=FIRST)
        ids = h_full.loc[(pd.to_datetime(h_full["game_date"]).dt.date == day)
                         & h_full["completed"] & ~h_full["vs_non_d1"].astype(bool),
                         "game_id"].tolist()
        c = compare(h_full, ht, cols, ids)
        c["day"] = day
        all_rows.append(c)
        m = models.get(int(h_full.loc[h_full["game_id"].isin(ids), "season"].max()))
        if m is not None and ids:
            a = h_full.set_index("game_id").loc[ids]
            b = ht.set_index("game_id").reindex(ids)
            pa, pb = cp.predict_frame(m, a), cp.predict_frame(m, b)
            act = (a["home_score"] - a["away_score"]).to_numpy(float)
            line = a["spread_line"].to_numpy(float)
            for lab, p in (("full build", pa), ("cut-off build", pb)):
                gap = p["pred_margin"].to_numpy(float) - line
                w = np.sign(gap) * np.sign(act - line)
                ok = np.isfinite(w) & (w != 0)
                pred_rows.append((day, lab, len(ids),
                                  np.nanmean(np.abs(p["pred_margin"].to_numpy(float) - act)),
                                  (w[ok] > 0).mean() if ok.any() else np.nan))
            say(f"  {day}: {len(ids)} games, mean |prediction change| "
                f"{np.nanmean(np.abs(pa['pred_margin'] - pb['pred_margin'])):.3f} pts margin, "
                f"{np.nanmean(np.abs(pa['pred_total'] - pb['pred_total'])):.3f} total")
    c = pd.concat(all_rows)
    s = c.groupby("feature").agg(games_differ=("games_differ", "sum"),
                                 games=("games", "sum"),
                                 mean_abs_diff=("mean_abs_diff", "mean"),
                                 max_abs_diff=("max_abs_diff", "max"))
    say(s.sort_values("games_differ", ascending=False).to_string())
    if pred_rows:
        say(pd.DataFrame(pred_rows, columns=["day", "build", "games", "margin MAE",
                                             "ATS win"]).to_string(index=False))
    say("  -> games_differ > 0 means that input used data from the test day "
        "or later.")


def truncation_kenpom(kp, g, box, days):
    head("1d. Truncation test, KenPom-side inputs (form, missing players, "
         "season-to-date matchup / shooting)")
    from kenpom_accuracy import add_inputs
    from kenpom_research import build as kp_build

    def feats(fr, p, gg):
        f = kp_build(fr)
        groups = {}
        f = add_inputs(f, gg, p, groups)
        groups["form"] = [c for c in f.columns
                          if (c[:2] in ("h_", "a_") and c.endswith(("_3", "_5")))
                          or c.startswith(("h_streak", "a_streak"))]
        cols = sorted({c for v in groups.values() for c in v})
        return f, cols

    full, cols = feats(kp, box, g)
    full["game_id"] = full["game_id"].astype(str)
    rows = []
    for day in days:
        d = pd.to_datetime(kp["game_date"]).dt.date
        kt = kp[d <= day].copy()
        k = pd.to_datetime(kt["game_date"]).dt.date == day
        kt.loc[k, ["home_score", "away_score"]] = np.nan
        bt = box[pd.to_datetime(box["game_date"]).dt.date < day]
        gt = g[pd.to_datetime(g["game_date"]).dt.date <= day]
        cut, _ = feats(kt, bt, gt)
        cut["game_id"] = cut["game_id"].astype(str)
        ids = full.loc[pd.to_datetime(full["game_date"]).dt.date == day,
                       "game_id"].tolist()
        c = compare(full, cut, cols, ids)
        c["day"] = day
        rows.append(c)
    c = pd.concat(rows)
    s = c.groupby("feature").agg(games_differ=("games_differ", "sum"),
                                 games=("games", "sum"),
                                 mean_abs_diff=("mean_abs_diff", "mean"))
    bad = s[s["games_differ"] > 0]
    say(f"  {len(s)} inputs checked on {len(days)} days; {len(bad)} differ")
    if len(bad):
        say(bad.sort_values("games_differ", ascending=False).to_string())


def live_roster_debut(g, box):
    head("1e. 'Live roster' players known only because they played (debuts)")
    cfg = cp.PredictConfig()
    av = cp.availability(g, box, cfg)
    q = cp.played_rows(box)
    q = q[q["team"] != NON_D1][["game_id", "team", "season", "player_id"]]
    tgi = cp.team_game_index(g)
    q = q.merge(tgi, on=["game_id", "team", "season"])
    rec = pd.concat([q.assign(n=q["n"] + k) for k in range(1, cfg.roster_games + 1)])
    rec = rec[["team", "season", "n", "player_id"]].merge(tgi, on=["team", "season", "n"])
    known = set(zip(rec["game_id"], rec["team"], rec["player_id"]))
    live = av["l"]
    deb = np.array([(a, b, c) not in known for a, b, c in
                    zip(live["game_id"], live["team"], live["player_id"])])
    tg = live.assign(deb=deb).groupby(["game_id", "team"])["deb"].sum()
    n_tg = len(tg)
    t5 = tgi[tgi["n"] >= 5]
    late = list(zip(t5["game_id"], t5["team"]))
    say(f"  live-roster rows {len(live)}; {deb.sum()} ({deb.mean():.1%}) are "
        f"players added only because the box shows them debuting that game")
    say(f"  team-games with at least one: {(tg > 0).sum()} of {n_tg} "
        f"({(tg > 0).mean():.1%}); after a team's 5th game: "
        f"{(tg[tg.index.isin(late)] > 0).mean():.1%}")
    say("  Live, a debuting player is unknown before tip, so these rows give "
        "the backtest information the live model never has.")


# ----------------------------------------------------------------------
# 2. Neutral sites
# ----------------------------------------------------------------------
def venues() -> pd.DataFrame:
    rows = []
    for p in glob.glob(os.path.join(CACHE_DIR, "days", "*.json")):
        try:
            with open(p) as fh:
                js = json.load(fh)
        except ValueError:
            continue
        for ev in js.get("events", []):
            comp = (ev.get("competitions") or [{}])[0]
            v = comp.get("venue") or {}
            addr = v.get("address") or {}
            rows.append((str(ev.get("id")), str(v.get("id", "")),
                         v.get("fullName", ""), addr.get("city", ""),
                         addr.get("state", ""), bool(comp.get("neutralSite", False))))
    return pd.DataFrame(rows, columns=["game_id", "venue_id", "venue", "city",
                                       "state", "neutral_raw"]).drop_duplicates("game_id")


def neutral_check(g, kp, oos):
    head("2. Neutral sites: ESPN flag vs the venue")
    v = venues()
    say(f"  venues parsed for {len(v)} games; with a venue id "
        f"{(v['venue_id'] != '').mean():.0%}")
    x = g[g["completed"] & ~g["vs_non_d1"]].merge(v, on="game_id", how="left")
    x = x[x["venue_id"].fillna("") != ""]
    home_games = x[~x["neutral"].astype(bool)]
    mode = (home_games.groupby(["season", "home_id"])["venue_id"]
            .agg(lambda s: s.value_counts().index[0]))
    share = (home_games.groupby(["season", "home_id"])["venue_id"]
             .agg(lambda s: s.value_counts().iloc[0] / len(s)))
    key = list(zip(x["season"], x["home_id"]))
    x["home_venue"] = mode.reindex(key).to_numpy()
    x["home_share"] = share.reindex(key).to_numpy()
    at_home = x["venue_id"] == x["home_venue"]
    neu = x["neutral"].astype(bool)
    x["cat"] = np.select(
        [~neu & at_home, ~neu & ~at_home, neu & at_home, neu & ~at_home],
        ["home, own arena", "home flag, OTHER arena", "neutral flag, home's OWN arena",
         "neutral, elsewhere"])
    say("  games by category and season:")
    say(pd.crosstab(x["cat"], x["season"]).to_string())
    # How off is each model in each category? Positive = home did better
    # than predicted.
    k = kp.drop_duplicates("game_id").set_index(kp.drop_duplicates("game_id")["game_id"].astype(str))
    x["kp_margin"] = k["kp_margin"].reindex(x["game_id"]).to_numpy()
    x["own_margin"] = k["model_margin"].reindex(x["game_id"]).to_numpy()
    if oos is not None:
        o = oos.set_index("game_id")
        x["trees_margin"] = o["pred_margin"].reindex(x["game_id"]).to_numpy()
    act = x["home_score"] - x["away_score"]
    rows = []
    for cat, part in x.groupby("cat"):
        a = act[part.index]
        r = {"category": cat, "n": len(part)}
        for c in ("spread_line", "own_margin", "kp_margin", "trees_margin"):
            if c in part:
                ok = part[c].notna()
                r[f"{c} resid"] = round(float((a[ok] - part.loc[ok, c]).mean()), 2)
        rows.append(r)
    say("\n  mean (actual home margin - prediction); a model that gives home "
        "court to a game that has none shows a NEGATIVE number:")
    say(pd.DataFrame(rows).to_string(index=False))
    sus = x[x["cat"] == "home flag, OTHER arena"]
    if len(sus):
        say("\n  most common 'home flag, other arena' venues (could be a second "
            "home arena; check the list):")
        say(sus.assign(t=sus["home_team"] + " @ " + sus["venue"].fillna("") + ", "
                       + sus["city"].fillna(""))["t"].value_counts().head(25).to_string())
        # ATS damage: betting our side in those games
        for c in ("own_margin", "kp_margin", "trees_margin"):
            if c not in sus:
                continue
            gap = sus[c] - sus["spread_line"]
            w = np.sign(gap) * np.sign(act[sus.index] - sus["spread_line"])
            ok = w.notna() & (w != 0)
            hb = ok & (gap > 0)
            say(f"  {c}: bets on the home side there win "
                f"{(w[hb] > 0).mean():.1%} (n={hb.sum()}), away side "
                f"{(w[ok & (gap < 0)] > 0).mean():.1%} (n={(ok & (gap < 0)).sum()})")
    return x[["game_id", "cat"]]


# ----------------------------------------------------------------------
# 3. Merges that drop rows
# ----------------------------------------------------------------------
def merge_check(g_raw, g, kp, box_raw_n, box, kp_full):
    head("3. Rows dropped silently in joins")
    say(f"  load_games: {box_raw_n.get('scoreboard_rows', 0)} raw scoreboard rows; "
        f"kept {len(g)} after dropping preseason/duplicates/same-name games")
    # Non-D1 pooling: real D1 teams pooled?
    pooled = []
    for side in ("home", "away"):
        x = g_raw[[f"{side}_team", f"{side}_id", f"{side}_conf", "season"]]
        x.columns = ["team", "id", "conf", "season"]
        pooled.append(x)
    t = pd.concat(pooled)
    cnt = t.groupby(["season", "team"]).size().rename("games").reset_index()
    small = cnt[cnt["games"] < 10]
    other = cnt[cnt["games"] >= 10].groupby("team")["season"].apply(set)
    small = small.assign(d1_other_season=small["team"].map(
        lambda n: n in other.index))
    say(f"  non-D1 pooling (backtest rule <10 games in the season): "
        f"{len(small)} team-seasons pooled; {small['d1_other_season'].sum()} of them "
        f"are D1 (10+ games) in another season:")
    say(small[small["d1_other_season"]].sort_values(["season", "games"],
                                                     ascending=[True, False])
        .head(30).to_string(index=False))
    # Renames: one ESPN id, two display names -> ratings restart from zero
    ids = t.groupby("id")["team"].apply(lambda s: sorted(set(s)))
    ren = ids[ids.map(len) > 1]
    say(f"\n  ESPN team ids with more than one display name: {len(ren)} (ratings "
        f"and form are keyed by NAME, so a rename starts the team from scratch)")
    for i, names in ren.head(20).items():
        say(f"    id {i}: {names}")
    # Live pooling in the current season (tools/tonight.py patch)
    cur = g_raw["season"].max()
    import tonight
    live = tonight.mark_non_d1(g_raw[g_raw["season"] >= cur - 1])
    lc = live[live["season"] == cur]
    raw_c = g_raw[g_raw["season"] == cur]
    lost = []
    for side in ("home", "away"):
        m = (lc[f"{side}_team"] == NON_D1).to_numpy() & (raw_c[f"{side}_conf"].astype(str) != "").to_numpy()
        lost += list(raw_c.loc[m, f"{side}_team"])
    c = Counter(lost)
    say(f"\n  live {cur} season: {sum(c.values())} team-games pooled as NON_D1 "
        f"(those games get no pick). Teams (with a conference id):")
    for n, k in c.most_common(25):
        say(f"    {n}: {k}")
    # KenPom match
    say("\n  ESPN -> KenPom FanMatch by season (completed D1 games):")
    kf = kp_full[kp_full["completed"].astype(bool) & ~kp_full["vs_non_d1"].astype(bool)
                 & (kp_full["season"] >= 2020)]
    tab = kf.groupby("season").agg(games=("game_id", "size"),
                                   kenpom=("kp_margin", lambda s: s.notna().mean()),
                                   archive=("kp_h_em", lambda s: s.notna().mean()),
                                   line=("spread_line", lambda s: s.notna().mean()))
    say(tab.round(3).to_string())
    miss = kf[kf["kp_margin"].isna()]
    tm = Counter(list(miss["home_team"]) + list(miss["away_team"]))
    say(f"  unmatched D1 games: {len(miss)}; teams in most of them:")
    for n, k in tm.most_common(20):
        say(f"    {n}: {k}")
    say("  unmatched by tier: " + ", ".join(
        f"{t_}: {v:.1%}" for t_, v in kf.assign(tier=tier(kf)).groupby("tier")["kp_margin"]
        .apply(lambda s: s.isna().mean()).items()))
    # Name-map collisions
    nm = pd.concat([kp_full[["season", "kp_home_name", "home_team"]].set_axis(["s", "kp", "espn"], axis=1),
                    kp_full[["season", "kp_away_name", "away_team"]].set_axis(["s", "kp", "espn"], axis=1)]).dropna()
    coll = nm.drop_duplicates().groupby(["s", "espn"])["kp"].nunique()
    say(f"  ESPN names mapped from 2+ KenPom names in a season: {(coll > 1).sum()}")
    coll2 = nm.drop_duplicates().groupby(["s", "kp"])["espn"].nunique()
    say(f"  KenPom names mapped to 2+ ESPN names in a season: {(coll2 > 1).sum()}")
    if (coll2 > 1).any():
        say("    " + "; ".join(f"{s} {k}" for (s, k) in coll2[coll2 > 1].index[:15]))
    # Lines
    d1 = g[g["completed"] & ~g["vs_non_d1"]]
    say("\n  completed D1 games with no spread line (dropped from every ATS test), "
        "by tier:")
    say(d1.assign(tier=tier(d1)).groupby(["season", "tier"])["spread_line"]
        .apply(lambda s: s.isna().mean()).unstack().round(3).to_string())
    # Box scores
    have = set(box["game_id"].astype(str))
    say(f"\n  box scores: {box_raw_n.get('box_rows_raw', 0)} parsed player rows, "
        f"{len(box)} kept after the team-id join "
        f"({box_raw_n.get('box_rows_raw', 0) - len(box)} dropped)")
    say("  D1 games without a box score by season: " + ", ".join(
        f"{s}: {v:.1%}" for s, v in d1.assign(b=~d1["game_id"].astype(str).isin(have))
        .groupby("season")["b"].mean().items()))
    # Odds API -> ESPN (DK / FD lines)
    try:
        snaps = pd.concat([pd.read_csv(f) for f in glob.glob("data/lines/games_*.csv")])
        ev = snaps.drop_duplicates("event_id")
        events = [{"id": r.event_id, "commence_time": r.commence_time,
                   "home_team": r.home_team, "away_team": r.away_team}
                  for r in ev.itertuples()]
        cur_g = g_raw[g_raw["season"] == cur]
        m = cbb_data.match_events(cur_g, events)
        um = ev[~ev["event_id"].isin(set(m))]
        say(f"\n  Odds API events {len(ev)}, matched to ESPN {len(m)}, unmatched "
            f"{len(um)}:")
        for r in um.head(25).itertuples():
            say(f"    {r.commence_time} {r.away_team} @ {r.home_team}")
        dup = Counter(m.values())
        say(f"  ESPN games matched by 2+ Odds API events: "
            f"{sum(1 for v in dup.values() if v > 1)}")
    except Exception as e:      # noqa: BLE001
        say(f"  Odds API check skipped: {e}")


# ----------------------------------------------------------------------
# 4/5. Overfitting and calibration
# ----------------------------------------------------------------------
def metrics(pred, line, act, sigma):
    """RMSE, log-loss of P(cover), ATS win %, ROI at -110 (all games, and
    3+ off the line)."""
    ok = np.isfinite(pred) & np.isfinite(line) & np.isfinite(act) & (act != line)
    pred, line, act = pred[ok], line[ok], act[ok]
    gap = pred - line
    p = norm.cdf(gap / sigma)          # P(result > line)
    y = (act > line).astype(float)
    win = ((gap > 0) == (act > line)).astype(float)[gap != 0]
    big = np.abs(gap) >= 3
    win3 = ((gap > 0) == (act > line)).astype(float)[big]
    return {"n": int(ok.sum()), "rmse": float(np.sqrt(((pred - act) ** 2).mean())),
            "mae": float(np.abs(pred - act).mean()),
            "logloss": logloss(p, y), "ats": float(win.mean()), "roi": roi(win),
            "ats3": float(win3.mean()) if len(win3) else np.nan,
            "roi3": roi(win3), "n3": int(big.sum()),
            "p": p, "y": y, "gap": gap}


def fmt_m(m):
    return (f"n={m['n']:6d} RMSE {m['rmse']:6.2f} MAE {m['mae']:5.2f} "
            f"logloss {m['logloss']:.4f} ATS {m['ats']:.1%} ROI {m['roi']:+.1%} | "
            f"3+ off: {m['ats3']:.1%} ROI {m['roi3']:+.1%} (n={m['n3']})")


def overfit_block(label, rows):
    """rows: list of (season, market, in-sample metrics, oos metrics)."""
    say(f"\n  -- {label} --")
    for season, market, ins, oos in rows:
        say(f"  {season} {market:7s} IN  {fmt_m(ins)}")
        say(f"  {season} {market:7s} OUT {fmt_m(oos)}")


def calib(label, p, y, gap):
    """Bucket by P(the side we bet covers)."""
    q = np.where(gap > 0, p, 1 - p)
    w = np.where(gap > 0, y, 1 - y)
    edges = [0.5, 0.52, 0.54, 0.56, 0.58, 0.62, 1.01]
    say(f"\n  {label}: predicted P(our side covers) vs actual")
    say(f"    {'bucket':12s} {'n':>6s} {'pred':>6s} {'actual':>7s} {'ROI -110':>9s}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (q >= lo) & (q < hi)
        if k.sum() == 0:
            continue
        say(f"    {lo:.2f}-{min(hi, 1):.2f}    {k.sum():6d} {q[k].mean():6.3f} "
            f"{w[k].mean():7.3f} {roi(w[k]):+9.1%}")
    # Brier and a fitted slope: actual = a + b * predicted
    k = np.isfinite(q)
    b = np.polyfit(q[k] - 0.5, w[k] - 0.5, 1)
    say(f"    calibration slope (1 = calibrated, 0 = no information): {b[0]:+.2f}")


def trees_overfit(h):
    head("4. Overfitting: player/trees model (the projection in the daily "
         "breakdown)")
    d = h[h["completed"] & ~h["vs_non_d1"].astype(bool) & (h["season"] >= FIRST)
          & h["model_margin"].notna()].copy()
    seasons = sorted(d["season"].unique())
    tests = [s for s in seasons if (d["season"] < s).sum() >= 8000]
    rows, oos_parts, models = [], [], {}
    store = {"spreads": [], "totals": []}
    for s in tests:
        tr, te = d[d["season"] < s], d[d["season"] == s]
        m = cp.fit_models(tr, trees=True)
        models[s] = m
        pi, po = cp.predict_frame(m, tr), cp.predict_frame(m, te)
        oos_parts.append(pd.DataFrame({"game_id": te["game_id"], "season": s,
                                       "pred_margin": po["pred_margin"],
                                       "pred_total": po["pred_total"]}))
        for market, pc, line, act in (
                ("spreads", "pred_margin", "spread_line", lambda x: x["home_score"] - x["away_score"]),
                ("totals", "pred_total", "total_line", lambda x: x["home_score"] + x["away_score"])):
            ai, ao = act(tr).to_numpy(float), act(te).to_numpy(float)
            sig = float(np.nanstd(ai - pi[pc].to_numpy(float)))
            mi = metrics(pi[pc].to_numpy(float), tr[line].to_numpy(float), ai, sig)
            mo = metrics(po[pc].to_numpy(float), te[line].to_numpy(float), ao, sig)
            rows.append((s, market, mi, mo))
            store[market].append(mo)
        say(f"  season {s} done ({time.time() - T0:.0f}s)")
    overfit_block("trees model, each season fitted on earlier seasons", rows)
    head("5. Calibration: player/trees model, out of sample")
    for market in ("spreads", "totals"):
        p = np.concatenate([m["p"] for m in store[market]])
        y = np.concatenate([m["y"] for m in store[market]])
        gp = np.concatenate([m["gap"] for m in store[market]])
        calib(f"trees {market} (P from normal error, sigma = training RMSE)", p, y, gp)
    oos = pd.concat(oos_parts) if oos_parts else None
    return oos, models


def kenpom_overfit(kp, g, box):
    head("4b. Overfitting: KenPom + all inputs (tools/kenpom_accuracy 'all')")
    from sklearn.ensemble import HistGradientBoostingRegressor
    from kenpom_accuracy import add_inputs
    from kenpom_research import build as kp_build
    fr = kp[kp["completed"].astype(bool) & kp["spread_line"].notna()
            & kp["kp_margin"].notna() & kp["kp_total"].notna()]
    f = kp_build(fr)
    groups = {}
    f = add_inputs(f, g, box, groups)
    groups["form"] = [c for c in f.columns
                      if (c[:2] in ("h_", "a_") and c.endswith(("_3", "_5")))
                      or c.startswith(("h_streak", "a_streak"))]
    groups["rest"] = [c for c in ("home_rest", "away_rest", "min_n", "neutral_f", "month") if c in f]
    groups["pace"] = ["kp_h_t", "kp_a_t", "tempo_diff", "tempo_avg"]
    base = ["kp_margin", "kp_total", "kp_wp", "kp_tempo", "kp_h_em", "kp_a_em",
            "kp_h_oe", "kp_a_oe", "kp_h_de", "kp_a_de", "kp_h_pre_em", "kp_a_pre_em"]
    feats = base + sorted({c for k in ("missing", "form", "rest", "matchup", "pace", "shooting")
                           for c in groups.get(k, [])})
    feats = [c for c in feats if c in f]
    rows_raw, rows_fit = [], []
    store = {"raw spreads": [], "raw totals": [], "fit spreads": [], "fit totals": []}
    for s in sorted(f["season"].unique()):
        if s < 2023:
            continue
        tr, te = f[f["season"] < s], f[f["season"] == s]
        for market, kc, line, act in (("spreads", "kp_margin", "spread_line", "actual_m"),
                                      ("totals", "kp_total", "total_line", "actual_t")):
            ytr = (tr[act] - tr[kc]).to_numpy(float)
            mdl = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.04,
                                                max_leaf_nodes=15, min_samples_leaf=300,
                                                l2_regularization=1.0, random_state=0)
            ok = np.isfinite(ytr)
            mdl.fit(tr[feats].astype(float)[ok], ytr[ok])
            pi = tr[kc].to_numpy(float) + mdl.predict(tr[feats].astype(float))
            po = te[kc].to_numpy(float) + mdl.predict(te[feats].astype(float))
            ai, ao = tr[act].to_numpy(float), te[act].to_numpy(float)
            sig_raw = float(np.nanstd(ai - tr[kc].to_numpy(float)))
            sig_fit = float(np.nanstd(ai - pi))
            ri = metrics(tr[kc].to_numpy(float), tr[line].to_numpy(float), ai, sig_raw)
            ro = metrics(te[kc].to_numpy(float), te[line].to_numpy(float), ao, sig_raw)
            fi = metrics(pi, tr[line].to_numpy(float), ai, sig_fit)
            fo = metrics(po, te[line].to_numpy(float), ao, sig_fit)
            rows_raw.append((s, market, ri, ro))
            rows_fit.append((s, market, fi, fo))
            store[f"raw {market}"].append(ro)
            store[f"fit {market}"].append(fo)
    overfit_block("KenPom FanMatch as published (no fitting: IN = earlier seasons)", rows_raw)
    overfit_block(f"KenPom + {len(feats) - len(base)} inputs, boosted trees on the "
                  "KenPom miss", rows_fit)
    head("5b. Calibration: KenPom, out of sample")
    for k, parts in store.items():
        if parts:
            calib(f"KenPom {k}", np.concatenate([m["p"] for m in parts]),
                  np.concatenate([m["y"] for m in parts]),
                  np.concatenate([m["gap"] for m in parts]))


def confidence_check(h, oos, g):
    head("5c. The breakdown's 1-5 spread confidence, replayed out of sample")
    import breakdown as bd
    d = h.merge(oos[["game_id", "pred_margin", "pred_total"]].rename(
        columns={"pred_margin": "pm", "pred_total": "pt"}), on="game_id")
    d = d[d["spread_line"].notna()]
    # last-10 average margin of each team before each game (any season)
    rows = []
    for side, sgn in (("home", 1), ("away", -1)):
        rows.append(pd.DataFrame({"game_id": g["game_id"], "team": g[f"{side}_team"],
                                  "commence": pd.to_datetime(g["commence"], utc=True),
                                  "m": sgn * (g["home_score"] - g["away_score"]),
                                  "done": g["completed"]}))
    t = pd.concat(rows)
    t = t[t["done"]].sort_values(["team", "commence"])
    t["form"] = t.groupby("team")["m"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    t["fn"] = t.groupby("team").cumcount()
    fi = t.set_index(["game_id", "team"])
    d["tier"] = tier(d)
    conf, win = [], []
    for r in d.itertuples():
        fh = fi.loc[(r.game_id, r.home_team)] if (r.game_id, r.home_team) in fi.index else None
        fa = fi.loc[(r.game_id, r.away_team)] if (r.game_id, r.away_team) in fi.index else None
        form = {"home": {"n": int(fh["fn"]) if fh is not None else 0,
                         "margin": float(fh["form"]) if fh is not None and np.isfinite(fh["form"]) else 0.0},
                "away": {"n": int(fa["fn"]) if fa is not None else 0,
                         "margin": float(fa["form"]) if fa is not None and np.isfinite(fa["form"]) else 0.0}}
        ln = {"home_margin": float(r.spread_line), "spread": -float(r.spread_line),
              "total": float(r.total_line) if np.isfinite(r.total_line) else None}
        gap = r.pm - r.spread_line
        b = {"home": r.home_team, "away": r.away_team, "tier": r.tier,
             "projection": {"margin": float(r.pm), "total": float(r.pt)},
             "line": ln, "form": form,
             "schedule": {"home_rest": r.home_rest if np.isfinite(r.home_rest) else None,
                          "away_rest": r.away_rest if np.isfinite(r.away_rest) else None,
                          "rest_diff": float(r.rest_diff)},
             "a_card": bd.a_card(r.tier, gap, float(r.spread_line)), "kenpom": {}}
        c = bd.assess(b)
        if not c:
            continue
        act = r.home_score - r.away_score
        if act == r.spread_line:
            continue
        home_pick = c["spread"]["side"] == r.home_team
        conf.append(c["spread"]["confidence"])
        win.append(float((act > r.spread_line) == home_pick))
    x = pd.DataFrame({"conf": conf, "win": win})
    say(x.groupby("conf")["win"].agg(n="size", cover="mean")
        .assign(roi=lambda z: [roi(x.loc[x["conf"] == c, "win"].to_numpy()) for c in z.index])
        .round(3).to_string())


FIRST = 2021


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box-cache", default=".cache_box")
    ap.add_argument("--days", default="2025-12-06,2026-01-24,2026-02-28")
    a = ap.parse_args()
    days = [pd.Timestamp(x).date() for x in a.days.split(",")]

    # KenPom joined to ESPN games exactly as tools/kenpom_backtest.py does,
    # but kept BEFORE its filters so unmatched games can be counted.
    from cbb_data import schedule_features
    from cbb_ratings import walk_forward
    from kenpom import load_kenpom
    fr = walk_forward(schedule_features(load_games(2019)), first_season=2020)
    kp_full = load_kenpom(fr, range(2020, int(fr["season"].max()) + 1))
    kp_full["game_id"] = kp_full["game_id"].astype(str)
    kp = kp_full[kp_full["completed"].astype(bool) & ~kp_full["vs_non_d1"].astype(bool)
                 & (kp_full["season"] >= 2020) & kp_full["kp_margin"].notna()].copy()
    say(f"KenPom frame: {len(kp_full)} rows, {len(kp)} completed D1 with KenPom "
        f"({time.time() - T0:.0f}s)")

    # ESPN games, raw (before pooling) and as the models see them
    raw_n = {}
    orig = cbb_data.mark_non_d1
    cbb_data.mark_non_d1 = lambda x, min_games=10: x.assign(vs_non_d1=False)
    try:
        g_raw = load_games(FIRST - 2, past_lines=False)
    finally:
        cbb_data.mark_non_d1 = orig
    raw_n["scoreboard_rows"] = sum(
        len(json.load(open(p)).get("events", []))
        for p in glob.glob(os.path.join(CACHE_DIR, "days", "*.json")))
    g = load_games(FIRST - 1, past_lines=True)
    g = g[g["season_type"].isin(["REG", "POST"])].reset_index(drop=True)
    g["game_id"] = g["game_id"].astype(str)
    box = load_box(g, cache_dir=a.box_cache)
    box["game_id"] = box["game_id"].astype(str)
    box["player_id"] = box["player_id"].astype(str)
    try:
        from cbb_players import parse_box
        n = 0
        for gid in g.loc[g["completed"], "game_id"]:
            p = os.path.join(a.box_cache, "box", f"{gid}.json")
            if os.path.exists(p):
                with open(p) as fh:
                    try:
                        n += len(parse_box(json.load(fh), gid))
                    except Exception:     # noqa: BLE001
                        pass
        raw_n["box_rows_raw"] = n
    except Exception:          # noqa: BLE001
        pass
    say(f"games {len(g)}, box rows {len(box)} ({time.time() - T0:.0f}s)")

    def run(name, fn, *args):
        try:
            return fn(*args)
        except Exception:      # noqa: BLE001
            say(f"\n!! {name} failed:\n{traceback.format_exc()[-1500:]}")
            return None

    run("kenpom timing", kenpom_timing, kp)
    run("live roster", live_roster_debut, g, box)
    h = cp.build(g, box, cp.PredictConfig(), first_season=FIRST, verbose=True)
    h["game_id"] = h["game_id"].astype(str)
    res = run("trees overfit", trees_overfit, h)
    oos, models = res if res else (None, {})
    run("truncation trees", truncation_trees, g, box, h, days, models)
    run("truncation kenpom", truncation_kenpom, kp, g, box, days)
    run("neutral", neutral_check, g, kp, oos)
    run("merges", merge_check, g_raw, g, kp, raw_n, box, kp_full)
    run("kenpom overfit", kenpom_overfit, kp, g, box)
    if oos is not None:
        run("confidence", confidence_check, h, oos, g)
    say(f"\ndone in {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()

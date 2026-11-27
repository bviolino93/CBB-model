import csv
from datetime import date

import pandas as pd

import books
import breakdown_path  # noqa: F401  (adds tools/ to the path)
import breakdown as bd


def write(tmp_path, rows):
    p = tmp_path / "games_2027.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["snapshot_utc", "event_id", "commence_time", "home_team",
                    "away_team", "book", "market", "outcome", "player",
                    "point", "price", "last_update"])
        w.writerows(rows)
    return str(tmp_path)


def test_best_lines_and_pick(tmp_path):
    tip = "2026-11-27T16:00:00Z"
    snap1, snap2 = "2026-11-27T12:30:00Z", "2026-11-27T15:00:00Z"
    ev = ["e1", tip, "Xavier Musketeers", "Virginia Cavaliers"]
    rows = [
        [snap1, *ev, "draftkings", "spreads", "Xavier Musketeers", "", 7.5, -110, ""],
        [snap2, *ev, "draftkings", "spreads", "Xavier Musketeers", "", 6.5, -110, ""],
        [snap2, *ev, "draftkings", "spreads", "Virginia Cavaliers", "", -6.5, -110, ""],
        [snap2, *ev, "fanduel", "spreads", "Xavier Musketeers", "", 6.5, -105, ""],
        [snap2, *ev, "fanduel", "spreads", "Virginia Cavaliers", "", -6.5, -115, ""],
        [snap2, *ev, "draftkings", "totals", "Over", "", 156.5, -110, ""],
        [snap2, *ev, "draftkings", "totals", "Under", "", 156.5, -110, ""],
        [snap2, *ev, "fanduel", "totals", "Over", "", 155.5, -110, ""],
        [snap2, *ev, "fanduel", "totals", "Under", "", 155.5, -110, ""],
        [snap2, *ev, "betmgm", "spreads", "Xavier Musketeers", "", 8.5, -110, ""],
    ]
    d = write(tmp_path, rows)
    g = pd.DataFrame({"game_id": ["g1"], "home_team": ["Xavier Musketeers"],
                      "away_team": ["Virginia Cavaliers"],
                      "commence": [pd.Timestamp(tip)]})
    rec = books.best_lines(g, date(2026, 11, 27), d)["g1"]
    assert rec["spread"]["draftkings"]["home"] == (6.5, -110)     # latest snapshot
    assert "betmgm" not in rec["spread"]
    assert books.pick_best(rec, "spread", "home") == ("FanDuel", 6.5, -105)
    assert books.pick_best(rec, "spread", "away") == ("DraftKings", -6.5, -110)
    assert books.pick_best(rec, "total", "over") == ("FanDuel", 155.5, -110)
    assert books.pick_best(rec, "total", "under") == ("DraftKings", 156.5, -110)
    b = {"home": "Xavier Musketeers", "books": rec,
         "call": {"spread": {"side": "Xavier Musketeers"},
                  "total": {"side": "Over"}}}
    bd.add_best_price(b)
    assert b["call"]["spread"]["book"] == "FanDuel"
    assert b["call"]["total"]["book_line"] == 155.5

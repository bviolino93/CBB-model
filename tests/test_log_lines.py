from datetime import datetime, timedelta, timezone

from log_lines import close_due

NOW = datetime(2026, 11, 14, 23, 50, tzinfo=timezone.utc)


def g(mins):
    return {"commence": NOW + timedelta(minutes=mins), "home_team": "A",
            "away_team": "B"}


def test_close_snapshot_once_per_wave():
    games = [g(10), g(10), g(40), g(200)]
    assert len(close_due(NOW, 30, games, {})) == 2
    # a snapshot 5 minutes ago already covered the games tipping in 10
    st = {"last_close": (NOW - timedelta(minutes=5)).isoformat()}
    assert close_due(NOW, 30, games, st) == []
    # one taken 45 minutes ago did not
    st = {"last_close": (NOW - timedelta(minutes=45)).isoformat()}
    assert len(close_due(NOW, 30, games, st)) == 2

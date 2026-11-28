# New college basketball model

A fresh model, built step by step on the tools already in this repo
(`cbb_data.py`, `cbb_players.py`, `kenpom.py`, `books.py`, `log_lines.py`).
Nothing in here changes the existing model or the daily breakdown.

| File | What it does |
| --- | --- |
| `kenpom_ml.py` | Underdog moneylines: KenPom's record and profit when it picks the market underdog, and on value dogs (KenPom chance above the no-vig price), from ESPN's pregame moneylines (run with tool = kenpom_ml) |
| `kenpom_ml_backtest.py` | Backtest of the underdog moneyline rule (KenPom has the dog above 50%) from 2020: by season, price, win chance, tier, month and venue (workflow "New model: underdog ML backtest") |
| `kenpom_day.py` | How KenPom's FanMatch predictions did against the spread and total on one date (or a range): record overall, by gap to the line and by conference tier, graded against ESPN's pregame line and, where logged, the DraftKings/FanDuel line |

## Run

On GitHub: Actions -> "New model: KenPom day" (needs `KENPOM_API_KEY`).
Locally, with the key set:

    python -m model.kenpom_day --date 2026-11-27
    python -m model.kenpom_day --date 2026-11-20 --to 2026-11-27

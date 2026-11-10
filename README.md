# College basketball model

Men's Division I basketball, built the way the NBA model is: our own line
from scores, tested walk-forward against the closing line before any
money goes on it, and closing-line value on every bet once it does.

Why college might be different from the NBA: 360 teams, most games with
thin betting markets (low-major conferences, early November), so the
closing line is likelier to be wrong in ways our own numbers can see.
The backtest checks that directly by market depth and season stage
instead of assuming it.

| File | What it does |
| --- | --- |
| `cbb_data.py` | Division I games, scores and a pregame spread/total per game from ESPN's public scoreboard (cached per day) and core odds API (past lines); non-D1 opponents pooled; conferences; matching The Odds API's team names to ESPN's |
| `cbb_ratings.py` | Recency-weighted ridge ratings for margin (blowouts capped) and total, refitted every game day on earlier games only: our own line, no market input |
| `backtest.py` | Our line against the close: slope, record at each gap and for the top N gaps a night, split by market depth (high-major / mixed / low-major), season stage and month; a blended line with its weight fitted on earlier seasons |
| `tools/tune_ratings.py` | Grid over half-life, carry-over from last season and the blowout cap, on tuning seasons only |
| `cbb_players.py` | Player box scores for every game from ESPN's game summaries (trimmed, cached per game, time-budgeted fetch) |
| `cbb_predict.py` | Player-based line, no market input: player impact (game-level adjusted plus-minus, carried across transfers), box value per 40, roster continuity, pace and efficiency ratings, four factors, schedule; ridge and boosted trees. Two availability sets: the live roster (what we can know before tip) and who actually played (hindsight, for comparison only) |
| `tools/model_lab.py` | Season-by-season test of each step of the player model against final scores and the closing line, with slices by market depth, season stage and month |
| `log_lines.py` | Snapshots spreads, totals and moneylines from The Odds API into `data/lines/`; `--close` takes the closing line once per wave of tips |

Seasons are labelled by the year they end (2027 = 2026-27), as ESPN does.
`spread_line` is the expected home margin (positive = home favoured).

## Run

    pip install -r requirements.txt
    python -m pytest -q tests
    python cbb_data.py --first 2019             # fill the cache, coverage table
    python backtest.py --first 2019 --split 2024
    ODDS_API_KEY=... python log_lines.py

On GitHub: Actions -> "Player model lab" tests the player-based line; "Backtest vs closing lines" (no secrets; the first
run fills the ESPN cache and may need a second run). "Log college lines"
and "Log closing lines" need the secret `ODDS_API_KEY`; together they
use about 1,400 credits a month in season.

## Not built yet

1. Daily picks with a graded record and closing-line value, and the dashboard

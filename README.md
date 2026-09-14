# Survivor Pool Optimizer

**Live tool: [jakerheingold.ai/survivor](https://jakerheingold.ai/survivor)**

An NFL survivor/eliminator pool picker. Each week you pick one team to win straight-up; lose and you're out. Each team can only be used once per season, and some weeks require two picks instead of one ("double-pick" weeks).

The tool doesn't recommend "highest win probability this week" — it solves for the pick sequence that maximizes probability of surviving the **entire remaining season**, holding strong teams in reserve for weeks where they're needed more and spending weaker teams on their best-relative-matchup weeks.

## Core concept: an assignment problem, not a greedy pick

Maximizing `P(survive entire season) = product of P(win) across all weekly picks` is equivalent (via log) to maximizing `sum of log(P(win))` across all picks, subject to each team being usable only once. That reframes it as a classic **linear assignment problem** — teams on one side of a bipartite graph, week-slots on the other, edge weight `log(win probability)`. Solved with the **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment` server-side for the initial full-season plan; a JS port runs client-side so re-optimization is instant every time you log a pick, with no server round-trip).

Double-pick weeks get two slot-columns instead of one. Bye weeks and already-played games get a large cost penalty so they're never selected.

## Data pipeline

Two free sources, no API key required:

1. **[nflverse `games.csv`](https://github.com/nflverse/nfldata)** — full NFL schedule including Vegas moneylines/spreads once a market opens for a game.
2. **[nfelo power ratings](https://www.nfeloapp.com/nfl-power-ratings/)** — a free, continuously-updated Elo rating per team with a QB-specific adjustment baked in (captures backup-QB situations, injuries to a starter, etc). No formal API, so this is scraped.

For each game: if Vegas moneylines are available, convert to implied probability and de-vig. Otherwise, fall back to Elo (`nfelo rating + QB adjustment`, standard logistic win probability, +48 point home-field advantage unless neutral site). Near-term weeks get market-accurate numbers; far-future weeks get a reasonable Elo estimate that improves as lines open up.

## How it runs

A GitHub Actions workflow re-runs the full pipeline automatically twice a week — Thursday and Saturday mornings — re-fetching both data sources fresh each time, and publishes the result to the live site. Every run is archived, so the site has a full history of how the model's recommended plan evolved week over week as odds moved and games were played.

## Files

- `build_model.py` — the full pipeline: fetches live data, computes win probabilities, builds the `team_week` lookup, runs the Hungarian assignment for the full-season baseline, writes the data JSON.
- `template.html` — the frontend (vanilla HTML/CSS/JS, no build step, no frameworks). Persists your logged picks in `localStorage`.
- `requirements.txt` — Python deps (`requests`, `beautifulsoup4`, `scipy`, `numpy`).

## Known limitation: injury/news lag

A breaking injury (e.g. a starting QB ruled out) only gets reflected once nfelo's own QB-adjustment number updates, which can lag real news by a day or two. A natural v2: a small manual-override table (`{team, week, adjustment}`) applied as a probability haircut before the optimizer runs, so real-time news can be incorporated immediately rather than waiting on the next scheduled data refresh.

---

Originally prototyped as a Claude.ai artifact, then rebuilt into this automated, hosted version with Claude Code.

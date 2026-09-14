"""
Survivor Pool Optimizer — model pipeline.

Fetches live NFL schedule/odds (nflverse) and power ratings (nfelo),
computes per-team-per-week win probabilities, solves the full-season
optimal pick assignment (Hungarian algorithm), renders the static
frontend, and writes a timestamped history snapshot.

Run manually:  python3 build_model.py
Run by CI:     invoked on a schedule by .github/workflows/survivor-update.yml
"""
import csv
import io
import json
import math
import os
import sys
from datetime import datetime, timezone

import requests
from scipy.optimize import linear_sum_assignment
import numpy as np

# ---- League-specific config (bump SEASON each year; edit DOUBLE_WEEKS to match your pool) ----
SEASON = 2026
WEEKS = list(range(1, 19))
DOUBLE_WEEKS = {6, 11, 12, 13, 16, 17, 18}

NFLVERSE_GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
# nfelo's own automated model output (same data that powers nfeloapp.com, but this
# updates ahead of the public site and is validated by season/week instead of scraped HTML).
NFELO_RATINGS_URL = "https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/elo_snapshot.csv"
UA = "Mozilla/5.0 (compatible; survivor-pool-optimizer/1.0; +https://jakerheingold.ai/survivor)"

# nfelo's team abbreviations that differ from nflverse's
TEAM_ALIASES = {"LAR": "LA", "OAK": "LV"}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PICKS_PATH = os.path.join(BASE_DIR, "picks.json")


def load_logged_picks():
    """The season's actual picks, checked into picks.json so they're baked into every
    build and visible to anyone loading the page — not just whoever's browser logged
    them. Update this file (week -> {picked: [...], recommended: [...]}) and rerun the
    pipeline whenever a real pick is made."""
    try:
        with open(PICKS_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def fetch_games():
    resp = requests.get(NFLVERSE_GAMES_URL, headers={"User-Agent": UA}, timeout=30)
    resp.raise_for_status()
    reader = csv.DictReader(io.StringIO(resp.text))
    return [r for r in reader if r["season"] == str(SEASON)]


def fetch_nfelo_ratings():
    resp = requests.get(NFELO_RATINGS_URL, headers={"User-Agent": UA}, timeout=30)
    resp.raise_for_status()
    reader = csv.DictReader(io.StringIO(resp.text))

    ratings = {}
    seasons_seen = set()
    for row in reader:
        seasons_seen.add(row["season"])
        abbr = TEAM_ALIASES.get(row["team"], row["team"])
        ratings[abbr] = (float(row["nfelo_base"]), float(row["qb_adj"]))

    if len(ratings) < 32:
        raise RuntimeError(f"Only parsed {len(ratings)}/32 teams from nfelo's elo_snapshot.csv — format may have changed.")
    if str(SEASON) not in seasons_seen:
        raise RuntimeError(
            f"nfelo's elo_snapshot.csv is on season(s) {seasons_seen}, not {SEASON} — "
            "their pipeline is behind and these ratings would be stale. Refusing to use them."
        )
    return ratings


def american_to_prob(ml):
    ml = float(ml)
    return -ml / (-ml + 100) if ml < 0 else 100 / (ml + 100)


def build_games(rows, ratings):
    def eff(team):
        r, q = ratings[team]
        return r + q

    def elo_prob(home, away, neutral):
        hfa = 0 if neutral else 48
        diff = (eff(home) + hfa) - eff(away)
        return 1 / (1 + 10 ** (-diff / 400))

    games = []
    for r in rows:
        week = int(r["week"])
        home, away = r["home_team"], r["away_team"]
        neutral = r["location"] == "Neutral"
        home_score, away_score = r["home_score"], r["away_score"]
        played = home_score not in (None, "") and away_score not in (None, "")

        ml_h, ml_a = r["home_moneyline"], r["away_moneyline"]
        if ml_h and ml_a:
            ph_raw, pa_raw = american_to_prob(ml_h), american_to_prob(ml_a)
            p_home = ph_raw / (ph_raw + pa_raw)
            source = "vegas"
        else:
            p_home = elo_prob(home, away, neutral)
            source = "elo"

        games.append({
            "week": week, "home": home, "away": away, "neutral": neutral,
            "p_home": round(p_home, 4), "source": source, "played": played,
            "home_score": home_score, "away_score": away_score, "gameday": r["gameday"],
        })
    return games


def build_team_week(games):
    team_week = {}
    for g in games:
        team_week[(g["home"], g["week"])] = {
            "opp": g["away"], "home": True, "p": g["p_home"], "played": g["played"],
            "won": bool(g["played"] and g["home_score"] and int(g["home_score"]) > int(g["away_score"])),
            "neutral": g["neutral"],
        }
        team_week[(g["away"], g["week"])] = {
            "opp": g["home"], "home": False, "p": round(1 - g["p_home"], 4), "played": g["played"],
            "won": bool(g["played"] and g["away_score"] and int(g["away_score"]) > int(g["home_score"])),
            "neutral": g["neutral"],
        }
    return team_week


def solve_full_season(teams, team_week):
    slot_cols = []
    for w in WEEKS:
        slot_cols.append((w, 0))
        if w in DOUBLE_WEEKS:
            slot_cols.append((w, 1))

    BYE_PENALTY = -50
    cost = np.zeros((len(teams), len(slot_cols)))
    for i, t in enumerate(teams):
        for j, (w, _) in enumerate(slot_cols):
            info = team_week.get((t, w))
            cost[i, j] = BYE_PENALTY if info is None else math.log(min(max(info["p"], 0.001), 0.999))

    row_ind, col_ind = linear_sum_assignment(-cost)
    assignment = {}
    for r_i, c_i in zip(row_ind, col_ind):
        if cost[r_i, c_i] > BYE_PENALTY + 1:
            w, _ = slot_cols[c_i]
            t = teams[r_i]
            assignment.setdefault(w, []).append({"team": t, "p": team_week[(t, w)]["p"]})
    return assignment


def compute_current_week(games):
    for w in WEEKS:
        week_games = [g for g in games if g["week"] == w]
        if week_games and any(not g["played"] for g in week_games):
            return w
    return WEEKS[-1]


def render_page(template_path, data, out_path):
    with open(template_path) as f:
        template = f.read()
    html = template.replace("__DATA_JSON__", json.dumps(data))
    with open(out_path, "w") as f:
        f.write(html)


def run_label():
    override = os.environ.get("SURVIVOR_RUN_LABEL")
    if override:
        return override
    wd = datetime.now(timezone.utc).weekday()  # Mon=0 .. Sun=6
    return {3: "thu", 5: "sat"}.get(wd, "manual")


def main():
    print("Fetching nflverse games...", file=sys.stderr)
    rows = fetch_games()
    print(f"  {len(rows)} games for season {SEASON}", file=sys.stderr)

    print("Scraping nfelo power ratings...", file=sys.stderr)
    ratings = fetch_nfelo_ratings()
    print(f"  {len(ratings)} teams", file=sys.stderr)

    games = build_games(rows, ratings)
    team_week = build_team_week(games)
    teams = sorted(ratings.keys())
    current_week = compute_current_week(games)
    optimal_assignment = solve_full_season(teams, team_week)

    now = datetime.now(timezone.utc)
    label = run_label()

    data = {
        "generated": now.strftime("%Y-%m-%d"),
        "generated_at": now.isoformat(),
        "run_label": label,
        "season": SEASON,
        "current_week": current_week,
        "ratings": {t: {"elo": ratings[t][0], "qb_adj": ratings[t][1]} for t in teams},
        "double_weeks": sorted(DOUBLE_WEEKS),
        "games": games,
        "team_week": {f"{t}|{w}": v for (t, w), v in team_week.items()},
        "optimal_assignment": optimal_assignment,
        "logged_picks": load_logged_picks(),
    }

    template_path = os.path.join(BASE_DIR, "template.html")
    out_path = os.path.join(BASE_DIR, "index.html")
    render_page(template_path, data, out_path)
    print(f"Wrote {out_path}", file=sys.stderr)

    history_dir = os.path.join(BASE_DIR, "history")
    os.makedirs(history_dir, exist_ok=True)
    snapshot_id = f"{SEASON}-w{current_week:02d}-{label}-{now.strftime('%Y%m%d')}"
    snapshot_path = os.path.join(history_dir, f"{snapshot_id}.json")
    with open(snapshot_path, "w") as f:
        json.dump(data, f)
    print(f"Wrote {snapshot_path}", file=sys.stderr)

    index_path = os.path.join(history_dir, "index.json")
    try:
        with open(index_path) as f:
            index = json.load(f)
    except FileNotFoundError:
        index = []
    index = [e for e in index if e["id"] != snapshot_id]
    index.append({
        "id": snapshot_id,
        "date": data["generated"],
        "generated_at": data["generated_at"],
        "run_label": label,
        "week": current_week,
        "file": f"history/{snapshot_id}.json",
    })
    index.sort(key=lambda e: e["generated_at"])
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"Updated {index_path} ({len(index)} runs)", file=sys.stderr)


if __name__ == "__main__":
    main()

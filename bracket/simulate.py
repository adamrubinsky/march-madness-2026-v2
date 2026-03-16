#!/usr/bin/env python3
"""
2026 NCAA Tournament Bracket Simulation Engine

Loads bracket_2026.json and Torvik stats, uses the trained logistic model
to simulate every game from R64 through Championship. Supports NAME_MAP
for bracket-to-Torvik name mapping.

Regional sites: East=DC, South=Houston, West=San Jose, Midwest=Chicago
Final Four: Indianapolis
"""

import json
import os
import pickle
import sys
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
BRACKET_DIR = Path(__file__).parent

# ── Bracket name → Torvik name mapping ──
# Torvik data uses specific name formats that may differ from bracket names
NAME_MAP: dict[str, str] = {
    "UConn": "Connecticut",
    "Ohio St.": "Ohio St.",
    "Michigan St.": "Michigan St.",
    "St. John's": "St. John's",
    "Miami (FL)": "Miami FL",
    "Miami (OH)": "Miami OH",
    "UNI": "Northern Iowa",
    "North Dakota St.": "North Dakota St.",
    "Utah St.": "Utah St.",
    "Iowa St.": "Iowa St.",
    "Texas Tech": "Texas Tech",
    "Texas A&M": "Texas A&M",
    "NC State": "N.C. State",
    "Kennesaw St.": "Kennesaw St.",
    "Wright St.": "Wright St.",
    "Tennessee St.": "Tennessee St.",
    "South Florida": "South Florida",
    "Saint Mary's": "Saint Mary's",
    "Saint Louis": "Saint Louis",
    "California Baptist": "Cal Baptist",
    "High Point": "High Point",
    "North Carolina": "North Carolina",
    "LIU": "LIU",
    "UMBC": "UMBC",
    "Queens": "Queens",
    "Prairie View": "Prairie View A&M",
    "SMU": "SMU",
    "UCF": "UCF",
    "BYU": "BYU",
    "VCU": "VCU",
}

# ── Column indices from Torvik data (same as logistic_model.py) ──
IDX_TEAM = 1
IDX_ADJOE = 4
IDX_ADJDE = 6
IDX_BARTHAG = 8
IDX_TEMPO = 44
IDX_EXP_RANK = 43

ROUND_NAMES = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8", "Final Four", "Championship"]

# ── Locked upset overrides ──
# Keyed by (team_a, team_b) tuple → forced P(team_a wins).
# Checked in both directions: (a,b) and (b,a).
# Set value to 1.0 to force team_a to win, 0.0 to force team_b to win.
# Vanderbilt always wins Round of 64 (5-seed upset lock).
# Add more overrides here as picks are finalized.
LOCKED_UPSETS: dict[tuple[str, str], float] = {
    ("Vanderbilt", "VCU"): 1.0,
}


def load_torvik_stats(year: int = 2026) -> dict[str, dict]:
    """Load Torvik stats into a dict keyed by team name."""
    fpath = DATA_DIR / f"torvik_{year}.json"
    with open(fpath) as f:
        raw = json.load(f)
    teams = {}
    for row in raw:
        name = row[IDX_TEAM]
        adjoe = row[IDX_ADJOE]
        adjde = row[IDX_ADJDE]
        teams[name] = {
            "adjoe": adjoe,
            "adjde": adjde,
            "adjem": adjoe - adjde,
            "barthag": row[IDX_BARTHAG],
            "tempo": row[IDX_TEMPO],
            "exp_rank": row[IDX_EXP_RANK],
        }
    return teams


def resolve_name(bracket_name: str, torvik_teams: dict[str, dict]) -> str:
    """Resolve a bracket name to a Torvik name using NAME_MAP, then fuzzy fallback."""
    # Direct match
    if bracket_name in torvik_teams:
        return bracket_name
    # NAME_MAP lookup
    mapped = NAME_MAP.get(bracket_name)
    if mapped and mapped in torvik_teams:
        return mapped
    # Fuzzy: try substring match
    lower = bracket_name.lower()
    for tname in torvik_teams:
        if lower in tname.lower() or tname.lower() in lower:
            return tname
    return bracket_name  # fallback — will use average stats


def get_team_stats(name: str, torvik: dict[str, dict]) -> dict:
    """Get stats for a team, falling back to median if not found."""
    resolved = resolve_name(name, torvik)
    if resolved in torvik:
        return torvik[resolved]
    # Fallback: median stats (roughly a bubble team)
    print(f"  ⚠ No Torvik data for '{name}' — using median stats")
    return {"adjoe": 105.0, "adjde": 105.0, "adjem": 0.0,
            "barthag": 0.5, "tempo": 68.0, "exp_rank": 180.0}


def compute_features(a: dict, b: dict) -> np.ndarray:
    """6-feature diff vector matching logistic_model.py."""
    return np.array([
        a["adjem"] - b["adjem"],
        a["adjoe"] - b["adjoe"],
        a["adjde"] - b["adjde"],
        a["tempo"] - b["tempo"],
        a["barthag"] - b["barthag"],
        a["exp_rank"] - b["exp_rank"],
    ])


def predict_win_prob(model, stats_a: dict, stats_b: dict,
                     seed_a: int, seed_b: int,
                     name_a: str = "", name_b: str = "") -> float:
    """Predict P(team_a wins) with locked overrides, variance regression + seed calibration."""
    # Check locked upset overrides (both directions)
    if (name_a, name_b) in LOCKED_UPSETS:
        return LOCKED_UPSETS[(name_a, name_b)]
    if (name_b, name_a) in LOCKED_UPSETS:
        return 1.0 - LOCKED_UPSETS[(name_b, name_a)]

    features = compute_features(stats_a, stats_b).reshape(1, -1)
    raw = model.predict_proba(features)[0][1]
    # Variance regression: 12% toward 50%
    prob = raw * 0.88 + 0.5 * 0.12
    # Seed-line calibration for close matchups
    calibration = {(5, 12): 0.64, (6, 11): 0.63, (7, 10): 0.61}
    matchup = (min(seed_a, seed_b), max(seed_a, seed_b))
    if matchup in calibration:
        target = calibration[matchup] if seed_a < seed_b else 1 - calibration[matchup]
        prob = 0.7 * prob + 0.3 * target
    return prob


def simulate_game(model, team_a: dict, team_b: dict,
                  torvik: dict, rng: np.random.RandomState) -> dict:
    """Simulate a single game. Returns the winner dict."""
    stats_a = get_team_stats(team_a["name"], torvik)
    stats_b = get_team_stats(team_b["name"], torvik)
    prob_a = predict_win_prob(model, stats_a, stats_b,
                              team_a["seed"], team_b["seed"],
                              team_a["name"], team_b["name"])
    winner = team_a if rng.random() < prob_a else team_b
    return winner, prob_a


def simulate_region(model, region_name: str, matchups: list,
                    torvik: dict, rng: np.random.RandomState) -> list:
    """Simulate a region from R64 through Elite 8. Returns [R64, R32, S16, E8] results."""
    rounds = []
    current = matchups  # list of [team_a, team_b] pairs

    for rd in range(4):  # R64, R32, Sweet 16, Elite 8
        round_results = []
        next_round = []
        for i in range(0, len(current), 2) if rd > 0 else range(len(current)):
            if rd == 0:
                pair = current[i]
                a, b = pair[0], pair[1]
            else:
                a = current[i]
                b = current[i + 1]
            winner, prob = simulate_game(model, a, b, torvik, rng)
            loser = b if winner is a else a
            round_results.append({
                "winner": winner, "loser": loser,
                "win_prob": prob if winner is a else 1 - prob,
            })
            next_round.append(winner)
        rounds.append({"round": ROUND_NAMES[rd], "region": region_name,
                        "games": round_results})
        current = next_round

    return rounds, current[0]  # rounds data + regional champion


def simulate_tournament(model, bracket: dict, torvik: dict,
                        seed: int = 42) -> dict:
    """Simulate the full tournament. Returns structured results."""
    rng = np.random.RandomState(seed)
    all_rounds = []
    regional_champs = {}

    for region_name, region_data in bracket["regions"].items():
        region_rounds, champ = simulate_region(
            model, region_name, region_data["matchups"], torvik, rng)
        all_rounds.extend(region_rounds)
        regional_champs[region_name] = champ
        site = region_data["site"]
        print(f"  {region_name} ({site}): {champ['name']} ({champ['seed']}-seed)")

    # Final Four: East vs West, South vs Midwest (standard NCAA pairing)
    print(f"\n{'='*60}")
    print(f"FINAL FOUR — Indianapolis")
    print(f"{'='*60}")

    ff_games = []
    # Semifinal 1: East vs West
    e_champ = regional_champs["East"]
    w_champ = regional_champs["West"]
    winner1, prob1 = simulate_game(model, e_champ, w_champ, torvik, rng)
    loser1 = w_champ if winner1 is e_champ else e_champ
    ff_games.append({"winner": winner1, "loser": loser1,
                     "win_prob": prob1 if winner1 is e_champ else 1 - prob1})
    print(f"  {e_champ['name']} vs {w_champ['name']} → {winner1['name']} ({prob1:.1%} / {1-prob1:.1%})")

    # Semifinal 2: South vs Midwest
    s_champ = regional_champs["South"]
    mw_champ = regional_champs["Midwest"]
    winner2, prob2 = simulate_game(model, s_champ, mw_champ, torvik, rng)
    loser2 = mw_champ if winner2 is s_champ else s_champ
    ff_games.append({"winner": winner2, "loser": loser2,
                     "win_prob": prob2 if winner2 is s_champ else 1 - prob2})
    print(f"  {s_champ['name']} vs {mw_champ['name']} → {winner2['name']} ({prob2:.1%} / {1-prob2:.1%})")

    all_rounds.append({"round": "Final Four", "region": "Final Four",
                        "games": ff_games})

    # Championship
    print(f"\n{'='*60}")
    print(f"CHAMPIONSHIP")
    print(f"{'='*60}")
    champion, prob_c = simulate_game(model, winner1, winner2, torvik, rng)
    loser_c = winner2 if champion is winner1 else winner1
    champ_game = {"winner": champion, "loser": loser_c,
                  "win_prob": prob_c if champion is winner1 else 1 - prob_c}
    all_rounds.append({"round": "Championship", "region": "Championship",
                        "games": [champ_game]})
    print(f"  {winner1['name']} vs {winner2['name']} → 🏆 {champion['name']} ({champion['seed']}-seed)")

    return {
        "champion": {"name": champion["name"], "seed": champion["seed"]},
        "runner_up": {"name": loser_c["name"], "seed": loser_c["seed"]},
        "final_four": {r: {"name": c["name"], "seed": c["seed"]}
                       for r, c in regional_champs.items()},
        "rounds": all_rounds,
    }


def run_monte_carlo(model, bracket: dict, torvik: dict,
                    n_sims: int = 10000) -> dict:
    """Run N tournament simulations, aggregate win frequencies per round."""
    from collections import Counter

    # Track wins per (region, round_index, game_index) → Counter of team names
    # Also track Final Four appearances, championship appearances, and titles
    round_wins: dict[tuple, Counter] = {}
    final_four_counts: Counter = Counter()
    champion_counts: Counter = Counter()
    runner_up_counts: Counter = Counter()

    for i in range(n_sims):
        # Suppress print output during bulk sims
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            result = simulate_tournament(model, bracket, torvik, seed=i)
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout

        champion_counts[result["champion"]["name"]] += 1
        runner_up_counts[result["runner_up"]["name"]] += 1
        for rname, rdata in result["final_four"].items():
            final_four_counts[rdata["name"]] += 1

        # Tally per-game winners across rounds
        for rd in result["rounds"]:
            round_name = rd["round"]
            region = rd["region"]
            for gi, game in enumerate(rd["games"]):
                key = (region, round_name, gi)
                if key not in round_wins:
                    round_wins[key] = Counter()
                round_wins[key][game["winner"]["name"]] += 1

    # Build consensus bracket: for each game slot, pick the most frequent winner
    # Re-run one sim with seed=0 just to get the bracket structure/matchup labels
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        template = simulate_tournament(model, bracket, torvik, seed=0)
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout

    consensus_rounds = []
    for rd in template["rounds"]:
        round_name = rd["round"]
        region = rd["region"]
        games = []
        for gi, game in enumerate(rd["games"]):
            key = (region, round_name, gi)
            counter = round_wins.get(key, Counter())
            top_team, top_count = counter.most_common(1)[0] if counter else ("???", 0)
            pct = top_count / n_sims
            games.append({
                "consensus_winner": top_team,
                "win_pct": round(pct, 4),
                "matchup": f"{game['winner']['name']} vs {game['loser']['name']}",
                "all_winners": {t: round(c / n_sims, 4) for t, c in counter.most_common()},
            })
        consensus_rounds.append({"round": round_name, "region": region, "games": games})

    top_champ, top_champ_n = champion_counts.most_common(1)[0]
    top_ff = {name: round(cnt / n_sims, 4) for name, cnt in final_four_counts.most_common()}

    mc_results = {
        "n_simulations": n_sims,
        "champion": {"name": top_champ, "win_pct": round(top_champ_n / n_sims, 4)},
        "champion_distribution": {t: round(c / n_sims, 4) for t, c in champion_counts.most_common(10)},
        "final_four_pct": top_ff,
        "consensus_bracket": consensus_rounds,
    }
    return mc_results


def print_monte_carlo_bracket(mc: dict) -> None:
    """Print a full bracket with consensus picks and probabilities."""
    n = mc["n_simulations"]
    print(f"\n{'='*70}")
    print(f"  MONTE CARLO BRACKET — {n:,} simulations")
    print(f"{'='*70}")

    for rd in mc["consensus_bracket"]:
        round_name = rd["round"]
        region = rd["region"]
        header = f"{region} — {round_name}" if region != round_name else round_name
        print(f"\n  ── {header} ──")
        for g in rd["games"]:
            winners = g["all_winners"]
            # Show top 2 contenders
            teams = list(winners.items())
            line = f"    {teams[0][0]} ({teams[0][1]:.0%})"
            if len(teams) > 1:
                line += f"  /  {teams[1][0]} ({teams[1][1]:.0%})"
            print(line)

    print(f"\n{'='*70}")
    print(f"  CHAMPION: {mc['champion']['name']} ({mc['champion']['win_pct']:.1%})")
    print(f"{'='*70}")

    print(f"\n  Top 10 Championship Contenders:")
    for name, pct in mc["champion_distribution"].items():
        bar = "█" * int(pct * 50)
        print(f"    {name:<20s} {pct:6.1%} {bar}")

    print(f"\n  Final Four Appearance Rates:")
    for name, pct in sorted(mc["final_four_pct"].items(), key=lambda x: -x[1])[:10]:
        print(f"    {name:<20s} {pct:6.1%}")


def main():
    # Load bracket
    bracket_path = BRACKET_DIR / "bracket_2026.json"
    with open(bracket_path) as f:
        bracket = json.load(f)

    # Load Torvik stats
    torvik = load_torvik_stats(2026)
    print(f"Loaded {len(torvik)} teams from Torvik 2026 data\n")

    # Load trained model
    model_path = MODEL_DIR / "logistic_model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Check NAME_MAP coverage
    all_bracket_teams = set()
    for region in bracket["regions"].values():
        for pair in region["matchups"]:
            for team in pair:
                all_bracket_teams.add(team["name"])
    missing = [t for t in all_bracket_teams if resolve_name(t, torvik) not in torvik]
    if missing:
        print(f"⚠ Teams without Torvik data: {missing}\n")

    # Monte Carlo mode
    if "--monte-carlo" in sys.argv:
        n = 10000
        for i, arg in enumerate(sys.argv):
            if arg == "--sims" and i + 1 < len(sys.argv):
                n = int(sys.argv[i + 1])
        print(f"Running {n:,} Monte Carlo simulations...")
        mc = run_monte_carlo(model, bracket, torvik, n_sims=n)
        print_monte_carlo_bracket(mc)
        # Save results
        out = BRACKET_DIR / "monte_carlo_results.json"
        with open(out, "w") as f:
            json.dump(mc, f, indent=2)
        print(f"\nResults saved to {out}")
        return

    # Single simulation mode (original behavior)
    print("=" * 60)
    print("2026 NCAA TOURNAMENT SIMULATION")
    print("=" * 60)
    print("\nRegional Champions:")
    results = simulate_tournament(model, bracket, torvik, seed=42)

    # Save results
    output_path = BRACKET_DIR / "simulation_results.json"
    serializable = {
        "champion": results["champion"],
        "runner_up": results["runner_up"],
        "final_four": results["final_four"],
    }
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

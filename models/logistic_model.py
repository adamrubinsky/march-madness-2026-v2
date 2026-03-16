#!/usr/bin/env python3
"""
Logistic Regression Model for March Madness Tournament Prediction

Trains sklearn LogisticRegression on 2008-2025 tournament data using 6 features:
AdjEM_diff, AdjOE_diff, AdjDE_diff, Tempo_diff, Barthag_diff, Exp_diff

Includes:
- 5-fold cross-validation accuracy reporting
- Tournament variance regression (12% toward 50%)
- Historical seed-line calibration for 5-12, 6-11, 7-10 matchups

Data schema (from mm-06h):
  [0]=rank, [1]=team, [2]=conf, [4]=AdjOE, [6]=AdjDE, [8]=Barthag,
  [44]=Tempo, [43]=experience_rank
  AdjEM = AdjOE - AdjDE (derived)
"""

import json
import os
import pickle
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# ── Column indices (from mm-06h schema) ──
IDX_RANK = 0
IDX_TEAM = 1
IDX_ADJOE = 4
IDX_ADJDE = 6
IDX_BARTHAG = 8
IDX_TEMPO = 44
IDX_EXP_RANK = 43

# ── Tournament seed-line matchups (first round) ──
# Each tuple: (higher_seed, lower_seed)
FIRST_ROUND_MATCHUPS = [
    (1, 16), (2, 15), (3, 14), (4, 13),
    (5, 12), (6, 11), (7, 10), (8, 9),
]

# ── Historical upset rates by seed matchup (approximate from NCAA data) ──
# Used to generate realistic training labels
HISTORICAL_UPSET_RATES = {
    (1, 16): 0.01, (2, 15): 0.06, (3, 14): 0.15, (4, 13): 0.20,
    (5, 12): 0.36, (6, 11): 0.37, (7, 10): 0.39, (8, 9): 0.49,
}

# ── Seed-line calibration targets for specific matchups ──
# Historical win rates for the higher seed in these matchups
SEED_CALIBRATION = {
    (5, 12): 0.64,  # 5-seeds win ~64%
    (6, 11): 0.63,  # 6-seeds win ~63%
    (7, 10): 0.61,  # 7-seeds win ~61%
}

# Variance regression factor: regress predictions 12% toward 50%
VARIANCE_REGRESSION = 0.12


def load_team_stats(data_dir: str, year: int) -> dict[str, dict]:
    """Load Torvik data for a year, return dict keyed by team name."""
    fpath = os.path.join(data_dir, f"torvik_{year}.json")
    if not os.path.exists(fpath):
        return {}
    with open(fpath) as f:
        raw = json.load(f)
    teams = {}
    for row in raw:
        name = row[IDX_TEAM]
        adjoe = row[IDX_ADJOE]
        adjde = row[IDX_ADJDE]
        teams[name] = {
            "rank": row[IDX_RANK],
            "adjoe": adjoe,
            "adjde": adjde,
            "adjem": adjoe - adjde,
            "barthag": row[IDX_BARTHAG],
            "tempo": row[IDX_TEMPO],
            "exp_rank": row[IDX_EXP_RANK],
        }
    return teams


def approximate_seeds(teams: dict[str, dict]) -> dict[int, list[str]]:
    """
    Approximate tournament seeds from Torvik rankings.
    Top 64 teams by rank → 4 regions × 16 seeds.
    Returns {seed: [team1, team2, team3, team4]} (4 teams per seed line).
    """
    sorted_teams = sorted(teams.items(), key=lambda x: x[1]["rank"])
    # Take top 64 as tournament field
    tourney_teams = sorted_teams[:64]
    seeds = {}
    for i, (name, _) in enumerate(tourney_teams):
        seed = (i // 4) + 1  # 4 teams per seed line
        seeds.setdefault(seed, []).append(name)
    return seeds


def compute_features(team_a: dict, team_b: dict) -> np.ndarray:
    """Compute 6 diff features: AdjEM, AdjOE, AdjDE, Tempo, Barthag, Exp."""
    return np.array([
        team_a["adjem"] - team_b["adjem"],
        team_a["adjoe"] - team_b["adjoe"],
        team_a["adjde"] - team_b["adjde"],  # lower DE is better, so diff sign matters
        team_a["tempo"] - team_b["tempo"],
        team_a["barthag"] - team_b["barthag"],
        team_a["exp_rank"] - team_b["exp_rank"],  # lower rank = more experienced
    ])


def log5_probability(barthag_a: float, barthag_b: float) -> float:
    """Log5 formula for head-to-head win probability."""
    num = barthag_a * (1 - barthag_b)
    den = num + barthag_b * (1 - barthag_a)
    return num / den if den > 0 else 0.5


def generate_training_data(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate training matchups from 2008-2025 tournament approximations.
    Creates matchups across all tournament rounds (R64 through championship).
    Labels derived from Barthag-based win probability blended with historical upset rates.
    Also generates cross-seed pairwise matchups from the top 64 teams for richer training.
    """
    X_all, y_all = [], []

    for year in range(2008, 2026):
        teams = load_team_stats(data_dir, year)
        if not teams:
            continue
        seeds = approximate_seeds(teams)

        # ── Round of 64 matchups (seed-based) ──
        for high_seed, low_seed in FIRST_ROUND_MATCHUPS:
            if high_seed not in seeds or low_seed not in seeds:
                continue
            high_teams = seeds[high_seed]
            low_teams = seeds[low_seed]
            upset_rate = HISTORICAL_UPSET_RATES.get((high_seed, low_seed), 0.5)

            for h_name, l_name in zip(high_teams, low_teams):
                h_stats, l_stats = teams[h_name], teams[l_name]
                p_win = log5_probability(h_stats["barthag"], l_stats["barthag"])
                blended_p = 0.6 * p_win + 0.4 * (1 - upset_rate)

                rng = np.random.RandomState(hash((year, high_seed, low_seed, h_name)) % (2**31))
                outcome = 1 if rng.random() < blended_p else 0

                features = compute_features(h_stats, l_stats)
                X_all.append(features)
                y_all.append(outcome)
                # Symmetric augmentation
                X_all.append(-features)
                y_all.append(1 - outcome)

        # ── Additional pairwise matchups from top-64 teams ──
        # Sample random matchups between tournament-caliber teams for richer training
        sorted_teams = sorted(teams.items(), key=lambda x: x[1]["rank"])[:64]
        rng_year = np.random.RandomState(year)
        n_extra = 60  # extra matchups per year
        for _ in range(n_extra):
            i, j = rng_year.choice(len(sorted_teams), size=2, replace=False)
            name_a, stats_a = sorted_teams[i]
            name_b, stats_b = sorted_teams[j]
            p_win = log5_probability(stats_a["barthag"], stats_b["barthag"])
            outcome = 1 if rng_year.random() < p_win else 0

            features = compute_features(stats_a, stats_b)
            X_all.append(features)
            y_all.append(outcome)
            X_all.append(-features)
            y_all.append(1 - outcome)

    return np.array(X_all), np.array(y_all)


def apply_variance_regression(prob: float, factor: float = VARIANCE_REGRESSION) -> float:
    """Regress probability toward 50% by the given factor."""
    return prob * (1 - factor) + 0.5 * factor


def apply_seed_calibration(
    prob: float, seed_a: int, seed_b: int
) -> float:
    """
    Calibrate probability for specific seed-line matchups (5-12, 6-11, 7-10).
    Blends model probability with historical seed-line win rate.
    """
    matchup = (min(seed_a, seed_b), max(seed_a, seed_b))
    if matchup not in SEED_CALIBRATION:
        return prob
    historical_rate = SEED_CALIBRATION[matchup]
    # If seed_a is the higher seed (lower number), use historical rate directly
    # Otherwise flip it
    if seed_a < seed_b:
        target = historical_rate
    else:
        target = 1 - historical_rate
    # Blend: 70% model, 30% historical calibration
    return 0.7 * prob + 0.3 * target


def train_model(data_dir: str) -> tuple[LogisticRegression, dict]:
    """
    Train logistic regression model and return model + metrics.
    """
    X, y = generate_training_data(data_dir)
    print(f"Training samples: {len(X)} ({np.sum(y)} wins, {len(y) - np.sum(y)} losses)")
    print(f"Features: AdjEM_diff, AdjOE_diff, AdjDE_diff, Tempo_diff, Barthag_diff, Exp_diff")

    # Train logistic regression
    model = LogisticRegression(max_iter=1000, random_state=42)

    # 5-fold cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"\n5-Fold Cross-Validation:")
    for i, score in enumerate(cv_scores):
        print(f"  Fold {i+1}: {score:.4f}")
    print(f"  Mean:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Fit on full dataset
    model.fit(X, y)

    # Report coefficients
    feature_names = ["AdjEM_diff", "AdjOE_diff", "AdjDE_diff", "Tempo_diff", "Barthag_diff", "Exp_diff"]
    print(f"\nModel Coefficients:")
    for name, coef in zip(feature_names, model.coef_[0]):
        print(f"  {name:15s}: {coef:+.6f}")
    print(f"  {'Intercept':15s}: {model.intercept_[0]:+.6f}")

    # Training accuracy
    train_acc = model.score(X, y)
    print(f"\nTraining Accuracy: {train_acc:.4f}")

    metrics = {
        "cv_scores": cv_scores.tolist(),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "train_accuracy": float(train_acc),
        "coefficients": dict(zip(feature_names, model.coef_[0].tolist())),
        "intercept": float(model.intercept_[0]),
        "n_samples": len(X),
    }

    return model, metrics


def predict_matchup(
    model: LogisticRegression,
    team_a: dict,
    team_b: dict,
    seed_a: int | None = None,
    seed_b: int | None = None,
) -> float:
    """
    Predict win probability for team_a over team_b.
    Applies variance regression and optional seed-line calibration.
    """
    features = compute_features(team_a, team_b).reshape(1, -1)
    raw_prob = model.predict_proba(features)[0][1]

    # Apply tournament variance regression (12% toward 50%)
    prob = apply_variance_regression(raw_prob)

    # Apply seed-line calibration if seeds provided
    if seed_a is not None and seed_b is not None:
        prob = apply_seed_calibration(prob, seed_a, seed_b)

    return prob


def main():
    """Train model, report metrics, save artifacts."""
    base_dir = Path(__file__).parent.parent
    data_dir = str(base_dir / "data")
    model_dir = Path(__file__).parent

    print("=" * 70)
    print("LOGISTIC REGRESSION MODEL: MARCH MADNESS TOURNAMENT PREDICTION")
    print("=" * 70)
    print()

    model, metrics = train_model(data_dir)

    # Save model
    model_path = model_dir / "logistic_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved: {model_path}")

    # Save metrics
    metrics_path = model_dir / "model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {metrics_path}")

    # Demo: predict a sample matchup using 2026 data
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS (2026 data)")
    print("=" * 70)
    teams_2026 = load_team_stats(data_dir, 2026)
    if teams_2026:
        # Show predictions for classic seed-line matchups
        seeds_2026 = approximate_seeds(teams_2026)
        for high_seed, low_seed in [(1, 16), (5, 12), (6, 11), (7, 10), (8, 9)]:
            if high_seed in seeds_2026 and low_seed in seeds_2026:
                h_name = seeds_2026[high_seed][0]
                l_name = seeds_2026[low_seed][0]
                h_stats = teams_2026[h_name]
                l_stats = teams_2026[l_name]
                prob = predict_matchup(model, h_stats, l_stats, high_seed, low_seed)
                print(f"  #{high_seed} {h_name:20s} vs #{low_seed} {l_name:20s} → {h_name} win: {prob:.1%}")

    # Report variance regression and calibration effects
    print("\n" + "=" * 70)
    print("VARIANCE REGRESSION & CALIBRATION")
    print("=" * 70)
    print(f"  Variance regression: {VARIANCE_REGRESSION:.0%} toward 50%")
    print(f"  Seed-line calibration targets:")
    for matchup, rate in SEED_CALIBRATION.items():
        print(f"    {matchup[0]}-seed vs {matchup[1]}-seed: higher seed wins {rate:.0%}")


if __name__ == "__main__":
    main()

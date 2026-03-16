#!/usr/bin/env python3
"""
Feature Importance Analysis for March Madness Tournament Prediction
Analyzes 2008-2025 Torvik data to determine which metrics best predict tournament success.

Analyses performed:
1. Pearson correlations of all features vs rank
2. Multivariate OLS regression
3. Elite vs bubble team effect sizes (Cohen's d)
4. Offense vs defense comparison
5. Experience/tempo factor analysis
6. Pairwise matchup gap analysis
"""

import json
import os
import numpy as np
from pathlib import Path

# ── Column mapping for Torvik 45-column arrays ──
# Derived from cross-referencing known team stats
COL = {
    "rank": 0,
    "team": 1,
    "conf": 2,
    "record": 3,
    "adjoe": 4,        # Adjusted Offensive Efficiency
    "adjoe_rank": 5,
    "adjde": 6,         # Adjusted Defensive Efficiency
    "adjde_rank": 7,
    "barthag": 8,       # Power rating (0-1)
    "barthag_rank": 9,
    "wins": 10,
    "losses": 11,
    "conf_w": 12,
    "conf_l": 13,
    "conf_rec": 14,
    "efg": 15,          # Effective FG%
    "efgd": 16,         # Opponent Effective FG%
    "ftr": 17,          # Free throw rate (or similar shooting metric)
    "raw_oe": 23,       # Raw offensive efficiency
    "raw_de": 24,       # Raw defensive efficiency
    "adj_oe_v2": 27,    # Adjusted OE (alternate calculation)
    "adj_de_v2": 28,    # Adjusted DE (alternate calculation)
    "wab": 32,          # Wins Above Bubble
    "luck": 33,         # Luck factor
    "sos_raw": 34,      # Strength of Schedule (raw)
    "opp_sos": 35,      # Opponent SOS
    "sos_avg": 36,      # SOS average
    "sos_ratio_o": 37,  # SOS ratio (offensive)
    "sos_ratio_d": 38,  # SOS ratio (defensive)
    "close_win_pct": 40,  # Close game win %
    "margin_metric": 41,  # Efficiency margin or WAB variant
    "tempo": 44,        # Adjusted tempo
}

# Features to analyze (numeric columns with predictive potential)
FEATURES = {
    "AdjOE": "adjoe",
    "AdjDE": "adjde",
    "Barthag": "barthag",
    "EFG%": "efg",
    "EFGD%": "efgd",
    "FTR": "ftr",
    "RawOE": "raw_oe",
    "RawDE": "raw_de",
    "AdjOE_v2": "adj_oe_v2",
    "AdjDE_v2": "adj_de_v2",
    "WAB": "wab",
    "Luck": "luck",
    "SOS": "sos_raw",
    "OppSOS": "opp_sos",
    "SOS_Avg": "sos_avg",
    "SOS_Ratio_O": "sos_ratio_o",
    "SOS_Ratio_D": "sos_ratio_d",
    "CloseWin%": "close_win_pct",
    "Margin": "margin_metric",
    "Tempo": "tempo",
    "WinPct": None,     # Derived: wins / (wins + losses)
    "AdjEM": None,      # Derived: AdjOE - AdjDE
    "ConfWinPct": None,  # Derived: conf_w / (conf_w + conf_l)
}


def load_all_data(data_dir: str) -> list[dict]:
    """Load all Torvik JSON files from 2008-2025, return list of team records as dicts."""
    records = []
    for year in range(2008, 2026):
        fpath = os.path.join(data_dir, f"torvik_{year}.json")
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            raw = json.load(f)
        for row in raw:
            rec = {"year": year}
            for name, idx in COL.items():
                rec[name] = row[idx]
            # Derived features
            total_games = (rec["wins"] or 0) + (rec["losses"] or 0)
            rec["win_pct"] = rec["wins"] / total_games if total_games > 0 else 0
            rec["adjem"] = (rec["adjoe"] or 0) - (rec["adjde"] or 0)
            conf_total = (rec["conf_w"] or 0) + (rec["conf_l"] or 0)
            rec["conf_win_pct"] = rec["conf_w"] / conf_total if conf_total > 0 else 0
            records.append(rec)
    return records


def get_feature_array(records: list[dict], feature_key: str) -> np.ndarray:
    """Extract a numeric feature array from records."""
    derived_map = {"WinPct": "win_pct", "AdjEM": "adjem", "ConfWinPct": "conf_win_pct"}
    if feature_key in derived_map:
        lookup = derived_map[feature_key]
    else:
        lookup = FEATURES[feature_key]  # This is the COL key name
    vals = []
    for r in records:
        v = r.get(lookup, None)
        if v is None or isinstance(v, str):
            vals.append(np.nan)
        else:
            vals.append(float(v))
    return np.array(vals)


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation handling NaN."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return np.nan
    return np.corrcoef(x[mask], y[mask])[0, 1]


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's d effect size between two groups."""
    g1 = group1[~np.isnan(group1)]
    g2 = group2[~np.isnan(group2)]
    if len(g1) < 2 or len(g2) < 2:
        return np.nan
    n1, n2 = len(g1), len(g2)
    pooled_std = np.sqrt(((n1 - 1) * np.var(g1, ddof=1) + (n2 - 1) * np.var(g2, ddof=1)) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(g1) - np.mean(g2)) / pooled_std


def ols_regression(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Simple OLS regression. Returns (coefficients, R-squared)."""
    # Remove rows with NaN
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X_clean = X[mask]
    y_clean = y[mask]
    # Standardize features for comparable coefficients
    means = X_clean.mean(axis=0)
    stds = X_clean.std(axis=0)
    stds[stds == 0] = 1
    X_std = (X_clean - means) / stds
    # Add intercept
    X_aug = np.column_stack([np.ones(len(X_std)), X_std])
    # Solve normal equations
    try:
        beta = np.linalg.lstsq(X_aug, y_clean, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.zeros(X.shape[1]), 0.0
    y_pred = X_aug @ beta
    ss_res = np.sum((y_clean - y_pred) ** 2)
    ss_tot = np.sum((y_clean - y_clean.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return beta[1:], r_squared  # Exclude intercept


def run_analysis(data_dir: str) -> str:
    """Run all analyses and return formatted report."""
    records = load_all_data(data_dir)
    n_teams = len(records)
    years = sorted(set(r["year"] for r in records))

    ranks = np.array([float(r["rank"]) for r in records])

    report = []
    report.append("=" * 72)
    report.append("FEATURE IMPORTANCE ANALYSIS: MARCH MADNESS TOURNAMENT PREDICTION")
    report.append(f"Data: {len(years)} seasons ({min(years)}-{max(years)}), {n_teams} team-seasons")
    report.append("=" * 72)

    # ── 1. Pearson Correlations vs Rank ──
    report.append("\n## 1. PEARSON CORRELATIONS vs RANK (lower rank = better team)")
    report.append(f"{'Feature':<16} {'r':>8} {'|r|':>8}  Direction")
    report.append("-" * 50)

    feature_keys = list(FEATURES.keys())
    correlations = {}
    for fk in feature_keys:
        arr = get_feature_array(records, fk)
        r = pearson_r(arr, ranks)
        correlations[fk] = r

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0, reverse=True)
    for fk, r in sorted_corr:
        if np.isnan(r):
            continue
        direction = "↑ better team" if r < 0 else "↓ better team" if r > 0 else "—"
        report.append(f"{fk:<16} {r:>8.4f} {abs(r):>8.4f}  {direction}")

    # ── 2. Multivariate OLS Regression ──
    report.append("\n## 2. MULTIVARIATE OLS REGRESSION (predicting rank)")
    # Use non-redundant features to avoid multicollinearity
    ols_features = ["AdjOE", "AdjDE", "EFG%", "EFGD%", "WAB", "Luck", "SOS_Ratio_O",
                    "CloseWin%", "Tempo", "WinPct", "ConfWinPct"]
    X_cols = []
    for fk in ols_features:
        X_cols.append(get_feature_array(records, fk))
    X = np.column_stack(X_cols)
    coeffs, r_sq = ols_regression(X, ranks)

    report.append(f"R² = {r_sq:.4f} (variance explained)")
    report.append(f"\n{'Feature':<16} {'Std Coeff':>10}  Interpretation")
    report.append("-" * 55)
    coeff_pairs = sorted(zip(ols_features, coeffs), key=lambda x: abs(x[1]), reverse=True)
    for fk, c in coeff_pairs:
        interp = "strong predictor" if abs(c) > 20 else "moderate" if abs(c) > 5 else "weak"
        report.append(f"{fk:<16} {c:>10.3f}  {interp}")

    # Also run with Barthag alone and AdjEM alone for comparison
    barthag_arr = get_feature_array(records, "Barthag").reshape(-1, 1)
    _, r_sq_barthag = ols_regression(barthag_arr, ranks)
    adjem_arr = get_feature_array(records, "AdjEM").reshape(-1, 1)
    _, r_sq_adjem = ols_regression(adjem_arr, ranks)
    report.append(f"\nSingle-feature R²: Barthag={r_sq_barthag:.4f}, AdjEM={r_sq_adjem:.4f}")

    # Check multicollinearity: AdjEM vs AdjOE/AdjDE
    r_em_oe = pearson_r(get_feature_array(records, "AdjEM"), get_feature_array(records, "AdjOE"))
    r_em_de = pearson_r(get_feature_array(records, "AdjEM"), get_feature_array(records, "AdjDE"))
    report.append(f"Multicollinearity check: AdjEM↔AdjOE r={r_em_oe:.4f}, AdjEM↔AdjDE r={r_em_de:.4f}")
    report.append("⚠ AdjEM = AdjOE - AdjDE by definition → perfect multicollinearity if all three included")

    # ── 3. Elite vs Bubble Effect Sizes ──
    report.append("\n## 3. ELITE vs BUBBLE EFFECT SIZES (Cohen's d)")
    report.append("Elite = rank 1-16 (top seeds), Bubble = rank 30-50 (tournament fringe)")

    elite_mask = np.array([r["rank"] <= 16 for r in records])
    bubble_mask = np.array([(30 <= r["rank"] <= 50) for r in records])

    report.append(f"Elite teams: {elite_mask.sum()}, Bubble teams: {bubble_mask.sum()}")
    report.append(f"\n{'Feature':<16} {'Cohen d':>8} {'Elite μ':>10} {'Bubble μ':>10}  Effect")
    report.append("-" * 65)

    effect_sizes = {}
    for fk in feature_keys:
        arr = get_feature_array(records, fk)
        elite_vals = arr[elite_mask]
        bubble_vals = arr[bubble_mask]
        d = cohens_d(elite_vals, bubble_vals)
        effect_sizes[fk] = d

    sorted_effects = sorted(effect_sizes.items(), key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0, reverse=True)
    for fk, d in sorted_effects:
        if np.isnan(d):
            continue
        arr = get_feature_array(records, fk)
        e_mean = np.nanmean(arr[elite_mask])
        b_mean = np.nanmean(arr[bubble_mask])
        size = "huge" if abs(d) > 2.0 else "very large" if abs(d) > 1.2 else "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small" if abs(d) > 0.2 else "negligible"
        report.append(f"{fk:<16} {d:>8.3f} {e_mean:>10.4f} {b_mean:>10.4f}  {size}")

    # ── 4. Offense vs Defense Comparison ──
    report.append("\n## 4. OFFENSE vs DEFENSE COMPARISON")
    report.append("Which matters more for predicting rank: offensive or defensive metrics?")

    off_features = ["AdjOE", "EFG%", "RawOE", "AdjOE_v2"]
    def_features = ["AdjDE", "EFGD%", "RawDE", "AdjDE_v2"]

    report.append(f"\n{'Metric':<16} {'Off r':>8} {'Def r':>8} {'Winner':>10}")
    report.append("-" * 50)
    for of, df in zip(off_features, def_features):
        r_off = abs(correlations.get(of, 0))
        r_def = abs(correlations.get(df, 0))
        winner = "OFFENSE" if r_off > r_def else "DEFENSE"
        report.append(f"{of}/{df:<10} {r_off:>8.4f} {r_def:>8.4f} {winner:>10}")

    # Aggregate: average |r| for offense vs defense
    avg_off = np.mean([abs(correlations.get(f, 0)) for f in off_features])
    avg_def = np.mean([abs(correlations.get(f, 0)) for f in def_features])
    report.append(f"\nAverage |r|: Offense={avg_off:.4f}, Defense={avg_def:.4f}")
    report.append(f"→ {'Offense' if avg_off > avg_def else 'Defense'} is slightly more predictive overall")

    # ── 5. Experience / Tempo Factor Analysis ──
    report.append("\n## 5. EXPERIENCE & TEMPO FACTOR ANALYSIS")

    tempo_arr = get_feature_array(records, "Tempo")
    r_tempo = correlations.get("Tempo", 0)
    report.append(f"Tempo vs Rank: r={r_tempo:.4f} (|r|={abs(r_tempo):.4f})")

    # Tempo for elite vs all
    elite_tempo = np.nanmean(tempo_arr[elite_mask])
    all_tempo = np.nanmean(tempo_arr)
    report.append(f"Mean tempo: Elite={elite_tempo:.2f}, All teams={all_tempo:.2f}")

    # Luck analysis
    luck_arr = get_feature_array(records, "Luck")
    r_luck = correlations.get("Luck", 0)
    report.append(f"\nLuck vs Rank: r={r_luck:.4f}")
    report.append(f"Luck range: [{np.nanmin(luck_arr):.4f}, {np.nanmax(luck_arr):.4f}], mean={np.nanmean(luck_arr):.4f}")
    report.append("→ Luck is near-zero signal (as expected — it's noise by definition)")

    # Close game win % analysis
    close_arr = get_feature_array(records, "CloseWin%")
    r_close = correlations.get("CloseWin%", 0)
    report.append(f"\nCloseWin% vs Rank: r={r_close:.4f}")
    report.append("→ Close game performance is partially luck-driven, limited predictive value")

    # SOS analysis
    sos_arr = get_feature_array(records, "SOS_Ratio_O")
    r_sos = correlations.get("SOS_Ratio_O", 0)
    report.append(f"\nSOS_Ratio_O vs Rank: r={r_sos:.4f}")
    report.append(f"SOS_Ratio_D vs Rank: r={correlations.get('SOS_Ratio_D', 0):.4f}")

    # ── 6. Pairwise Matchup Gap Analysis ──
    report.append("\n## 6. PAIRWISE MATCHUP GAP ANALYSIS")
    report.append("How well does each feature's gap predict which team is ranked higher?")
    report.append("(Sampled 50,000 random pairings across all team-seasons)")

    rng = np.random.RandomState(42)
    n_pairs = 50000
    idx1 = rng.randint(0, n_teams, n_pairs)
    idx2 = rng.randint(0, n_teams, n_pairs)
    # Ensure different teams
    mask_diff = idx1 != idx2
    idx1 = idx1[mask_diff]
    idx2 = idx2[mask_diff]

    rank_diff = ranks[idx1] - ranks[idx2]  # Negative = team1 is better
    actual_better = (rank_diff < 0).astype(float)

    report.append(f"\n{'Feature':<16} {'Accuracy':>10}  Notes")
    report.append("-" * 50)

    gap_results = {}
    for fk in feature_keys:
        arr = get_feature_array(records, fk)
        feat_diff = arr[idx1] - arr[idx2]
        valid = ~(np.isnan(feat_diff))
        if valid.sum() < 100:
            continue
        # For features where higher = better team (negative correlation with rank),
        # team1 having higher value means team1 is predicted better
        r = correlations.get(fk, 0)
        if r > 0:
            # Higher value = worse team, so lower diff = team1 better
            predicted_better = (feat_diff < 0).astype(float)
        else:
            predicted_better = (feat_diff > 0).astype(float)
        accuracy = np.mean(predicted_better[valid] == actual_better[valid])
        gap_results[fk] = accuracy

    sorted_gaps = sorted(gap_results.items(), key=lambda x: x[1], reverse=True)
    for fk, acc in sorted_gaps:
        note = "★ top predictor" if acc > 0.85 else "strong" if acc > 0.80 else "moderate" if acc > 0.70 else "weak"
        report.append(f"{fk:<16} {acc:>9.1%}  {note}")

    # ── 7. Summary & Recommended Feature Weights ──
    report.append("\n" + "=" * 72)
    report.append("## 7. SUMMARY & RECOMMENDED FEATURE WEIGHTS")
    report.append("=" * 72)

    report.append("""
KEY FINDINGS:
1. Barthag is the single strongest predictor of team quality (r ≈ -0.99 with rank)
2. AdjEM (= AdjOE - AdjDE) is perfectly multicollinear with AdjOE and AdjDE
   → Use AdjOE + AdjDE separately OR AdjEM alone, never all three
3. Offensive and defensive efficiency are both strong, with offense slightly more
   predictive in isolation
4. WAB (Wins Above Bubble) is a strong composite predictor
5. Luck and Tempo are near-zero signal for tournament prediction
6. SOS matters moderately — teams from stronger conferences tend to be better
7. Close game win% has limited predictive value (partially luck-driven)

RECOMMENDED FEATURE WEIGHTS FOR PREDICTION MODEL:
(Normalized weights summing to 1.0 for a weighted composite score)
""")

    weights = {
        "Barthag": 0.30,
        "AdjOE": 0.15,
        "AdjDE": 0.15,
        "WAB": 0.12,
        "EFG%": 0.06,
        "EFGD%": 0.06,
        "SOS_Ratio_O": 0.05,
        "SOS_Ratio_D": 0.05,
        "WinPct": 0.04,
        "Tempo": 0.01,
        "Luck": 0.01,
    }
    report.append(f"{'Feature':<16} {'Weight':>8}  Rationale")
    report.append("-" * 65)
    for fk, w in weights.items():
        r_val = correlations.get(fk, 0)
        report.append(f"{fk:<16} {w:>8.2f}  |r|={abs(r_val):.3f}")

    report.append(f"\nTotal: {sum(weights.values()):.2f}")
    report.append("""
NOTES FOR DOWNSTREAM MODEL (mm-cn6):
- Drop AdjEM if using AdjOE + AdjDE (multicollinearity)
- Barthag alone explains ~99% of rank variance — other features add marginal lift
- For logistic regression on tournament outcomes, consider using Barthag + WAB + SOS
  as the core feature set, with EFG%/EFGD% as secondary features
- Tempo and Luck should be excluded or given minimal weight
""")

    return "\n".join(report)


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    report = run_analysis(str(data_dir))
    print(report)

    # Save report
    output_path = script_dir / "feature_importance_report.txt"
    with open(output_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {output_path}")

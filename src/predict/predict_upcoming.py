import json
import pandas as pd
import numpy as np
from src.predict.simulate_match import load_models, predict_match_outcome

def get_team_form_stats(team_name, stats_df, prefix):
    # Get the most recent form row (exclude matches after today)
    row = stats_df[stats_df['team'] == team_name].sort_values('date').tail(1)
    if row.empty:
        return {
            f"{prefix}_form_goals_scored": 0,
            f"{prefix}_form_goals_conceded": 0,
            f"{prefix}_form_xg_for": 0,
            f"{prefix}_form_xg_against": 0
        }
    r = row.iloc[0]
    return {
        f"{prefix}_form_goals_scored": r['form_goals_scored'],
        f"{prefix}_form_goals_conceded": r['form_goals_conceded'],
        f"{prefix}_form_xg_for": r['form_xg_for'],
        f"{prefix}_form_xg_against": r['form_xg_against'],
    }

def prepare_rolling_stats(historic_csv, rolling_window=5):
    df = pd.read_csv(historic_csv, parse_dates=['date']).sort_values('date')

    # Use correct historical xG column names: home_xg and away_xg
    home = df[['date', 'home_team', 'home_goals', 'away_goals', 'home_xg', 'away_xg']].copy()
    home.columns = ['date', 'team', 'goals_scored', 'goals_conceded', 'xg_for', 'xg_against']

    away = df[['date', 'away_team', 'away_goals', 'home_goals', 'away_xg', 'home_xg']].copy()
    away.columns = ['date', 'team', 'goals_scored', 'goals_conceded', 'xg_for', 'xg_against']

    stats = pd.concat([home, away]).sort_values(['team', 'date'])

    # Compute rolling averages with shift so current match isn't included
    stats['form_goals_scored'] = stats.groupby('team')['goals_scored'].transform(
        lambda x: x.rolling(rolling_window, min_periods=1).mean().shift(1)
    )
    stats['form_goals_conceded'] = stats.groupby('team')['goals_conceded'].transform(
        lambda x: x.rolling(rolling_window, min_periods=1).mean().shift(1)
    )
    stats['form_xg_for'] = stats.groupby('team')['xg_for'].transform(
        lambda x: x.rolling(rolling_window, min_periods=1).mean().shift(1)
    )
    stats['form_xg_against'] = stats.groupby('team')['xg_against'].transform(
        lambda x: x.rolling(rolling_window, min_periods=1).mean().shift(1)
    )

    return stats

def predict_upcoming_matches(
    lineup_xg_json="data/processed/team_fixture_xg_estimates.json",
    historical_csv="data/raw/epl_historical_results.csv",
    model_home_path="models/model_home.pkl",
    model_away_path="models/model_away.pkl",
    rolling_window=5
):
    # Load fixture-level lineup xG JSON
    lineup_xg = pd.read_json(lineup_xg_json)

    # Prepare rolling stats from historic data using correct columns
    stats = prepare_rolling_stats(historical_csv, rolling_window)

    # Load trained Poisson models
    model_home, model_away = load_models(model_home_path, model_away_path)

    # Predict for each upcoming fixture
    for _, row in lineup_xg.iterrows():
        home = row['team']
        away = row['opponent']

        # Get rolling form stats for both teams
        home_form = get_team_form_stats(home, stats, 'home')
        away_form = get_team_form_stats(away, stats, 'away')

        # Build feature dictionary matching model input
        features = {
            'home_avg_xg': row['home_avg_xg'],
            'away_avg_xg': row['away_avg_xg'],
            'home_field': 1,
        }
        features.update(home_form)
        features.update(away_form)

        df_features = pd.DataFrame([features])

        # Predict
        pred = predict_match_outcome(model_home, model_away, df_features)

        # Extract probability matrix for scorelines
        prob_matrix = pred['score_probabilities']
        max_goals = prob_matrix.shape[0] - 1

        # Most likely scoreline
        idx = np.unravel_index(np.argmax(prob_matrix, axis=None), prob_matrix.shape)
        most_likely_score = (idx[0], idx[1])
        most_likely_prob = prob_matrix[idx]

        # Top 3 most probable scorelines
        flat = prob_matrix.flatten()
        top3_idx = np.argpartition(flat, -3)[-3:]
        top3_scores = [np.unravel_index(i, prob_matrix.shape) for i in top3_idx]
        top3_probs = [prob_matrix[i] for i in top3_scores]

        print(f"{home} vs {away}")
        print(f"Expected goals: {pred['lambda_home']:.2f}-{pred['lambda_away']:.2f}")
        print(f"Most likely scoreline: {most_likely_score[0]}-{most_likely_score[1]} ({most_likely_prob:.2%})")
        print("Top 3 scorelines:")
        for (h, a), prob in sorted(zip(top3_scores, top3_probs), key=lambda tup: -tup[1]):
            print(f"  {h}-{a}: {prob:.2%}")
        # If you want to show the full matrix for small max_goals:
        print("Probability matrix (rows: home goals 0-6, cols: away goals 0-6):")
        print(np.round(prob_matrix, 3))
        print(f"Probabilities: Home {pred['home_win_prob']:.2%}, Draw {pred['draw_prob']:.2%}, Away {pred['away_win_prob']:.2%}")
        print("-" * 30)

if __name__ == "__main__":
    predict_upcoming_matches()

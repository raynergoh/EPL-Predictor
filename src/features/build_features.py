import pandas as pd
import json
import os

def build_features(team_xg_estimates_path, historical_matches_csv, output_path, rolling_window=5):
    # Load lineup-based xG features
    with open(team_xg_estimates_path) as f:
        team_xg_data = json.load(f)
    team_xg_df = pd.DataFrame(team_xg_data)

    # Load match results
    matches_df = pd.read_csv(historical_matches_csv, parse_dates=['date'])
    matches_df = matches_df.sort_values('date')

    # Merge in lineup xG (for each row: home and away xG based on teams and opponent)
    home_xg = team_xg_df[['team', 'opponent', 'team_xg_avg_adj']].copy()
    home_xg.rename(columns={'team': 'home_team', 'opponent': 'away_team', 'team_xg_avg_adj': 'home_avg_xg'}, inplace=True)
    away_xg = team_xg_df[['team', 'opponent', 'team_xg_avg_adj']].copy()
    away_xg.rename(columns={'team': 'away_team', 'opponent': 'home_team', 'team_xg_avg_adj': 'away_avg_xg'}, inplace=True)
    matches_df = matches_df.merge(home_xg, on=['home_team', 'away_team'], how='left')
    matches_df = matches_df.merge(away_xg, on=['home_team', 'away_team'], how='left')

    matches_df['home_avg_xg'] = matches_df['home_avg_xg'].fillna(matches_df['home_avg_xg'].median())
    matches_df['away_avg_xg'] = matches_df['away_avg_xg'].fillna(matches_df['away_avg_xg'].median())
    matches_df['home_field'] = 1

    # Feature engineering: form and defensive stats
    def create_team_stats(df):
        home = df[['date', 'home_team', 'home_goals', 'away_goals', 'home_avg_xg', 'away_avg_xg']].copy()
        home.columns = ['date', 'team', 'goals_scored', 'goals_conceded', 'xg_for', 'xg_against']
        away = df[['date', 'away_team', 'away_goals', 'home_goals', 'away_avg_xg', 'home_avg_xg']].copy()
        away.columns = ['date', 'team', 'goals_scored', 'goals_conceded', 'xg_for', 'xg_against']
        return pd.concat([home, away]).sort_values(['team', 'date'])

    team_stats = create_team_stats(matches_df)
    team_stats['form_goals_scored'] = (
        team_stats.groupby('team')['goals_scored']
        .transform(lambda x: x.rolling(rolling_window, min_periods=1).mean().shift(1))
    )
    team_stats['form_goals_conceded'] = (
        team_stats.groupby('team')['goals_conceded']
        .transform(lambda x: x.rolling(rolling_window, min_periods=1).mean().shift(1))
    )
    team_stats['form_xg_for'] = (
        team_stats.groupby('team')['xg_for']
        .transform(lambda x: x.rolling(rolling_window, min_periods=1).mean().shift(1))
    )
    team_stats['form_xg_against'] = (
        team_stats.groupby('team')['xg_against']
        .transform(lambda x: x.rolling(rolling_window, min_periods=1).mean().shift(1))
    )

    # Merge these team rolling features back for home and away team
    rolling_cols = ['team', 'date', 'form_goals_scored', 'form_goals_conceded', 'form_xg_for', 'form_xg_against']
    home_rolling = team_stats[rolling_cols].copy()
    home_rolling.columns = ['home_team', 'date'] + [f'home_{c}' for c in rolling_cols[2:]]
    away_rolling = team_stats[rolling_cols].copy()
    away_rolling.columns = ['away_team', 'date'] + [f'away_{c}' for c in rolling_cols[2:]]

    out = matches_df.merge(home_rolling, on=['home_team', 'date'], how='left')
    out = out.merge(away_rolling, on=['away_team', 'date'], how='left')

    # Fill missing rolling features with median (common for season start)
    rolling_feature_cols = [c for c in out.columns if c.startswith('home_') or c.startswith('away_')]
    for col in rolling_feature_cols:
        if out[col].dtype != 'O':  # Only numeric cols
            out[col] = out[col].fillna(out[col].median())

    # Select features and labels
    feature_cols = ['home_avg_xg', 'away_avg_xg', 'home_field'] + rolling_feature_cols
    features = out[feature_cols]
    labels_home = out['home_goals']
    labels_away = out['away_goals']

    os.makedirs(output_path, exist_ok=True)
    features.to_csv(os.path.join(output_path, 'features.csv'), index=False)
    labels_home.to_csv(os.path.join(output_path, 'labels_home.csv'), index=False)
    labels_away.to_csv(os.path.join(output_path, 'labels_away.csv'), index=False)

    print(f"Saved features to {os.path.join(output_path, 'features.csv')}")
    print(f"Saved home labels to {os.path.join(output_path, 'labels_home.csv')}")
    print(f"Saved away labels to {os.path.join(output_path, 'labels_away.csv')}")

if __name__ == "__main__":
    build_features(
        team_xg_estimates_path="data/processed/team_xg_estimates.json",
        historical_matches_csv="data/raw/epl_historical_results.csv",
        output_path="data/processed",
        rolling_window=5  # Or another value as desired
    )

import pandas as pd

def calculate_team_xg(team_fixtures_lineups, player_xg_csv="data/raw/understat_player_xg.csv"):
    """
    Calculate xG values for each team (based on predicted lineups), return enriched list of dicts.
    """
    # --- Load player xG mapping ---
    player_xg_df = pd.read_csv(player_xg_csv)
    # Use xG_per90 if available, else xG
    if "xG90" in player_xg_df.columns:
        player_xg_map = dict(zip(player_xg_df["player_name"], player_xg_df["xG90"]))
    else:
        player_xg_map = dict(zip(player_xg_df["player_name"], player_xg_df["xG"]))

    team_xg_estimates = []
    for entry in team_fixtures_lineups:
        team = entry["team"]
        opponent = entry["opponent"]
        home_away = entry["home_away"]
        lineup = entry["predicted_lineup"]

        # Look up xG for each player, default to 0 if not found
        player_xgs = [float(player_xg_map.get(player, 0.0)) for player in lineup]
        team_xg_sum = sum(player_xgs)
        team_xg_avg = team_xg_sum / len(player_xgs) if player_xgs else 0.0

        # Apply home/away adjustment to total xG
        if home_away == "H":
            adj_factor = 1.10
        elif home_away == "A":
            adj_factor = 0.95
        else:
            adj_factor = 1.0

        team_xg_sum_adj = team_xg_sum * adj_factor
        team_xg_avg_adj = team_xg_sum_adj / len(player_xgs) if player_xgs else 0.0

        team_xg_estimates.append({
            "team": team,
            "opponent": opponent,
            "home_away": home_away,
            "predicted_lineup": lineup,
            "team_xg_sum": round(team_xg_sum, 3),
            "team_xg_avg": round(team_xg_avg, 3),
            "team_xg_sum_adj": round(team_xg_sum_adj, 3),
            "team_xg_avg_adj": round(team_xg_avg_adj, 3)
        })

    return team_xg_estimates

def merge_xg_to_fixtures(team_xg_estimates):
    # Convert to DataFrame for easy pivot/merge
    df = pd.DataFrame(team_xg_estimates)

    # Split into home and away based on home_away flag
    home_df = df[df['home_away'] == 'H'].copy()
    away_df = df[df['home_away'] == 'A'].copy()

    # Rename columns for clarity before merge
    home_df = home_df.rename(columns={
        'team': 'home_team',
        'opponent': 'away_team',
        'team_xg_avg_adj': 'home_avg_xg',
        'predicted_lineup': 'home_predicted_lineup'
    })

    away_df = away_df.rename(columns={
        'team': 'away_team',
        'opponent': 'home_team',
        'team_xg_avg_adj': 'away_avg_xg',
        'predicted_lineup': 'away_predicted_lineup'
    })

    # Merge on fixture keys (home_team and away_team)
    merged = pd.merge(home_df, away_df,
                      on=['home_team', 'away_team'],
                      suffixes=('_home', '_away'))

    # Select & reorder columns in desired format
    fixtures_df = merged[['home_team', 'away_team', 'home_predicted_lineup',
                          'away_predicted_lineup', 'home_avg_xg', 'away_avg_xg']]

    # Rename columns (if you want to match your spec)
    fixtures_df.columns = ['team', 'opponent', 'home_predicted_lineup',
                           'away_predicted_lineup', 'home_avg_xg', 'away_avg_xg']
    
    # Add home_away flag as "H" because this row represents the home side
    fixtures_df = fixtures_df.copy()
    fixtures_df['home_away'] = 'H'

    # Convert to list of dicts for json serialization
    return fixtures_df.to_dict(orient='records')


# If you want to allow direct script running:
if __name__ == "__main__":
    import json
    with open("data/processed/ffscout_next_match_lineups.json") as f:
        team_fixtures_lineups = json.load(f)
    results = calculate_team_xg(team_fixtures_lineups)
    df = pd.DataFrame(results)
    df.to_csv("data/processed/team_xg_estimates.csv", index=False)
    print(df)

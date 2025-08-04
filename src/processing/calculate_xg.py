import pandas as pd

def calculate_team_xg(team_fixtures_lineups, player_xg_csv="data/understat_player_xg.csv"):
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

# If you want to allow direct script running:
if __name__ == "__main__":
    import json
    with open("data/processed/ffscout_next_match_lineups.json") as f:
        team_fixtures_lineups = json.load(f)
    results = calculate_team_xg(team_fixtures_lineups)
    df = pd.DataFrame(results)
    df.to_csv("data/processed/team_xg_estimates.csv", index=False)
    print(df)

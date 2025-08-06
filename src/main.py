import json
import os
import logging

from src.scraping.scrape_fixtures_lineups import scrape_ffscout_lineups_and_fixtures
from src.processing.calculate_xg import calculate_team_xg, merge_xg_to_fixtures
from src.predict.predict_upcoming import predict_upcoming_matches

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] - %(message)s')

def save_json(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    logging.info(f"Saved data to {filepath}")

def main():
    logging.info("Scraping predicted lineups and next fixtures from Fantasy Football Scout...")
    team_fixtures_lineups = scrape_ffscout_lineups_and_fixtures()

    logging.info(f"Calculating xG values for {len(team_fixtures_lineups)} teams based on predicted lineups...")
    team_xg_estimates = calculate_team_xg(team_fixtures_lineups)

    logging.info("Merging home and away xG estimates into fixture-level data...")
    team_fixtures = merge_xg_to_fixtures(team_xg_estimates)

    for entry in team_xg_estimates:
        logging.info(
            f"{entry['team']} ({'Home' if entry['home_away'] == 'H' else 'Away'}) "
            f"vs {entry['opponent']}: sum_xG={entry['team_xg_sum_adj']}, avg_xG={entry['team_xg_avg_adj']}"
        )

    # Save raw and merged xG data for downstream use
    save_json(team_fixtures_lineups, "data/processed/ffscout_next_match_lineups.json")
    save_json(team_xg_estimates, "data/processed/team_xg_estimates.json")       # per-team xG (optional)
    save_json(team_fixtures, "data/processed/team_fixture_xg_estimates.json")   # merged per fixture (home & away)

    # Run predictions using merged fixture-level xG data
    predict_upcoming_matches(
        lineup_xg_json="data/processed/team_fixture_xg_estimates.json",
        historical_csv="data/raw/epl_historical_results.csv",
        model_home_path="models/model_home.pkl",
        model_away_path="models/model_away.pkl",
        rolling_window=5
    )

if __name__ == "__main__":
    main()

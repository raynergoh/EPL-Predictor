import json
import os
import logging

from src.scraping.scrape_fixtures_lineups import scrape_ffscout_lineups_and_fixtures
from src.processing.calculate_xg import calculate_team_xg  # we will add this function wrapper to your calculate_xg.py

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
    # calculate_team_xg accepts the list of team fixture-lineups and returns a list of dicts with xG info added
    team_xg_estimates = calculate_team_xg(team_fixtures_lineups)

    for entry in team_xg_estimates:
        logging.info(
            f"{entry['team']} ({'Home' if entry['home_away'] == 'H' else 'Away'}) "
            f"vs {entry['opponent']}: sum_xG={entry['team_xg_sum_adj']}, avg_xG={entry['team_xg_avg_adj']}"
        )

    # Save raw and xG enriched files for downstream use
    save_json(team_fixtures_lineups, "data/processed/ffscout_next_match_lineups.json")
    save_json(team_xg_estimates, "data/processed/team_xg_estimates.json")

if __name__ == "__main__":
    main()

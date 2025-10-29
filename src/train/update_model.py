"""
Continuous Model Training - Update with Current Season Data

This script updates the Poisson model with the latest match results from the current season.
It fetches recent results, appends them to historical data, and retrains the model.

Usage:
    python -m src.train.update_model
    
The script will:
1. Fetch latest results from the current season (2025/26)
2. Append new results to historical data
3. Retrain the Poisson GLM model
4. Save updated model and coefficients
"""

import sys
from pathlib import Path
import pandas as pd
import requests
from datetime import datetime, timedelta
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from train.train_poisson_model import train_poisson_model

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
HISTORICAL_DATA = DATA_DIR / "epl_historical_results.csv"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelUpdater:
    """Updates Poisson model with current season data."""
    
    def __init__(self):
        """Initialize model updater."""
        self.api_url = "https://footballapi.pulselive.com/football/fixtures"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
    
    def fetch_recent_results(self, days_back: int = 90) -> pd.DataFrame:
        """
        Fetch recent match results from Premier League API.
        
        Args:
            days_back: Number of days to look back for results
            
        Returns:
            DataFrame with recent match results
        """
        logger.info(f"Fetching match results from last {days_back} days...")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Fetch from API
        params = {
            'comps': '1',  # Premier League
            'compSeasons': '578',  # 2025/26 season
            'page': '0',
            'pageSize': '100',
            'sort': 'desc',
            'statuses': 'C',  # Completed matches only
            'altIds': 'true'
        }
        
        try:
            response = requests.get(self.api_url, params=params, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Parse results
            results = []
            for match in data.get('content', []):
                # Only include completed matches with scores
                if match.get('status') != 'C':
                    continue
                
                teams = match.get('teams', [])
                if len(teams) < 2:
                    continue
                
                home_team = teams[0].get('team', {}).get('name', '')
                away_team = teams[1].get('team', {}).get('name', '')
                home_score = teams[0].get('score', 0)
                away_score = teams[1].get('score', 0)
                
                # Extract date
                kickoff = match.get('kickoff', {})
                date_str = kickoff.get('label', '')
                
                if home_team and away_team and date_str:
                    results.append({
                        'date': date_str,
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_goals': home_score,
                        'away_goals': away_score,
                        'home_xg': None,  # Not available from this API
                        'away_xg': None
                    })
            
            df = pd.DataFrame(results)
            logger.info(f"✓ Fetched {len(df)} recent match results")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching recent results: {e}")
            return pd.DataFrame()
    
    def update_historical_data(self, new_results: pd.DataFrame) -> bool:
        """
        Append new results to historical data file.
        
        Args:
            new_results: DataFrame with new match results
            
        Returns:
            True if update successful
        """
        if new_results.empty:
            logger.warning("No new results to add")
            return False
        
        logger.info("Updating historical data...")
        
        # Load existing historical data
        historical_df = pd.read_csv(HISTORICAL_DATA)
        
        # Convert dates to comparable format
        historical_df['date'] = pd.to_datetime(historical_df['date']).dt.strftime('%Y-%m-%d')
        new_results['date'] = pd.to_datetime(new_results['date']).dt.strftime('%Y-%m-%d')
        
        # Find truly new matches (not duplicates)
        existing_matches = set(
            historical_df.apply(
                lambda row: f"{row['date']}_{row['home_team']}_{row['away_team']}", 
                axis=1
            )
        )
        
        new_matches = new_results[
            ~new_results.apply(
                lambda row: f"{row['date']}_{row['home_team']}_{row['away_team']}", 
                axis=1
            ).isin(existing_matches)
        ]
        
        if new_matches.empty:
            logger.info("No new matches to add (all already in historical data)")
            return False
        
        # Append new matches
        updated_df = pd.concat([historical_df, new_matches], ignore_index=True)
        updated_df = updated_df.sort_values('date')
        
        # Save updated data
        updated_df.to_csv(HISTORICAL_DATA, index=False)
        logger.info(f"✓ Added {len(new_matches)} new matches to historical data")
        logger.info(f"  Total matches in historical data: {len(updated_df)}")
        
        return True
    
    def retrain_model(self):
        """Retrain Poisson model with updated data."""
        logger.info("\nRetraining Poisson model with updated data...")
        logger.info("="*70)
        
        # Call the main training function
        train_poisson_model()
        
        logger.info("="*70)
        logger.info("✓ Model retraining complete!")


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("MODEL UPDATE PIPELINE - Add Current Season Data")
    print("="*70)
    
    updater = ModelUpdater()
    
    # Step 1: Fetch recent results
    print("\n[1/3] Fetching recent match results...")
    recent_results = updater.fetch_recent_results(days_back=120)  # Last 4 months
    
    if recent_results.empty:
        print("\n⚠ No recent results fetched. Model not updated.")
        return
    
    print(f"\n✓ Found {len(recent_results)} completed matches")
    print("\nSample of recent results:")
    print(recent_results.head(10).to_string(index=False))
    
    # Step 2: Update historical data
    print("\n" + "="*70)
    print("[2/3] Updating historical data...")
    updated = updater.update_historical_data(recent_results)
    
    if not updated:
        print("\n✓ Historical data already up to date. No retraining needed.")
        return
    
    # Step 3: Retrain model
    print("\n" + "="*70)
    print("[3/3] Retraining model...")
    updater.retrain_model()
    
    print("\n" + "="*70)
    print("✅ MODEL UPDATE COMPLETE!")
    print("="*70)
    print("\nThe model has been updated with the latest match results.")
    print("You can now run weekly predictions with the updated model:")
    print("  python -m src.weekly.predict_weekly")
    print("="*70)


if __name__ == "__main__":
    main()

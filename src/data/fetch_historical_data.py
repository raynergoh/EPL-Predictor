"""
Fetch historical EPL data from football-data.co.uk.

Based on methodology from: https://artiebits.com/blog/predicting-football-results-with-statistical-modelling/

This script scrapes EPL historical results from football-data.co.uk and combines
multiple seasons into a single CSV file for model training.
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime
import logging


def fetch_season_data(url: str) -> pd.DataFrame:
    """
    Fetch data for a single season from football-data.co.uk.
    
    Args:
        url: URL to the CSV file for a specific season
        
    Returns:
        DataFrame with match data for that season
    """
    season = url.split("/")[4]
    print(f"Getting data for {season}")
    
    temp_df = pd.read_csv(url)
    temp_df["Season"] = season
    
    # Clean the data:
    # - Drop columns with too many missing values
    # - Remove the 'Div' column (redundant)
    # - Parse dates with day-first format
    # - Drop remaining rows with missing values
    temp_df = (
        temp_df.dropna(axis="columns", thresh=temp_df.shape[0] - 30)
        .drop("Div", axis=1, errors='ignore')
        .assign(Date=lambda df: pd.to_datetime(df.Date, dayfirst=True))
        .dropna()
    )
    
    return temp_df


def fetch_data(competition: str, page: str) -> pd.DataFrame:
    """
    Fetch historical data for a competition from football-data.co.uk.
    
    Args:
        competition: Name of the competition (e.g., "Premier League")
        page: Page path on football-data.co.uk (e.g., "englandm.php")
        
    Returns:
        DataFrame with all historical match data combined
    """
    base_url = "https://www.football-data.co.uk/"
    response = requests.get(f"{base_url}{page}")
    soup = BeautifulSoup(response.content, "lxml")

    # Find the table containing links to the data
    tables = soup.find_all(
        "table", attrs={"align": "center", "cellspacing": "0", "width": "800"}
    )
    
    if len(tables) < 2:
        raise ValueError("Could not find data table on page")
    
    table = tables[1]
    body = table.find_all("td", attrs={"valign": "top"})[1]

    # Extract links and link texts from the table
    links = [link.get("href") for link in body.find_all("a")]
    link_texts = [link.text for link in body.find_all("a")]

    # Filter the links for the given competition name and exclude unwanted data
    # The [:-12] slice excludes the last 12 links which are typically summary files
    data_urls = [
        f"{base_url}{links[i]}"
        for i, text in enumerate(link_texts)
        if text == competition
    ][:-12]
    
    if not data_urls:
        raise ValueError(f"No data found for competition: {competition}")

    print(f"Found {len(data_urls)} seasons of data")

    # Fetch data from all URLs and concatenate them into a single DataFrame
    data_frames = list(map(fetch_season_data, data_urls))
    merged_df = (
        pd.concat(data_frames, ignore_index=True)
        .dropna(axis=1)
        .dropna()
        .sort_values(by="Date")
    )

    return merged_df


def save_epl_data(output_file: str = "data/raw/epl_historical_results.csv"):
    """
    Fetch EPL historical data and save to CSV.
    
    Args:
        output_file: Path where the CSV file will be saved
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Fetch the data
    print("Fetching EPL historical data from football-data.co.uk...")
    df = fetch_data("Premier League", "englandm.php")
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"\nSuccessfully saved {len(df)} matches to {output_file}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Seasons: {', '.join(sorted(df['Season'].unique()))}")


if __name__ == "__main__":
    save_epl_data()


def fetch_and_update_data(data_path: str = 'data/raw/epl_historical_results.csv', 
                         force_fetch: bool = False,
                         cache_hours: float = 24.0) -> bool:
    """
    Fetch latest EPL data from football-data.co.uk and update historical file.
    
    This checks the existing file first, then only fetches new data if needed.
    Uses intelligent caching to avoid unnecessary fetches:
    - If file was updated within cache_hours (default 24h), skip fetch
    - This prevents repeated fetches when football-data.co.uk hasn't updated yet
    
    Args:
        data_path: Path to historical results CSV
        force_fetch: Force fetch even if cache is valid (default: False)
        cache_hours: Hours to cache data before re-fetching (default: 24)
        
    Returns:
        True if data was updated, False if no update needed or fetch failed
    """
    logger = logging.getLogger(__name__)
    data_file = Path(data_path)
    
    try:
        # Check if we need to fetch at all
        if data_file.exists() and not force_fetch:
            # Check file modification time - this is when we last successfully fetched
            import time
            file_age_hours = (time.time() - data_file.stat().st_mtime) / 3600
            
            # Skip fetch if file was modified within cache window
            # This prevents hammering football-data.co.uk when data hasn't changed
            if file_age_hours < cache_hours:
                old_df = pd.read_csv(data_path, dtype={'Season': str})
                old_df['Date'] = pd.to_datetime(old_df['Date'])
                latest_match_date = old_df['Date'].max()
                
                logger.info(f"Data file updated {file_age_hours:.1f} hours ago (< {cache_hours}h cache), skipping fetch")
                print(f"✓ Data file recently fetched ({file_age_hours:.1f}h ago), skipping re-fetch")
                print(f"  {len(old_df)} matches, latest: {latest_match_date.date()}")
                return False
        
        # Fetch data from football-data.co.uk
        logger.info("Fetching latest EPL data from football-data.co.uk...")
        print("📥 Fetching latest EPL data from football-data.co.uk...")
        
        df = fetch_data("Premier League", "englandm.php")
        
        # Compare with existing file
        updated = False
        
        if data_file.exists():
            old_df = pd.read_csv(data_path, dtype={'Season': str})
            old_count = len(old_df)
            new_count = len(df)
            
            if new_count > old_count:
                updated = True
                logger.info(f"Data updated: {old_count} → {new_count} matches (+{new_count - old_count})")
                print(f"✓ Data updated: {old_count} → {new_count} matches (+{new_count - old_count} new)")
                
                # Save the updated data
                df.to_csv(data_path, index=False)
                latest_date = df['Date'].max()
                print(f"  Latest match: {latest_date}")
            else:
                logger.info(f"Data unchanged: {new_count} matches")
                print(f"✓ Data already up to date ({new_count} matches)")
                # Touch file to update timestamp - prevents re-fetching for cache_hours
                data_file.touch()
                latest_date = old_df['Date'].max()
                print(f"  Latest match: {latest_date}")
        else:
            updated = True
            logger.info(f"Created new data file with {len(df)} matches")
            print(f"✓ Created new data file with {len(df)} matches")
            
            # Save the new data file
            df.to_csv(data_path, index=False)
            latest_date = df['Date'].max()
            print(f"  Latest match: {latest_date}")
        
        return updated
        
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        print(f"⚠️  Failed to fetch latest data: {e}")
        print("   Using existing data file...")
        
        # Check existing file
        if data_file.exists():
            df = pd.read_csv(data_path, dtype={'Season': str})
            latest_date = df['Date'].max()
            print(f"✓ Using existing data ({len(df)} matches, latest: {latest_date})")
            return False
        else:
            logger.error(f"No data file found: {data_path}")
            raise FileNotFoundError(f"Cannot proceed - no data file found: {data_path}")


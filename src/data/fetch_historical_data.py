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

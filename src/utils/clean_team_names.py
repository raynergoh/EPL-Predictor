"""
Clean and standardize team names in historical data.

Ensures consistency between football-data.co.uk format and Premier League API format.
This script should be run after fetching historical data to normalize team names.
"""

import pandas as pd
from pathlib import Path


def get_team_name_mapping():
    """
    Return mapping from various team name formats to standardized format.
    
    Standard format matches football-data.co.uk convention which is used
    consistently in our historical data.
    """
    return {
        # Standardized names (already correct)
        'Arsenal': 'Arsenal',
        'Aston Villa': 'Aston Villa',
        'Bournemouth': 'Bournemouth',
        'Brentford': 'Brentford',
        'Brighton': 'Brighton',
        'Burnley': 'Burnley',
        'Chelsea': 'Chelsea',
        'Crystal Palace': 'Crystal Palace',
        'Everton': 'Everton',
        'Fulham': 'Fulham',
        'Ipswich': 'Ipswich',
        'Leeds': 'Leeds',
        'Leicester': 'Leicester',
        'Liverpool': 'Liverpool',
        'Luton': 'Luton',
        'Man City': 'Man City',
        'Man United': 'Man United',
        'Newcastle': 'Newcastle',
        'Nott\'m Forest': 'Nott\'m Forest',
        'Sheffield United': 'Sheffield United',
        'Southampton': 'Southampton',
        'Sunderland': 'Sunderland',
        'Tottenham': 'Tottenham',
        'West Ham': 'West Ham',
        'Wolves': 'Wolves',
        
        # Alternative spellings from football-data.co.uk
        'Man Utd': 'Man United',
        'Nott\'m Forest': 'Nott\'m Forest',
        'Nottingham Forest': 'Nott\'m Forest',
        'Nottm Forest': 'Nott\'m Forest',
        'Sheffield Utd': 'Sheffield United',
        
        # Premier League API full names
        'AFC Bournemouth': 'Bournemouth',
        'Brighton & Hove Albion': 'Brighton',
        'Brighton and Hove Albion': 'Brighton',
        'Ipswich Town': 'Ipswich',
        'Leeds United': 'Leeds',
        'Leicester City': 'Leicester',
        'Luton Town': 'Luton',
        'Manchester City': 'Man City',
        'Manchester United': 'Man United',
        'Newcastle United': 'Newcastle',
        'Tottenham Hotspur': 'Tottenham',
        'West Ham United': 'West Ham',
        'Wolverhampton Wanderers': 'Wolves',
        
        # Historical teams
        'Birmingham': 'Birmingham',
        'Birmingham City': 'Birmingham',
        'Blackburn': 'Blackburn',
        'Blackburn Rovers': 'Blackburn',
        'Blackpool': 'Blackpool',
        'Bolton': 'Bolton',
        'Bolton Wanderers': 'Bolton',
        'Cardiff': 'Cardiff',
        'Cardiff City': 'Cardiff',
        'Charlton': 'Charlton',
        'Charlton Athletic': 'Charlton',
        'Derby': 'Derby',
        'Derby County': 'Derby',
        'Huddersfield': 'Huddersfield',
        'Huddersfield Town': 'Huddersfield',
        'Hull': 'Hull',
        'Hull City': 'Hull',
        'Middlesbrough': 'Middlesbrough',
        'Norwich': 'Norwich',
        'Norwich City': 'Norwich',
        'Portsmouth': 'Portsmouth',
        'QPR': 'QPR',
        'Queens Park Rangers': 'QPR',
        'Reading': 'Reading',
        'Stoke': 'Stoke',
        'Stoke City': 'Stoke',
        'Swansea': 'Swansea',
        'Swansea City': 'Swansea',
        'Watford': 'Watford',
        'West Brom': 'West Brom',
        'West Bromwich Albion': 'West Brom',
        'Wigan': 'Wigan',
        'Wigan Athletic': 'Wigan',
    }


def clean_team_name(name: str) -> str:
    """
    Clean and standardize a single team name.
    
    Args:
        name: Raw team name from any source
        
    Returns:
        Standardized team name
    """
    mapping = get_team_name_mapping()
    return mapping.get(name, name)


def clean_historical_data(
    input_file: str = "data/raw/epl_historical_results.csv",
    output_file: str = None
):
    """
    Clean team names in historical data CSV.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (defaults to overwriting input)
    """
    if output_file is None:
        output_file = input_file
    
    print(f"Loading historical data from {input_file}...")
    df = pd.read_csv(input_file, dtype={'Season': str})
    
    original_count = len(df)
    print(f"Loaded {original_count} matches")
    
    # Get unique team names before cleaning
    home_teams_before = set(df['HomeTeam'].unique())
    away_teams_before = set(df['AwayTeam'].unique())
    all_teams_before = home_teams_before | away_teams_before
    print(f"\nUnique teams before cleaning: {len(all_teams_before)}")
    print("Teams:", sorted(all_teams_before))
    
    # Clean team names
    print("\nCleaning team names...")
    df['HomeTeam'] = df['HomeTeam'].apply(clean_team_name)
    df['AwayTeam'] = df['AwayTeam'].apply(clean_team_name)
    
    # Get unique team names after cleaning
    home_teams_after = set(df['HomeTeam'].unique())
    away_teams_after = set(df['AwayTeam'].unique())
    all_teams_after = home_teams_after | away_teams_after
    print(f"\nUnique teams after cleaning: {len(all_teams_after)}")
    print("Teams:", sorted(all_teams_after))
    
    # Show changes
    teams_removed = all_teams_before - all_teams_after
    if teams_removed:
        print(f"\n✓ Cleaned {len(teams_removed)} team name variations:")
        for team in sorted(teams_removed):
            # Find what it was mapped to
            standardized = clean_team_name(team)
            if standardized != team:
                print(f"  • {team} → {standardized}")
    
    # Save cleaned data
    print(f"\nSaving cleaned data to {output_file}...")
    df.to_csv(output_file, index=False)
    print(f"✓ Saved {len(df)} matches")
    
    return df


if __name__ == "__main__":
    import sys
    
    # Allow command-line arguments for input/output files
    input_file = sys.argv[1] if len(sys.argv) > 1 else "data/raw/epl_historical_results.csv"
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    clean_historical_data(input_file, output_file)

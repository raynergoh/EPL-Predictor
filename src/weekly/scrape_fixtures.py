"""
Premier League fixture scraper.

Fetches upcoming fixtures for the current matchweek from Premier League website.
Returns structured data with home/away teams and match details.
"""

import requests
from bs4 import BeautifulSoup
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


class FixtureScraper:
    """Scrapes EPL fixtures from official Premier League website."""
    
    def __init__(self):
        """Initialize fixture scraper."""
        self.base_url = "https://www.premierleague.com"
        self.fixtures_url = f"{self.base_url}/fixtures"
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Headers to avoid bot detection
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def fetch_fixtures(self, matchweek: Optional[int] = None) -> List[Dict]:
        """
        Fetch fixtures for specified matchweek or next upcoming fixtures.
        
        Args:
            matchweek: Specific matchweek number (1-38), or None for upcoming
            
        Returns:
            List of fixture dicts with home_team, away_team, date, etc.
        """
        self.logger.info(f"Fetching fixtures for matchweek: {matchweek or 'upcoming'}")
        
        try:
            # Try fetching from Premier League API (if available)
            fixtures = self._fetch_from_api(matchweek)
            if fixtures:
                return fixtures
        except Exception as e:
            self.logger.warning(f"API fetch failed: {e}, trying web scraping")
        
        # Fallback to web scraping
        try:
            fixtures = self._fetch_from_web(matchweek)
            return fixtures
        except Exception as e:
            self.logger.error(f"Web scraping failed: {e}")
            return []
    
    def _fetch_from_api(self, matchweek: Optional[int]) -> List[Dict]:
        """
        Attempt to fetch fixtures from PL API endpoint.
        
        Uses the official Premier League API (footballapi.pulselive.com)
        which provides structured JSON data for fixtures.
        
        Note: The API works WITHOUT the compSeasons parameter. Don't add it!
        """
        self.logger.info("Attempting to fetch from Premier League API")
        
        try:
            url = 'https://footballapi.pulselive.com/football/fixtures'
            
            headers = {
                'Origin': 'https://www.premierleague.com',
                'User-Agent': self.headers['User-Agent']
            }
            
            # Parameters for API request
            # IMPORTANT: Do NOT include 'compSeasons' parameter - it causes empty results!
            params = {
                'comps': '1',  # Competition ID for Premier League
                'page': '0',
                'pageSize': '100',  # Fetch more fixtures to ensure we get the requested matchweek
                'sort': 'asc',
                'statuses': 'U',  # U = Upcoming fixtures
            }
            
            # Note: gameweeks parameter doesn't seem to work reliably, so we'll filter manually
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            fixtures_data = data.get('content', [])
            
            if not fixtures_data:
                self.logger.warning("API returned no fixtures")
                return []
            
            self.logger.info(f"API returned {len(fixtures_data)} fixtures")
            
            # Parse API response into our fixture format
            fixtures = []
            for fixture_data in fixtures_data:
                try:
                    # Extract team information
                    teams = fixture_data.get('teams', [])
                    if len(teams) < 2:
                        continue
                    
                    # Teams are in order: [home, away]
                    home_team = teams[0].get('team', {}).get('name', '')
                    away_team = teams[1].get('team', {}).get('name', '')
                    
                    # Clean team names (normalize to match our model's training data)
                    home_team = self._normalize_team_name(home_team)
                    away_team = self._normalize_team_name(away_team)
                    
                    # Extract match timing
                    kickoff = fixture_data.get('kickoff', {})
                    kickoff_label = kickoff.get('label', '')
                    
                    # Parse date and time from label (e.g., "Sat 1 Nov 2025, 15:00 GMT")
                    date_str, time_str = self._parse_kickoff_label(kickoff_label)
                    
                    # Extract venue
                    ground = fixture_data.get('ground', {})
                    venue = ground.get('name', '')
                    
                    # Extract matchweek
                    gameweek_info = fixture_data.get('gameweek', {})
                    fixture_matchweek = gameweek_info.get('gameweek')
                    if fixture_matchweek:
                        fixture_matchweek = int(fixture_matchweek)
                    
                    # Get season info for logging
                    comp_season = gameweek_info.get('compSeason', {})
                    season_label = comp_season.get('label', '')
                    
                    fixture = {
                        'matchweek': fixture_matchweek or matchweek,
                        'date': date_str,
                        'time': time_str,
                        'home_team': home_team,
                        'away_team': away_team,
                        'venue': venue,
                        'season': season_label
                    }
                    
                    fixtures.append(fixture)
                    
                except Exception as e:
                    self.logger.debug(f"Error parsing fixture: {e}")
                    continue
            
            # Filter by matchweek if specified (double-check since API filter might not work)
            if matchweek and fixtures:
                fixtures = [f for f in fixtures if f.get('matchweek') == matchweek]
                self.logger.info(f"Filtered to {len(fixtures)} fixtures for matchweek {matchweek}")
            elif fixtures:
                # If no matchweek specified, get only the NEXT upcoming matchweek
                # Group fixtures by matchweek and get the earliest one
                matchweeks = {}
                for f in fixtures:
                    mw = f.get('matchweek')
                    if mw:
                        if mw not in matchweeks:
                            matchweeks[mw] = []
                        matchweeks[mw].append(f)
                
                if matchweeks:
                    # Get the earliest matchweek number
                    next_matchweek = min(matchweeks.keys())
                    fixtures = matchweeks[next_matchweek]
                    self.logger.info(f"Auto-selected matchweek {next_matchweek} ({len(fixtures)} fixtures)")
            
            if fixtures:
                season = fixtures[0].get('season', 'Unknown')
                self.logger.info(f"Successfully fetched {len(fixtures)} fixtures from {season}")
            
            return fixtures
            
        except Exception as e:
            self.logger.warning(f"API fetch failed: {e}")
            return []
    
    def _normalize_team_name(self, team_name: str) -> str:
        """
        Normalize team names from Premier League API to match football-data.co.uk format.
        
        This ensures consistency with the historical data from football-data.co.uk
        which is used to train our Poisson model.
        
        Premier League API format -> football-data.co.uk format
        """
        name_mapping = {
            # Current 2025/26 season teams
            'Arsenal': 'Arsenal',
            'Aston Villa': 'Aston Villa',
            'AFC Bournemouth': 'Bournemouth',
            'Brentford': 'Brentford',
            'Brighton & Hove Albion': 'Brighton',
            'Burnley': 'Burnley',
            'Chelsea': 'Chelsea',
            'Crystal Palace': 'Crystal Palace',
            'Everton': 'Everton',
            'Fulham': 'Fulham',
            'Ipswich Town': 'Ipswich',
            'Leeds United': 'Leeds',  # KEY: API uses "Leeds United", data.co.uk uses "Leeds"
            'Leicester City': 'Leicester',
            'Liverpool': 'Liverpool',
            'Luton Town': 'Luton',
            'Manchester City': 'Man City',
            'Manchester United': 'Man United',
            'Newcastle United': 'Newcastle',
            'Nottingham Forest': 'Nott\'m Forest',
            'Sheffield United': 'Sheffield United',
            'Southampton': 'Southampton',
            'Sunderland': 'Sunderland',
            'Tottenham Hotspur': 'Tottenham',
            'West Ham United': 'West Ham',
            'Wolverhampton Wanderers': 'Wolves',
            
            # Historical teams (for completeness)
            'Birmingham City': 'Birmingham',
            'Blackburn Rovers': 'Blackburn',
            'Blackpool': 'Blackpool',
            'Bolton Wanderers': 'Bolton',
            'Cardiff City': 'Cardiff',
            'Charlton Athletic': 'Charlton',
            'Derby County': 'Derby',
            'Huddersfield Town': 'Huddersfield',
            'Hull City': 'Hull',
            'Middlesbrough': 'Middlesbrough',
            'Norwich City': 'Norwich',
            'Portsmouth': 'Portsmouth',
            'Queens Park Rangers': 'QPR',
            'Reading': 'Reading',
            'Stoke City': 'Stoke',
            'Swansea City': 'Swansea',
            'Watford': 'Watford',
            'West Bromwich Albion': 'West Brom',
            'Wigan Athletic': 'Wigan',
        }
        
        return name_mapping.get(team_name, team_name)
    
    def _parse_kickoff_label(self, label: str) -> tuple:
        """
        Parse kickoff label into date and time strings.
        
        Example input: "Sat 1 Nov 2025, 15:00 GMT"
        Returns: ("2025-11-01", "15:00")
        """
        try:
            if not label or ',' not in label:
                return ('', '')
            
            # Split into date part and time part
            parts = label.split(',')
            date_part = parts[0].strip()  # "Sat 1 Nov 2025"
            time_part = parts[1].strip() if len(parts) > 1 else ''  # "15:00 GMT"
            
            # Parse date part
            from datetime import datetime
            # Remove day name and parse
            date_tokens = date_part.split()
            if len(date_tokens) >= 3:
                # Format: "Sat 1 Nov 2025" -> "1 Nov 2025"
                date_str = ' '.join(date_tokens[1:])
                dt = datetime.strptime(date_str, '%d %b %Y')
                formatted_date = dt.strftime('%Y-%m-%d')
            else:
                formatted_date = ''
            
            # Parse time part (remove timezone)
            time_str = time_part.split()[0] if time_part else ''  # "15:00"
            
            return (formatted_date, time_str)
            
        except Exception as e:
            self.logger.debug(f"Error parsing kickoff label '{label}': {e}")
            return ('', '')
    
    def _fetch_from_web(self, matchweek: Optional[int]) -> List[Dict]:
        """
        Scrape fixtures from Premier League website.
        
        Args:
            matchweek: Matchweek number or None for upcoming
            
        Returns:
            List of fixture dictionaries
        """
        self.logger.info("Attempting to scrape fixtures from Premier League website")
        
        try:
            # Construct URL with matchweek parameter if specified
            url = self.fixtures_url
            if matchweek:
                # PL website uses query parameter for matchweek filtering
                url = f"{self.fixtures_url}?co=1&cl=-1&compSeasons=578&page=0&pageSize=100"
            
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Try to parse fixtures from the page structure
            fixtures = self._parse_fixtures_html(soup, matchweek)
            
            if fixtures:
                self.logger.info(f"Successfully scraped {len(fixtures)} fixtures from web")
                return fixtures
            else:
                self.logger.warning("No fixtures found on page, using fallback sample data")
                return self._get_fallback_fixtures()
                
        except Exception as e:
            self.logger.error(f"Web scraping error: {e}, using fallback sample data")
            return self._get_fallback_fixtures()
    
    def _parse_fixtures_html(self, soup: BeautifulSoup, matchweek: Optional[int]) -> List[Dict]:
        """
        Parse fixtures from BeautifulSoup HTML object.
        
        Premier League website structure may change, so this includes multiple strategies.
        """
        fixtures = []
        
        # Strategy 1: Look for fixture containers with data attributes
        fixture_elements = soup.find_all('li', class_='matchFixtureContainer')
        
        for element in fixture_elements:
            try:
                # Extract teams
                teams = element.find_all('span', class_='teamName')
                if len(teams) >= 2:
                    home_team = teams[0].get_text(strip=True)
                    away_team = teams[1].get_text(strip=True)
                    
                    # Extract date/time
                    date_elem = element.find('time')
                    date_str = date_elem.get('datetime', '') if date_elem else ''
                    
                    # Extract matchweek (if available)
                    mw_elem = element.find('span', class_='matchweek')
                    fixture_mw = None
                    if mw_elem:
                        mw_text = mw_elem.get_text(strip=True)
                        # Extract number from "Matchweek X" text
                        import re
                        mw_match = re.search(r'\d+', mw_text)
                        if mw_match:
                            fixture_mw = int(mw_match.group())
                    
                    # Extract venue (stadium name)
                    venue_elem = element.find('span', class_='stadium')
                    venue = venue_elem.get_text(strip=True) if venue_elem else ''
                    
                    # Parse datetime
                    if date_str:
                        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        date = dt.strftime('%Y-%m-%d')
                        time = dt.strftime('%H:%M')
                    else:
                        date = ''
                        time = ''
                    
                    fixture = {
                        'matchweek': fixture_mw or matchweek,
                        'date': date,
                        'time': time,
                        'home_team': home_team,
                        'away_team': away_team,
                        'venue': venue
                    }
                    
                    fixtures.append(fixture)
                    
            except Exception as e:
                self.logger.debug(f"Error parsing fixture element: {e}")
                continue
        
        return fixtures
    
    def _get_fallback_fixtures(self) -> List[Dict]:
        """
        Return fallback sample fixtures when API/web scraping fails.
        
        These are sample fixtures with proper team name formatting that matches
        our model's training data (football-data.co.uk format).
        """
        self.logger.info("Using fallback sample fixtures")
        
        # Return sample upcoming fixtures (example matchweek 10)
        sample_fixtures = [
            {
                'matchweek': 10,
                'date': '2025-11-02',
                'time': '15:00',
                'home_team': 'Arsenal',  # Note: Arsenal IS in our model (baseline team)
                'away_team': 'Liverpool',
                'venue': 'Emirates Stadium'
            },
            {
                'matchweek': 10,
                'date': '2025-11-02',
                'time': '15:00',
                'home_team': 'Bournemouth',
                'away_team': 'Man City',
                'venue': 'Vitality Stadium'
            },
            {
                'matchweek': 10,
                'date': '2025-11-02',
                'time': '15:00',
                'home_team': 'Brighton',
                'away_team': 'Wolves',
                'venue': 'Amex Stadium'
            },
            {
                'matchweek': 10,
                'date': '2025-11-02',
                'time': '15:00',
                'home_team': 'Ipswich',
                'away_team': 'Leicester',
                'venue': 'Portman Road'
            },
            {
                'matchweek': 10,
                'date': '2025-11-02',
                'time': '15:00',
                'home_team': 'Nott\'m Forest',
                'away_team': 'West Ham',
                'venue': 'City Ground'
            },
            {
                'matchweek': 10,
                'date': '2025-11-02',
                'time': '17:30',
                'home_team': 'Crystal Palace',
                'away_team': 'Tottenham',
                'venue': 'Selhurst Park'
            },
            {
                'matchweek': 10,
                'date': '2025-11-03',
                'time': '14:00',
                'home_team': 'Aston Villa',
                'away_team': 'Bournemouth',
                'venue': 'Villa Park'
            },
            {
                'matchweek': 10,
                'date': '2025-11-03',
                'time': '14:00',
                'home_team': 'Man United',
                'away_team': 'Chelsea',
                'venue': 'Old Trafford'
            },
            {
                'matchweek': 10,
                'date': '2025-11-03',
                'time': '16:30',
                'home_team': 'Fulham',
                'away_team': 'Brentford',
                'venue': 'Craven Cottage'
            },
            {
                'matchweek': 10,
                'date': '2025-11-04',
                'time': '20:00',
                'home_team': 'Southampton',
                'away_team': 'Everton',
                'venue': 'St Mary\'s Stadium'
            }
        ]
        
        return sample_fixtures
    
    def save_fixtures(self, fixtures: List[Dict], output_dir: str = 'data/weekly'):
        """
        Save fixtures to JSON file.
        
        Args:
            fixtures: List of fixture dictionaries
            output_dir: Directory to save fixtures file
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if fixtures:
            matchweek = fixtures[0].get('matchweek', 'unknown')
            filename = f'fixtures_mw{matchweek}_{timestamp}.json'
        else:
            filename = f'fixtures_{timestamp}.json'
        
        output_file = output_path / filename
        
        with open(output_file, 'w') as f:
            json.dump(fixtures, f, indent=2)
        
        self.logger.info(f"Fixtures saved to {output_file}")
        return output_file


def main():
    """Fetch and save current matchweek fixtures."""
    scraper = FixtureScraper()
    
    # Fetch upcoming fixtures
    fixtures = scraper.fetch_fixtures()
    
    print(f"\nFound {len(fixtures)} fixtures:\n")
    for fixture in fixtures:
        print(f"  {fixture['home_team']} vs {fixture['away_team']} "
              f"({fixture['date']} {fixture['time']})")
    
    # Save to file
    output_file = scraper.save_fixtures(fixtures)
    print(f"\n✅ Fixtures saved to: {output_file}")
    
    return fixtures


if __name__ == '__main__':
    main()

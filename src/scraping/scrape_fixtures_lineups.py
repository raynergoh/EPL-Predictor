import requests
from bs4 import BeautifulSoup
import re
import logging

from src.utils.cleaning import clean_team_name, clean_player_name

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

def scrape_ffscout_lineups_and_fixtures(url='https://www.fantasyfootballscout.co.uk/team-news/'):
    logger.info(f"Requesting Fantasy Football Scout team news page: {url}")
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")

    lineup_divs = soup.find_all("div", {"class": re.compile('formation')})
    team_headers = soup.find_all('header')
    next_match_divs = soup.find_all("div", class_="next-match")
    logger.info(f"Found {len(lineup_divs)} lineup divs, {len(team_headers)} headers, {len(next_match_divs)} next-match tags")

    if len(lineup_divs) > 20:
        lineup_divs = lineup_divs[1:21]
    else:
        lineup_divs = lineup_divs[:20]
    logger.info(f"Processing first {len(lineup_divs)} team lineups")

    all_team_data = []

    for i in range(len(lineup_divs)):
        # Team name, cleaning
        team_name_raw = team_headers[i+2].text.split('Next')[0].strip()
        team_name = clean_team_name(team_name_raw)
        logger.info(f"Parsing team: {team_name_raw} -> {team_name}")

        # Next match info
        next_match_html = next_match_divs[i]
        opp_text = next_match_html.text.strip().replace('Next Match:', '').strip().strip('" ')
        m = re.match(r"(.+?)\s*\((H|A)\)", opp_text)
        if m:
            opponent_raw = m.group(1).strip()
            home_away = m.group(2)
            opponent = clean_team_name(opponent_raw)
        else:
            opponent, home_away = None, None
            logger.warning(f"Could not parse opponent/home_away for {team_name}: got '{opp_text}'")

        # Predicted lineup (player name cleaning)
        players = [
            clean_player_name(li['title'])
            for li in lineup_divs[i].find_all('li')
            if li.has_attr('title')
        ]
        logger.info(f"Predicted lineup for {team_name}: {players}")

        all_team_data.append({
            "team": team_name,
            "opponent": opponent,
            "home_away": home_away,
            "predicted_lineup": players
        })

    logger.info(f"Scraped predicted lineups and fixtures for {len(all_team_data)} teams")
    return all_team_data

if __name__ == "__main__":
    team_fixtures_lineups = scrape_ffscout_lineups_and_fixtures()
    for entry in team_fixtures_lineups:
        print(entry)

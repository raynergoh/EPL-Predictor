import asyncio
import aiohttp
import pandas as pd
import os
from understat import Understat

async def fetch_and_save_understat_results(league='EPL', seasons=[2023, 2024], output_file="data/raw/epl_historical_results.csv"):
    matches = []
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        for year in seasons:
            print(f"Fetching season {year} data...")
            results = await understat.get_league_results(league, year)
            for match in results:
                matches.append({
                    "date": match["datetime"],
                    "home_team": match["h"]["title"],
                    "away_team": match["a"]["title"],
                    "home_goals": int(match["goals"]["h"]),
                    "away_goals": int(match["goals"]["a"]),
                    "home_xg": float(match["xG"]["h"]),
                    "away_xg": float(match["xG"]["a"]),
                    # add fields as needed from match
                })
    df = pd.DataFrame(matches)
    df["date"] = pd.to_datetime(df["date"])
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Saved historical match data to {output_file}")

if __name__ == "__main__":
    asyncio.run(fetch_and_save_understat_results())

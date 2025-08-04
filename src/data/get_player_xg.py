import aiohttp
import asyncio
import pandas as pd
from understat import Understat

async def get_league_players(league="EPL", season=2024, out_csv="data/understat_player_xg.csv"):
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        players = await understat.get_league_players(league, season)
        df = pd.DataFrame(players)
        df.to_csv(out_csv, index=False)
        print(f"Wrote {len(df)} player records to {out_csv}")

if __name__ == "__main__":
    asyncio.run(get_league_players())

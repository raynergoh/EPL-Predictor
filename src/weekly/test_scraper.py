"""
Test scraper to understand Premier League website structure.
"""

import requests
from bs4 import BeautifulSoup
import json
import re

def test_premierleague_page():
    """Test fetching data from Premier League website."""
    
    url = "https://www.premierleague.com/en/matches"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    
    print(f"Fetching: {url}\n")
    response = requests.get(url, headers=headers, timeout=15)
    print(f"Status: {response.status_code}\n")
    
    # Parse HTML
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Look for matchweek information
    scripts = soup.find_all('script')
    for script in scripts:
        script_text = script.string
        if script_text and 'ACTIVE_PL_MATCHWEEK_ID' in script_text:
            print("Found matchweek script:")
            # Extract matchweek ID
            match = re.search(r"ACTIVE_PL_MATCHWEEK_ID = '(\d+)'", script_text)
            if match:
                print(f"  Current matchweek: {match.group(1)}\n")
        
        # Look for fixture data in scripts
        if script_text and ('fixture' in script_text.lower() or 'match' in script_text.lower()):
            if len(script_text) > 1000:  # Likely contains data
                print(f"Found potential data script ({len(script_text)} chars)")
                # Try to find JSON data
                if '{' in script_text:
                    print("  Contains JSON-like data\n")
    
    # Look for fixture containers in HTML
    print("\nLooking for fixture elements...")
    
    # Common class patterns used by Premier League
    fixture_patterns = [
        'fixture',
        'match',
        'fixtures-container',
        'matchList',
        'match-card',
        'fixture-card'
    ]
    
    for pattern in fixture_patterns:
        elements = soup.find_all(class_=re.compile(pattern, re.I))
        if elements:
            print(f"  Found {len(elements)} elements matching '{pattern}'")
    
    # Check for data attributes
    data_elements = soup.find_all(attrs={'data-fixture': True})
    if data_elements:
        print(f"\n  Found {len(data_elements)} elements with data-fixture attribute")
    
    data_elements = soup.find_all(attrs={'data-match': True})
    if data_elements:
        print(f"  Found {len(data_elements)} elements with data-match attribute")
    
    # Save full HTML for inspection
    with open('data/debug_pl_page.html', 'w', encoding='utf-8') as f:
        f.write(soup.prettify())
    print("\n✓ Saved full HTML to data/debug_pl_page.html for inspection")

if __name__ == '__main__':
    test_premierleague_page()

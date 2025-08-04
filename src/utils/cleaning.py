import re

def clean_team_name(team_name: str) -> str:
    mapping = {
        "Brighton and Hove Albion": "Brighton",
        "Tottenham Hotspur": "Tottenham",
        "West Ham United": "West Ham",
        # Add more mappings as needed
    }
    team_name = team_name.strip()
    return mapping.get(team_name, team_name)

def clean_player_name(name: str) -> str:
    """
    If name has format "Surname (FirstName)", returns "FirstName Surname".
    Otherwise returns as is.
    Handles special cases (e.g., "Raya Martin (David)" -> "David Raya").
    """
    name = name.strip()
    # Special mappings for known naming issues
    manual_player_map = {
        "Raya Martin (David)": "David Raya",
        # Add more if needed
    }
    if name in manual_player_map:
        return manual_player_map[name]

    match = re.match(r'^(.+?)\s*\((.+?)\)$', name)
    if match:
        last_name = match.group(1).strip()
        first_name = match.group(2).strip()
        return f"{first_name} {last_name}"
    else:
        return name

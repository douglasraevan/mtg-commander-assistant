import requests
import time
from typing import List, Dict
import logging

def fetch_scryfall_data(card_list: List[str]) -> List[Dict]:
    """
    Fetches card data from Scryfall API for a list of card names.
    
    Args:
        card_list: List of card names to look up
        
    Returns:
        List of card data dictionaries from Scryfall
    """
    BASE_URL = "https://api.scryfall.com"
    RATE_LIMIT_DELAY = 0.1  # 100ms between requests to respect Scryfall's rate limits
    
    results = []
    
    for card_name in card_list:
        try:
            # Clean up card name and handle split cards
            clean_name = card_name.split('//')[0].strip()
            
            # Remove quantity prefix if present (e.g., "2 Mountain" -> "Mountain")
            if clean_name[0].isdigit():
                clean_name = ' '.join(clean_name.split()[1:])
            
            # Make API request
            response = requests.get(
                f"{BASE_URL}/cards/named",
                params={"fuzzy": clean_name},
                timeout=10
            )
            
            if response.status_code == 200:
                results.append(response.json())
            else:
                logging.warning(f"Failed to fetch data for card: {card_name}. Status code: {response.status_code}")
            
            # Respect rate limiting
            time.sleep(RATE_LIMIT_DELAY)
            
        except Exception as e:
            logging.error(f"Error fetching data for card {card_name}: {str(e)}")
            continue
    
    return results

# Example usage:
if __name__ == "__main__":
    # Read card list from file
    with open("kaust-current-20240524-030310.txt", "r") as f:
        cards = [line.strip() for line in f if line.strip()]
    
    # Fetch data
    card_data = fetch_scryfall_data(cards)
    
    # Print some basic info as verification
    for card in card_data:
        print(f"{card['name']}: {card['mana_cost']} - {card['type_line']}")

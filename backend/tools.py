from langchain.tools import tool
from backend.api_client import client
import json

def resolve_location_logic(query: str) -> str:
    """
    Search for a location (suburb/state) to get the correct location details.
    """
    results = client.auto_complete(query)
    return json.dumps(results)

@tool
def resolve_location(query: str) -> str:
    """
    Search for a location (suburb/state) to get the correct location details.
    Use this when the user provides a suburb name to ensure strictly matching location is found.
    Returns a list of matching locations with their State and Slug.
    """
    return resolve_location_logic(query)

@tool
def search_properties(location_slug: str, state: str, min_price: int = None, max_price: int = None, min_bedrooms: int = None, max_bedrooms: int = None) -> str:
    """
    Search for properties in a specific location.
    'location_slug' and 'state' MUST be obtained from the 'resolve_location' tool results.
    'min_price', 'max_price', 'min_bedrooms', and 'max_bedrooms' are optional filters.
    """
    results = client.search_listings(
        location_slug=location_slug,
        state=state,
        min_price=min_price,
        max_price=max_price,
        min_bedrooms=min_bedrooms,
        max_bedrooms=max_bedrooms
    )
    results = results["tieredResults"][0]["results"]
    return json.dumps(results)

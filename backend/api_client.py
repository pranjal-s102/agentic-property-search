import os
import requests
from dotenv import load_dotenv

load_dotenv()

RAPIDAPI_HOST = "realty-in-au.p.rapidapi.com"

class RealtyClient:
    def __init__(self):
        self.base_url = f"https://{RAPIDAPI_HOST}"
        
    @property
    def headers(self):
        key = os.getenv("RAPIDAPI_KEY")
        if not key:
            raise ValueError("RAPIDAPI_KEY not found in environment variables")
        return {
            "X-RapidAPI-Key": key,
            "X-RapidAPI-Host": RAPIDAPI_HOST
        }

    def auto_complete(self, query: str):
        """
        Get auto-complete suggestions for a location query.
        """
        url = f"{self.base_url}/auto-complete"
        querystring = {"query": query}
        
        try:
            response = requests.get(url, headers=self.headers, params=querystring)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error in auto_complete: {e}")
            return {"error": str(e)}

    def search_listings(self, location_slug: str, state: str, min_price: int = None, max_price: int = None, min_bedrooms: int = None, max_bedrooms: int = None, property_type: list = None, keywords: list = None, surrounding: bool = True, channel: str = "buy"):
        """
        Search for property listings.
        Note: The API parameters might need adjustment based on specific endpoint docs.
        Using 'properties/list' based on assumption/plan.
        """
        url = f"{self.base_url}/properties/list"
        
        # Construct search options
        
        # Filters
        # Filters
        filters = {}
        if min_price:
            filters["minimumPrice"] = int(min_price)
        if max_price:
            filters["maximumPrice"] = int(max_price)
        if min_bedrooms:
            filters["minimumBedrooms"] = int(min_bedrooms)
        if max_bedrooms:
            filters["maximumBedrooms"] = int(max_bedrooms)
        if property_type:
            filters["propertyTypes"] = property_type


        # Payload structure depends on the specific API. 
        # Assuming a GET request with query params or POST with body.
        # RapidAPI Realty in AU usually uses GET with complex query params or POST.
        # Let's try to align with a common pattern or what was implied. 
        # The prompt asked to fetch from: https://rapidapi.com/apidojo/api/realty-in-au/playground/apiendpoint_ad510095-1a60-4fbe-bd60-903cbfbe5533
        # That endpoint usually takes query params.
        
        # Attempting to map parameters to what likely works for this API:
        # It often requires a 'limit', 'offset', 'state', 'suburb' etc.
        
        # Since I don't have the exact docs open, I will implement a generic mapped request 
        # based on the inputs and refine if validation fails.
        
        querystring = {
            "channel": channel,
            "searchLocation": location_slug+', '+state, # slug is often the suburb name in these APIs
            "page": 1,
            "pageSize": 5,
            "sortType": "relevance"
        }
        
        if max_price:
            querystring["maximumPrice"] = str(max_price)
        if min_bedrooms:
            querystring["minimumBedrooms"] = str(min_bedrooms)
        if max_bedrooms:
            querystring["maximumBedrooms"] = str(max_bedrooms)
            
        # Surrounding Suburbs flag (assuming 'surroundingSuburbs' is the correct param key for this API)
        querystring["surroundingSuburbs"] = "true" if surrounding else "false"

        if keywords:
            querystring["keywords"] = ",".join(keywords)

        try:
            response = requests.get(url, headers=self.headers, params=querystring)
            # Debugging: Print raw response to see what's wrong
            response.raise_for_status()

            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error in search_listings: {e}")
            return {"error": str(e)}
        except ValueError as e:
             # JSON Decode error
            print(f"JSON Decode Error: {e}")
            return {"error": "Failed to parse API response. See server logs for raw output."}

    def fetch_all_listings(self, location_slug: str, state: str, channel: str = "buy", keywords: list = None, max_pages: int = 50) -> list:
        """
        Fetch ALL listings for a suburb (up to max_pages * 20 = 1000 properties).
        
        Guardian Rule: This should ONLY be called when:
        - suburb_confirmed == True
        - suburb is NOT in indexed_suburbs
        
        Args:
            location_slug: The suburb slug (e.g., "richmond")
            state: The state code (e.g., "VIC")
            channel: "buy" or "rent"
            max_pages: Maximum pages to fetch (default 50 = 1000 properties)
            
        Returns:
            List of all property dicts
        """
        url = f"{self.base_url}/properties/list"
        all_listings = []
        
        # for page in range(1, max_pages + 1):
        querystring = {
            "channel": channel,
            "searchLocation": f"{location_slug}, {state}",
            # "page": page,
            "pageSize": 1000,  # Max per page
            "pageSize": 1000,  # Max per page
            "sortType": "relevance"
        }
        
        if keywords:
            querystring["keywords"] = ",".join(keywords)
        
        try:
            response = requests.get(url, headers=self.headers, params=querystring)
            response.raise_for_status()
            data = response.json()
            
            # Extract listings from response
            listings = []
            if isinstance(data, dict):
                tiered = data.get("tieredResults", [])
                for tier in tiered:
                    listings.extend(tier.get("results", []))
            elif isinstance(data, list):
                listings = data
            
            # if not listings:
            #     # No more results
            #     break
                
            all_listings.extend(listings)
            print(f"[API] Fetched pages : {len(listings)} properties")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching page : {e}")
            # break
            return
        except ValueError as e:
            print(f"JSON Decode Error on page : {e}")
            # break
            return 
        
        print(f"[API] Total fetched for {location_slug}: {len(all_listings)} properties")
        return all_listings

client = RealtyClient()

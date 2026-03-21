"""
Cache Module for Property Search.

Handles disk-based caching of suburb data with TTL.
Rules:
- Cache by suburb_slug
- Invalidate after 24 hours (86400 seconds)
- File-based persistence in 'cache_data' directory
"""

import os
import json
import time
from typing import Optional, Dict, List

# Cache directory within backend
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache_data")
TTL_SECONDS = 86400  # 24 hours

class SuburbCache:
    def __init__(self):
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR, exist_ok=True)
            
    def _get_path(self, suburb_slug: str) -> str:
        safe_slug = "".join([c if c.isalnum() or c in "-_" else "_" for c in suburb_slug.lower()])
        return os.path.join(CACHE_DIR, f"{safe_slug}.json")
    
    def get(self, suburb_slug: str) -> Optional[List[Dict]]:
        """
        Retrieve cached listings for a suburb if valid.
        Returns None if missing or expired.
        """
        path = self._get_path(suburb_slug)
        if not os.path.exists(path):
            return None
            
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            timestamp = data.get("timestamp", 0)
            if time.time() - timestamp > TTL_SECONDS:
                print(f"[Cache] Expired for {suburb_slug}")
                return None
                
            print(f"[Cache] Hit for {suburb_slug}")
            return data.get("listings")
        except Exception as e:
            print(f"[Cache] Read error: {e}")
            return None
            
    def set(self, suburb_slug: str, listings: List[Dict]):
        """Save listings to cache with current timestamp."""
        path = self._get_path(suburb_slug)
        data = {
            "timestamp": time.time(),
            "listings": listings
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            print(f"[Cache] Saved {len(listings)} listings for {suburb_slug}")
        except Exception as e:
            print(f"[Cache] Write error: {e}")

# Singleton
suburb_cache = SuburbCache()

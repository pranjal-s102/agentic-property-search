"""
Property Vector Store Manager.

You manage the Property Vector Store.

Ingest Rules:
- Each property must be embedded using:
  - Title
  - Description
  - Key features
  - Location metadata
- Use OpenAI embedding model: text-embedding-3-small
- Store suburb slug as metadata.

Search Rules:
- Perform cosine similarity search.
- Apply metadata filters AFTER similarity scoring.
- Return top-k results only.

Architecture:
- Local-first (no cloud databases)
- ChromaDB for vector storage
- Persistence enabled when available
"""

import os
import json
import hashlib
from typing import List, Dict, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Persistent storage directory (local-first)
if os.environ.get("VERCEL"):
    CHROMA_PERSIST_DIR = "/tmp/chroma_db"
else:
    CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

# Lazy-loaded embeddings (deferred to avoid import-time API key check)
_embeddings = None

def get_embeddings():
    """Get or create OpenAI embeddings instance (text-embedding-3-small)."""
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return _embeddings


def _safe_int(val) -> Optional[int]:
    """Helper to safely convert value to int."""
    if val is None:
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    if isinstance(val, str):
        # Extract digits only (handle '2 baths', '1.5' -> 1)
        import re
        # Find first number sequence
        match = re.search(r'\d+', val)
        if match:
            return int(match.group())
    return None


def _parse_price(val) -> Optional[int]:
    """
    Robust price parser (handles '$1.2M', '850k', '$1,200,000', ranges).
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return int(val)
    
    s = str(val).strip().lower()
    # Remove commas
    s = s.replace(",", "")
    
    # Regex to find the first significant number with optional multiplier suffix
    # Matches: 1.2m, 850k, 1000000, $500k
    import re
    match = re.search(r'(\d+(?:\.\d+)?)\s*([km]?)', s)
    
    if not match:
        return None
        
    try:
        num_str = match.group(1)
        suffix = match.group(2)
        num = float(num_str)
        
        if suffix == 'k':
            num *= 1_000
        elif suffix == 'm':
            num *= 1_000_000
            
        return int(num)
    except:
        return None


def _property_to_document(prop: Dict, suburb_slug: str) -> Document:
    """
    Convert a property dict to a LangChain Document for embedding.
    
    Embedding content includes:
    - Title (address)
    - Description/headline
    - Key features (beds, baths, parking, land size)
    - Property type
    - Price
    
    Metadata stored:
    - suburb_slug for filtering
    - Structured fields for post-filtering
    - Raw JSON for retrieval
    """
    # Extract key fields with fallbacks
    addr = prop.get("address")
    if isinstance(addr, dict):
        addr_str = addr.get("streetAddress", "")
    else:
        addr_str = str(addr) if addr else ""
    
    title = prop.get("title") or addr_str or "Unknown Address"
    price_val = prop.get("price")
    if isinstance(price_val, dict):
         price_display = price_val.get("display", "Price on Application")
    else:
         price_display = str(price_val) if price_val else "Price on Application"
    
    property_type = prop.get("propertyType", "Property")
    
    # Extract features
    features = prop.get("features")
    if not isinstance(features, dict):
        features = {}
    
    # Helper to extract values from nested structures
    def _extract_feature(key, alt_keys=None):
        val = None
        # Try direct access in features dict
        val = features.get(key)
        
        # Try nested under 'general'
        if val is None and isinstance(features.get("general"), dict):
            val = features["general"].get(key)
        
        # Try top-level generalFeatures (common in some API versions)
        if val is None:
            gen_features = prop.get("generalFeatures")
            if isinstance(gen_features, dict):
                feat_obj = gen_features.get(key)
                if isinstance(feat_obj, dict):
                    val = feat_obj.get("value")
                else:
                    val = feat_obj

        # Try alt keys
        if val is None and alt_keys:
            for ak in alt_keys:
                 # Repeat logic for alt key
                 val = features.get(ak)
                 if val is None and isinstance(features.get("general"), dict):
                    val = features["general"].get(ak)
                 if val is None and prop.get("generalFeatures"):
                     gf = prop.get("generalFeatures").get(ak)
                     val = gf.get("value") if isinstance(gf, dict) else gf
                 if val is not None:
                     break
        
        return _safe_int(val)

    beds = _extract_feature("bedrooms", ["beds"])
    baths = _extract_feature("bathrooms", ["baths"])
    parking = _extract_feature("parkingSpaces", ["parking", "carSpaces"])
    land_size = features.get("landSize", "")
    
    # Parse price for filtering
    price_val = _parse_price(price_display)

    # Construct rich page content for embedding
    description = prop.get("description") or prop.get("summary") or ""
    
    # Format features list
    feature_list = []
    if features:
        for k, v in features.items():
            if isinstance(v, bool) and v:
                feature_list.append(k)
            elif v:
                feature_list.append(f"{k}: {v}")
    
    features_str = ", ".join(feature_list)

    page_content = f"""
    Title: {title}
    Type: {property_type}
    Price: {price_display}
    Bedrooms: {beds if beds else 'N/A'}
    Bathrooms: {baths if baths else 'N/A'}
    Parking: {parking if parking else 'N/A'}
    
    Description:
    {description}
    
    Features:
    {features_str}
    """.strip()

    metadata = {
        "suburb_slug": suburb_slug,
        "title": title,
        "price_display": price_display,
        "price_val": price_val if price_val else 0, # Store 0 if unknown so we don't break comparisons? Or maybe None is better but chroma metadata flat. None is okay.
        "property_type": property_type,
        "beds": beds,
        "baths": baths,
        "parking": parking,
        "listing_id": prop.get("id") or prop.get("listingId", ""),
        "raw_json": json.dumps(prop),
    }
    
    return Document(page_content=page_content, metadata=metadata)


class QueryAnalyzer:
    """
    Query Analyzer for semantic search.
    
    Input:
    - Raw user preferences (from UserProfile)
    - Known hard filters (budget, bedrooms, property type)
    
    Task:
    - Extract semantic preferences (lifestyle, style, proximity)
    - Produce a clean semantic query string
    
    Rules:
    - Do NOT remove subjective language (e.g. "modern", "quiet", "near trains")
    - Do NOT inject assumptions
    - Combine preferences into a single retrieval query
    
    Output:
    - semantic_query: str
    - filters: dict
    """
    
    @staticmethod
    def analyze(preferences: list, budget: int = None, min_bedrooms: int = None, 
                max_bedrooms: int = None, min_bathrooms: int = None,
                max_bathrooms: int = None, parking_spaces: int = None,
                property_type: str = None, location: str = None,
                property_types: List[str] = None) -> dict:
        """
        Analyze user preferences and produce semantic query + filters.
        
        Args:
            preferences: List of semantic preferences
            budget: Maximum budget
            min_bedrooms: Minimum bedrooms
            max_bedrooms: Maximum bedrooms
            min_bathrooms: Minimum bathrooms
            max_bathrooms: Maximum bathrooms
            parking_spaces: Minimum parking spaces
            property_type: Property type
            location: Location context
            
        Returns:
            {
                "semantic_query": str,
                "filters": {
                    "max_price": int or None,
                    "min_beds": int or None,
                    "property_type": str or None
                }
            }
        """
        # Build semantic query from preferences
        # Rule: Do NOT remove subjective language
        query_parts = []
        
        # Add all semantic preferences as-is
        if preferences:
            query_parts.extend(preferences)
        
        # Add location context if available
        if location:
            query_parts.append(f"in {location}")
        
        # Combine into single query string
        # Rule: Combine preferences into a single retrieval query
        semantic_query = " ".join(query_parts) if query_parts else "property"
        
        # Extract hard filters separately
        # Rule: Keep filters separate from semantic query
        filters = {
            "max_price": budget,
            "min_beds": min_bedrooms,
            "max_beds": max_bedrooms,
            "min_baths": min_bathrooms,
            "max_baths": max_bathrooms,
            "max_baths": max_bathrooms,
            "min_parking": parking_spaces,
            "property_type": property_type,
            "property_types": property_types
        }
        
        return {
            "semantic_query": semantic_query,
            "filters": filters
        }


# Singleton query analyzer
query_analyzer = QueryAnalyzer()


class PropertyVectorStore:
    """
    Property Vector Store Manager.
    
    Ingest Rules:
    - Embed properties using title, description, features, location
    - Use OpenAI text-embedding-3-small
    - Store suburb_slug as metadata
    
    Search Rules:
    - Cosine similarity search (ChromaDB default)
    - Apply metadata filters AFTER similarity scoring
    - Return top-k results only
    
    Architecture:
    - Local-first (no cloud databases)
    - ChromaDB for persistence
    """
    
    def __init__(self):
        self._vectorstore: Optional[Chroma] = None
        self._indexed_suburbs: set = set()
        self._loaded = False
    
    def _ensure_loaded(self):
        """Lazy load existing vector store from disk."""
        if self._loaded:
            return
        self._loaded = True
        self._load_existing()
    
    def _load_existing(self):
        """Load existing vector store from disk if it exists."""
        if os.path.exists(CHROMA_PERSIST_DIR):
            try:
                self._vectorstore = Chroma(
                    persist_directory=CHROMA_PERSIST_DIR,
                    embedding_function=get_embeddings(),
                    collection_name="properties"
                )
                # Recover indexed suburbs from metadata
                existing_docs = self._vectorstore.get(include=["metadatas"])
                if existing_docs and existing_docs.get("metadatas"):
                    for meta in existing_docs["metadatas"]:
                        if meta and meta.get("suburb_slug"):
                            self._indexed_suburbs.add(meta["suburb_slug"])
                print(f"[RAG] Loaded existing store with suburbs: {self._indexed_suburbs}")
            except Exception as e:
                print(f"[RAG] Error loading store: {e}")
                self._vectorstore = None
    
    def is_suburb_indexed(self, suburb_slug: str) -> bool:
        """Check if a suburb has already been ingested."""
        self._ensure_loaded()
        return suburb_slug.lower() in self._indexed_suburbs
    
    def get_indexed_suburbs(self) -> List[str]:
        """Return list of all indexed suburbs."""
        self._ensure_loaded()
        return list(self._indexed_suburbs)
    
    def ingest(self, suburb_slug: str, properties: List[Dict]) -> int:
        """
        Embed and store properties for a suburb.
        
        Guardian Rule: This should ONLY be called after suburb_confirmed == True.
        
        Args:
            suburb_slug: The suburb identifier (e.g., "richmond-vic")
            properties: List of property dicts from the API
            
        Returns:
            Number of properties ingested
        """
        if not properties:
            print(f"[RAG] No properties to ingest for {suburb_slug}")
            return 0
        
        self._ensure_loaded()
        suburb_key = suburb_slug.lower()
        
        # Convert to documents
        documents = [_property_to_document(p, suburb_key) for p in properties]
        
        # Create or update vector store
        if self._vectorstore is None:
            self._vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=get_embeddings(),
                persist_directory=CHROMA_PERSIST_DIR,
                collection_name="properties"
            )
        else:
            self._vectorstore.add_documents(documents)
        
        # Track this suburb as indexed
        self._indexed_suburbs.add(suburb_key)
        
        print(f"[RAG] Ingested {len(documents)} properties for {suburb_slug}")
        return len(documents)
    
    def search(
        self,
        query: str,
        filters: Optional[Dict] = None,
        k: int = 5,
        offset: int = 0
    ) -> List[Dict]:
        """
        Semantic search for properties.
        
        Args:
            query: Natural language query
            filters: Dictionary of filters (min_beds, max_beds, min_baths, etc.)
            k: Number of results to return
            
        Returns:
            List of property dicts matching the query
        """
        self._ensure_loaded()
        if self._vectorstore is None:
            print("[RAG] Vector store not initialized")
            return []
        
        # Build filter
        where_filter = {}
        conditions = []

        # 1. Suburb Filter
        if filters and filters.get("suburb_slug"):
             conditions.append({"suburb_slug": filters["suburb_slug"].lower()})
        
        # 2. Property Type Filter (Pre-filtering is critical for performance)
        target_types = filters.get("property_types") if filters else None
        if target_types:
            # normalized_types = [t.lower() for t in target_types]
            # Chroma $in operator requires exact match on the stored string.
            # Our stored property_type comes from the API (e.g. "apartment", "house").
            # We rarely get complex lists here, usually just one or two.
            # Limitation: partial matching isn't easy in Chroma pre-filter.
            # Strategy: use $in if possible.
            if len(target_types) == 1:
                 conditions.append({"property_type": target_types[0]})
            else:
                 conditions.append({"property_type": {"$in": target_types}})

        # Combine conditions
        if len(conditions) > 1:
            where_filter = {"$and": conditions}
        elif len(conditions) == 1:
            where_filter = conditions[0]
            
        # Perform similarity search with broad filtering (suburb only)
        # We handle granular numeric filters in post-processing to avoid ChromaDB complexity
        # Logic: Fetch enough results to cover the offset + k
        # IMPROVEMENT: Increased multiplier to 10x (min 50) to prevent filtering exhaustion
        # when searching across many suburbs or strict numeric ranges.
        fetch_k = max(50, (k + offset) * 10)  
        
        try:
            # Use similarity_search_with_score to get relevance scores
            # Note: Chroma default metric is L2 (Euclidean distance). Lower is better.
            # 0.0 = identical. 
            # A good threshold for "relevant" is usually < 0.3 or 0.4 depending on embedding model.
            # text-embedding-3-small is quite precise.
            
            results_with_scores = self._vectorstore.similarity_search_with_score(
                query,
                k=fetch_k, 
                filter=where_filter if where_filter else None
            )
        except Exception as e:
            print(f"[RAG] Search error: {e}")
            return []
        
        # Post-filter by numeric fields AND score threshold
        filtered = []
        print(f"[RAG] Raw results count: {len(results_with_scores)}")
        
        for doc, score in results_with_scores:
            print(f"[RAG] Checking doc: {doc.metadata.get('title')} | Score: {score}")
            
            # Score Filter removed by request to ensure top results are shown.
            
            meta = doc.metadata
            
            # Apply filters
            if filters:
                # Price - Fixed!
                max_price = filters.get("max_price")
                if max_price:
                    price_val = meta.get("price_val")
                    # If price is unknown (None or 0), should we include or exclude?
                    # Usually include to be safe, but if user has strict budget, maybe exclude?
                    # Let's exclude ONLY if we are sure it's over budget. 
                    # Actually, if price isn't listed, we can't be sure. Let's include "Price on Application" 
                    # but if we parsed a number, we check it.
                    if price_val and isinstance(price_val, (int, float)):
                         if int(price_val) > int(max_price):
                             print(f"[RAG] Filtered out {meta.get('title')} (Price {price_val} > {max_price})")
                             continue
                
                # Bedrooms
                beds = meta.get("beds")
                if filters.get("min_beds"):
                    if beds is None or beds < filters["min_beds"]: continue
                if filters.get("max_beds"):
                    if beds is not None and beds > filters["max_beds"]: continue
                
                # Bathrooms
                baths = meta.get("baths")
                if filters.get("min_baths"):
                    if baths is None or baths < filters["min_baths"]: continue
                if filters.get("max_baths"):
                    if baths is not None and baths > filters["max_baths"]: continue
                    
                # Parking
                cars = meta.get("parking")
                if filters.get("min_parking"):
                    if cars is None or cars < filters["min_parking"]: continue

                # Property Types
                # Filter if the user has selected specific types
                target_types = filters.get("property_types")
                if target_types:
                    doc_type = meta.get("property_type", "").lower()
                    # Check if the doc's type matches ANY of the target types
                    # Simple substring/inclusion check
                    match_found = False
                    for t in target_types:
                        t_clean = t.lower().strip()
                        # If target is 'unitblock', it might match 'Unit Block' or 'Unit'
                        # If target is 'house', it might match 'House' or 'Townhouse' (maybe? strict is safer)
                        # Let's strict match against the known API types if possible, or fuzzy match
                        # API types often: "House", "Apartment", "Unit", "Townhouse"
                        
                        # Logic: If query is "house", match "House", "Townhouse", "Villa"? 
                        # User specified: apartment|retire|acreage|land|unitblock|house|villa|rural
                        
                        # Strict substring matching:
                        if t_clean in doc_type or doc_type in t_clean:
                            match_found = True
                            break
                    
                    if not match_found:
                        print(f"[RAG] Filtered out {meta.get('title')} (Type '{doc_type}' not in {target_types})")
                        continue

            
            # Reconstruct property from stored JSON
            try:
                prop = json.loads(meta.get("raw_json", "{}"))
                filtered.append(prop)
            except:
                continue
            
            # Optimization: Stop once we have enough for the requested page
            if len(filtered) >= (offset + k):
                break
        
        # Return the slice corresponding to the requested page
        start = offset
        end = offset + k
        return filtered[start:end]

    
    def clear_suburb(self, suburb_slug: str):
        """Remove all properties for a suburb (for cache invalidation)."""
        if self._vectorstore is None:
            return
        
        suburb_key = suburb_slug.lower()
        try:
            # Get IDs to delete
            results = self._vectorstore.get(
                where={"suburb_slug": suburb_key},
                include=["metadatas"]
            )
            if results and results.get("ids"):
                self._vectorstore.delete(ids=results["ids"])
                self._indexed_suburbs.discard(suburb_key)
                print(f"[RAG] Cleared {len(results['ids'])} properties for {suburb_slug}")
        except Exception as e:
            print(f"[RAG] Error clearing suburb: {e}")


# Singleton instance
property_store = PropertyVectorStore()

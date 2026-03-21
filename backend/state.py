import operator
from typing import Annotated, List, Optional, Dict, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage

class UserProfile(BaseModel):
    name: Optional[str] = Field(None, description="The user's name for personalized greetings.")
    intent: Optional[Literal["buy", "rent", "sell"]] = Field(None, description="The user's intent: buy, rent, or sell.")
    location: Optional[str] = Field(None, description="The preferred location (suburb/state) for the property.")
    location_slug: Optional[str] = Field(None, description="The resolved location slug for API calls.")
    state: Optional[str] = Field(None, description="The resolved state code (e.g. VIC, NSW).")
    budget: Optional[int] = Field(None, description="The maximum budget in dollars.")
    budget_period: Optional[Literal["weekly", "monthly", "unknown"]] = Field(None, description="The confirmed period for the budget: 'weekly', 'monthly', or 'unknown' if unclear.")
    bedrooms: Optional[int] = Field(None, description="The minimum number of bedrooms.")
    preferences: List[str] = Field(default_factory=list, description="Long-term semantic preferences (e.g. 'modern style', 'close to transport').")
    include_surrounding: Optional[bool] = Field(None, description="Whether to include surrounding suburbs. None=Unknown, True=Yes, False=No.")
    
    # Comprehensive Filters
    min_bedrooms: Optional[int] = Field(None, description="Minimum number of bedrooms.")
    max_bedrooms: Optional[int] = Field(None, description="Maximum number of bedrooms.")
    min_bathrooms: Optional[int] = Field(None, description="Minimum number of bathrooms.")
    max_bathrooms: Optional[int] = Field(None, description="Maximum number of bathrooms.")
    parking_spaces: Optional[int] = Field(None, description="Minimum number of parking spaces.")
    property_types: Optional[List[str]] = Field(None, description="List of preferred property types (e.g. ['house', 'apartment', 'villa']).")
    keywords: Optional[List[str]] = Field(None, description="List of specific facilities/keywords (e.g. ['pool', 'garage', 'gym', 'aircon']).")
    search_radius: Optional[int] = Field(None, description="Search radius in kilometers (e.g. 10, 50).")

class AgentState(TypedDict):
    """
    The central state of the agent, persisting across the graph execution.
    It is designed to be serializable and explicitly tracks the conversation and functional state.
    """
    
    # --- Conversation History ---
    # Holds the raw input from the user and responses from the agent.
    # Annotated with operator.add to ensure messages are appended, not overwritten.
    messages: Annotated[List[BaseMessage], operator.add]
    
    # --- Entity Memory (Structured Data) ---
    # The 'Extracted Criteria'. This is the source of truth for the property search.
    # Separating this from 'messages' ensures we don't rely on the LLM's context window 
    # to remember parameters perfectly every time.
    user_profile: UserProfile
    
    # Explicitly tracks which fields (from UserProfile) are still needed before a search can run.
    # Calculated by the Router node.
    missing_fields: List[str]
    
    # --- Search Context ---
    # The raw results from the API. Storing this allows the 'Analyzer' and 'Presenter' 
    # to access the data without re-running the tool.
    listings: List[Dict]
    
    # Flag to prevent infinite search loops. The Router checks this to decide 
    # whether to search or analyze.
    search_executed: bool
    
    # Explicit status tracking for the search operation.
    # Used for deterministic routing: success -> analyzer, empty -> empty_handler, error -> presenter.
    search_status: Literal["not_started", "success", "empty", "error"]
    
    # Explicit error message if search_status is 'error'.
    # Allows the Presenter to explain what went wrong to the user.
    error_message: Optional[str]

    # --- Pagination Control ---
    # Tracks the current offset for search results (e.g. 0, 5, 10).
    search_offset: int
    
    # --- Internal Reasoning ---
    # The 'Analyzer' node's output. Separation of 'Analysis' (reasoning) from 
    # 'Presentation' (user-facing text) improves quality.
    analysis: Optional[str]
    recommendation: Optional[str]
    
    # --- Control Flow ---
    # The explicit decision from the Router node on where to go next.
    # Used by the conditional edges in the graph.
    next_node: Optional[str]
    
    # --- RAG Control (Guardian Enforcement) ---
    # Tracks whether the user has confirmed the suburb for property search.
    # Guardian Rule: NEVER fetch listings unless suburb_confirmed == True.
    suburb_confirmed: bool
    
    # Tracks which suburbs have been ingested into the vector store.
    # Guardian Rule: NEVER fetch listings for a suburb in this list.
    indexed_suburbs: List[str]
    
    # --- Greeting Flow ---
    # Tracks if this is the first interaction in a new session.
    # Used by Router to trigger greeting flow.
    is_first_interaction: bool
    
    # --- Off-Topic Detection ---
    # Tracks if the last message was off-topic or inappropriate.
    # Used by Router to redirect to off_topic_handler.
    off_topic_detected: bool

    # --- Robustness & Loop Detection ---
    # Tracks how many times we've tried to get the SAME field but failed.
    # Used to trigger "Help Mode" or simpler prompts.
    consecutive_failures: int
    last_target_field: Optional[str]

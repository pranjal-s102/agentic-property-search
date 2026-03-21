import json
from datetime import datetime
from typing import Literal, List, Dict, Optional
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from backend.state import AgentState, UserProfile
from backend.tools import resolve_location, search_properties, resolve_location_logic
from backend.rag import property_store, query_analyzer
from backend.cache import suburb_cache
# Re-import api_client directly if needed for cleaner search logic, 
# but utilizing the existing tool wrappers is fine if they return data usable by the node.
# However, the user asked for "Search execution node", so we might call the logic directly.
from backend.api_client import client

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# --- prompts ---
INTERVIEWER_SYSTEM = """You are a friendly Australian Real Estate Agent.
Your goal is to collect missing information from the user to help find their perfect property.
You need to know:
- Location (Suburb)
- Budget (Max price)
- Bedrooms (Min count)
- Property Type (House, Apartment, etc.)

Current missing fields: {missing_fields}

Ask ONE clear, natural question to get the missing information.
If the user mentions a location, verify if it's specific enough (State).
Context:
{user_context}
"""

ANALYZER_SYSTEM = """You are an expert Property Analyst.

GOAL: Provide a factual analysis of the listings found relative to the User Profile.

INPUT DATA:
User Profile: {user_profile}
Listings: {listings}

OUTPUT FORMAT:
Provide a concise analysis covering:
1. **Summary Match**: How well these listings match the criteria (e.g. "Found 3 properties in Richmond under $800k").
2. **Price Analysis**: Range, median, or value assessment.
3. **Features & Trade-offs**: Note if criteria are barely met (e.g. "Only 1 has 2 bathrooms") or exceeded.

CONSTRAINTS:
- **NO ACTIONS**: Do NOT suggest contacting agents or visiting.
- **NO RECOMMENDATIONS**: Do NOT suggest broadening search (that is for the Presenter).
- **NO HALLUCINSATIONS**: Stick strictly to the provided listing data.
- Keep it internal and analytical (this is for the Presenter node to read, not the user).
"""

PRESENTER_SYSTEM = """You are the Property Results Presenter.

GOAL: Present results clearly and concisely.

INPUT:
- Analysis of findings
- Ranked list of properties (Top 5)

RULES:
- Emphasize WHY each property matches the user's preferences (e.g. "This has the quiet garden you wanted").
- Be neutral and informative.
- Do NOT mention technically details like "RAG", "embeddings", or "cosine similarity".
- Focus on relevance to the user's specific request.

SCENARIO 1: SEARCH SUCCESS
1. Summarize findings naturally (e.g. "I found a few great options in Richmond matching your criteria.").
2. Highlight the best matches using the analysis.
3. Suggest clear next steps (e.g. "Would you like to see more details on any of these?").
4. **REQUIRED**: Output the property cards in the strictly formatted JSON block at the end.

SCENARIO 2: SEARCH ERROR
1. Apologize professionally.
2. Explain what went wrong (simple terms).
3. Ask if they want to try again.
4. Do NOT output the JSON block.

OUTPUT JSON FORMAT (Success Only):
At the very end of your response, strictly follow this format:
```json_properties
[
  {
    "title": "123 Example St, Richmond",
    "price": "$850,000",
    "beds": 2,
    "baths": 1,
    "image": "url_if_available",
    "link": "link_if_available"
  }
]
```
"""

EMPTY_HANDLER_SYSTEM = """You are a helpful Real Estate Agent.

CONTEXT: 
The user's search returned NO RESULTS.
User Profile: {user_profile}

GOAL: 
1. Acknowledge no results were found (be empathetic but professional).
2. Briefly explain WHY (refer to specific constraints like budget or location).
3. Propose 2-3 concrete refinements to the search. 
   - Example: "Increase budget to $X" or "Include neighboring suburb Y".
4. Do NOT assume what the user wants. Ask them which refinement they prefer.
5. Do NOT apply changes automatically.

Output should be short and encouraging.
"""

GUARDIAN_SYSTEM = """You are a smart data extractor. 
Your job is to update the UserProfile based on the conversation history.
Update ONLY the fields that have changed or act as new information.
Keep existing values if not contradicted.

Current Profile:
{current_profile}
"""

# --- Nodes ---

class GuardianUpdater(BaseModel):
    """
    Patch model for UserProfile.
    Fields are Optional to indicate "partial update".
    """
    name: Optional[str] = Field(None, description="User's name if they introduce themselves (e.g. 'I'm Alex', 'My name is Sarah').")
    intent: Optional[Literal["buy", "rent", "sell"]] = Field(None, description="Updates intent if explicitly changed.")
    location: Optional[str] = Field(None, description="Full location string (e.g. 'Richmond, VIC'). Normalize to 'Suburb, State'.")
    budget: Optional[int] = Field(None, description="Max budget in integers. Normalize '800k' to 800000.")
    min_bedrooms: Optional[int] = Field(None, description="Min bedrooms as integer.")
    max_bedrooms: Optional[int] = Field(None, description="Max bedrooms as integer.")
    # bedrooms: Optional[int] = Field(None, description="Legacy field. Use min_bedrooms instead.")
    min_bathrooms: Optional[int] = Field(None, description="Min bathrooms as integer.")
    max_bathrooms: Optional[int] = Field(None, description="Max bathrooms as integer.")
    parking_spaces: Optional[int] = Field(None, description="Min parking spaces as integer.")
    include_surrounding: Optional[bool] = Field(None, description="True if user wants surrounding suburbs, False if they want strict location only.")
    new_preferences: Optional[List[str]] = Field(None, description="Extract semantic search preferences (e.g. 'high ceilings', 'modern', 'near park'). Exclude technical needs like '3 bedrooms'.")
    suburb_confirmation: Optional[bool] = Field(None, description="True if user confirms the suburb (e.g. 'yes', 'correct', 'go ahead', 'search there'). False if user rejects it. None if not applicable.")
    off_topic: Optional[bool] = Field(None, description="True ONLY if the message is completely unrelated to property search (e.g., jokes, illegal acts). Set to False if the user is answering ANY question asked by the agent (e.g. 'Name?', 'Budget?', 'Location?').")
    budget_period: Optional[Literal["weekly", "monthly", "unknown"]] = Field(None, description="The period for the budget if specified (e.g. 'pw', 'per week' -> 'weekly', 'pm', 'per month' -> 'monthly'). If just a number is given, set 'unknown'.")
    load_more: Optional[bool] = Field(None, description="True if the user explicitly asks to see MORE results or the NEXT batch (e.g. 'show me more', 'next 5', 'what else is there').")
    property_types: Optional[List[str]] = Field(None, description="Extract property types. Allowed values: apartment, retire, acreage, land, unitblock, house, villa, rural. If user says 'any', 'anything', or 'all', extract ['any'].")
    keywords: Optional[List[str]] = Field(None, description="Keywords for facilities (e.g. pool, garage, aircon, pets).")
    search_radius: Optional[int] = Field(None, description="Search radius in kilometers. Extract only the number (e.g. '50km' -> 50).")

def guardian_node(state: AgentState):
    """
    Extracts entities from the latest message and updates the UserProfile.
    Implements a strict 'Patch' strategy with Semantic Memory support.
    """
    messages = state["messages"]
    current_profile = state.get("user_profile") or UserProfile()
    
    extractor_llm = llm.with_structured_output(GuardianUpdater)
    
    last_message = messages[-1].content.lower()
    
    # Get the last agent message for context (if exists)
    agent_context = ""
    if len(messages) >= 2:
        last_agent_msg = messages[-2]
        if isinstance(last_agent_msg, AIMessage):
            agent_context = f"""
    Last Question Asked by Agent:
    "{last_agent_msg.content}"
    """

    prompt_text = f"""
    You are a precise data extractor for an Australian real estate agent.
    Your goal is to update the UserProfile and Semantic Preferences based *only* on the new information in the LAST message.
    
    Current Profile State:
    {current_profile.model_dump_json()}
    
    {agent_context}
    
    Last User Message:
    "{messages[-1].content}"
    
    CRITICAL INSTRUCTIONS:
        - If Agent asked for bedrooms and user says "2", extract min_bedrooms=2.
        - If Agent asked for budget and user says "2000", extract budget=2000.
        - If Agent asked about "surrounding suburbs" and user says "yes", extract include_surrounding=True.
        - If Agent asked for Name and user says a short phrase (e.g. "Alex", "yesh"), extract name="Alex".
        - If user says "any", "anything", "don't care" for a COMPULSORY field (like property type), extract the special value needed (e.g. ['any'] for types).
        - **Unlimited Budget**: If user says "any budget", "no limit", "anything", set `budget=100000000` (100 Million).
        - **Any Bedrooms**: If user says "any bedrooms", "studio or more", "anything", set `min_bedrooms=1`.
        - **Auto-Confirm**: If user says "Search in [Suburb]", "Find properties in [Suburb]", or "Show me [Suburb]", set `location=[Suburb]` AND `suburb_confirmation=True`. Don't make them confirm again.
        - **Keywords**: If user explicitly asks for "pool", "gym", "pets", "balcony", extract them to `keywords`.
        - **No Keywords**: If user says "no features", "standard", "basic", or "no" to a features question, extract `keywords=[]` (empty list).
        - **Radius**: If user says "within 50km", "50km radius", "near X (20km)", extract `search_radius` (int).

        - **COMPOUND RESPONSES ("Yes... No...")**:
            * If user says "Yes [to search], No [to surrounding]", extract `suburb_confirmation=True` AND `include_surrounding=False`.
            * If user says "Yes, but no surrounding", extract `suburb_confirmation=True` AND `include_surrounding=False`.
            * If user says "Yes, search nearby" or "Yes, include surrounding", extract `suburb_confirmation=True` AND `include_surrounding=True`.
            * If user says "Just search there", extract `suburb_confirmation=True`.
            * If user says "yes yes", "yep yep", or "yeah yeah", extract `suburb_confirmation=True`.
            * If user gives mixed signals (e.g. "no no yes no no yes"), trust the **FINAL** word or the affirmative if it appears to be a correction.
    
    2. **BUDGET UPDATES**: If the user mentions ANY of these phrases, you MUST extract and update the budget:
        - "adjust my budget", "change my budget", "update budget", "new budget"
        - "increase budget", "raise budget", "budget to", "budget is"
        - "I can afford", "my max is", "up to", "willing to pay"
        - ANY mention of a monetary value (e.g., "10k", "$3000", "3 thousand", "10000")
        - **CURRENCY CONVERSION**: The system ONLY understands AUD.
            * If user says "USD" or "US Dollars", multiply by 1.6 to get AUD (e.g. $1000 USD -> 1600).
            * If user says "EUR", multiply by 1.7.
            * If user says "GBP", multiply by 2.0.
    
    3. **NORMALIZATION**: 
        - Budget: 
            * If user specifies "pw", "per week", "weekly" -> Set budget_period="weekly".
            * If user specifies "pm", "per month", "monthly" -> Set budget_period="monthly".
            * If NO period is specified (e.g. just "2000") -> Set budget_period="unknown" and extract raw numbers.
        - Location: Use "Suburb, State" format (e.g., "Richmond, VIC")
    
    4. **NO INFERENCE**: Do not guess values not mentioned.
        - Do NOT infer 'buy' intent from words like "house", "home", "property".
        - Only set intent if user explicitly says "buy", "rent", "lease", "purchase".
    
    5. **OFF-TOPIC RULES**:
        - A message is ONLY off-topic if it is gibberish, an unrelated joke, or illegal.
        - "Give me a random property" is NOT off-topic. It is a request (even if missing info).
        - "Anything" is NOT off-topic.
    
    6. **PARTIAL UPDATES**: Return None for unchanged fields, but ALWAYS extract budget if any monetary value is mentioned.
    """
    
    try:
        print(f"[Guardian] Processing message: {messages[-1].content}")
        patch = extractor_llm.invoke(prompt_text)
        print(f"[Guardian] Extracted patch: name={patch.name}, intent={patch.intent}, budget={patch.budget}, min_beds={patch.min_bedrooms}, max_beds={patch.max_bedrooms}, location={patch.location}")
        
        # Merge Strategy
        new_profile_data = current_profile.model_dump()
        
        # Track changes for cleaner logging/debugging
        params_changed = False
        
        # 0. Update Name (does NOT trigger search reset)
        if patch.name and patch.name != current_profile.name:
            new_profile_data["name"] = patch.name
            print(f"[Guardian] Name updated: {current_profile.name} -> {patch.name}")
        
        # 1. Update Standard Fields - track if they actually changed
        if patch.intent and patch.intent != current_profile.intent:
            new_profile_data["intent"] = patch.intent
            params_changed = True
            print(f"[Guardian] Intent changed: {current_profile.intent} -> {patch.intent}")
        
        # Budget update - always update if extracted, even if same value triggers re-search
        if patch.budget is not None:
            if patch.budget != current_profile.budget:
                new_profile_data["budget"] = patch.budget
                params_changed = True
                print(f"[Guardian] Budget changed: {current_profile.budget} -> {patch.budget}")
            else:
                print(f"[Guardian] Budget extracted but unchanged: {patch.budget}")
        
        # Budget Period logic
        if patch.budget_period:
            new_profile_data["budget_period"] = patch.budget_period
            print(f"[Guardian] Budget period updated: {patch.budget_period}")
            
            # Normalize if Monthly -> Weekly
            # Only if budget is set
            current_budget = new_profile_data.get("budget")
            if patch.budget_period == "monthly" and current_budget:
                # Convert to weekly
                weekly_budget = int(current_budget / 4.33)
                new_profile_data["budget"] = weekly_budget
                params_changed = True
                print(f"[Guardian] Normalized monthly budget {current_budget} -> {weekly_budget} pw")
        
        if patch.min_bedrooms is not None:
            new_profile_data["min_bedrooms"] = patch.min_bedrooms
            # Sync with legacy 'bedrooms' field for router compatibility
            new_profile_data["bedrooms"] = patch.min_bedrooms
            params_changed = True
            print(f"[Guardian] Min Bedrooms updated: {patch.min_bedrooms}")
            
        if patch.max_bedrooms is not None:
            new_profile_data["max_bedrooms"] = patch.max_bedrooms
            params_changed = True
            print(f"[Guardian] Max Bedrooms updated: {patch.max_bedrooms}")
        
        if patch.min_bathrooms is not None:
            new_profile_data["min_bathrooms"] = patch.min_bathrooms
            # Sync with max if not specified? No, let's keep it simple.
            params_changed = True
            print(f"[Guardian] Min Bathrooms updated: {patch.min_bathrooms}")

        if patch.max_bathrooms is not None:
            new_profile_data["max_bathrooms"] = patch.max_bathrooms
            params_changed = True
            print(f"[Guardian] Max Bathrooms updated: {patch.max_bathrooms}")

        if patch.parking_spaces is not None:
            new_profile_data["parking_spaces"] = patch.parking_spaces
            params_changed = True
            print(f"[Guardian] Parking Spaces updated: {patch.parking_spaces}")
        if patch.include_surrounding is not None and patch.include_surrounding != current_profile.include_surrounding:
            new_profile_data["include_surrounding"] = patch.include_surrounding
            params_changed = True
        
        if patch.location:
            new_profile_data["location"] = patch.location
            if new_profile_data["location"] != current_profile.location:
                new_profile_data["location_slug"] = None
                new_profile_data["state"] = None
                params_changed = True
        
        if patch.property_types:
            # Append or Replace? Usually users refine, but here let's replace if they specify new ones.
            # actually if they say "and villas", we might want to append. 
            # But the extractor usually extracts the full list from the context if we are lucky.
            # Let's simple Replace for now as it's cleaner.
            existing_types = set(current_profile.property_types or [])
            new_types = set(patch.property_types)
            
            # If the user says "houses", we get ['house'].
            if existing_types != new_types:
                new_profile_data["property_types"] = list(new_types)
                params_changed = True
                print(f"[Guardian] Property Types updated: {new_profile_data['property_types']}")
        
        if patch.keywords is not None:
            # Replace keywords logic
            new_profile_data["keywords"] = patch.keywords
            params_changed = True
            print(f"[Guardian] Keywords updated: {patch.keywords}")

        if patch.search_radius is not None:
            new_profile_data["search_radius"] = patch.search_radius
            params_changed = True
            print(f"[Guardian] Search Radius updated: {patch.search_radius}km")


        # 2. Append Preferences (Set Logic)
        # 2. Append Preferences (Set Logic)
        if patch.new_preferences:
            existing = set(current_profile.preferences)
            old_set = existing.copy()
            for p in patch.new_preferences:
                existing.add(p.lower()) # Normalize to lowercase for de-dupe
            
            if existing != old_set:
                new_profile_data["preferences"] = list(existing)
                params_changed = True
                print(f"[Guardian] Preferences updated: {new_profile_data['preferences']}")
            else:
                 # Ensure we keep the existing list format even if no change
                 new_profile_data["preferences"] = list(existing)

        # 3. Location Resolution (Refactored for clarity)
        if new_profile_data.get("location") and not new_profile_data.get("location_slug"):
            try:
                loc_json = resolve_location_logic(new_profile_data["location"])
                loc_data = json.loads(loc_json)
                
                loc_list = []
                if isinstance(loc_data, list):
                    loc_list = loc_data
                elif isinstance(loc_data, dict):
                    # Handle the actual API format: _embedded.suggestions
                    if "_embedded" in loc_data and "suggestions" in loc_data["_embedded"]:
                        loc_list = loc_data["_embedded"]["suggestions"]
                    elif "results" in loc_data:
                        loc_list = loc_data["results"]
                    elif "embedded" in loc_data and "results" in loc_data["embedded"]:
                        loc_list = loc_data["embedded"]["results"]
                    else:
                        print(f"Warning: Unexpected location data keys: {loc_data.keys()}")
                
                # Filter to only suburb-type results (addresses won't work for search)
                suburb_results = [r for r in loc_list if r.get("type") == "suburb"]
                
                if suburb_results:
                    best = suburb_results[0]
                    # API returns: 'id' as the slug, 'source.state' for state, 'display.text' for title
                    new_profile_data["location_slug"] = best.get("id") or best.get("slug")
                    # State is nested inside 'source' object
                    source_info = best.get("source", {})
                    new_profile_data["state"] = source_info.get("state") or best.get("state")
                    display_info = best.get("display", {})
                    new_profile_data["location"] = display_info.get("text") or best.get("title") or new_profile_data["location"]
                else:
                    # No valid suburb found - keep location but clear slug so interviewer asks again
                    print(f"Warning: No suburb found for '{new_profile_data['location']}'. Results were: {[r.get('type') for r in loc_list]}")
            except Exception as e:
                with open("debug_output.txt", "a", encoding="utf-8") as f:
                    f.write(f"Location Resolution Error: {e}\\n")
                print(f"Location Resolution Error: {e}")
        
        # 4. State Reset Logic
        # If parameters changed, we must invalidate previous search results
        state_updates = {"user_profile": UserProfile(**new_profile_data)}
        
        # 5. GUARDIAN: Handle off-topic detection
        if patch.off_topic is True:
            print("[Guardian] Off-topic message detected - flagging for handler")
            state_updates["off_topic_detected"] = True
            return state_updates  # Early return - skip other processing
        else:
            state_updates["off_topic_detected"] = False
        
        # 6. GUARDIAN: Handle suburb confirmation
        if patch.suburb_confirmation is True:
            print("[Guardian] User confirmed suburb - setting suburb_confirmed=True")
            state_updates["suburb_confirmed"] = True
        elif patch.suburb_confirmation is False:
            print("[Guardian] User rejected suburb - resetting location")
            # User said "no" - clear location so interviewer asks again
            new_profile_data["location"] = None
            new_profile_data["location_slug"] = None
            new_profile_data["state"] = None
            state_updates["user_profile"] = UserProfile(**new_profile_data)
            state_updates["suburb_confirmed"] = False
        
        if params_changed:
            print("Search parameters changed - resetting search state")
            state_updates["search_executed"] = False
            state_updates["search_status"] = "not_started"
            state_updates["listings"] = []  # Clear old listings
            state_updates["missing_fields"] = []  # Force router to re-eval
            state_updates["search_offset"] = 0  # RESET pagination on new search
            # Also reset suburb confirmation if location changed
            if patch.location:
                # CRITICAL: Only reset confirmation if location changed meaningfully
                # Prevents loop if LLM re-extracts "Redfern" from "Properties in Redfern"
                if new_profile_data["location"] != current_profile.location:
                    state_updates["suburb_confirmed"] = False
                    print(f"[Guardian] Location changed ({current_profile.location} -> {new_profile_data['location']}). Resetting confirmation.")
                else:
                    print(f"[Guardian] Location extracted '{patch.location}' but matches current. NOT resetting confirmation.")
        
        # 6. GUARDIAN: Handle Pagination (Load More)
        # Only process if NO parameters changed (pure pagination request)
        if patch.load_more and not params_changed:
            print("[Guardian] Load More detected - incrementing offset")
            current_offset = state.get("search_offset", 0)
            state_updates["search_offset"] = current_offset + 5
            
            # Force search re-execution for the next page
            state_updates["search_executed"] = False
            state_updates["search_status"] = "not_started"
            # Ensure we don't accidentally ask for confirmation again if we just want more
            state_updates["suburb_confirmed"] = True 

        return state_updates
        
    except Exception as e:
        print(f"Guardian Error: {e}")
        return {} # No state update on error

# --- Greeter Node ---
GREETER_SYSTEM = """You are a friendly Australian Real Estate Agent.

GOAL: Greet the user warmly and introduce yourself.

GUIDELINES:
- Be warm and welcoming
- Introduce yourself as their Australian real estate assistant
- Ask for their name naturally
- Keep it brief (1-2 sentences)
- Use a friendly, professional tone
"""

def greeter_node(state: AgentState):
    """
    Greeting Agent - First point of contact for new conversations.
    
    Task: Greet the user warmly and ask for their name.
    This only runs on first interaction (is_first_interaction == True).
    """
    profile = state["user_profile"]
    
    # If name is already known (shouldn't happen on first interaction, but safety check)
    if profile.name:
        greeting = f"Welcome back, {profile.name}! How can I help you with your property search today?"
    else:
        greeting = "G'day! 👋 I'm your Australian Real Estate Agent. I'm here to help you find your perfect property. Before we get started, what's your name?"
    
    return {
        "messages": [AIMessage(content=greeting)], 
        "is_first_interaction": False
    }

def router_node(state: AgentState):
    """
    Determines the next step in the workflow based ONLY on the state.
    NO LLM calls allowed here. Outputs ONLY the next node name.
    
    ROUTING RULES:
    - If location_slug is missing → location_resolver_node
    - If location_slug exists AND suburb_confirmed == false → confirmer_node
    - If suburb_confirmed == true → search_node
    - If user rejects suburb → Guardian resets location_slug, router goes to location_resolver_node
    
    Routing Decision Table:
    | Condition | Decision |
    | :--- | :--- |
    | search_executed + empty | empty_handler |
    | search_executed + error | presenter |
    | search_executed + success | analyzer |
    | location_slug missing | location_resolver |
    | other fields missing | interviewer |
    | suburb_confirmed == false | confirmer |
    | suburb_confirmed == true | search |
    """
    profile = state["user_profile"]
    search_executed = state.get("search_executed", False)
    search_status = state.get("search_status", "not_started")
    suburb_confirmed = state.get("suburb_confirmed", False)
    is_first_interaction = state.get("is_first_interaction", True)  # Default True for new sessions
    
    print(f"[Router] is_first={is_first_interaction}, name={profile.name}, search_executed={search_executed}, search_status={search_status}, suburb_confirmed={suburb_confirmed}, location_slug={profile.location_slug}")
    
    # 0. GREETING FLOW - Highest priority for new sessions
    # If first interaction and no name yet, greet the user
    if is_first_interaction and not profile.name:
        print("[Router] First interaction without name → greeter")
        return {"next_node": "greeter"}
    
    # 0.5 OFF-TOPIC CHECK - Second highest priority
    # If the Guardian detected an off-topic message, handle it gracefully
    off_topic_detected = state.get("off_topic_detected", False)
    if off_topic_detected:
        print("[Router] Off-topic message detected → off_topic_handler")
        return {"next_node": "off_topic_handler"}
    
    # If first interaction but name exists (from greeting), mark as not first anymore
    if is_first_interaction:
        # This will be handled naturally as we proceed
        pass
    
    # 1. Handle Post-Search States (Highest Priority to avoid loops)
    if search_executed:
        if search_status == "empty":
            return {"next_node": "empty_handler"}
        elif search_status == "error":
            return {"next_node": "presenter"}
        elif search_status == "success":
            return {"next_node": "analyzer"}
        return {"next_node": "presenter"}

    # 2. RULE: If location_slug is missing → location_resolver_node
    if not profile.location_slug:
        print("[Router] location_slug missing → location_resolver")
        return {"next_node": "location_resolver", "missing_fields": ["location"]}
    
    # 3. Check for other missing fields
    required_fields = ["intent"]
    
    if profile.intent == "buy":
        required_fields.extend(["budget", "bedrooms", "property_types"])
    elif profile.intent == "rent":
        if profile.budget and (not profile.budget_period or profile.budget_period == "unknown"):
             # EXCEPTION: If budget is massive (indicating "unlimited"), skip period check
             if profile.budget >= 90000000: # 90 Million+ treated as unlimited
                 pass
             else:
                 print("[Router] Budget set but period unknown -> Clarify")
                 # Loop Detection Logic
                 last_field = state.get("last_target_field")
                 failures = state.get("consecutive_failures", 0)
                 
                 target = "budget_period"
                 if last_field == target:
                     failures += 1
                 else:
                     failures = 0
                     
                 return {
                     "next_node": "interviewer", 
                     "missing_fields": [target],
                     "last_target_field": target,
                     "consecutive_failures": failures
                 }
             
        required_fields.extend(["budget", "bedrooms", "property_types"])
    elif profile.intent == "sell":
        return {"next_node": "interviewer", "missing_fields": ["details_about_selling (Feature not implemented)"]}
        
    missing = []
    for field in required_fields:
        val = getattr(profile, field)
        if val is None:
            missing.append(field)
        elif field == "property_types" and val == ["any"]:
            # "Any" is a valid selection, so it's not missing
            pass
        elif isinstance(val, list) and len(val) == 0:
            missing.append(field)
            
    if missing:
        # Loop Detection Logic
        target = missing[0]
        last_field = state.get("last_target_field")
        failures = state.get("consecutive_failures", 0)
        
        if last_field == target:
            failures += 1
        else:
            failures = 0
            
        print(f"[Router] Missing {target}. Failures: {failures}")
        return {
            "next_node": "interviewer", 
            "missing_fields": missing,
            "last_target_field": target,
            "consecutive_failures": failures
        }
    
    # 4. RULE: If location_slug exists AND suburb_confirmed == false → confirmer
    if not suburb_confirmed:
        print("[Router] Guardian: Suburb not confirmed → confirmer")
        return {"next_node": "confirmer"}

    # 5.5 Check for Keywords (Features)
    if profile.keywords is None:
        print("[Router] Surrounding confirmed, asking for features -> features_check_node")
        return {"next_node": "features_check_node"}
    
    # 6. RULE: If suburb_confirmed == true → search
    return {"next_node": "search"}

INTERVIEWER_SYSTEM = """You are a friendly Australian Real Estate Agent.

GOAL: Ask the user for their **{target_field}**.

USER'S NAME: {user_name}

CONTEXT:
User Profile so far:
{user_context}

GUIDELINES:
- Address the user by their name ({user_name}) naturally in your response.
- Ask ONE clear, concise question about the {target_field}.
- Provide an EXAMPLE of what you expect (e.g. "{example}").
- Do NOT mention that you are "missing fields" or explain your logic.
- Do NOT ask for other information yet.
- Be natural and conversational.
"""

# ... (rest of prompts) ...

def interviewer_node(state: AgentState):
    """
    Asks the user for missing info.
    Strategy: identifies the HIGHEST PRIORITY missing field and asks about that ONLY.
    Priority Order (implicit in Router): Location -> Budget -> Bedrooms.
    """
    missing = state["missing_fields"]
    profile = state["user_profile"]
    
    # Fail-safe
    if not missing:
        return {"messages": [AIMessage(content="How can I help you regarding real estate today?")]}
    
    # Select top priority field
    target = missing[0]
    
    # Map technical field names to natural language if needed
    field_map = {
        "intent": "whether you want to Buy or Rent",
        "location": "Preferred Suburb and State",
        "budget": "Maximum Budget",
        "budget_period": "clarification on whether your budget is weekly or monthly",
        "bedrooms": "Minimum Bedrooms",
        "include_surrounding": "preference for Surrounding Suburbs",
        "property_types": "Preferred Property Type (e.g. House, Apartment, Villa)",
        "details_about_selling (Feature not implemented)": "details about the property you want to sell"
    }
    
    readable_target = field_map.get(target, target)
    
    # Define examples for each field
    field_examples = {
        "intent": "I'm looking to Buy",
        "location": "Richmond, VIC",
        "budget": "$850,000" if profile.intent != "rent" else "$500 per week",
        "budget_period": "Weekly",
        "bedrooms": "at least 3 bedrooms",
        "include_surrounding": "Yes, I'm open to nearby suburbs",
        "property_types": "House and Apartment",
        "keywords": "Pool, Gym, and Garage"
    }
    
    example_text = field_examples.get(target, "some value")
    
    # Get user's name for personalized response
    user_name = profile.name or "friend"
    
    # Robustness: Check for failures
    failures = state.get("consecutive_failures", 0)
    extra_instructions = ""
    
    if failures > 0:
        print(f"[Interviewer] Retry mode active (Failure {failures}) for {target}")
        extra_instructions = f"""
        \nIMPORTANT: The user ALREADY failed to provide this information {failures} time(s).
        They might be confused or answering vaguely.
        
        1. Be EXTREMELY clear and direct. 
        2. Give explicit examples of what they should type.
        3. Explain WHY you need it simply.
        4. If asking for budget period, explicitly ask: "Is that per week or per month?"
        5. DO NOT be polite if it sacrifices clarity. Focus on getting the data.
        """
    
    prompt = INTERVIEWER_SYSTEM.format(
        target_field=readable_target,
        user_name=user_name,
        example=example_text,
        user_context=profile.model_dump_json(exclude_none=True)
    ) + extra_instructions
    
    # We include full history so the agent knows what has been said, preventing repetition
    messages = [SystemMessage(content=prompt)] + state["messages"]
    
    response = llm.invoke(messages)
    return {"messages": [response]}

# --- Location Resolver Node ---
LOCATION_RESOLVER_SYSTEM = """You are a friendly Australian Real Estate Agent.

GOAL: Ask the user for their preferred LOCATION (Suburb and State).

CONTEXT:
The user wants to {intent} a property but hasn't specified a valid location yet.

GUIDELINES:
- Ask clearly for the Suburb and State (e.g., "Richmond, VIC")
- Be natural and conversational
- Do NOT ask about budget or bedrooms yet
- If they mentioned a vague location, ask them to be more specific

Example: "Which suburb are you interested in? Please include the state, like 'Richmond, VIC'."
"""

def location_resolver_node(state: AgentState):
    """
    Asks the user for their preferred location.
    
    Router Rule: This node is called when location_slug is missing.
    """
    profile = state["user_profile"]
    
    prompt = LOCATION_RESOLVER_SYSTEM.format(
        intent=profile.intent or "find"
    )
    
    messages = [SystemMessage(content=prompt)] + state["messages"]
    response = llm.invoke(messages)
    
    return {"messages": [response]}

# --- Confirmer Node ---
# You are the Suburb Confirmation Agent.
# Task: Ask user to confirm suburb before searching.
# Rules: Be concise, explicit, do NOT fetch properties, do NOT assume confirmation.

def confirmer_node(state: AgentState):
    """
    Suburb Confirmation Agent.
    
    Context: A suburb has been resolved but not confirmed.
    Task: Ask user to confirm the suburb before searching.
    
    Rules:
    - Be concise and explicit
    - Mention the resolved suburb clearly
    - Do NOT fetch properties
    - Do NOT assume confirmation
    """
    profile = state["user_profile"]
    display_location = profile.location or "Unknown"
    
    # Response format: "I found {display_location}. Should I search for properties there?"
    if profile.search_radius:
         confirmation_message = f"I found {display_location}. Should I search within {profile.search_radius}km of there?"
    else:
         confirmation_message = f"I found {display_location}. Should I search for properties there? (I can also check surrounding suburbs if you like!)"
    
    return {"messages": [AIMessage(content=confirmation_message)]}

def surrounding_check_node(state: AgentState):
    """
    Asks the user if they want to include surrounding suburbs.
    Run AFTER suburb is confirmed.
    """
    return {"messages": [AIMessage(content="Would you like to include properties in surrounding suburbs as well?")]}

def features_check_node(state: AgentState):
    """
    Asks the user for specific keywords/facilities.
    """
    return {"messages": [AIMessage(content="Do you have any specific features in mind, like a pool, gym, garage, or pet-friendly?")]}


def expand_radius(center: str, radius: int) -> List[str]:
    """
    Asks the LLM to list major suburbs within a radius.
    """
    try:
        print(f"[Expand Radius] Finding suburbs within {radius}km of {center}")
        # Need to be clearer about output format or parsing will fail
        prompt = f"""
        List up to 15 major residential suburbs within {radius}km of {center}.
        Return ONLY a perfectly formatted comma-separated list of names. No bullets, no newlines, no other text.
        Example: Suburb A, Suburb B, Suburb C
        """
        response = llm.invoke(prompt)
        content = response.content.strip()
        suburbs = [s.strip() for s in content.split(',') if s.strip()]
        print(f"[Expand Radius] Found: {suburbs}")
        return suburbs
    except Exception as e:
        print(f"[Expand Radius] Error: {e}")
        return []


def search_node(state: AgentState):
    """
    Semantic Property Search Agent.
    
    Preconditions:
    - suburb_confirmed == true (enforced by Router)
    
    Responsibilities:
    1. Determine if the suburb is already indexed.
    2. If NOT indexed:
       - Fetch ALL available listings for the suburb
       - Ingest listings into the local vector store
       - Record suburb in indexed_suburbs
    3. Perform semantic search using:
       - User natural language preferences
       - Hard filters (budget, bedrooms, etc.)
    4. Retrieve TOP 5 most relevant properties.
    5. Return ranked results for presentation.
    
    Rules:
    - Fetch listings ONLY ONCE per suburb
    - Never re-fetch if suburb is already indexed
    - Always use semantic similarity as the primary ranking signal
    - Hard filters reduce the candidate set but do not replace semantic ranking
    """
    profile = state["user_profile"]
    indexed_suburbs = state.get("indexed_suburbs", [])
    log_messages = []
    
    # 1. Parameter Validation
    if not profile.location_slug:
        return {
            "search_executed": True, 
            "search_status": "error", 
            "error_message": "Location slug was missing. Please try selecting the location again."
        }
    
    channel = profile.intent or "buy"  # "buy" or "rent"
    suburb_name = profile.location.split(',')[0].strip() if profile.location else ""
    state_code = profile.state or "VIC"
    
    # --- Radius Logic ---
    search_locations = [suburb_name]
    radius_key_suffix = ""

    # Use explicit radius OR default to 5km if "include_surrounding" is checked
    search_radius = profile.search_radius
    if not search_radius and profile.include_surrounding:
        search_radius = 5

    if search_radius:
        print(f"[Search] Expanding radius {search_radius}km around {suburb_name}")
        neighbors = expand_radius(f"{suburb_name}, {state_code}", search_radius)
        if neighbors:
            search_locations.extend(neighbors)
            # Dedupe
            search_locations = list(set(search_locations))
            # Update suburb_name to comma-separated for API call
            suburb_name = ",".join(search_locations) 
            radius_key_suffix = f"-r{search_radius}"
            
            log_messages.append(AIMessage(content=f"🗺️ **Radius Search**: Expanded to include {len(search_locations)-1} neighboring suburbs (e.g. {', '.join(neighbors[:3])}...)"))

    # CRITICAL: Include channel AND keywords in the key to differentiate cached data
    keywords_str = "-".join(sorted(profile.keywords or []))
    base_key = f"{profile.location_slug.lower()}{radius_key_suffix}"
    suburb_key = f"{base_key}-{channel}-{keywords_str}" if keywords_str else f"{base_key}-{channel}"
    
    try:
        # 2. GUARDIAN CHECK: Is suburb already indexed?
        if not property_store.is_suburb_indexed(suburb_key):
            print(f"[Search] Suburb {suburb_key} not indexed. Checking cache...")
            
            # Check Cache
            all_listings = suburb_cache.get(suburb_key)
            

            if not all_listings:
                print(f"[Search] Cache miss for {suburb_key}. Fetching from API...")
                # Fetch ALL listings for this suburb (or list of suburbs)
                # Note: suburb_name might be a comma-separated list now
                all_listings = client.fetch_all_listings(
                    location_slug=suburb_name,
                    state=state_code,
                    channel=channel,
                    keywords=profile.keywords
                )
                
                if all_listings:
                    # Save to cache
                    suburb_cache.set(suburb_key, all_listings)
            
            if not all_listings:
                return {
                    "listings": [], 
                    "search_executed": True, 
                    "search_status": "empty",
                    "error_message": None,
                    "indexed_suburbs": indexed_suburbs  # Don't add failed suburb
                }
            
            # Ingest into vector store
            print(f"[Search] Ingesting {len(all_listings)} properties...")
            log_messages.append(AIMessage(content=f"🔄 **RAG System**: Cache miss. Ingesting {len(all_listings)} properties for **{suburb_name}** into local vector store..."))
            
            property_store.ingest(suburb_key, all_listings)
            indexed_suburbs = list(set(indexed_suburbs + [suburb_key]))
            print(f"[Search] Ingested {len(all_listings)} properties for {suburb_key}")
        else:
            print(f"[Search] Suburb {suburb_key} already indexed. Using RAG.")
            log_messages.append(AIMessage(content=f"⚡ **RAG System**: Suburb **{suburb_name}** is already indexed. Performing fast local search."))
        
        # 3. Semantic Search using Query Analyzer
        # Query Analyzer extracts semantic preferences and produces clean query
        analysis = query_analyzer.analyze(
            preferences=profile.preferences or [],
            budget=profile.budget,
            min_bedrooms=profile.min_bedrooms,
            max_bedrooms=profile.max_bedrooms,
            min_bathrooms=profile.min_bathrooms,
            max_bathrooms=profile.max_bathrooms,
            parking_spaces=profile.parking_spaces,
            property_type=None,  # Could be added to UserProfile
            location=suburb_name,
            property_types=None if profile.property_types == ["any"] else profile.property_types
        )
        
        semantic_query = analysis["semantic_query"]
        filters = analysis["filters"]
        
        # Log Query Generation
        print(f"[Search] Semantic query: {semantic_query}")
        print(f"[Search] Filters: {filters}")
        log_messages.append(AIMessage(content=f"🔍 **RAG Query Generated**:\n> **Semantic**: \"{semantic_query}\"\n> **Filters**: `{filters}`"))
        
        # Search the vector store with semantic query + filters
        current_offset = state.get("search_offset", 0)
        print(f"[Search] Executing search with offset={current_offset}")

        listings = property_store.search(
            query=semantic_query,
            filters={
                "suburb_slug": suburb_key,
                **filters
            },
            k=5,
            offset=current_offset
        )
        
        # Log Result Count
        log_messages.append(AIMessage(content=f"✅ **RAG Search Complete**: Found **{len(listings)}** properties matching your criteria."))
        
        # 4. Handle Results
        if not listings:
            return {
                "listings": [], 
                "search_executed": True, 
                "search_status": "empty",
                "error_message": None,
                "indexed_suburbs": indexed_suburbs,
                "messages": log_messages
            }
            
        print(f"[Search] Found {len(listings)} listings.")
        
        return {
            "listings": listings, 
            "search_executed": True, 
            "search_status": "success",
            "error_message": None, 
            "indexed_suburbs": indexed_suburbs,
            "messages": log_messages
        }
        
    except Exception as e:
        print(f"Search Execution Error: {e}")
        return {
            "listings": [], 
            "search_executed": True, 
            "search_status": "error",
            "error_message": f"System error during search: {str(e)}",
            "indexed_suburbs": indexed_suburbs
        }

def analyzer_node(state: AgentState):
    """
    Analyzes search results.
    Purpose: Convert raw data into insights for the Presenter.
    """
    try:
        profile = state["user_profile"]
        listings = state["listings"]
        
        # Pre-processing: Calculate stats to help the LLM (Optional but good for 'No Hallucination')
        # We pass simplified data to save tokens and focus the LLM
        simple_listings = []
        
        for l in listings:
            # Extract price for stats if possible
            raw_price = l.get("price")
            if isinstance(raw_price, dict):
                price_str = raw_price.get("display", "")
            else:
                price_str = str(raw_price) if raw_price else ""
            # Naive extraction - just pass the string to LLM, it handles it well enough for analysis
            
            # Safe address extraction
            title = l.get("title")
            addr = l.get("address")
            if not title and isinstance(addr, dict):
                title = addr.get("streetAddress")
            elif not title:
                title = str(addr) if addr else "Unknown Address"

            simple_listings.append({
                "address": title,
                "price": price_str,
                "features": l.get("features"),
                "propertyType": l.get("propertyType")
            })
        
        prompt = ANALYZER_SYSTEM.format(
            user_profile=profile.model_dump_json(exclude_none=True),
            listings=json.dumps(simple_listings, indent=2)
        )
        
        print(f"[Analyzer] Generating analysis for {len(simple_listings)} properties...")
        response = llm.invoke([SystemMessage(content=prompt)])
        return {"analysis": response.content}
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print(f"[Analyzer] CRASH: {e}")
        with open("debug_output.txt", "a", encoding="utf-8") as f:
            f.write(f"\n[Analyzer Crash] {e}\n{err}\n")
        # Return dummy analysis to prevent hard crash
        return {"analysis": "I found some properties but encountered an error analyzing them in detail."}

def presenter_node(state: AgentState):
    """
    Presents the findings (or errors) to the user.
    """
    search_status = state.get("search_status", "not_started")
    
    # 1. Error Handling
    if search_status == "error":
        error_msg = state.get("error_message", "Unknown error")
        prompt = f"""
        CONTEXT: The property search failed.
        Error Details: {error_msg}
        
        Task: Write a helpful error message to the user. See PRESENTER_SYSTEM for tone.
        """
        messages = [SystemMessage(content=PRESENTER_SYSTEM), HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        return {"messages": [response]}
        
    # 2. Success Handling
    analysis = state.get("analysis")
    listings = state.get("listings")
    
    # Validation: Ensure we actually have data
    if not analysis or not listings:
        # Fallback if something weird happened
        return {"messages": [AIMessage(content="I found some properties but I'm having trouble summarizing them right now.")]}
    
    # Optimize Listing Data for LLM Context
    # Raw listings can be huge, causing context overflow or confusion.
    display_listings = []
    
    with open("debug_output.txt", "a", encoding="utf-8") as f:
        f.write(f"Presenter Node: Processing {len(listings)} listings.\\n")
        
    for l in listings[:5]:
        # Extract features safely - API uses nested 'general' structure
        feats = (l.get("features") or {}).get("general", {})
        
        # Build the full realestate.com.au URL from prettyUrl
        pretty_url = l.get("prettyUrl", "")
        full_link = f"https://www.realestate.com.au/{pretty_url}" if pretty_url else ""
        
        # Build full image URL from mainImage
        # Format: https://i3.au.reastatic.net/800x600/{hash}/image.jpg
        main_image = l.get("mainImage") or {}
        if main_image.get("uri"):
            server = main_image.get("server", "https://i2.au.reastatic.net").rstrip("/")
            uri = main_image.get("uri", "").lstrip("/")
            image_url = f"{server}/800x600/{uri}"
        else:
            image_url = "https://via.placeholder.com/400x300?text=No+Image"
        
        raw_price = l.get("price")
        if isinstance(raw_price, dict):
            display_price = raw_price.get("display", "Contact Agent")
        else:
            display_price = str(raw_price) if raw_price else "Contact Agent"

        item = {
            "title": l.get("title") or (l.get("address") or {}).get("streetAddress") or "Property",
            "price": display_price,
            "beds": feats.get("bedrooms", "-"),
            "baths": feats.get("bathrooms", "-"),
            "carspaces": feats.get("parkingSpaces", "-"),
            "image": image_url,
            "link": full_link,
            "listingId": l.get("listingId", ""),
            "description": (l.get("description") or "No description available")[:150] + "..."
        }
        display_listings.append(item)
    
    with open("debug_output.txt", "a", encoding="utf-8") as f:
        f.write(f"Display Listings: {len(display_listings)}\\n")
        
    prompt = f"""
    CONTEXT: Search succeeded.
    
    Analysis derived from findings:
    {analysis}
    
    Listings Data:
    {json.dumps(display_listings, indent=2)}
    
    Task: Present these findings to the user.
    CRITICAL: You MUST include the `json_properties` block at the end of your response so the user can see the cards.
    Use the data provided above to populate the JSON block.
    """
    
    messages = [SystemMessage(content=PRESENTER_SYSTEM), HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    content = response.content
    
    # ALWAYS replace/append our structured data - don't rely on LLM to generate correct JSON
    import re
    # Remove any LLM-generated json_properties block
    content = re.sub(r'```json_properties\s*[\s\S]*?```', '', content).strip()
    
    # Always append our structured data
    if display_listings:
        json_block = f"\n\n```json_properties\n{json.dumps(display_listings, indent=2)}\n```"
        content += json_block
    
    response = AIMessage(content=content)
        
    return {"messages": [response]}
    
def empty_handler_node(state: AgentState):
    """
    Handles empty results intelligently.
    """
    profile = state["user_profile"]
    
    prompt = EMPTY_HANDLER_SYSTEM.format(
        user_profile=profile.model_dump_json(exclude_none=True)
    )
    
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    
    return {"messages": [response]}

def off_topic_handler_node(state: AgentState):
    """
    Handles off-topic or inappropriate queries gracefully.
    
    Guardrail: When the user asks something unrelated to property search
    (e.g., jokes, illegal activities, personal advice), this node responds
    politely and redirects them back to property-related topics.
    """
    profile = state["user_profile"]
    user_name = profile.name or "there"
    
    response_text = f"""I appreciate the chat, {user_name}! 😊 However, that question is a bit outside my expertise as a property search assistant.

I'm here to help you find the perfect property to **buy**, **rent**, or **sell** in Australia. 

Is there anything property-related I can help you with? For example:
- Searching for properties in a different suburb
- Adjusting your budget or bedroom requirements
- Exploring nearby areas

Just let me know how I can assist with your property search! 🏡"""
    
    return {"messages": [AIMessage(content=response_text)]}

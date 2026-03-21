import os
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from backend.state import AgentState
from backend.nodes import (
    guardian_node, 
    router_node,
    greeter_node,
    interviewer_node,
    location_resolver_node,
    confirmer_node,
    search_node, 
    analyzer_node, 
    presenter_node, 
    empty_handler_node,
    off_topic_handler_node,
    surrounding_check_node,
    features_check_node
)

# Ensure OPENAI_API_KEY is allowed
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")

def route_step(state: AgentState):
    """
    Router function for Conditional Edge.
    Extracts the decision made by the Router Node.
    """
    return state.get("next_node")

# Define Graph
workflow = StateGraph(AgentState)

workflow.add_node("guardian", guardian_node)
workflow.add_node("router", router_node)
workflow.add_node("greeter", greeter_node)
workflow.add_node("interviewer", interviewer_node)
workflow.add_node("location_resolver", location_resolver_node)
workflow.add_node("confirmer", confirmer_node)
workflow.add_node("search", search_node)
workflow.add_node("analyzer", analyzer_node)
workflow.add_node("presenter", presenter_node)
workflow.add_node("empty_handler", empty_handler_node)
workflow.add_node("off_topic_handler", off_topic_handler_node)
workflow.add_node("surrounding_check_node", surrounding_check_node)
workflow.add_node("features_check_node", features_check_node)

# Set Entry
workflow.set_entry_point("guardian")

# Edges
# 1. Guardian -> Router
workflow.add_edge("guardian", "router")

# 2. Router -> Conditional
# The Router determines the next step based on state (missing fields, search status).
workflow.add_conditional_edges(
    "router",
    route_step,
    {
        "greeter": "greeter",
        "interviewer": "interviewer",
        "location_resolver": "location_resolver",
        "confirmer": "confirmer",
        "search": "search",
        "empty_handler": "empty_handler",
        "presenter": "presenter",
        "off_topic_handler": "off_topic_handler",
        "surrounding_check_node": "surrounding_check_node",
        "features_check_node": "features_check_node",
        "analyzer": "analyzer",
    }
)

# 3. Node -> END (Validation & Interaction points)
workflow.add_edge("greeter", END)             # Wait for user to provide name
workflow.add_edge("interviewer", END)         # Wait for user input
workflow.add_edge("location_resolver", END)   # Wait for user to specify location
workflow.add_edge("confirmer", END)           # Wait for user to confirm suburb
workflow.add_edge("presenter", END)           # Wait for user input
workflow.add_edge("empty_handler", END)       # Wait for user input
workflow.add_edge("off_topic_handler", END)   # Wait for user input after off-topic response
workflow.add_edge("surrounding_check_node", END) # Wait for answer
workflow.add_edge("features_check_node", END) # Wait for answer

# 4. Search -> Router
# Search updates 'search_executed' and 'search_status'.
# Router checks these to send to Analyzer or EmptyHandler.
workflow.add_edge("search", "router")

# 5. Analyzer -> Presenter
# Analyzer creates insights, Presenter formats them.
workflow.add_edge("analyzer", "presenter")

# Checkpointer for Stateful conversation
checkpointer = MemorySaver()

# Compile
graph = workflow.compile(checkpointer=checkpointer)

__all__ = ["graph"]

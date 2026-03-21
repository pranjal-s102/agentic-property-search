from fastapi import FastAPI, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
from backend.agent import graph
from langchain_core.messages import HumanMessage
# from langchain_classic.agents import AgentExecutor # Removed
# from langchain_classic.memory import ConversationBufferMemory # Removed
import uvicorn
import os
import uuid

app = FastAPI(title="Property Agent Chatbot")

# Store active agent executors per session
# agent_executors = {} # Removed global dict

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None # Optional

class ChatResponse(BaseModel):
    response: str
    session_id: str

# Mount Frontend
# Mount Frontend removed from here, moved to bottom

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        session_id = request.session_id
        if not session_id:
            session_id = str(uuid.uuid4())

        if request.message.strip().lower() == "pineapple":
             # Optimization: In LangGraph with checkpointer, we might not need to explicit clear memory manually 
             # if we just stop using the thread_id, but here strictly speaking we don't have a specific deletion method exposed easily.
             # We can just let the user know.
            return ChatResponse(response="Session terminated. Goodbye!", session_id=session_id)
        
        # Prepare the input for the graph
        # We append the new user message to the state
        inputs = {"messages": [HumanMessage(content=request.message)]}
        
        # Config contains the thread_id for persistence
        config = {"configurable": {"thread_id": session_id}}
        
        # Invoke the graph
        # Since we are using an async server, we should use ainvoke if possible, but the graph compilation returns a Runnable.
        # Runnable has ainvoke.
        
        # We need to get the final state or the events. 
        # For a simple chat application, we usually want the last message from the agent.
        
        final_state = await graph.ainvoke(inputs, config=config)
        
        # Extract the last message content
        output_messages = final_state["messages"]
        last_message = output_messages[-1]
        output = last_message.content
        
        return ChatResponse(response=output, session_id=session_id)
    except Exception as e:
        # In a real app we might log this and return a generic error
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Mount Frontend last to avoid shadowing API routes
if os.path.isdir("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)

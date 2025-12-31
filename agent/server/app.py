import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent.agents import PlannerAgent, ExecutorAgent, CriticAgent
from agent.core.protocol import AgentMessage, Role

app = FastAPI()

# Determine role from environment
ROLE_MAP = {
    "planner": PlannerAgent,
    "executor": ExecutorAgent,
    "critic": CriticAgent
}

agent_role = os.getenv("AGENT_ROLE", "planner")
if agent_role not in ROLE_MAP:
    raise ValueError(f"Invalid AGENT_ROLE: {agent_role}")

agent = ROLE_MAP[agent_role]()

@app.get("/health")
async def health():
    return {"status": "ok", "role": agent.role}

@app.post("/v1/message")
async def process_message(message: AgentMessage):
    try:
        response = agent.process_message(message)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

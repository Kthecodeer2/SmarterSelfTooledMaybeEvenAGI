import os
import uvicorn
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from agent.agents import PlannerAgent, ExecutorAgent, CriticAgent
from agent.core.protocol import AgentMessage, Role

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent_server")

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

try:
    agent = ROLE_MAP[agent_role]()
    logger.info(f"Initialized agent with role: {agent_role}")
except Exception as e:
    logger.critical(f"Failed to initialize agent: {e}")
    raise

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "error": str(exc)},
    )

@app.get("/health")
async def health():
    return {"status": "ok", "role": agent.role}

@app.post("/v1/message")
async def process_message(message: AgentMessage):
    try:
        response = agent.process_message(message)
        return response
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

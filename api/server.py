"""
FastAPI Server
Single /chat endpoint with JSON in/out.

DESIGN DECISIONS:
- Minimal API surface
- Full request/response logging option
- Memory management endpoints
- Health check endpoint
- Optimized for local M4 use
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.core.orchestrator import Orchestrator, AgentResponse, get_orchestrator
from agent.core.llm_interface import LLMConfig
from config.settings import config as app_config


# Configure logging
log_dir = Path(app_config.api.log_dir)
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Request/Response Models
class ChatRequest(BaseModel):
    """Chat request body"""
    message: str
    include_logs: bool = False  # Include verification logs in response
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Explain how to implement a binary search in Python",
                "include_logs": False
            }
        }


class ChatResponse(BaseModel):
    """Chat response body"""
    response: str
    confidence: float
    reasoning_mode: str
    task_type: str
    was_refused: bool
    
    # Optional detailed info
    verification_log: Optional[list[dict]] = None
    memory_context: Optional[str] = None
    code_analysis: Optional[dict] = None
    research_analysis: Optional[dict] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "response": "Binary search implementation...",
                "confidence": 0.92,
                "reasoning_mode": "deep",
                "task_type": "coding",
                "was_refused": False
            }
        }


class MemoryRequest(BaseModel):
    """Memory add request"""
    content: str
    tag: str = "project"  # temporary, project, permanent
    category: str = "fact"  # environment, preference, constraint, goal, fact


class MemoryResponse(BaseModel):
    """Memory response"""
    id: str
    content: str
    tag: str
    category: str


class MemoryListResponse(BaseModel):
    """Memory list response"""
    memories: list[dict]
    count: int


class StatusResponse(BaseModel):
    """System status response"""
    status: str
    model_loaded: bool
    model_path: str
    memory_count: int
    version: str


# Global orchestrator instance
orchestrator: Optional[Orchestrator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global orchestrator
    
    # Startup
    logger.info("Starting AI Agent API...")
    
    # Initialize LLM config from settings
    llm_config = LLMConfig(
        model_path=app_config.model.model_path,
        n_ctx=app_config.model.n_ctx,
        n_gpu_layers=app_config.model.n_gpu_layers,
        n_batch=app_config.model.n_batch,
        n_threads=app_config.model.n_threads,
        temperature=app_config.model.temperature,
        top_p=app_config.model.top_p,
        max_tokens=app_config.model.max_tokens,
        repeat_penalty=app_config.model.repeat_penalty
    )
    
    # Create orchestrator
    orchestrator = get_orchestrator(
        llm_config=llm_config,
        memory_path=app_config.memory.persist_directory,
        enable_logging=app_config.api.enable_logging
    )
    
    logger.info("AI Agent API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Agent API...")
    if orchestrator:
        orchestrator.clear_session()
    logger.info("AI Agent API shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Local AI Agent",
    description="A local-first AI agent with verification, memory, and specialized modes",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API info"""
    return {
        "name": "Local AI Agent",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/chat",
            "memory": "/memory",
            "status": "/status",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get system status"""
    global orchestrator
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    model_info = orchestrator.get_model_status()
    memories = orchestrator.list_memories()
    
    return StatusResponse(
        status="ready" if model_info.get("loaded") else "model_not_loaded",
        model_loaded=model_info.get("loaded", False),
        model_path=model_info.get("path", ""),
        memory_count=len(memories),
        version="1.0.0"
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    
    Processes user message through the full agent pipeline:
    1. Input sanitization
    2. Task classification
    3. Mode selection
    4. Memory retrieval
    5. LLM generation
    6. Verification
    7. Personality filtering
    
    Returns response with confidence and metadata.
    """
    global orchestrator
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    logger.info(f"Chat request: {request.message[:100]}...")
    
    try:
        # Process through orchestrator
        result = orchestrator.process(request.message)
        
        # Build response
        response = ChatResponse(
            response=result.response,
            confidence=result.confidence,
            reasoning_mode=result.reasoning_mode,
            task_type=result.task_type,
            was_refused=result.was_refused
        )
        
        # Include optional logs if requested
        if request.include_logs:
            response.verification_log = result.verification_log
            response.memory_context = result.memory_context
            response.code_analysis = result.code_analysis
            response.research_analysis = result.research_analysis
        
        logger.info(f"Chat response: confidence={result.confidence:.2f}, mode={result.reasoning_mode}")
        
        return response
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/memory", response_model=MemoryResponse)
async def add_memory(request: MemoryRequest):
    """Add a memory entry"""
    global orchestrator
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        memory_id = orchestrator.add_memory(
            content=request.content,
            tag=request.tag,
            category=request.category
        )
        
        return MemoryResponse(
            id=memory_id,
            content=request.content,
            tag=request.tag,
            category=request.category
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/memory", response_model=MemoryListResponse)
async def list_memories(
    tag: Optional[str] = None,
    category: Optional[str] = None
):
    """List all memories, optionally filtered"""
    global orchestrator
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    memories = orchestrator.list_memories(tag=tag, category=category)
    
    return MemoryListResponse(
        memories=memories,
        count=len(memories)
    )


@app.delete("/memory/{memory_id}")
async def delete_memory(memory_id: str):
    """Delete a memory by ID"""
    global orchestrator
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    success = orchestrator.delete_memory(memory_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    return {"status": "deleted", "id": memory_id}


@app.post("/session/clear")
async def clear_session():
    """Clear current session (temporary memories and conversation history)"""
    global orchestrator
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    orchestrator.clear_session()
    
    return {"status": "session_cleared"}


@app.post("/model/load")
async def load_model():
    """Explicitly load the model into memory"""
    global orchestrator
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        success = orchestrator.load_model()
        return {"status": "loaded" if success else "failed"}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Load error: {str(e)}")


# Entry point for running directly
def main():
    """Run the server"""
    import uvicorn
    
    uvicorn.run(
        "api.server:app",
        host=app_config.api.host,
        port=app_config.api.port,
        reload=False,  # Disable for production
        log_level="info"
    )


if __name__ == "__main__":
    main()

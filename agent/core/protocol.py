from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import uuid
from datetime import datetime, timezone

class Role(str, Enum):
    PLANNER = "planner"
    EXECUTOR = "executor"
    CRITIC = "critic"
    ORCHESTRATOR = "orchestrator"
    USER = "user"

class MessageType(str, Enum):
    TASK_ASSIGNMENT = "task_assignment"
    PLAN_PROPOSAL = "plan_proposal"
    EXECUTION_RESULT = "execution_result"
    CRITIQUE = "critique"
    FEEDBACK = "feedback"
    ERROR = "error"

class AgentMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sender: Role
    receiver: Role
    type: MessageType
    content: str
    metadata: Dict[str, Any] = {}
    conversation_id: str

class SubGoal(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    status: str = "pending"  # pending, in_progress, completed, failed
    dependencies: List[str] = []
    result: Optional[str] = None
    critique: Optional[str] = None

class Plan(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    goal_id: str
    subgoals: List[SubGoal]
    strategy_explanation: str

class ExecutionResult(BaseModel):
    success: bool
    output: str
    artifacts: List[str] = []
    metrics: Dict[str, float] = {}
    error: Optional[str] = None

class CritiqueResult(BaseModel):
    approved: bool
    feedback: str
    score: float = 0.0
    suggestions: List[str] = []

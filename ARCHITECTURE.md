# Autonomous Multi-Agent System Architecture

## Overview
This system implements a self-learning, autonomous agent architecture composed of three specialized agents (Planner, Executor, Critic) running in isolated containers, coordinated by a central control plane.

## Architecture

### 1. Agents
- **Planner (Agent A)**: BREAKS DOWN high-level goals into executable sub-goals. Uses semantic memory to recall successful strategies.
- **Executor (Agent B)**: EXECUTES sub-goals using tools (Shell, File I/O, Web). It is the only agent with side-effect capabilities.
- **Critic (Agent C)**: REVIEWS plans and execution results. Enforces safety and quality standards before actions are finalized.

### 2. Communication
- Agents communicate via HTTP (FastAPI) using a strict JSON schema (`AgentMessage`).
- The **Coordinator** acts as the message bus and state manager.
- **Fallback Mechanisms**: 
    - The Coordinator implements exponential backoff retries for agent communication (up to 3 attempts).
    - Agents implement internal retries for LLM generation and JSON parsing to handle transient model failures.

### 3. Memory & Learning
- **ChromaDB** is used for long-term vector memory.
- **Self-Learning Loop**: 
    - **Concept**: The system treats every mission as a data point.
    - **Mechanism**:
        1. When a plan succeeds, the Planner stores the strategy and goal in ChromaDB with a "lesson" tag.
        2. When a plan fails (rejected by Critic or execution error), the failure reason is stored.
        3. **Context Injection**: Before generating a new plan, the Planner queries ChromaDB for similar past goals.
        4. **Outcome**: If the Planner sees a similar past failure, it avoids that strategy. If it sees a success, it mimics it.
    - This allows the system to "learn" from mistakes without model fine-tuning.

### 4. Infrastructure
- **Fully Docked**: Each agent runs in its own container (`python:3.14-slim` base).
- **Isolation**: The Executor acts within a specific workspace volume, isolated from the host system except for defined bind mounts.
- **Updates**: When `Dockerfile` or dependencies change, run `docker-compose build` to rebuild the images. The generic `agent` image is shared across services but configured via environment variables.

## System Diagram

```mermaid
graph TD
    User[User Goal] --> Coordinator[Coordinator / Control Plane]
    
    subgraph "Docker Network"
        Coordinator <-->|HTTP| Planner[Planner Agent]
        Coordinator <-->|HTTP| Critic[Critic Agent]
        Coordinator <-->|HTTP| Executor[Executor Agent]
        
        Planner <--> Memory[(Memory Store)]
        Critic <--> Memory
        Executor <--> Memory
    end
    
    Executor -->|Tools| FS[File System / Shell]
```

## Setup & Execution

### Prerequisites
1. Docker & Docker Compose installed.
2. A GGUF LLM model (e.g., Qwen2.5-7B) placed in `./models/model.gguf`.

### Running the System
1. Build and Start Services:
   ```bash
   docker-compose up --build -d
   ```

2. Run a Mission:
   ```bash
   docker-compose run coordinator
   ```
   *Modify the goal in `agent/core/coordinator.py` main block to change the mission.*

### Development
- **Agents Code**: `agent/agents.py`
- **Protocol**: `agent/core/protocol.py`
- **Coordinator**: `agent/core/coordinator.py`

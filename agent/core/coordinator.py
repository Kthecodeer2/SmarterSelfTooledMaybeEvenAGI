import os
import httpx
import logging
import json
import uuid
import asyncio
from typing import Dict, Any, Optional
from agent.core.protocol import AgentMessage, Role, MessageType, Plan, CritiqueResult, ExecutionResult
from agent.memory.memory_store import get_memory_store, MemoryTag

logger = logging.getLogger("coordinator")
logging.basicConfig(level=logging.INFO)

class Coordinator:
    def __init__(self):
        self.endpoints = {
            Role.PLANNER: os.getenv("PLANNER_URL", "http://planner:8000"),
            Role.EXECUTOR: os.getenv("EXECUTOR_URL", "http://executor:8000"),
            Role.CRITIC: os.getenv("CRITIC_URL", "http://critic:8000")
        }
        self.conversation_id = str(uuid.uuid4())
        self.memory = get_memory_store()

    async def _send(self, role: Role, message: AgentMessage, retries: int = 3) -> Optional[AgentMessage]:
        url = f"{self.endpoints[role]}/v1/message"
        last_error = None
        
        for i in range(retries):
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    resp = await client.post(url, json=message.model_dump())
                    resp.raise_for_status()
                    if not resp.content or resp.text == "null":
                        return None
                    return AgentMessage(**resp.json())
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                logger.warning(f"Connection error to {role} (attempt {i+1}/{retries}): {e}")
                last_error = e
                await asyncio.sleep(2 ** i) # Exponential backoff
            except Exception as e:
                logger.error(f"Unexpected error sending to {role}: {e}")
                raise e
                
        logger.error(f"Failed to send message to {role} after {retries} attempts")
        return None

    async def _plan_goal(self, user_goal: str) -> Optional[Plan]:
        """Collaborate with Planner and Critic to create an approved plan."""
        # 1. Ask Planner
        msg = AgentMessage(
            sender=Role.ORCHESTRATOR,
            receiver=Role.PLANNER,
            type=MessageType.TASK_ASSIGNMENT,
            content=user_goal,
            conversation_id=self.conversation_id
        )
        
        plan_msg = await self._send(Role.PLANNER, msg)
        if not plan_msg or plan_msg.type == MessageType.ERROR:
            logger.error("Planning failed or Planner unreachable")
            return None

        logger.info("Plan received, sending to Critic...")
        
        # 2. Ask Critic
        critic_msg = await self._send(Role.CRITIC, plan_msg)
        if not critic_msg:
             logger.error("Critic silent")
             return None

        critique_data = json.loads(critic_msg.content)
        if not critique_data.get("approved"):
             logger.warning(f"Plan rejected: {critique_data.get('feedback')}")
             self.memory.add(
                 f"Goal '{user_goal}' failed. Plan rejected: {critique_data.get('feedback')}",
                 tag=MemoryTag.PROJECT,
                 category="lesson"
             )
             return None

        logger.info("Plan approved.")
        try:
            plan_data = json.loads(plan_msg.content)
            # Reconstruct Plan object to ensure validity
            return Plan(
                goal_id=msg.id,
                subgoals=plan_data.get("subgoals", []),
                strategy_explanation=plan_data.get("strategy_explanation", "")
            )
        except Exception as e:
            logger.error(f"Failed to parse approved plan: {e}")
            return None

    async def _execute_plan(self, plan: Plan) -> tuple[bool, list[str]]:
        """Execute the approved plan via Executor."""
        all_success = True
        execution_log = []

        for sg in plan.subgoals:
            logger.info(f"Executing subgoal: {sg.description}")
            exec_msg = AgentMessage(
                sender=Role.ORCHESTRATOR,
                receiver=Role.EXECUTOR,
                type=MessageType.TASK_ASSIGNMENT,
                content=sg.description,
                conversation_id=self.conversation_id,
                metadata={"subgoal_id": sg.id}
            )
            
            desc = sg.description

            res_msg = await self._send(Role.EXECUTOR, exec_msg)
            if res_msg:
                try:
                    res_data = json.loads(res_msg.content)
                    success = res_data.get('success')
                    output = res_data.get('output', '')[:200]
                    logger.info(f"Result: {success} - {output}")
                    execution_log.append(f"Task: {desc} -> {success} ({output})")
                    if not success:
                        all_success = False
                except json.JSONDecodeError:
                    logger.error("Failed to decode execution result")
                    all_success = False
            else:
                logger.error("Executor unreachable")
                all_success = False
        
        return all_success, execution_log

    async def run(self, user_goal: str):
        logger.info(f"Starting goal: {user_goal}")
        
        plan = await self._plan_goal(user_goal)
        if not plan:
            return

        success, log = await self._execute_plan(plan)
        
        if success:
            logger.info("Mission Complete")
            self.memory.add(
                 f"Goal '{user_goal}' succeeded. Strategy: {plan.strategy_explanation}",
                 tag=MemoryTag.PROJECT,
                 category="lesson"
             )
        else:
            logger.error("Mission Failed")
            self.memory.add(
                 f"Goal '{user_goal}' failed during execution. Log: {'; '.join(log)}",
                 tag=MemoryTag.PROJECT,
                 category="lesson"
             )

if __name__ == "__main__":
    coordinator = Coordinator()
    # Example usage
    asyncio.run(coordinator.run("Analyze the current directory and create a summary file."))

import os
import httpx
import logging
import json
import uuid
from typing import Dict, Any
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

    async def _send(self, role: Role, message: AgentMessage) -> AgentMessage:
        url = f"{self.endpoints[role]}/v1/message"
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=message.model_dump())
            resp.raise_for_status()
            if not resp.content or resp.text == "null":
                return None
            return AgentMessage(**resp.json())

    async def run(self, user_goal: str):
        logger.info(f"Starting goal: {user_goal}")
        
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
            logger.error("Planning failed")
            return

        logger.info("Plan received, sending to Critic...")
        
        # 2. Ask Critic
        # The Planner might have already targeted CRITIC, but we route it
        critic_msg = await self._send(Role.CRITIC, plan_msg)
        
        if not critic_msg:
             logger.error("Critic silent")
             return

        critique_data = json.loads(critic_msg.content)
        if not critique_data.get("approved"):
             logger.warning(f"Plan rejected: {critique_data.get('feedback')}")
             # Store failure lesson
             self.memory.add(
                 f"Goal '{user_goal}' failed. Plan rejected: {critique_data.get('feedback')}",
                 tag=MemoryTag.PROJECT,
                 category="lesson"
             )
             return

        logger.info("Plan approved. Executing...")
        plan_data = json.loads(plan_msg.content)
        subgoals = plan_data.get("subgoals", [])
        
        all_success = True
        execution_log = []

        for sg in subgoals:
            logger.info(f"Executing subgoal: {sg['description']}")
            exec_msg = AgentMessage(
                sender=Role.ORCHESTRATOR,
                receiver=Role.EXECUTOR,
                type=MessageType.TASK_ASSIGNMENT,
                content=sg['description'],
                conversation_id=self.conversation_id,
                metadata={"subgoal_id": sg.get("id")}
            )
            
            res_msg = await self._send(Role.EXECUTOR, exec_msg)
            if res_msg:
                res_data = json.loads(res_msg.content)
                success = res_data.get('success')
                output = res_data.get('output', '')[:200]
                logger.info(f"Result: {success} - {output}")
                execution_log.append(f"Task: {sg['description']} -> {success} ({output})")
                if not success:
                    all_success = False
        
        if all_success:
            logger.info("Mission Complete")
            self.memory.add(
                 f"Goal '{user_goal}' succeeded. Strategy: {plan_data.get('strategy_explanation')}",
                 tag=MemoryTag.PROJECT,
                 category="lesson"
             )
        else:
            logger.error("Mission Failed")
            self.memory.add(
                 f"Goal '{user_goal}' failed during execution. Log: {'; '.join(execution_log)}",
                 tag=MemoryTag.PROJECT,
                 category="lesson"
             )

if __name__ == "__main__":
    import asyncio
    coordinator = Coordinator()
    # Example usage
    asyncio.run(coordinator.run("Analyze the current directory and create a summary file."))

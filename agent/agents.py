import json
import logging
from typing import Dict, List, Optional, Any
from agent.core.protocol import AgentMessage, Role, MessageType, Plan, SubGoal, ExecutionResult, CritiqueResult
from agent.core.llm_interface import LLMInterface, get_llm_interface
from agent.memory.memory_store import MemoryStore, get_memory_store, MemoryTag
from agent.tools.definitions import get_tools_map

logger = logging.getLogger(__name__)

class BaseAgent:
    def __init__(self, role: Role, name: str):
        self.role = role
        self.name = name
        self.llm = get_llm_interface()
        self.memory = get_memory_store()
        self.logger = logging.getLogger(f"agent.{name}")

    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        raise NotImplementedError

    def _query_memory(self, query: str) -> str:
        try:
            entries = self.memory.query(query, n_results=3)
            return "\n".join([f"- {e.content}" for e in entries])
        except Exception as e:
            self.logger.warning(f"Memory query failed: {e}")
            return ""

    def _store_memory(self, content: str, category: str):
        try:
            self.memory.add(content, tag=MemoryTag.PROJECT, category=category)
        except Exception as e:
            self.logger.warning(f"Memory storage failed: {e}")

class PlannerAgent(BaseAgent):
    def __init__(self):
        super().__init__(Role.PLANNER, "Planner")

    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        self.logger.info(f"Planner received: {message.type}")
        
        try:
            if message.type == MessageType.TASK_ASSIGNMENT:
                return self._create_plan(message)
            elif message.type == MessageType.CRITIQUE:
                 return self._refine_plan(message)
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return AgentMessage(
                sender=self.role,
                receiver=Role.ORCHESTRATOR,
                type=MessageType.ERROR,
                content=f"Internal Planner Error: {str(e)}",
                conversation_id=message.conversation_id
            )
        return None

    def _create_plan(self, message: AgentMessage) -> AgentMessage:
        goal = message.content
        context = self._query_memory(goal)
        
        prompt = f"""
        You are the Planner Agent.
        Goal: {goal}
        Context: {context}
        
        Create a detailed execution plan with distinct steps (subgoals).
        Each subgoal must be actionable by an Executor using Shell, File I/O, or Web tools.
        
        Output JSON format:
        {{
            "strategy_explanation": "...",
            "subgoals": [
                {{"description": "...", "dependencies": []}}
            ]
        }}
        """
        
        # Retry logic for LLM generation/parsing
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            response = self.llm.generate(prompt)
            try:
                # Robust JSON extraction
                text = response.text
                start = text.find('{')
                end = text.rfind('}') + 1
                if start == -1 or end == -1:
                    raise ValueError("No JSON found in response")
                
                json_str = text[start:end]
                data = json.loads(json_str)
                
                # Validate structure
                if "subgoals" not in data:
                     raise ValueError("Missing 'subgoals' field")
                
                plan = Plan(
                    goal_id=message.id,
                    subgoals=[SubGoal(description=sg["description"], dependencies=sg.get("dependencies", [])) for sg in data.get("subgoals", [])],
                    strategy_explanation=data.get("strategy_explanation", "")
                )
                
                return AgentMessage(
                    sender=self.role,
                    receiver=Role.CRITIC,
                    type=MessageType.PLAN_PROPOSAL,
                    content=plan.model_dump_json(),
                    conversation_id=message.conversation_id
                )
            except Exception as e:
                last_error = e
                self.logger.warning(f"Plan generation attempt {attempt+1} failed: {e}")
                # Optional: Add error to prompt for next attempt (not implemented here for brevity)

        self.logger.error(f"Failed to generate plan after {max_retries} attempts: {last_error}")
        return AgentMessage(
            sender=self.role,
            receiver=Role.ORCHESTRATOR,
            type=MessageType.ERROR,
            content=f"Planning failed: {str(last_error)}",
            conversation_id=message.conversation_id
        )

    def _refine_plan(self, message: AgentMessage) -> AgentMessage:
        # Simplified placeholder
        return AgentMessage(
            sender=self.role,
            receiver=Role.ORCHESTRATOR,
            type=MessageType.ERROR,
            content="Plan rejected, refinement needed (Not fully implemented)",
            conversation_id=message.conversation_id
        )

class ExecutorAgent(BaseAgent):
    def __init__(self):
        super().__init__(Role.EXECUTOR, "Executor")
        self.tools = get_tools_map()

    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        if message.type == MessageType.TASK_ASSIGNMENT:
            return self._execute_task(message)
        return None

    def _execute_task(self, message: AgentMessage) -> AgentMessage:
        subgoal = message.content
        
        prompt = f"""
        You are the Executor Agent.
        Task: {subgoal}
        Tools available: {list(self.tools.keys())}
        
        Decide the best tool to use and the arguments.
        Output JSON:
        {{
            "tool": "tool_name",
            "args": {{ ... }},
            "reasoning": "..."
        }}
        """
        
        # Simple retry for tool selection logic
        max_retries = 2
        result_output = "Execution failed"
        success = False
        
        for attempt in range(max_retries):
            response = self.llm.generate(prompt)
            try:
                text = response.text
                start = text.find('{')
                end = text.rfind('}') + 1
                if start != -1 and end != -1:
                    cmd = json.loads(text[start:end])
                    tool_name = cmd.get("tool")
                    if tool_name in self.tools:
                        tool = self.tools[tool_name]
                        args = cmd.get("args", {})
                        
                        # Execute safely
                        try:
                            tool_res = tool.execute(**args)
                            success = tool_res.success
                            result_output = tool_res.output if success else tool_res.error
                        except Exception as tool_err:
                            success = False
                            result_output = f"Tool Execution Exception: {str(tool_err)}"
                        
                        break # Valid tool call attempted, stop retrying LLM
                    else:
                        raise ValueError(f"Unknown tool: {tool_name}")
            except Exception as e:
                self.logger.warning(f"Execution decision attempt {attempt+1} failed: {e}")
                if attempt == max_retries - 1:
                    result_output = f"Failed to select tool: {e}"

        exec_result = ExecutionResult(success=success, output=str(result_output))
        
        return AgentMessage(
            sender=self.role,
            receiver=Role.ORCHESTRATOR,
            type=MessageType.EXECUTION_RESULT,
            content=exec_result.model_dump_json(),
            conversation_id=message.conversation_id,
            metadata={"subgoal_id": message.metadata.get("subgoal_id")}
        )

class CriticAgent(BaseAgent):
    def __init__(self):
        super().__init__(Role.CRITIC, "Critic")

    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        if message.type == MessageType.PLAN_PROPOSAL:
            return self._critique_plan(message)
        elif message.type == MessageType.EXECUTION_RESULT:
            return self._critique_execution(message)
        return None

    def _critique_plan(self, message: AgentMessage) -> AgentMessage:
        plan_json = message.content
        prompt = f"""
        You are the Critic Agent.
        Review this plan for feasibility, safety, and completeness.
        Plan: {plan_json}
        
        Output JSON:
        {{
            "approved": true/false,
            "feedback": "...",
            "score": 0.0-1.0
        }}
        """
        response = self.llm.generate(prompt)
        
        critique = CritiqueResult(approved=False, feedback="Failed to parse critique", score=0)
        try:
            text = response.text
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end != -1:
                data = json.loads(text[start:end])
                critique = CritiqueResult(**data)
        except Exception as e:
            self.logger.error(f"Critique parsing failed: {e}")
            
        return AgentMessage(
            sender=self.role,
            receiver=Role.PLANNER if not critique.approved else Role.ORCHESTRATOR,
            type=MessageType.CRITIQUE,
            content=critique.model_dump_json(),
            conversation_id=message.conversation_id
        )

    def _critique_execution(self, message: AgentMessage) -> AgentMessage:
        return None

import json
import logging
from typing import Dict, List, Optional
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
        entries = self.memory.query(query, n_results=3)
        return "\n".join([f"- {e.content}" for e in entries])

    def _store_memory(self, content: str, category: str):
        self.memory.add(content, tag=MemoryTag.PROJECT, category=category)

class PlannerAgent(BaseAgent):
    def __init__(self):
        super().__init__(Role.PLANNER, "Planner")

    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        self.logger.info(f"Planner received: {message.type}")
        
        if message.type == MessageType.TASK_ASSIGNMENT:
            return self._create_plan(message)
        elif message.type == MessageType.CRITIQUE:
             # If critique is negative, refine plan
             return self._refine_plan(message)
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
        response = self.llm.generate(prompt)
        try:
            # Simple parsing attempt (robustness would need json repair)
            # Find first { and last }
            text = response.text
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end != -1:
                data = json.loads(text[start:end])
                
                plan = Plan(
                    goal_id=message.id, # Linking to original task ID
                    subgoals=[SubGoal(description=sg["description"], dependencies=sg.get("dependencies", [])) for sg in data.get("subgoals", [])],
                    strategy_explanation=data.get("strategy_explanation", "")
                )
                
                return AgentMessage(
                    sender=self.role,
                    receiver=Role.CRITIC, # Send to critic first
                    type=MessageType.PLAN_PROPOSAL,
                    content=plan.model_dump_json(),
                    conversation_id=message.conversation_id
                )
        except Exception as e:
            self.logger.error(f"Failed to parse plan: {e}")
            return AgentMessage(
                sender=self.role,
                receiver=Role.ORCHESTRATOR,
                type=MessageType.ERROR,
                content=f"Planning failed: {str(e)}",
                conversation_id=message.conversation_id
            )
        return None

    def _refine_plan(self, message: AgentMessage) -> AgentMessage:
        # Simplified: Just log for now, or retry planning based on feedback
        # In a real system, we'd feed the critique back into the prompt
        critique = json.loads(message.content)
        if critique.get("approved"):
             # If approved, send to Executor (Orchestrator handles routing usually, but here we can signal)
             # Actually, if Planner gets an approved critique, it might forward to Executor.
             # But let's assume Orchestrator routes "Approved Plan" to Executor.
             pass
        
        return AgentMessage(
            sender=self.role,
            receiver=Role.ORCHESTRATOR, # Ask orchestrator to handle retry/failure
            type=MessageType.ERROR,
            content="Plan rejected, refinement needed (Not fully implemented)",
            conversation_id=message.conversation_id
        )

class ExecutorAgent(BaseAgent):
    def __init__(self):
        super().__init__(Role.EXECUTOR, "Executor")
        self.tools = get_tools_map()

    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        if message.type == MessageType.TASK_ASSIGNMENT: # Executor receives a subgoal as a task
            return self._execute_task(message)
        return None

    def _execute_task(self, message: AgentMessage) -> AgentMessage:
        subgoal = message.content # Description of what to do
        
        # Self-Prompting Loop for execution
        # 1. Think: What tool do I need?
        # 2. Act: Call tool
        # 3. Observe: Check result
        
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
        
        response = self.llm.generate(prompt)
        result_output = "No execution attempted"
        success = False
        
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
                    # Execute
                    tool_res = tool.execute(**args)
                    success = tool_res.success
                    result_output = tool_res.output if success else tool_res.error
                else:
                    result_output = f"Unknown tool: {tool_name}"
        except Exception as e:
            result_output = f"Execution error: {e}"

        exec_result = ExecutionResult(success=success, output=str(result_output))
        
        return AgentMessage(
            sender=self.role,
            receiver=Role.ORCHESTRATOR, # Report back to Orchestrator
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
        except:
            pass
            
        return AgentMessage(
            sender=self.role,
            receiver=Role.PLANNER if not critique.approved else Role.ORCHESTRATOR,
            type=MessageType.CRITIQUE,
            content=critique.model_dump_json(),
            conversation_id=message.conversation_id
        )

    def _critique_execution(self, message: AgentMessage) -> AgentMessage:
        # Simple pass-through for now, or check if output matches expectation
        # For simplicity, we assume execution result is self-contained for now
        return None

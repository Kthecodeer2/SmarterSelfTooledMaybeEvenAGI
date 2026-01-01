import unittest
from unittest.mock import MagicMock, patch
import json
from datetime import datetime
from agent.agents import PlannerAgent, ExecutorAgent, CriticAgent
from agent.core.protocol import AgentMessage, Role, MessageType, Plan, SubGoal, ExecutionResult, CritiqueResult

class TestAgents(unittest.TestCase):
    def setUp(self):
        # Mock LLM and MemoryStore globally for agents
        self.llm_patcher = patch('agent.agents.get_llm_interface')
        self.memory_patcher = patch('agent.agents.get_memory_store')
        self.mock_get_llm = self.llm_patcher.start()
        self.mock_get_memory = self.memory_patcher.start()
        
        self.mock_llm = MagicMock()
        self.mock_memory = MagicMock()
        self.mock_get_llm.return_value = self.mock_llm
        self.mock_get_memory.return_value = self.mock_memory

    def tearDown(self):
        self.llm_patcher.stop()
        self.memory_patcher.stop()

    def test_planner_creates_plan(self):
        agent = PlannerAgent()
        msg = AgentMessage(
            sender=Role.ORCHESTRATOR,
            receiver=Role.PLANNER,
            type=MessageType.TASK_ASSIGNMENT,
            content="Test Goal",
            conversation_id="123"
        )
        
        # Mock LLM response for plan
        plan_json = json.dumps({
            "strategy_explanation": "Test Strategy",
            "subgoals": [{"description": "Step 1", "dependencies": []}]
        })
        self.mock_llm.generate.return_value = MagicMock(text=plan_json)
        
        response = agent.process_message(msg)
        
        self.assertIsNotNone(response)
        self.assertEqual(response.type, MessageType.PLAN_PROPOSAL)
        self.assertIn("Step 1", response.content)

    def test_executor_executes_task(self):
        agent = ExecutorAgent()
        msg = AgentMessage(
            sender=Role.ORCHESTRATOR,
            receiver=Role.EXECUTOR,
            type=MessageType.TASK_ASSIGNMENT,
            content="Print hello",
            conversation_id="123",
            metadata={"subgoal_id": "sg1"}
        )
        
        # Mock LLM response for tool selection
        tool_json = json.dumps({
            "tool": "shell",
            "args": {"command": "echo hello"},
            "reasoning": "Simple echo"
        })
        self.mock_llm.generate.return_value = MagicMock(text=tool_json)
        
        # Mock Tool execution
        with patch.dict(agent.tools, {"shell": MagicMock()}):
            mock_tool = agent.tools["shell"]
            mock_tool.execute.return_value = MagicMock(success=True, output="hello")
            
            response = agent.process_message(msg)
            
            self.assertIsNotNone(response)
            self.assertEqual(response.type, MessageType.EXECUTION_RESULT)
            result = json.loads(response.content)
            self.assertTrue(result["success"])
            self.assertEqual(result["output"], "hello")

    def test_critic_validates_plan(self):
        agent = CriticAgent()
        plan = Plan(goal_id="g1", subgoals=[SubGoal(description="s1")], strategy_explanation="strat")
        msg = AgentMessage(
            sender=Role.PLANNER,
            receiver=Role.CRITIC,
            type=MessageType.PLAN_PROPOSAL,
            content=plan.model_dump_json(),
            conversation_id="123"
        )
        
        # Mock LLM response
        critique_json = json.dumps({
            "approved": True,
            "feedback": "Good plan",
            "score": 0.9
        })
        self.mock_llm.generate.return_value = MagicMock(text=critique_json)
        
        response = agent.process_message(msg)
        
        self.assertIsNotNone(response)
        self.assertEqual(response.type, MessageType.CRITIQUE)
        result = json.loads(response.content)
        self.assertTrue(result["approved"])

if __name__ == '__main__':
    unittest.main()

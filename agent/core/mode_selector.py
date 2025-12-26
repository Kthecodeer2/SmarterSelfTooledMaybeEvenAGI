"""
Mode Selector Module
Selects reasoning mode (FAST/DEEP) and injects internal control tokens
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel

from .task_classifier import TaskClassification, TaskType, RiskDomain
from .input_pipeline import ProcessedInput


class ReasoningMode(str, Enum):
    """Reasoning modes for the agent"""
    FAST = "fast"      # Quick responses for trivial tasks
    DEEP = "deep"      # Chain-of-thought for complex tasks
    VERIFY = "verify"  # Verification mode for high-risk tasks


class ControlTokens(str, Enum):
    """
    Internal control tokens - NEVER exposed to users.
    These are injected into system prompts only.
    """
    FAST = "<FAST>"
    DEEP = "<DEEP>"
    VERIFY = "<VERIFY>"
    REFUSE_OK = "<REFUSE_OK>"  # Permission to refuse if uncertain


class ModeSelection(BaseModel):
    """Result of mode selection"""
    reasoning_mode: ReasoningMode
    control_tokens: list[str]
    system_prompt_modifier: str
    requires_cot: bool  # Chain of thought
    max_thinking_steps: int


class ModeSelector:
    """
    Selects appropriate reasoning mode based on task classification.
    Injects internal control tokens for the LLM.
    
    DESIGN DECISION: Control tokens are system-internal only.
    They're injected into system prompts, never in user-visible output.
    """
    
    # Keywords that indicate need for deep thinking
    DEEP_THINKING_TRIGGERS = [
        "explain in detail", "step by step", "analyze",
        "design", "architect", "debug", "optimize",
        "security audit", "code review", "trade-off",
        "compare and contrast", "prove", "derive",
        "complex", "advanced", "difficult",
    ]
    
    # System prompt modifiers for each mode
    FAST_MODIFIER = """
You are in FAST mode. Provide direct, concise answers.
- No unnecessary explanations
- Get to the point immediately
- Skip pleasantries
"""
    
    DEEP_MODIFIER = """
You are in DEEP thinking mode. Think carefully before responding.
- Break down the problem into components
- Consider edge cases and potential issues
- Verify your reasoning step by step
- Only provide an answer when confident
- If uncertain, explain exactly why
"""
    
    VERIFY_MODIFIER = """
You are in VERIFICATION mode. This is a high-stakes task.
- Double-check all claims and calculations
- Identify potential errors in your reasoning
- Consider: "What would make this answer wrong?"
- Flag any assumptions you're making
- If confidence < 90%, explain uncertainty
- Refusing to answer is acceptable if uncertain
"""
    
    def __init__(self, deep_triggers: Optional[list[str]] = None):
        self.deep_triggers = deep_triggers or self.DEEP_THINKING_TRIGGERS
    
    def needs_deep_thinking(self, processed: ProcessedInput) -> bool:
        """Check if input requires deep thinking based on content"""
        text = processed.cleaned.lower()
        
        # Long inputs likely need more thought
        if processed.word_count > 50:
            return True
        
        # Code always gets deep thinking
        if processed.has_code:
            return True
        
        # Check for trigger keywords
        return any(trigger in text for trigger in self.deep_triggers)
    
    def select(
        self,
        processed: ProcessedInput,
        classification: TaskClassification
    ) -> ModeSelection:
        """
        Select reasoning mode based on input and classification.
        Returns mode with appropriate control tokens and prompts.
        """
        control_tokens = []
        
        # High-risk tasks always get VERIFY mode
        if classification.is_high_risk:
            mode = ReasoningMode.VERIFY
            control_tokens = [
                ControlTokens.DEEP.value,
                ControlTokens.VERIFY.value,
                ControlTokens.REFUSE_OK.value
            ]
            modifier = self.VERIFY_MODIFIER
            max_steps = 10
            
        # Trivial tasks get FAST mode
        elif classification.task_type == TaskType.TRIVIAL:
            mode = ReasoningMode.FAST
            control_tokens = [ControlTokens.FAST.value]
            modifier = self.FAST_MODIFIER
            max_steps = 2
            
        # Coding and research tasks, or complex inputs, get DEEP mode
        elif (classification.task_type in [TaskType.CODING, TaskType.RESEARCH] 
              or self.needs_deep_thinking(processed)):
            mode = ReasoningMode.DEEP
            control_tokens = [ControlTokens.DEEP.value]
            if classification.requires_verification:
                control_tokens.append(ControlTokens.VERIFY.value)
            modifier = self.DEEP_MODIFIER
            max_steps = 5
            
        # Default to FAST for simple conversation
        else:
            mode = ReasoningMode.FAST
            control_tokens = [ControlTokens.FAST.value]
            modifier = self.FAST_MODIFIER
            max_steps = 3
        
        return ModeSelection(
            reasoning_mode=mode,
            control_tokens=control_tokens,
            system_prompt_modifier=modifier,
            requires_cot=mode in [ReasoningMode.DEEP, ReasoningMode.VERIFY],
            max_thinking_steps=max_steps
        )


# Singleton instance
_selector: Optional[ModeSelector] = None


def get_mode_selector() -> ModeSelector:
    """Get or create singleton ModeSelector instance"""
    global _selector
    if _selector is None:
        _selector = ModeSelector()
    return _selector

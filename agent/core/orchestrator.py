"""
Orchestrator Module
Wires all components together for end-to-end processing.

DESIGN DECISIONS:
- Single entry point for all requests
- Manages flow: input -> classify -> mode -> generate -> verify -> respond
- Handles memory retrieval and storage
- Applies personality filtering
- Logs for debugging
"""

import logging
from typing import Optional
from pydantic import BaseModel

from .input_pipeline import InputPipeline, ProcessedInput, get_input_pipeline
from .task_classifier import TaskClassifier, TaskClassification, TaskType, get_task_classifier
from .mode_selector import ModeSelector, ModeSelection, ReasoningMode, get_mode_selector
from .llm_interface import LLMInterface, LLMResponse, LLMConfig, get_llm_interface
from .verification import VerificationLayer, VerificationResult, get_verification_layer
from .confidence import ConfidenceScorer, ConfidenceScore, get_confidence_scorer
from .retry_loop import RetryLoop, RetryResult, get_retry_loop
from .refusal import RefusalHandler, RefusalResponse, get_refusal_handler
from ..memory import MemoryStore, MemoryTag, get_memory_store
from ..coding import CodeAnalyzer, get_code_analyzer
from ..research import ResearchMode, get_research_mode
from ..personality import ResponseFilter, get_response_filter


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentResponse(BaseModel):
    """Complete agent response with metadata"""
    response: str
    confidence: float
    reasoning_mode: str
    task_type: str
    verification_passed: bool
    was_refused: bool
    retry_count: int
    
    # Optional detailed logs
    verification_log: Optional[list[dict]] = None
    memory_context: Optional[str] = None
    code_analysis: Optional[dict] = None
    research_analysis: Optional[dict] = None


class ConversationContext(BaseModel):
    """Maintains conversation state"""
    history: list[dict] = []  # [{"role": "user/assistant", "content": "..."}]
    environment: dict = {}  # Remembered environment info
    max_history: int = 10
    
    def add_user_message(self, content: str):
        """Add user message to history"""
        self.history.append({"role": "user", "content": content})
        self._trim_history()
    
    def add_assistant_message(self, content: str):
        """Add assistant message to history"""
        self.history.append({"role": "assistant", "content": content})
        self._trim_history()
    
    def _trim_history(self):
        """Keep history within limits"""
        if len(self.history) > self.max_history * 2:
            # Keep first exchange (for context) and recent history
            self.history = self.history[:2] + self.history[-(self.max_history * 2 - 2):]
    
    def get_history_for_prompt(self) -> list[dict]:
        """Get history formatted for LLM prompt"""
        return self.history[:-1] if self.history else []  # Exclude current message


class Orchestrator:
    """
    Main orchestrator that coordinates all agent components.
    
    Flow:
    1. Input Pipeline: Sanitize input
    2. Task Classifier: Determine task type and risk
    3. Mode Selector: Choose reasoning mode
    4. Memory: Retrieve relevant context
    5. LLM Interface: Generate response
    6. Verification: Check response quality
    7. Retry Loop: Retry if needed
    8. Refusal: Handle failures gracefully
    9. Personality Filter: Apply style rules
    10. Memory: Store learned information
    
    All optimized for MacBook Air M4.
    """
    
    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        memory_path: str = "./memory_store",
        enable_logging: bool = True
    ):
        # Initialize components
        self.input_pipeline = get_input_pipeline()
        self.task_classifier = get_task_classifier()
        self.mode_selector = get_mode_selector()
        self.llm = get_llm_interface(llm_config)
        self.verifier = get_verification_layer()
        self.scorer = get_confidence_scorer()
        self.retry_loop = get_retry_loop()
        self.refusal_handler = get_refusal_handler()
        self.memory = get_memory_store(memory_path)
        self.code_analyzer = get_code_analyzer()
        self.research_mode = get_research_mode()
        self.response_filter = get_response_filter()
        
        self.enable_logging = enable_logging
        self.context = ConversationContext()
    
    def _log(self, message: str, level: str = "info"):
        """Log message if logging is enabled"""
        if self.enable_logging:
            getattr(logger, level)(message)
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response from LLM"""
        response = self.llm.generate(prompt)
        return response.text
    
    def _extract_environment_info(self, text: str) -> dict:
        """Extract environment information from user input"""
        env_info = {}
        
        # OS detection
        os_patterns = {
            "macos": ["mac", "macos", "osx", "darwin", "m1", "m2", "m3", "m4"],
            "linux": ["linux", "ubuntu", "debian", "fedora", "centos"],
            "windows": ["windows", "win10", "win11"],
        }
        
        lower_text = text.lower()
        for os_name, patterns in os_patterns.items():
            if any(p in lower_text for p in patterns):
                env_info["os"] = os_name
                break
        
        # Architecture detection
        if any(p in lower_text for p in ["arm", "m1", "m2", "m3", "m4", "apple silicon"]):
            env_info["arch"] = "arm64"
        elif any(p in lower_text for p in ["x86", "amd64", "intel"]):
            env_info["arch"] = "x86_64"
        
        # Python version detection
        import re
        python_match = re.search(r"python\s*(\d+\.\d+)", lower_text)
        if python_match:
            env_info["python_version"] = python_match.group(1)
        
        return env_info
    
    def _store_environment_info(self, env_info: dict):
        """Store extracted environment info in memory"""
        if env_info:
            for key, value in env_info.items():
                # Check if we already have this info
                existing = self.memory.query(
                    f"environment {key}",
                    category_filter="environment",
                    n_results=1
                )
                
                if not existing:
                    self.memory.add(
                        content=f"{key}: {value}",
                        tag=MemoryTag.PERMANENT,
                        category="environment"
                    )
                    self._log(f"Stored environment info: {key}={value}")
    
    def process(self, user_input: str) -> AgentResponse:
        """
        Process user input and generate response.
        
        This is the main entry point for the agent.
        
        Args:
            user_input: Raw user input text
            
        Returns:
            AgentResponse with full response and metadata
        """
        self._log(f"Processing input: {user_input[:100]}...")
        
        # Step 1: Input Pipeline - Sanitize input
        processed = self.input_pipeline.process(user_input)
        self._log(f"Input sanitized, word count: {processed.word_count}")
        
        # Step 2: Task Classifier - Determine task type
        classification = self.task_classifier.classify(processed)
        self._log(f"Task classified: {classification.task_type}, high-risk: {classification.is_high_risk}")
        
        # Step 3: Mode Selector - Choose reasoning mode
        mode = self.mode_selector.select(processed, classification)
        self._log(f"Reasoning mode: {mode.reasoning_mode}")
        
        # Step 4: Memory - Retrieve relevant context
        memory_context = self.memory.get_context_for_query(
            processed.cleaned,
            max_memories=5
        )
        if memory_context:
            self._log("Retrieved memory context")
        
        # Step 5: Extract and store environment info
        env_info = self._extract_environment_info(processed.cleaned)
        if env_info:
            self._store_environment_info(env_info)
            self.context.environment.update(env_info)
        
        # Step 6: Build prompt with all context
        prompt = self.llm.build_prompt(
            user_message=processed.cleaned,
            mode_modifier=mode.system_prompt_modifier,
            conversation_history=self.context.get_history_for_prompt(),
            context=memory_context
        )
        
        # Step 7: Run through retry loop
        retry_result = self.retry_loop.run(
            generate_fn=self._generate_response,
            prompt=prompt,
            classification=classification,
            mode=mode,
            code_blocks=processed.code_blocks
        )
        
        self._log(f"Retry loop completed: {retry_result.attempts} attempts, confidence: {retry_result.final_confidence}")
        
        # Step 8: Handle refusal if needed
        if retry_result.refused:
            self._log("Response refused due to low confidence")
            
            # Get verification result for refusal context
            verification = self.verifier.verify(
                retry_result.final_response,
                processed.code_blocks
            )
            confidence = self.scorer.score(
                retry_result.final_response,
                classification,
                mode,
                verification
            )
            
            refusal = self.refusal_handler.generate_refusal(
                reason="Confidence too low after verification",
                response_text=retry_result.final_response,
                classification=classification,
                verification=verification,
                confidence=confidence
            )
            
            final_response = refusal.formatted_response
            was_refused = True
        else:
            final_response = retry_result.final_response
            was_refused = False
        
        # Step 9: Code analysis for coding tasks
        code_analysis = None
        if classification.task_type == TaskType.CODING and processed.code_blocks:
            for code_block in processed.code_blocks:
                analysis = self.code_analyzer.analyze(code_block)
                if analysis.has_critical or analysis.has_warnings:
                    code_analysis = {
                        "language": analysis.language,
                        "issues": [i.dict() for i in analysis.issues],
                        "security_score": analysis.security_score
                    }
                    break
        
        # Step 10: Research analysis for research tasks
        research_analysis = None
        if classification.task_type == TaskType.RESEARCH:
            analysis = self.research_mode.analyze(final_response)
            research_analysis = {
                "claims": len(analysis.claims),
                "assumptions": analysis.assumptions,
                "limitations": analysis.limitations,
                "confidence": analysis.confidence
            }
        
        # Step 11: Apply personality filter
        filtered_response = self.response_filter.apply_all_filters(
            text=final_response,
            is_question=processed.is_question
        )
        
        # Step 12: Update conversation context
        self.context.add_user_message(processed.cleaned)
        self.context.add_assistant_message(filtered_response)
        
        # Build final response
        return AgentResponse(
            response=filtered_response,
            confidence=retry_result.final_confidence,
            reasoning_mode=mode.reasoning_mode.value,
            task_type=classification.task_type.value,
            verification_passed=retry_result.accepted,
            was_refused=was_refused,
            retry_count=retry_result.attempts - 1,
            verification_log=retry_result.verification_log if self.enable_logging else None,
            memory_context=memory_context if memory_context else None,
            code_analysis=code_analysis,
            research_analysis=research_analysis
        )
    
    def add_memory(
        self,
        content: str,
        tag: str = "project",
        category: str = "fact"
    ) -> str:
        """
        Manually add a memory entry.
        
        Args:
            content: Memory content
            tag: Memory tag (temporary, project, permanent)
            category: Category (environment, preference, constraint, goal, fact)
            
        Returns:
            Memory ID
        """
        tag_enum = MemoryTag(tag)
        entry = self.memory.add(content, tag_enum, category)
        return entry.id
    
    def list_memories(
        self,
        tag: Optional[str] = None,
        category: Optional[str] = None
    ) -> list[dict]:
        """List all memories, optionally filtered"""
        tag_enum = MemoryTag(tag) if tag else None
        entries = self.memory.list_all(tag_enum, category)
        return [
            {
                "id": e.id,
                "content": e.content,
                "tag": e.tag.value,
                "category": e.category,
                "created_at": e.created_at
            }
            for e in entries
        ]
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID"""
        return self.memory.delete(memory_id)
    
    def clear_session(self):
        """Clear temporary data for new session"""
        self.context = ConversationContext()
        self.memory.clear_temporary()
        self._log("Session cleared")
    
    def get_model_status(self) -> dict:
        """Get current model status"""
        return self.llm.get_model_info()
    
    def load_model(self) -> bool:
        """Explicitly load the model"""
        return self.llm.load_model()


# Singleton instance
_orchestrator: Optional[Orchestrator] = None


def get_orchestrator(
    llm_config: Optional[LLMConfig] = None,
    memory_path: str = "./memory_store",
    enable_logging: bool = True
) -> Orchestrator:
    """Get or create singleton Orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator(llm_config, memory_path, enable_logging)
    return _orchestrator

"""
Retry Loop Module
Implements the verification-retry loop for ensuring answer quality.

DESIGN DECISIONS:
- Max retries to prevent infinite loops
- Feedback injection for improvement
- Escalation path to refusal
- Logging for debugging
"""

from typing import Optional, Callable
from pydantic import BaseModel

from .confidence import ConfidenceScore, ConfidenceScorer, get_confidence_scorer
from .verification import VerificationResult, VerificationLayer, get_verification_layer
from .task_classifier import TaskClassification
from .mode_selector import ModeSelection


class RetryResult(BaseModel):
    """Result of the retry loop"""
    final_response: str
    final_confidence: float
    attempts: int
    accepted: bool
    refused: bool
    verification_log: list[dict]


class RetryLoop:
    """
    Implements the verification-retry loop.
    
    Flow:
    1. Generate response
    2. Verify response
    3. Score confidence
    4. If confidence >= 0.9: accept
    5. If 0.6 <= confidence < 0.9: retry with feedback
    6. If confidence < 0.6 after max retries: refuse
    """
    
    RETRY_PROMPT_TEMPLATE = """
Your previous response had issues:
{issues}

Self-critique: {critique}

Please provide a corrected response that addresses these concerns.
Be more careful and precise this time.
"""
    
    def __init__(
        self,
        max_retries: int = 2,
        verifier: Optional[VerificationLayer] = None,
        scorer: Optional[ConfidenceScorer] = None
    ):
        self.max_retries = max_retries
        self.verifier = verifier or get_verification_layer()
        self.scorer = scorer or get_confidence_scorer()
    
    def _format_issues(self, verification: VerificationResult) -> str:
        """Format verification issues for feedback"""
        parts = []
        
        if verification.issues:
            parts.append("Issues: " + "; ".join(verification.issues))
        
        if verification.warnings:
            parts.append("Warnings: " + "; ".join(verification.warnings))
        
        critical_code = [
            i for i in verification.code_issues 
            if i.get("severity") == "critical"
        ]
        if critical_code:
            parts.append("Code issues: " + "; ".join(
                i["description"] for i in critical_code
            ))
        
        return "\n".join(parts) if parts else "General quality concerns"
    
    def _build_retry_prompt(
        self,
        original_prompt: str,
        verification: VerificationResult
    ) -> str:
        """Build prompt for retry attempt"""
        issues = self._format_issues(verification)
        
        feedback = self.RETRY_PROMPT_TEMPLATE.format(
            issues=issues,
            critique=verification.self_critique
        )
        
        return f"{original_prompt}\n\n{feedback}"
    
    def run(
        self,
        generate_fn: Callable[[str], str],
        prompt: str,
        classification: TaskClassification,
        mode: ModeSelection,
        code_blocks: Optional[list[str]] = None
    ) -> RetryResult:
        """
        Run the retry loop.
        
        Args:
            generate_fn: Function that takes prompt and returns response text
            prompt: Initial prompt to send
            classification: Task classification
            mode: Selected reasoning mode
            code_blocks: Any code blocks from user input
            
        Returns:
            RetryResult with final response and metadata
        """
        verification_log = []
        current_prompt = prompt
        attempts = 0
        
        while attempts <= self.max_retries:
            attempts += 1
            
            # Generate response
            response_text = generate_fn(current_prompt)
            
            # Verify response
            verification = self.verifier.verify(
                response_text,
                code_blocks=code_blocks
            )
            
            # Score confidence
            confidence = self.scorer.score(
                response_text,
                classification,
                mode,
                verification,
                retry_count=attempts - 1
            )
            
            # Log this attempt
            verification_log.append({
                "attempt": attempts,
                "confidence": confidence.score,
                "action": confidence.action,
                "issues": verification.issues,
                "warnings": verification.warnings,
            })
            
            # Decision based on confidence
            if confidence.action == "accept":
                return RetryResult(
                    final_response=response_text,
                    final_confidence=confidence.score,
                    attempts=attempts,
                    accepted=True,
                    refused=False,
                    verification_log=verification_log
                )
            
            elif confidence.action == "refuse":
                # Below retry threshold - refuse
                return RetryResult(
                    final_response=response_text,
                    final_confidence=confidence.score,
                    attempts=attempts,
                    accepted=False,
                    refused=True,
                    verification_log=verification_log
                )
            
            else:  # retry
                if attempts > self.max_retries:
                    # Max retries exceeded
                    break
                
                # Build retry prompt with feedback
                current_prompt = self._build_retry_prompt(prompt, verification)
        
        # Exhausted retries - accept with caveat or refuse
        final_confidence = verification_log[-1]["confidence"] if verification_log else 0.0
        
        return RetryResult(
            final_response=response_text,
            final_confidence=final_confidence,
            attempts=attempts,
            accepted=final_confidence >= 0.6,  # Accept if at least 0.6
            refused=final_confidence < 0.6,
            verification_log=verification_log
        )


# Singleton instance
_retry_loop: Optional[RetryLoop] = None


def get_retry_loop(max_retries: int = 2) -> RetryLoop:
    """Get or create singleton RetryLoop instance"""
    global _retry_loop
    if _retry_loop is None:
        _retry_loop = RetryLoop(max_retries=max_retries)
    return _retry_loop

"""
Confidence Scoring Module
Scores response confidence 0-1, penalizing contradictions and failures.

DESIGN DECISIONS:
- Base confidence from task classification
- Penalties for verification failures
- Bonuses for well-supported answers
- Threshold-based decision making (accept/retry/refuse)
"""

from typing import Optional
from pydantic import BaseModel

from .task_classifier import TaskClassification, TaskType
from .mode_selector import ModeSelection, ReasoningMode
from .verification import VerificationResult


class ConfidenceScore(BaseModel):
    """Confidence score with breakdown"""
    score: float  # 0.0 to 1.0
    base_score: float
    adjustments: dict[str, float]
    reasoning: str
    action: str  # "accept", "retry", "refuse"


class ConfidenceScorer:
    """
    Calculates confidence scores for LLM responses.
    
    Scoring factors:
    - Task type (trivial = high base, high-risk = lower base)
    - Verification results (issues reduce confidence)
    - Response quality markers (uncertainty words, sources cited)
    - Mode appropriateness (deep thinking for complex tasks)
    """
    
    # Base confidence by task type
    BASE_CONFIDENCE = {
        TaskType.TRIVIAL: 0.95,
        TaskType.CONVERSATION: 0.85,
        TaskType.CODING: 0.75,
        TaskType.RESEARCH: 0.7,
        TaskType.HIGH_RISK: 0.6,
    }
    
    # Thresholds for decision making
    ACCEPT_THRESHOLD = 0.9
    RETRY_THRESHOLD = 0.6
    
    # Uncertainty markers that reduce confidence
    UNCERTAINTY_MARKERS = [
        "i think", "probably", "might", "maybe", "perhaps",
        "not sure", "uncertain", "i believe", "possibly",
        "could be", "seems like", "appears to",
    ]
    
    # Confidence markers that increase confidence
    CONFIDENCE_MARKERS = [
        "according to", "source:", "documented", "specification",
        "standard", "defined as", "officially", "verified",
    ]
    
    def __init__(
        self,
        accept_threshold: float = 0.9,
        retry_threshold: float = 0.6
    ):
        self.accept_threshold = accept_threshold
        self.retry_threshold = retry_threshold
    
    def _count_markers(self, text: str, markers: list[str]) -> int:
        """Count occurrences of markers in text"""
        text_lower = text.lower()
        return sum(1 for m in markers if m in text_lower)
    
    def _analyze_response_quality(self, response_text: str) -> dict[str, float]:
        """Analyze response for quality markers"""
        adjustments = {}
        
        # Count uncertainty markers
        uncertainty_count = self._count_markers(response_text, self.UNCERTAINTY_MARKERS)
        if uncertainty_count > 0:
            adjustments["uncertainty_markers"] = -0.05 * min(uncertainty_count, 3)
        
        # Count confidence markers (sources, etc.)
        confidence_count = self._count_markers(response_text, self.CONFIDENCE_MARKERS)
        if confidence_count > 0:
            adjustments["confidence_markers"] = 0.03 * min(confidence_count, 3)
        
        # Check response length
        word_count = len(response_text.split())
        if word_count < 10:
            adjustments["too_short"] = -0.1
        elif word_count > 500:
            # Long responses for complex questions are good
            adjustments["comprehensive"] = 0.02
        
        # Check for structured response (lists, code blocks)
        if any(marker in response_text for marker in ["1.", "2.", "- ", "```"]):
            adjustments["structured"] = 0.02
        
        return adjustments
    
    def _determine_action(self, score: float) -> str:
        """Determine action based on confidence score"""
        if score >= self.accept_threshold:
            return "accept"
        elif score >= self.retry_threshold:
            return "retry"
        else:
            return "refuse"
    
    def _generate_reasoning(
        self,
        adjustments: dict[str, float],
        action: str
    ) -> str:
        """Generate human-readable reasoning for the score"""
        parts = []
        
        for key, value in adjustments.items():
            if value < 0:
                parts.append(f"Penalty from {key.replace('_', ' ')}: {value:.2f}")
            elif value > 0:
                parts.append(f"Bonus from {key.replace('_', ' ')}: +{value:.2f}")
        
        if not parts:
            parts.append("No significant adjustments")
        
        parts.append(f"Action: {action}")
        
        return "; ".join(parts)
    
    def score(
        self,
        response_text: str,
        classification: TaskClassification,
        mode: ModeSelection,
        verification: VerificationResult,
        retry_count: int = 0
    ) -> ConfidenceScore:
        """
        Calculate confidence score for a response.
        
        Args:
            response_text: The LLM's response
            classification: Task classification result
            mode: Selected reasoning mode
            verification: Verification results
            retry_count: Number of retries already attempted
            
        Returns:
            ConfidenceScore with breakdown and action
        """
        # Get base confidence from task type
        base_score = self.BASE_CONFIDENCE.get(classification.task_type, 0.7)
        
        adjustments = {}
        
        # Apply verification adjustments
        if verification.confidence_adjustment != 0:
            adjustments["verification"] = verification.confidence_adjustment
        
        # Add verification issue penalties
        if verification.issues:
            adjustments["issues"] = -0.1 * len(verification.issues)
        
        if verification.warnings:
            adjustments["warnings"] = -0.03 * len(verification.warnings)
        
        # Analyze response quality
        quality_adjustments = self._analyze_response_quality(response_text)
        adjustments.update(quality_adjustments)
        
        # Mode appropriateness
        if mode.reasoning_mode == ReasoningMode.DEEP and classification.task_type == TaskType.TRIVIAL:
            adjustments["mode_mismatch"] = -0.05  # Overthinking simple task
        elif mode.reasoning_mode == ReasoningMode.FAST and classification.task_type == TaskType.HIGH_RISK:
            adjustments["mode_mismatch"] = -0.15  # Underthinking complex task
        
        # Retry penalty (slight reduction in confidence for retried answers)
        if retry_count > 0:
            adjustments["retry_penalty"] = -0.05 * retry_count
        
        # Calculate final score
        total_adjustment = sum(adjustments.values())
        final_score = max(0.0, min(1.0, base_score + total_adjustment))
        
        # Determine action
        action = self._determine_action(final_score)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(adjustments, action)
        
        return ConfidenceScore(
            score=final_score,
            base_score=base_score,
            adjustments=adjustments,
            reasoning=reasoning,
            action=action
        )


# Singleton instance
_scorer: Optional[ConfidenceScorer] = None


def get_confidence_scorer(
    accept_threshold: float = 0.9,
    retry_threshold: float = 0.6
) -> ConfidenceScorer:
    """Get or create singleton ConfidenceScorer instance"""
    global _scorer
    if _scorer is None:
        _scorer = ConfidenceScorer(accept_threshold, retry_threshold)
    return _scorer

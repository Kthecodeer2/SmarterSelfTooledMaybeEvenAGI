"""
Refusal Path Module
Handles cases where the agent cannot provide a confident answer.

DESIGN DECISIONS:
- Always explain why we're refusing
- State what information is missing
- Never hallucinate to fill gaps
- Offer alternative approaches when possible
"""

from typing import Optional
from pydantic import BaseModel

from .verification import VerificationResult
from .confidence import ConfidenceScore
from .task_classifier import TaskClassification, TaskType, RiskDomain


class RefusalResponse(BaseModel):
    """Structured refusal response"""
    refused: bool
    reason: str
    missing_information: list[str]
    suggestions: list[str]
    partial_answer: Optional[str] = None
    formatted_response: str


class RefusalHandler:
    """
    Generates appropriate refusal responses when the agent
    cannot provide a confident answer.
    
    Refusal reasons:
    - Insufficient confidence
    - Missing critical information
    - Safety concerns
    - Out of scope / capability
    """
    
    REFUSAL_TEMPLATE = """I cannot provide a confident answer to this question.

**Reason**: {reason}

{missing_info_section}

{suggestions_section}

{partial_section}"""
    
    MISSING_INFO_TEMPLATE = """**Missing Information**:
{items}"""
    
    SUGGESTIONS_TEMPLATE = """**Suggestions**:
{items}"""
    
    PARTIAL_TEMPLATE = """**What I can say**:
{partial}"""
    
    def __init__(self):
        pass
    
    def _identify_missing_information(
        self,
        classification: TaskClassification,
        verification: VerificationResult
    ) -> list[str]:
        """Identify what information is missing to answer confidently"""
        missing = []
        
        # High-risk domains often need sources
        if classification.is_high_risk:
            if RiskDomain.FINANCE in classification.risk_domains:
                missing.append("Verified financial data or official sources")
            if RiskDomain.SECURITY in classification.risk_domains:
                missing.append("Security context and threat model")
            if RiskDomain.MEDICAL in classification.risk_domains:
                missing.append("Medical professional consultation required")
            if RiskDomain.LEGAL in classification.risk_domains:
                missing.append("Legal professional consultation required")
        
        # Code analysis issues
        if verification.code_issues:
            critical = [i for i in verification.code_issues if i["severity"] == "critical"]
            if critical:
                missing.append("Code security review before implementation")
        
        # Numerical claims without verification
        unverified = [c for c in verification.numerical_claims if not c.get("verified", True)]
        if unverified:
            missing.append("Verified data sources for numerical claims")
        
        return missing
    
    def _generate_suggestions(
        self,
        classification: TaskClassification,
        reason: str
    ) -> list[str]:
        """Generate helpful suggestions for the user"""
        suggestions = []
        
        if classification.task_type == TaskType.CODING:
            suggestions.append("Try breaking down the problem into smaller parts")
            suggestions.append("Provide more context about your environment and constraints")
        
        elif classification.task_type == TaskType.RESEARCH:
            suggestions.append("Consult authoritative sources directly")
            suggestions.append("Consider seeking expert opinion for this domain")
        
        elif classification.is_high_risk:
            if RiskDomain.FINANCE in classification.risk_domains:
                suggestions.append("Consult a qualified financial advisor")
            if RiskDomain.MEDICAL in classification.risk_domains:
                suggestions.append("Consult a healthcare professional")
            if RiskDomain.LEGAL in classification.risk_domains:
                suggestions.append("Consult a legal professional")
            if RiskDomain.SECURITY in classification.risk_domains:
                suggestions.append("Perform a formal security audit")
        
        if not suggestions:
            suggestions.append("Try rephrasing the question with more specific details")
        
        return suggestions
    
    def _extract_partial_answer(
        self,
        response_text: str,
        confidence: ConfidenceScore
    ) -> Optional[str]:
        """
        Extract any partial answer that might still be useful.
        Only include if we have something meaningful to say.
        """
        # If confidence is very low, don't provide partial
        if confidence.score < 0.4:
            return None
        
        # If response is very short, probably not useful
        if len(response_text.split()) < 20:
            return None
        
        # Truncate to reasonable length if needed
        words = response_text.split()
        if len(words) > 100:
            partial = " ".join(words[:100]) + "..."
        else:
            partial = response_text
        
        return partial
    
    def _format_list_items(self, items: list[str]) -> str:
        """Format list items as bullet points"""
        return "\n".join(f"- {item}" for item in items)
    
    def generate_refusal(
        self,
        reason: str,
        response_text: str,
        classification: TaskClassification,
        verification: VerificationResult,
        confidence: ConfidenceScore
    ) -> RefusalResponse:
        """
        Generate a structured refusal response.
        
        Args:
            reason: Why we're refusing
            response_text: The original response (may contain partial info)
            classification: Task classification
            verification: Verification results
            confidence: Confidence score
            
        Returns:
            RefusalResponse with all details
        """
        # Identify missing information
        missing_info = self._identify_missing_information(classification, verification)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(classification, reason)
        
        # Try to extract partial answer
        partial = self._extract_partial_answer(response_text, confidence)
        
        # Build formatted response
        missing_section = ""
        if missing_info:
            missing_section = self.MISSING_INFO_TEMPLATE.format(
                items=self._format_list_items(missing_info)
            )
        
        suggestions_section = ""
        if suggestions:
            suggestions_section = self.SUGGESTIONS_TEMPLATE.format(
                items=self._format_list_items(suggestions)
            )
        
        partial_section = ""
        if partial:
            partial_section = self.PARTIAL_TEMPLATE.format(partial=partial)
        
        formatted = self.REFUSAL_TEMPLATE.format(
            reason=reason,
            missing_info_section=missing_section,
            suggestions_section=suggestions_section,
            partial_section=partial_section
        ).strip()
        
        # Clean up extra newlines
        while "\n\n\n" in formatted:
            formatted = formatted.replace("\n\n\n", "\n\n")
        
        return RefusalResponse(
            refused=True,
            reason=reason,
            missing_information=missing_info,
            suggestions=suggestions,
            partial_answer=partial,
            formatted_response=formatted
        )
    
    def should_refuse(
        self,
        classification: TaskClassification,
        confidence: ConfidenceScore
    ) -> tuple[bool, str]:
        """
        Determine if we should refuse to answer.
        
        Returns:
            (should_refuse, reason)
        """
        # Confidence too low
        if confidence.action == "refuse":
            return True, "Confidence too low to provide a reliable answer"
        
        # High-risk domains with insufficient confidence
        if classification.is_high_risk and confidence.score < 0.85:
            domain_names = [d.value for d in classification.risk_domains if d != RiskDomain.NONE]
            return True, f"High-risk domain ({', '.join(domain_names)}) requires higher confidence"
        
        return False, ""


# Singleton instance
_handler: Optional[RefusalHandler] = None


def get_refusal_handler() -> RefusalHandler:
    """Get or create singleton RefusalHandler instance"""
    global _handler
    if _handler is None:
        _handler = RefusalHandler()
    return _handler

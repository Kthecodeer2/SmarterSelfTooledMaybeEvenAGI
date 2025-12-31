"""
Research Mode Module
Handles research tasks with proper attribution and verification.

DESIGN DECISIONS:
- Clear assumptions and known limitations
- Numerical claims require sources or "estimate" label
- Reproducible outputs where possible
- Avoid unverified speculation
- Cross-check conflicting information
"""

import re
from typing import Optional
from pydantic import BaseModel
import httpx


class Claim(BaseModel):
    """A claim made in research output"""
    text: str
    type: str  # "fact", "estimate", "consensus", "speculation"
    source: Optional[str] = None
    confidence: float  # 0-1


class ResearchOutput(BaseModel):
    """Structured research output"""
    summary: str
    claims: list[Claim]
    assumptions: list[str]
    limitations: list[str]
    sources: list[str]
    reproducibility_notes: Optional[str] = None
    confidence: float


class ResearchMode:
    """
    Research mode with proper attribution and verification.
    
    Features:
    - Extract and categorize claims
    - Mark sources or label as estimates
    - List assumptions and limitations
    - Avoid speculation
    - Optional web verification
    """
    
    # Patterns that indicate speculation
    SPECULATION_MARKERS = [
        "probably", "might", "could be", "perhaps",
        "possibly", "may", "i think", "i believe",
        "it seems", "appears to", "likely"
    ]
    
    # Patterns that indicate estimates
    ESTIMATE_MARKERS = [
        "approximately", "around", "roughly", "about",
        "estimated", "circa", "nearly", "close to"
    ]
    
    # Patterns that indicate consensus
    CONSENSUS_MARKERS = [
        "generally accepted", "widely known", "common knowledge",
        "well established", "consensus", "mainstream view"
    ]
    
    # Patterns that indicate facts
    FACT_MARKERS = [
        "according to", "source:", "documented", "officially",
        "measured", "recorded", "published", "stated"
    ]
    
    def __init__(self, enable_web_verification: bool = False):
        """
        Args:
            enable_web_verification: Whether to verify claims via web search
                                     Disabled by default for privacy
        """
        self.enable_web_verification = enable_web_verification
        self._http_client = None
    
    def _get_http_client(self) -> httpx.Client:
        """Lazy init HTTP client"""
        if self._http_client is None:
            self._http_client = httpx.Client(timeout=10.0)
        return self._http_client
    
    def _detect_claim_type(self, text: str) -> str:
        """Detect the type of claim from its text"""
        lower_text = text.lower()
        
        # Check for fact indicators
        if any(marker in lower_text for marker in self.FACT_MARKERS):
            return "fact"
        
        # Check for consensus indicators
        if any(marker in lower_text for marker in self.CONSENSUS_MARKERS):
            return "consensus"
        
        # Check for estimate indicators
        if any(marker in lower_text for marker in self.ESTIMATE_MARKERS):
            return "estimate"
        
        # Check for speculation indicators
        if any(marker in lower_text for marker in self.SPECULATION_MARKERS):
            return "speculation"
        
        # Default to estimate if contains numbers without source
        if re.search(r"\d+", text):
            return "estimate"
        
        return "fact"  # Default assumption
    
    def extract_claims(self, text: str) -> list[Claim]:
        """Extract and categorize claims from text"""
        claims = []
        
        # Split into sentences
        sentences = re.split(r"[.!?]+", text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 10:
                continue
            
            # Skip meta-statements
            if sentence.lower().startswith(("i ", "you ", "we ")):
                continue
            
            claim_type = self._detect_claim_type(sentence)
            
            # Try to extract source
            source = None
            source_match = re.search(r"according to (.+?)(?:,|$)", sentence, re.IGNORECASE)
            if source_match:
                source = source_match.group(1).strip()
                claim_type = "fact"
            
            # Confidence based on claim type
            confidence_map = {
                "fact": 0.9,
                "consensus": 0.8,
                "estimate": 0.6,
                "speculation": 0.4
            }
            
            claims.append(Claim(
                text=sentence,
                type=claim_type,
                source=source,
                confidence=confidence_map.get(claim_type, 0.5)
            ))
        
        return claims
    
    def extract_assumptions(self, text: str) -> list[str]:
        """Extract assumptions from text"""
        assumptions = []
        
        # Look for explicit assumptions
        assumption_patterns = [
            r"assum(?:e|ing|ption)[^.]*\.",
            r"given that[^.]*\.",
            r"assuming[^.]*\.",
            r"if we assume[^.]*\."
        ]
        
        for pattern in assumption_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            assumptions.extend([m.strip() for m in matches])
        
        # Add implicit assumptions for technical content
        if "code" in text.lower() or "function" in text.lower():
            assumptions.append("Code examples assume a compatible runtime environment")
        
        if "performance" in text.lower():
            assumptions.append("Performance characteristics may vary by environment")
        
        return list(set(assumptions))  # Deduplicate
    
    def extract_limitations(self, text: str, claims: list[Claim]) -> list[str]:
        """Identify limitations of the research output"""
        limitations = []
        
        # Check for speculative claims
        speculation_count = sum(1 for c in claims if c.type == "speculation")
        if speculation_count > 0:
            limitations.append(f"{speculation_count} claims are speculative and need verification")
        
        # Check for unsourced numerical claims
        unsourced_numbers = sum(
            1 for c in claims 
            if c.type == "estimate" and c.source is None
        )
        if unsourced_numbers > 0:
            limitations.append(f"{unsourced_numbers} numerical claims lack cited sources")
        
        # Check for recency
        if not any(marker in text.lower() for marker in ["2024", "2025", "recent", "latest"]):
            limitations.append("Information may not reflect the most recent developments")
        
        # No web verification
        if not self.enable_web_verification:
            limitations.append("Claims have not been verified against external sources")
        
        return limitations
    
    def extract_sources(self, text: str) -> list[str]:
        """Extract cited sources from text"""
        sources = []
        
        # Look for explicit citations
        citation_patterns = [
            r"according to ([^,\n]+)",
            r"source:\s*([^\n]+)",
            r"\(([^)]+\d{4}[^)]*)\)",  # Academic-style citations
            r"https?://[^\s]+",  # URLs
        ]
        
        for pattern in citation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            sources.extend([m.strip() for m in matches if m.strip()])
        
        return list(set(sources))  # Deduplicate
    
    def analyze(self, text: str) -> ResearchOutput:
        """
        Analyze text for research quality.
        
        Args:
            text: The research text to analyze
            
        Returns:
            ResearchOutput with claims, assumptions, limitations
        """
        claims = self.extract_claims(text)
        assumptions = self.extract_assumptions(text)
        sources = self.extract_sources(text)
        limitations = self.extract_limitations(text, claims)
        
        # Calculate overall confidence
        if claims:
            avg_confidence = sum(c.confidence for c in claims) / len(claims)
        else:
            avg_confidence = 0.5
        
        # Adjust for limitations
        confidence_penalty = len(limitations) * 0.05
        final_confidence = max(0.1, avg_confidence - confidence_penalty)
        
        # Generate reproducibility notes
        reproducibility = None
        if "experiment" in text.lower() or "test" in text.lower():
            reproducibility = "To reproduce: follow the methodology described above"
        
        return ResearchOutput(
            summary=text[:200] + "..." if len(text) > 200 else text,
            claims=claims,
            assumptions=assumptions,
            limitations=limitations,
            sources=sources,
            reproducibility_notes=reproducibility,
            confidence=final_confidence
        )
    
    def format_research_response(
        self,
        content: str,
        analysis: ResearchOutput
    ) -> str:
        """
        Format research response with proper annotations.
        
        Adds:
        - Assumption markers
        - Source citations
        - Limitation notes
        """
        output_parts = [content, "\n\n---\n"]
        
        # Add assumptions
        if analysis.assumptions:
            output_parts.append("\n**Assumptions**:")
            for assumption in analysis.assumptions:
                output_parts.append(f"\n- {assumption}")
        
        # Add limitations
        if analysis.limitations:
            output_parts.append("\n\n**Limitations**:")
            for limitation in analysis.limitations:
                output_parts.append(f"\n- {limitation}")
        
        # Add sources
        if analysis.sources:
            output_parts.append("\n\n**Sources**:")
            for source in analysis.sources:
                output_parts.append(f"\n- {source}")
        
        # Add confidence note
        output_parts.append(f"\n\n*Confidence: {analysis.confidence:.0%}*")
        
        return "".join(output_parts)
    
    def verify_claim_web(self, claim: str) -> Optional[dict]:
        """
        Verify a claim using web search.
        
        DISABLED BY DEFAULT for privacy.
        Override enable_web_verification in constructor.
        
        Returns:
            dict with verification result or None if disabled
        """
        if not self.enable_web_verification:
            return None
        
        # This would integrate with a search API
        # For now, return placeholder
        return {
            "verified": False,
            "reason": "Web verification not implemented",
            "sources": []
        }


# Singleton instance
_research: Optional[ResearchMode] = None


def get_research_mode(enable_web_verification: bool = False) -> ResearchMode:
    """Get or create singleton ResearchMode instance"""
    global _research
    if _research is None:
        _research = ResearchMode(enable_web_verification)
    return _research

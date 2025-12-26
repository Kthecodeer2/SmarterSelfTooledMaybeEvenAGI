"""
Response Filter Module
Applies personality rules to responses.

DESIGN DECISIONS:
- Minimal politeness
- Zero motivational fluff
- Clear opinions when justified
- Politely call out bad ideas
- No "as an AI" phrasing
- No filler sentences
"""

import re
from typing import Optional
from pydantic import BaseModel


class FilterResult(BaseModel):
    """Result of response filtering"""
    original: str
    filtered: str
    changes_made: list[str]
    fluff_removed: int


class ResponseFilter:
    """
    Filters LLM responses to match desired personality.
    
    Removes:
    - Excessive politeness
    - Motivational fluff
    - "As an AI" phrases
    - Filler sentences
    
    Preserves:
    - Technical accuracy
    - Clear opinions
    - Direct communication
    """
    
    # Phrases to remove or replace
    FLUFF_PATTERNS = [
        # AI identity phrases
        (r"As an AI( language model)?[,.]?\s*", ""),
        (r"I'm an AI( assistant)?[,.]?\s*", ""),
        (r"Being an AI[,.]?\s*", ""),
        (r"I don't have (personal )?(feelings|opinions|preferences)[,.]?\s*", ""),
        
        # Excessive politeness
        (r"I'd be happy to help( you)?[.!]?\s*", ""),
        (r"I'd love to help( you)?[.!]?\s*", ""),
        (r"Great question[.!]?\s*", ""),
        (r"That's a great question[.!]?\s*", ""),
        (r"Thank you for asking[.!]?\s*", ""),
        (r"Thanks for sharing[.!]?\s*", ""),
        (r"I appreciate you asking[.!]?\s*", ""),
        
        # Filler phrases
        (r"I think it's important to (note|mention) that\s*", ""),
        (r"It's worth (noting|mentioning) that\s*", "Note: "),
        (r"Let me explain[.:]?\s*", ""),
        (r"Let me break this down[.:]?\s*", ""),
        (r"To answer your question[,:]?\s*", ""),
        (r"In my opinion[,]?\s*", ""),
        
        # Motivational fluff
        (r"You're doing great[.!]?\s*", ""),
        (r"Keep up the good work[.!]?\s*", ""),
        (r"Don't give up[.!]?\s*", ""),
        (r"You've got this[.!]?\s*", ""),
        (r"I believe in you[.!]?\s*", ""),
        
        # Hedging (sometimes appropriate, but often excessive)
        (r"I could be wrong, but\s*", ""),
        (r"I'm not entirely sure, but\s*", "Uncertain: "),
        (r"If I'm not mistaken[,]?\s*", ""),
        
        # Unnecessary preambles
        (r"^Sure[,!]?\s*", ""),
        (r"^Of course[,!]?\s*", ""),
        (r"^Absolutely[,!]?\s*", ""),
        (r"^Certainly[,!]?\s*", ""),
        (r"^Definitely[,!]?\s*", ""),
    ]
    
    # Sentences to remove entirely
    REMOVE_SENTENCES = [
        r"Is there anything else I can help you with\?",
        r"Feel free to ask if you have any (more )?questions[.!]",
        r"Let me know if you need anything else[.!]",
        r"I hope this helps[.!]",
        r"Hope that helps[.!]",
        r"Please let me know if you have any questions[.!]",
        r"Don't hesitate to ask[.!]",
    ]
    
    def __init__(self):
        # Compile patterns
        self.fluff_compiled = [
            (re.compile(p, re.IGNORECASE), r)
            for p, r in self.FLUFF_PATTERNS
        ]
        self.remove_compiled = [
            re.compile(p, re.IGNORECASE)
            for p in self.REMOVE_SENTENCES
        ]
    
    def filter(self, text: str) -> FilterResult:
        """
        Filter response to match personality guidelines.
        
        Args:
            text: Original response text
            
        Returns:
            FilterResult with filtered text and change log
        """
        changes = []
        fluff_count = 0
        result = text
        
        # Apply fluff pattern replacements
        for pattern, replacement in self.fluff_compiled:
            if pattern.search(result):
                fluff_count += 1
                changes.append(f"Removed: {pattern.pattern[:30]}...")
                result = pattern.sub(replacement, result)
        
        # Remove filler sentences
        for pattern in self.remove_compiled:
            if pattern.search(result):
                fluff_count += 1
                changes.append(f"Removed sentence: {pattern.pattern[:30]}...")
                result = pattern.sub("", result)
        
        # Clean up whitespace
        result = re.sub(r"\n{3,}", "\n\n", result)
        result = re.sub(r"  +", " ", result)
        result = result.strip()
        
        return FilterResult(
            original=text,
            filtered=result,
            changes_made=changes,
            fluff_removed=fluff_count
        )
    
    def add_opinion_markers(self, text: str) -> str:
        """
        Add opinion markers where appropriate.
        Ensures opinions are clearly labeled.
        """
        # Replace wishy-washy phrasing with clear opinion markers
        replacements = [
            (r"I think\s+", "[Opinion] "),
            (r"I believe\s+", "[Opinion] "),
            (r"In my view\s+", "[Opinion] "),
        ]
        
        result = text
        for pattern, replacement in replacements:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        return result
    
    def ensure_direct_response(self, text: str, is_question: bool) -> str:
        """
        Ensure response is direct, especially for questions.
        """
        if not is_question:
            return text
        
        lines = text.strip().split("\n")
        if not lines:
            return text
        
        first_line = lines[0].strip()
        
        # If first line is just preamble, it will have been removed
        # Check if response actually answers the question
        if len(first_line) < 20 and not any(c in first_line for c in ".!?"):
            # Very short first line - might be incomplete
            pass
        
        return text
    
    def call_out_bad_idea(self, text: str, issue: str) -> str:
        """
        Add a polite but direct callout for bad ideas.
        Used when verification detects problems.
        """
        callout = f"\n\n⚠️ **Concern**: {issue}\n"
        return text + callout
    
    def apply_all_filters(
        self,
        text: str,
        is_question: bool = False,
        issues: Optional[list[str]] = None
    ) -> str:
        """
        Apply all personality filters to response.
        
        Args:
            text: Original response
            is_question: Whether input was a question
            issues: Any issues to call out
            
        Returns:
            Filtered response text
        """
        # Main filtering
        result = self.filter(text)
        filtered = result.filtered
        
        # Ensure direct response
        filtered = self.ensure_direct_response(filtered, is_question)
        
        # Add callouts for issues
        if issues:
            for issue in issues:
                filtered = self.call_out_bad_idea(filtered, issue)
        
        return filtered


# Singleton instance
_filter: Optional[ResponseFilter] = None


def get_response_filter() -> ResponseFilter:
    """Get or create singleton ResponseFilter instance"""
    global _filter
    if _filter is None:
        _filter = ResponseFilter()
    return _filter

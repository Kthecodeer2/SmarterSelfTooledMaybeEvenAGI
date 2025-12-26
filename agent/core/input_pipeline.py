"""
Input Pipeline Module
Sanitizes and normalizes user input, strips control tokens
"""

import re
from typing import Optional
from pydantic import BaseModel


class ProcessedInput(BaseModel):
    """Cleaned and processed user input"""
    original: str
    cleaned: str
    detected_language: str = "en"
    has_code: bool = False
    code_blocks: list[str] = []
    is_question: bool = False
    word_count: int = 0


class InputPipeline:
    """
    Sanitizes user input to prevent injection attacks
    and normalizes text for consistent processing.
    """
    
    # Control tokens that should never come from user input
    # These are internal system tokens only
    FORBIDDEN_TOKENS = [
        "<FAST>", "<DEEP>", "<VERIFY>", "<REFUSE_OK>",
        "<|system|>", "<|user|>", "<|assistant|>",
        "<|im_start|>", "<|im_end|>",
        "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>",
        "<s>", "</s>", "<pad>", "</pad>",
    ]
    
    # Regex for code blocks
    CODE_BLOCK_PATTERN = re.compile(r"```[\w]*\n?(.*?)```", re.DOTALL)
    
    # Regex for inline code
    INLINE_CODE_PATTERN = re.compile(r"`([^`]+)`")
    
    def __init__(self):
        # Build regex pattern for forbidden tokens
        escaped = [re.escape(t) for t in self.FORBIDDEN_TOKENS]
        self.forbidden_pattern = re.compile("|".join(escaped), re.IGNORECASE)
    
    def sanitize(self, text: str) -> str:
        """
        Remove potentially malicious control tokens from user input.
        These tokens are system-internal and should never appear in user messages.
        """
        if not text:
            return ""
        
        # Strip forbidden control tokens
        cleaned = self.forbidden_pattern.sub("", text)
        
        # Remove null bytes and other control characters (except newline, tab)
        cleaned = "".join(
            c for c in cleaned 
            if c.isprintable() or c in "\n\t\r"
        )
        
        # Normalize whitespace
        cleaned = re.sub(r"\r\n", "\n", cleaned)
        cleaned = re.sub(r"\t", "    ", cleaned)
        
        # Remove excessive newlines (more than 2 consecutive)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        
        # Strip leading/trailing whitespace
        cleaned = cleaned.strip()
        
        return cleaned
    
    def extract_code_blocks(self, text: str) -> list[str]:
        """Extract code blocks from markdown-style text"""
        blocks = self.CODE_BLOCK_PATTERN.findall(text)
        return [b.strip() for b in blocks if b.strip()]
    
    def detect_language(self, text: str) -> str:
        """
        Basic language detection.
        ASSUMPTION: Most users will use English. 
        For production, use a proper language detection library.
        """
        # Check for common non-ASCII character ranges
        if any("\u4e00" <= c <= "\u9fff" for c in text):
            return "zh"  # Chinese
        if any("\u3040" <= c <= "\u30ff" for c in text):
            return "ja"  # Japanese
        if any("\uac00" <= c <= "\ud7af" for c in text):
            return "ko"  # Korean
        if any("\u0600" <= c <= "\u06ff" for c in text):
            return "ar"  # Arabic
        if any("\u0400" <= c <= "\u04ff" for c in text):
            return "ru"  # Russian
        
        return "en"  # Default to English
    
    def is_question(self, text: str) -> bool:
        """Detect if input is a question"""
        question_markers = ["?", "what", "why", "how", "when", "where", "who", "which", "can you", "could you", "would you", "is it", "are there"]
        lower_text = text.lower()
        return any(marker in lower_text for marker in question_markers)
    
    def process(self, raw_input: str) -> ProcessedInput:
        """
        Full input processing pipeline.
        Returns structured ProcessedInput with all metadata.
        """
        if not raw_input:
            return ProcessedInput(
                original="",
                cleaned="",
                word_count=0
            )
        
        cleaned = self.sanitize(raw_input)
        code_blocks = self.extract_code_blocks(cleaned)
        
        return ProcessedInput(
            original=raw_input,
            cleaned=cleaned,
            detected_language=self.detect_language(cleaned),
            has_code=bool(code_blocks) or bool(self.INLINE_CODE_PATTERN.search(cleaned)),
            code_blocks=code_blocks,
            is_question=self.is_question(cleaned),
            word_count=len(cleaned.split())
        )


# Singleton instance for efficiency
_pipeline: Optional[InputPipeline] = None


def get_input_pipeline() -> InputPipeline:
    """Get or create singleton InputPipeline instance"""
    global _pipeline
    if _pipeline is None:
        _pipeline = InputPipeline()
    return _pipeline

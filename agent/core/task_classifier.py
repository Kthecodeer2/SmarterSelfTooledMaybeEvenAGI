"""
Task Classifier Module
Classifies user input into task types and detects risk domains
"""

import re
from enum import Enum
from typing import Optional
from pydantic import BaseModel

from .input_pipeline import ProcessedInput


class TaskType(str, Enum):
    """Types of tasks the agent can handle"""
    TRIVIAL = "trivial"          # Simple questions, greetings
    CODING = "coding"            # Code generation, debugging, review
    RESEARCH = "research"        # Analysis, explanations, comparisons
    HIGH_RISK = "high_risk"      # Security, finance, physics, math
    CONVERSATION = "conversation" # General chat


class RiskDomain(str, Enum):
    """High-risk domains requiring verification"""
    SECURITY = "security"
    FINANCE = "finance"
    PHYSICS = "physics"
    MATH = "math"
    MEDICAL = "medical"
    LEGAL = "legal"
    SYSTEMS = "systems"
    NONE = "none"


class TaskClassification(BaseModel):
    """Result of task classification"""
    task_type: TaskType
    risk_domains: list[RiskDomain]
    is_high_risk: bool
    requires_verification: bool
    requires_code_analysis: bool
    confidence: float  # How confident we are in this classification


class TaskClassifier:
    """
    Classifies user input into task types and detects risk domains.
    Uses keyword matching and pattern detection for efficiency.
    No LLM call needed - this is fast preprocessing.
    """
    
    # Coding keywords
    CODING_KEYWORDS = [
        "code", "function", "class", "implement", "debug", "fix", "error",
        "compile", "runtime", "exception", "bug", "syntax", "api", "endpoint",
        "database", "sql", "query", "algorithm", "data structure", "refactor",
        "optimize", "performance", "test", "unit test", "integration",
        "python", "javascript", "typescript", "rust", "go", "java", "c++",
        "react", "vue", "angular", "django", "flask", "fastapi", "node",
        "docker", "kubernetes", "aws", "gcp", "azure", "terraform",
        "git", "github", "gitlab", "ci/cd", "pipeline", "deploy",
    ]
    
    # Research keywords
    RESEARCH_KEYWORDS = [
        "explain", "analyze", "compare", "research", "investigate",
        "what is", "how does", "why does", "difference between",
        "pros and cons", "trade-off", "best practice", "recommend",
        "history of", "overview", "summary", "literature", "study",
        "theory", "concept", "principle", "methodology", "approach",
    ]
    
    # Risk domain patterns
    RISK_PATTERNS = {
        RiskDomain.SECURITY: [
            r"\bsecurity\b", r"\bvulnerab", r"\bexploit", r"\battack",
            r"\bencrypt", r"\bdecrypt", r"\bpassword", r"\bauth",
            r"\bpermission", r"\baccess control", r"\binjection",
            r"\bxss\b", r"\bcsrf\b", r"\bsql injection", r"\brce\b",
            r"\bprivilege", r"\bmalware", r"\bphishing", r"\bhacking",
        ],
        RiskDomain.FINANCE: [
            r"\bfinance\b", r"\btrading\b", r"\binvestment\b", r"\bstock\b",
            r"\bcrypto", r"\bbitcoin\b", r"\bethereum\b", r"\bblockchain\b",
            r"\btax\b", r"\baccounting\b", r"\bbudget\b", r"\bloan\b",
            r"\binterest rate\b", r"\bmortgage\b", r"\bportfolio\b",
        ],
        RiskDomain.PHYSICS: [
            r"\bphysics\b", r"\bquantum\b", r"\brelativity\b", r"\bthermodynamic",
            r"\belectromagnet", r"\bnuclear\b", r"\bparticle\b", r"\bwave\b",
            r"\benergy\b", r"\bforce\b", r"\bmomentum\b", r"\bgravity\b",
        ],
        RiskDomain.MATH: [
            r"\bmath", r"\bcalcul", r"\balgebra", r"\bgeometry\b",
            r"\bstatistic", r"\bprobability\b", r"\bproof\b", r"\btheorem\b",
            r"\bequation\b", r"\bderivative\b", r"\bintegral\b", r"\bmatrix\b",
            r"\bvector\b", r"\btensor\b", r"\beigenvalue\b",
        ],
        RiskDomain.MEDICAL: [
            r"\bmedical\b", r"\bhealth\b", r"\bdisease\b", r"\bsymptom",
            r"\bdiagnos", r"\btreatment\b", r"\bmedication\b", r"\bdrug\b",
            r"\bsurgery\b", r"\bpatient\b", r"\bdoctor\b", r"\bhospital\b",
        ],
        RiskDomain.LEGAL: [
            r"\blegal\b", r"\blaw\b", r"\bcontract\b", r"\blawsuit\b",
            r"\bcourt\b", r"\bjudge\b", r"\battorney\b", r"\blawyer\b",
            r"\bliability\b", r"\bcopyright\b", r"\bpatent\b", r"\btrademark\b",
        ],
        RiskDomain.SYSTEMS: [
            r"\bsystem design\b", r"\barchitecture\b", r"\bscalability\b",
            r"\bdistributed\b", r"\bmicroservice", r"\bload balanc",
            r"\bcaching\b", r"\breplication\b", r"\bconsensus\b",
            r"\bfault tolerance\b", r"\bhigh availability\b",
        ],
    }
    
    # Trivial patterns - greetings, simple acknowledgments
    TRIVIAL_PATTERNS = [
        r"^hi\b", r"^hello\b", r"^hey\b", r"^thanks\b", r"^thank you\b",
        r"^ok\b", r"^okay\b", r"^yes\b", r"^no\b", r"^sure\b",
        r"^bye\b", r"^goodbye\b", r"^see you\b",
    ]
    
    def __init__(self):
        # Compile patterns for efficiency
        self.risk_patterns_compiled = {
            domain: [re.compile(p, re.IGNORECASE) for p in patterns]
            for domain, patterns in self.RISK_PATTERNS.items()
        }
        self.trivial_patterns_compiled = [
            re.compile(p, re.IGNORECASE) for p in self.TRIVIAL_PATTERNS
        ]
        self.coding_pattern = re.compile(
            "|".join(r"\b" + re.escape(kw) + r"\b" for kw in self.CODING_KEYWORDS),
            re.IGNORECASE
        )
        self.research_pattern = re.compile(
            "|".join(r"\b" + re.escape(kw) + r"\b" for kw in self.RESEARCH_KEYWORDS),
            re.IGNORECASE
        )
    
    def detect_risk_domains(self, text: str) -> list[RiskDomain]:
        """Detect which risk domains the input touches"""
        domains = []
        for domain, patterns in self.risk_patterns_compiled.items():
            for pattern in patterns:
                if pattern.search(text):
                    domains.append(domain)
                    break  # One match per domain is enough
        return domains if domains else [RiskDomain.NONE]
    
    def is_trivial(self, processed: ProcessedInput) -> bool:
        """Check if input is trivial (greeting, simple acknowledgment)"""
        if processed.word_count > 10:
            return False
        
        text = processed.cleaned.lower()
        return any(p.search(text) for p in self.trivial_patterns_compiled)
    
    def is_coding_task(self, processed: ProcessedInput) -> bool:
        """Check if input is a coding task"""
        if processed.has_code:
            return True
        return bool(self.coding_pattern.search(processed.cleaned))
    
    def is_research_task(self, text: str) -> bool:
        """Check if input is a research/analysis task"""
        return bool(self.research_pattern.search(text))
    
    def classify(self, processed: ProcessedInput) -> TaskClassification:
        """
        Classify the processed input into task type and risk domains.
        Returns confidence score based on pattern match strength.
        """
        text = processed.cleaned
        risk_domains = self.detect_risk_domains(text)
        is_high_risk = RiskDomain.NONE not in risk_domains
        
        # Determine task type
        if self.is_trivial(processed):
            task_type = TaskType.TRIVIAL
            confidence = 0.95
        elif is_high_risk:
            task_type = TaskType.HIGH_RISK
            confidence = 0.85
        elif self.is_coding_task(processed):
            task_type = TaskType.CODING
            confidence = 0.9
        elif self.is_research_task(text):
            task_type = TaskType.RESEARCH
            confidence = 0.85
        else:
            task_type = TaskType.CONVERSATION
            confidence = 0.7
        
        return TaskClassification(
            task_type=task_type,
            risk_domains=risk_domains,
            is_high_risk=is_high_risk,
            requires_verification=is_high_risk or task_type in [TaskType.CODING, TaskType.RESEARCH],
            requires_code_analysis=task_type == TaskType.CODING or processed.has_code,
            confidence=confidence
        )


# Singleton instance
_classifier: Optional[TaskClassifier] = None


def get_task_classifier() -> TaskClassifier:
    """Get or create singleton TaskClassifier instance"""
    global _classifier
    if _classifier is None:
        _classifier = TaskClassifier()
    return _classifier

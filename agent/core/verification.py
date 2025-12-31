"""
Verification Layer Module
Performs logical consistency checks, numerical validation, code analysis,
and adversarial self-critique.

DESIGN DECISIONS:
- All verification happens locally, no external API calls
- Static analysis uses AST parsing for security
- Numerical checks use regex + basic math validation
- Adversarial critique generates counter-arguments
"""

import re
import ast
from typing import Optional
from pydantic import BaseModel


class VerificationResult(BaseModel):
    """Result of verification checks"""
    passed: bool
    issues: list[str]
    warnings: list[str]
    numerical_claims: list[dict]  # {claim, value, verified}
    code_issues: list[dict]  # {type, description, severity}
    self_critique: str  # "What would make this wrong?"
    confidence_adjustment: float  # -1.0 to +0.0 adjustment


class VerificationLayer:
    """
    Verifies LLM responses for consistency, accuracy, and safety.
    
    Checks performed:
    1. Logical consistency - contradictions in response
    2. Numerical sanity - math claims are reasonable
    3. Code static analysis - security issues, bugs
    4. Adversarial critique - what could be wrong?
    """
    
    # Contradiction patterns
    CONTRADICTION_PATTERNS = [
        (r"\bis\b.*\bbut\s+is\s+not\b", "Potential contradiction: 'is' followed by 'is not'"),
        (r"\balways\b.*\bnever\b", "Contradiction: 'always' and 'never' in same context"),
        (r"\bimpossible\b.*\bpossible\b", "Contradiction: 'impossible' and 'possible'"),
        (r"\bcorrect\b.*\bincorrect\b", "Contradiction: 'correct' and 'incorrect'"),
    ]
    
    # Numerical claim patterns
    NUMBER_PATTERNS = [
        r"(\d+(?:\.\d+)?)\s*%",  # Percentages
        r"\$(\d+(?:,\d{3})*(?:\.\d{2})?)",  # Dollar amounts
        r"(\d+(?:\.\d+)?)\s*(?:GB|MB|KB|TB)",  # Data sizes
        r"(\d+(?:\.\d+)?)\s*(?:ms|seconds?|minutes?|hours?)",  # Time
        r"(\d+(?:\.\d+)?)\s*(?:x|times)\s+(?:faster|slower)",  # Performance claims
    ]
    
    # Code security patterns (simplified static analysis)
    SECURITY_PATTERNS = [
        (r"eval\s*\(", "CRITICAL: Use of eval() - code injection risk"),
        (r"exec\s*\(", "CRITICAL: Use of exec() - code injection risk"),
        (r"__import__\s*\(", "WARNING: Dynamic import - potential security risk"),
        (r"subprocess\..*shell\s*=\s*True", "WARNING: Shell=True in subprocess - injection risk"),
        (r"os\.system\s*\(", "WARNING: os.system() - prefer subprocess"),
        (r"pickle\.loads?\s*\(", "WARNING: Pickle deserialization - potential RCE"),
        (r"yaml\.load\s*\([^,)]*\)", "WARNING: yaml.load without Loader - use safe_load"),
        (r"password\s*=\s*[\"'][^\"']+[\"']", "CRITICAL: Hardcoded password detected"),
        (r"api_key\s*=\s*[\"'][^\"']+[\"']", "CRITICAL: Hardcoded API key detected"),
        (r"\.format\s*\([^)]*request\.", "WARNING: Potential format string injection"),
        (r"sql.*\+.*input|input.*\+.*sql", "CRITICAL: Potential SQL injection"),
    ]
    
    def __init__(self):
        # Compile patterns for efficiency
        self.contradiction_compiled = [
            (re.compile(p, re.IGNORECASE), msg) 
            for p, msg in self.CONTRADICTION_PATTERNS
        ]
        self.number_compiled = [
            re.compile(p, re.IGNORECASE) for p in self.NUMBER_PATTERNS
        ]
        self.security_compiled = [
            (re.compile(p, re.IGNORECASE), msg) 
            for p, msg in self.SECURITY_PATTERNS
        ]
    
    def check_contradictions(self, text: str) -> list[str]:
        """Check for logical contradictions in text"""
        issues = []
        for pattern, msg in self.contradiction_compiled:
            if pattern.search(text):
                issues.append(msg)
        return issues
    
    def extract_numerical_claims(self, text: str) -> list[dict]:
        """Extract and validate numerical claims"""
        claims = []
        for pattern in self.number_compiled:
            matches = pattern.findall(text)
            for match in matches:
                # Basic sanity check - is the number reasonable?
                try:
                    value = float(match.replace(",", ""))
                    verified = True
                    warning = None
                    
                    # Check for suspicious values
                    if "%" in text and value > 100:
                        verified = False
                        warning = "Percentage > 100%"
                    
                    claims.append({
                        "value": value,
                        "raw": match,
                        "verified": verified,
                        "warning": warning
                    })
                except ValueError:
                    pass
        
        return claims
    
    def analyze_code(self, code: str) -> list[dict]:
        """
        Static analysis of code for security issues and bugs.
        Uses regex for common patterns and AST for Python code.
        """
        issues = []
        
        # Regex-based security checks
        for pattern, msg in self.security_compiled:
            if pattern.search(code):
                severity = "critical" if "CRITICAL" in msg else "warning"
                issues.append({
                    "type": "security",
                    "description": msg.replace("CRITICAL: ", "").replace("WARNING: ", ""),
                    "severity": severity
                })
        
        # AST-based analysis for Python code
        try:
            tree = ast.parse(code)
            issues.extend(self._analyze_ast(tree))
        except SyntaxError:
            # Not valid Python or might be another language
            pass
        
        return issues
    
    def _analyze_ast(self, tree: ast.AST) -> list[dict]:
        """Analyze Python AST for potential issues"""
        issues = []
        
        for node in ast.walk(tree):
            # Check for bare except clauses
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    issues.append({
                        "type": "code_quality",
                        "description": "Bare 'except:' clause catches all exceptions including KeyboardInterrupt",
                        "severity": "warning"
                    })
            
            # Check for mutable default arguments
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for default in node.args.defaults + node.args.kw_defaults:
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        issues.append({
                            "type": "bug_prediction",
                            "description": f"Mutable default argument in function '{node.name}'",
                            "severity": "warning"
                        })
            
            # Check for assert statements (removed in optimized code)
            if isinstance(node, ast.Assert):
                issues.append({
                    "type": "code_quality",
                    "description": "Assert statements are removed with -O flag",
                    "severity": "info"
                })
        
        return issues
    
    def generate_self_critique(self, text: str) -> str:
        """
        Generate adversarial self-critique.
        This is a template - actual critique would come from LLM.
        """
        critiques = []
        
        # Check for uncertainty markers that should trigger doubt
        if any(word in text.lower() for word in ["probably", "likely", "might", "perhaps", "maybe"]):
            critiques.append("Response contains uncertainty markers - claims may not be fully verified")
        
        # Check for unsourced claims
        if not any(marker in text.lower() for marker in ["according to", "source:", "cited", "reference"]):
            critiques.append("No sources cited - claims should be independently verified")
        
        # Check for absolute claims
        if any(word in text.lower() for word in ["always", "never", "definitely", "certainly"]):
            critiques.append("Contains absolute claims - consider edge cases")
        
        # Check for code without error handling
        if "```" in text and "try:" not in text and "except" not in text:
            critiques.append("Code lacks explicit error handling")
        
        return "; ".join(critiques) if critiques else "No obvious issues found"
    
    def calculate_confidence_adjustment(
        self,
        issues: list[str],
        warnings: list[str],
        code_issues: list[dict],
        numerical_claims: list[dict]
    ) -> float:
        """
        Calculate confidence adjustment based on verification results.
        Returns a negative adjustment (0.0 to -1.0).
        """
        adjustment = 0.0
        
        # Each issue reduces confidence
        adjustment -= len(issues) * 0.15
        
        # Warnings have smaller impact
        adjustment -= len(warnings) * 0.05
        
        # Critical code issues have big impact
        critical_count = sum(1 for i in code_issues if i.get("severity") == "critical")
        adjustment -= critical_count * 0.2
        
        # Unverified numerical claims
        unverified = sum(1 for c in numerical_claims if not c.get("verified", True))
        adjustment -= unverified * 0.1
        
        # Clamp to valid range
        return max(-1.0, min(0.0, adjustment))
    
    def verify(
        self,
        response_text: str,
        code_blocks: Optional[list[str]] = None
    ) -> VerificationResult:
        """
        Run full verification on a response.
        
        Args:
            response_text: The LLM's response text
            code_blocks: Extracted code blocks (if any)
            
        Returns:
            VerificationResult with all checks
        """
        issues = []
        warnings = []
        
        # Check for contradictions
        contradiction_issues = self.check_contradictions(response_text)
        issues.extend(contradiction_issues)
        
        # Extract and verify numerical claims
        numerical_claims = self.extract_numerical_claims(response_text)
        for claim in numerical_claims:
            if claim.get("warning"):
                warnings.append(f"Numerical issue: {claim['warning']}")
        
        # Analyze code blocks
        code_issues = []
        all_code = code_blocks or []
        
        # Also extract inline code from response
        inline_code = re.findall(r"```[\w]*\n?(.*?)```", response_text, re.DOTALL)
        all_code.extend(inline_code)
        
        for code in all_code:
            if code.strip():
                code_issues.extend(self.analyze_code(code))
        
        # Generate self-critique
        self_critique = self.generate_self_critique(response_text)
        
        # Calculate confidence adjustment
        confidence_adjustment = self.calculate_confidence_adjustment(
            issues, warnings, code_issues, numerical_claims
        )
        
        # Determine if verification passed
        critical_issues = [i for i in code_issues if i.get("severity") == "critical"]
        passed = len(issues) == 0 and len(critical_issues) == 0
        
        return VerificationResult(
            passed=passed,
            issues=issues,
            warnings=warnings,
            numerical_claims=numerical_claims,
            code_issues=code_issues,
            self_critique=self_critique,
            confidence_adjustment=confidence_adjustment
        )


# Singleton instance
_verifier: Optional[VerificationLayer] = None


def get_verification_layer() -> VerificationLayer:
    """Get or create singleton VerificationLayer instance"""
    global _verifier
    if _verifier is None:
        _verifier = VerificationLayer()
    return _verifier

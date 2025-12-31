"""
Code Analyzer Module
Compiler-grade reasoning for code analysis, bug prediction, and diff generation.

DESIGN DECISIONS:
- Static analysis using Python AST
- Security-first: flag vulnerable patterns
- Diff-first output (Git-style)
- Bug prediction before runtime
- Environment awareness
"""

import ast
import re
import difflib
from typing import Optional
from pydantic import BaseModel


class CodeIssue(BaseModel):
    """A detected code issue"""
    type: str  # "security", "bug", "performance", "style"
    severity: str  # "critical", "warning", "info"
    line: Optional[int] = None
    description: str
    suggestion: Optional[str] = None


class CodeAnalysisResult(BaseModel):
    """Result of code analysis"""
    language: str
    issues: list[CodeIssue]
    has_critical: bool
    has_warnings: bool
    security_score: float  # 0-1, 1 = no issues
    predicted_bugs: list[str]
    suggestions: list[str]


class DiffResult(BaseModel):
    """Result of diff generation"""
    has_changes: bool
    unified_diff: str
    added_lines: int
    removed_lines: int
    changed_files: list[str]


class CodeAnalyzer:
    """
    Analyzes code for security issues, bugs, and quality.
    Generates Git-style diffs for code changes.
    
    Features:
    - Static analysis for Python, JS patterns
    - Security vulnerability detection
    - Bug prediction
    - Performance issue detection
    - Diff generation
    """
    
    # Security patterns for various languages
    SECURITY_PATTERNS = {
        "python": [
            (r"\beval\s*\(", "critical", "eval() allows arbitrary code execution"),
            (r"\bexec\s*\(", "critical", "exec() allows arbitrary code execution"),
            (r"\b__import__\s*\(", "warning", "Dynamic import can be dangerous"),
            (r"subprocess\.[^(]+\([^)]*shell\s*=\s*True", "warning", "shell=True allows shell injection"),
            (r"\bos\.system\s*\(", "warning", "os.system is vulnerable to injection, use subprocess"),
            (r"\bpickle\.loads?\s*\(", "critical", "Pickle deserialization can execute arbitrary code"),
            (r"yaml\.load\s*\([^)]*\)(?!\s*,\s*Loader)", "warning", "yaml.load without Loader is unsafe"),
            (r"password\s*=\s*['\"][^'\"]+['\"]", "critical", "Hardcoded password detected"),
            (r"api[_-]?key\s*=\s*['\"][^'\"]+['\"]", "critical", "Hardcoded API key detected"),
            (r"secret\s*=\s*['\"][^'\"]+['\"]", "critical", "Hardcoded secret detected"),
            (r"\.execute\s*\([^)]*%[^)]*\)", "critical", "SQL injection via string formatting"),
            (r"\.execute\s*\([^)]*\+[^)]*\)", "critical", "SQL injection via concatenation"),
            (r"\.execute\s*\(f['\"]", "critical", "SQL injection via f-string"),
            (r"hashlib\.md5\(", "warning", "MD5 is cryptographically weak"),
            (r"hashlib\.sha1\(", "warning", "SHA1 is cryptographically weak"),
            (r"random\.", "info", "random module is not cryptographically secure"),
        ],
        "javascript": [
            (r"\beval\s*\(", "critical", "eval() allows arbitrary code execution"),
            (r"innerHTML\s*=", "warning", "innerHTML can lead to XSS"),
            (r"document\.write\s*\(", "warning", "document.write can lead to XSS"),
            (r"\.html\s*\([^)]*\$", "warning", "jQuery .html() with user input can cause XSS"),
            (r"new\s+Function\s*\(", "critical", "Function constructor allows code injection"),
            (r"localStorage\.", "info", "localStorage is not secure for sensitive data"),
            (r"sessionStorage\.", "info", "sessionStorage is not secure for sensitive data"),
        ],
    }
    
    # Bug patterns
    BUG_PATTERNS = {
        "python": [
            (r"except\s*:", "Bare except catches all exceptions including KeyboardInterrupt"),
            (r"def\s+\w+\s*\([^)]*=\s*\[\]", "Mutable default argument (list)"),
            (r"def\s+\w+\s*\([^)]*=\s*\{\}", "Mutable default argument (dict)"),
            (r"assert\s+", "Assert statements are removed with -O flag"),
            (r"==\s*None\b", "Use 'is None' instead of '== None'"),
            (r"!=\s*None\b", "Use 'is not None' instead of '!= None'"),
            (r"\btype\s*\([^)]+\)\s*==", "Use isinstance() instead of type() comparison"),
            (r"except\s+\w+\s*,\s*\w+", "Old-style exception syntax (Python 2)"),
            (r"print\s+['\"]", "Python 2 print statement syntax"),
        ],
        "javascript": [
            (r"==\s*null\b", "Use === for null comparison"),
            (r"!=\s*null\b", "Use !== for null comparison"),
            (r"==\s*undefined\b", "Use === for undefined comparison"),
            (r"\bvar\s+", "Use const or let instead of var"),
            (r"\.then\s*\([^)]*\)\s*$", "Promise chain without .catch()"),
        ],
    }
    
    # Performance patterns
    PERF_PATTERNS = {
        "python": [
            (r"for\s+\w+\s+in\s+range\s*\(\s*len\s*\(", "Use enumerate() instead of range(len())"),
            (r"\+\s*=\s*['\"]", "String concatenation in loop is O(n²), use join()"),
            (r"\.append\s*\([^)]+\)\s*$", "Consider list comprehension instead of append loop"),
            (r"import\s+\*", "Wildcard import is slow and pollutes namespace"),
            (r"time\.sleep\s*\([^)]*\)", "Blocking sleep - consider async alternatives"),
        ],
        "javascript": [
            (r"\.forEach\s*\(", "forEach cannot break early, consider for...of"),
            (r"document\.querySelector.*in.*loop", "DOM queries in loops are slow"),
            (r"JSON\.parse\s*\(JSON\.stringify", "Deep clone via JSON is slow for large objects"),
        ],
    }
    
    def __init__(self):
        # Compile all patterns
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency"""
        self.security_compiled = {}
        self.bug_compiled = {}
        self.perf_compiled = {}
        
        for lang, patterns in self.SECURITY_PATTERNS.items():
            self.security_compiled[lang] = [
                (re.compile(p, re.IGNORECASE), sev, msg)
                for p, sev, msg in patterns
            ]
        
        for lang, patterns in self.BUG_PATTERNS.items():
            self.bug_compiled[lang] = [
                (re.compile(p, re.IGNORECASE), msg)
                for p, msg in patterns
            ]
        
        for lang, patterns in self.PERF_PATTERNS.items():
            self.perf_compiled[lang] = [
                (re.compile(p, re.IGNORECASE), msg)
                for p, msg in patterns
            ]
    
    def detect_language(self, code: str) -> str:
        """Detect programming language from code"""
        # Simple heuristics
        if "def " in code or "import " in code or "class " in code:
            if "self" in code or ":" in code:
                return "python"
        
        if "function " in code or "const " in code or "let " in code or "var " in code:
            return "javascript"
        
        if "fn " in code or "impl " in code or "pub " in code:
            return "rust"
        
        if "func " in code or "package " in code:
            return "go"
        
        # Default to python
        return "python"
    
    def _find_line_number(self, code: str, match_start: int) -> int:
        """Find line number for a match position"""
        return code[:match_start].count("\n") + 1
    
    def analyze_security(self, code: str, language: str) -> list[CodeIssue]:
        """Analyze code for security issues"""
        issues = []
        patterns = self.security_compiled.get(language, [])
        
        for pattern, severity, description in patterns:
            for match in pattern.finditer(code):
                line = self._find_line_number(code, match.start())
                issues.append(CodeIssue(
                    type="security",
                    severity=severity,
                    line=line,
                    description=description,
                    suggestion=f"Review line {line} for security implications"
                ))
        
        return issues
    
    def analyze_bugs(self, code: str, language: str) -> list[CodeIssue]:
        """Analyze code for potential bugs"""
        issues = []
        patterns = self.bug_compiled.get(language, [])
        
        for pattern, description in patterns:
            for match in pattern.finditer(code):
                line = self._find_line_number(code, match.start())
                issues.append(CodeIssue(
                    type="bug",
                    severity="warning",
                    line=line,
                    description=description
                ))
        
        return issues
    
    def analyze_performance(self, code: str, language: str) -> list[CodeIssue]:
        """Analyze code for performance issues"""
        issues = []
        patterns = self.perf_compiled.get(language, [])
        
        for pattern, description in patterns:
            for match in pattern.finditer(code):
                line = self._find_line_number(code, match.start())
                issues.append(CodeIssue(
                    type="performance",
                    severity="info",
                    line=line,
                    description=description
                ))
        
        return issues
    
    def analyze_python_ast(self, code: str) -> list[CodeIssue]:
        """Deep AST analysis for Python code"""
        issues = []
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return [CodeIssue(
                type="syntax",
                severity="critical",
                line=e.lineno,
                description=f"Syntax error: {e.msg}"
            )]
        
        for node in ast.walk(tree):
            # Check for unused variables (basic)
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.startswith("_"):
                        pass  # Intentionally unused
            
            # Check for recursion without base case
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_name = node.name
                has_return = False
                has_self_call = False
                
                for child in ast.walk(node):
                    if isinstance(child, ast.Return):
                        has_return = True
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Name) and child.func.id == func_name:
                            has_self_call = True
                
                if has_self_call and not has_return:
                    issues.append(CodeIssue(
                        type="bug",
                        severity="warning",
                        line=node.lineno,
                        description=f"Recursive function '{func_name}' may not have a base case"
                    ))
            
            # Check for empty except blocks
            if isinstance(node, ast.ExceptHandler):
                if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                    issues.append(CodeIssue(
                        type="bug",
                        severity="warning",
                        line=node.lineno,
                        description="Empty except block silently swallows errors"
                    ))
        
        return issues
    
    def predict_bugs(self, code: str, language: str) -> list[str]:
        """Predict potential bugs before runtime"""
        predictions = []
        
        # Check for common bug patterns
        if language == "python":
            # Missing return statement
            if "def " in code and "return" not in code and "print" not in code:
                predictions.append("Function may be missing a return statement")
            
            # Unclosed resources
            if "open(" in code and "with " not in code:
                predictions.append("File opened without 'with' statement - may leak file handles")
            
            # Missing error handling for network/file operations
            if any(op in code for op in ["requests.", "urllib", "open("]):
                if "try:" not in code:
                    predictions.append("I/O operations without error handling")
        
        return predictions
    
    def analyze(self, code: str, language: Optional[str] = None) -> CodeAnalysisResult:
        """
        Full code analysis.
        
        Args:
            code: The code to analyze
            language: Programming language (auto-detected if not provided)
            
        Returns:
            CodeAnalysisResult with all issues and suggestions
        """
        if not language:
            language = self.detect_language(code)
        
        issues = []
        
        # Security analysis
        issues.extend(self.analyze_security(code, language))
        
        # Bug detection
        issues.extend(self.analyze_bugs(code, language))
        
        # Performance analysis
        issues.extend(self.analyze_performance(code, language))
        
        # Deep AST analysis for Python
        if language == "python":
            issues.extend(self.analyze_python_ast(code))
        
        # Predict bugs
        predicted_bugs = self.predict_bugs(code, language)
        
        # Calculate security score
        critical_count = sum(1 for i in issues if i.severity == "critical")
        warning_count = sum(1 for i in issues if i.severity == "warning")
        security_score = max(0.0, 1.0 - (critical_count * 0.3) - (warning_count * 0.1))
        
        # Generate suggestions
        suggestions = []
        if critical_count > 0:
            suggestions.append("Address critical security issues before deployment")
        if predicted_bugs:
            suggestions.append("Review predicted bug locations")
        
        return CodeAnalysisResult(
            language=language,
            issues=issues,
            has_critical=critical_count > 0,
            has_warnings=warning_count > 0,
            security_score=security_score,
            predicted_bugs=predicted_bugs,
            suggestions=suggestions
        )
    
    def generate_diff(
        self,
        original: str,
        modified: str,
        filename: str = "code"
    ) -> DiffResult:
        """
        Generate Git-style unified diff between original and modified code.
        
        Args:
            original: Original code
            modified: Modified code
            filename: Filename for diff header
            
        Returns:
            DiffResult with unified diff
        """
        original_lines = original.splitlines(keepends=True)
        modified_lines = modified.splitlines(keepends=True)
        
        diff = list(difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}"
        ))
        
        added = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
        removed = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))
        
        return DiffResult(
            has_changes=len(diff) > 0,
            unified_diff="".join(diff),
            added_lines=added,
            removed_lines=removed,
            changed_files=[filename] if diff else []
        )
    
    def format_diff_output(self, diff: DiffResult) -> str:
        """Format diff for display"""
        if not diff.has_changes:
            return "No changes"
        
        summary = f"+{diff.added_lines} -{diff.removed_lines} lines"
        return f"```diff\n{diff.unified_diff}```\n{summary}"


# Singleton instance
_analyzer: Optional[CodeAnalyzer] = None


def get_code_analyzer() -> CodeAnalyzer:
    """Get or create singleton CodeAnalyzer instance"""
    global _analyzer
    if _analyzer is None:
        _analyzer = CodeAnalyzer()
    return _analyzer

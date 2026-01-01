import subprocess
import os
import shutil
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import httpx
import glob

class ToolResult(BaseModel):
    success: bool
    output: str
    error: Optional[str] = None

class BaseTool:
    name: str = "base_tool"
    description: str = "Base tool"

    def execute(self, **kwargs) -> ToolResult:
        raise NotImplementedError

class ShellTool(BaseTool):
    name: str = "shell"
    description: str = "Execute shell commands. Use carefully."

    def execute(self, command: str, timeout: int = 30) -> ToolResult:
        try:
            # Security check: basic strictness
            forbidden = ["rm -rf /", ":(){ :|:& };:"]
            if any(f in command for f in forbidden):
                return ToolResult(success=False, output="", error="Command blocked by safety filter")

            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=timeout,
                executable="/bin/bash"
            )
            success = result.returncode == 0
            output = result.stdout if success else result.stdout + result.stderr
            return ToolResult(success=success, output=output.strip(), error=None if success else result.stderr.strip())
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output="", error=f"Command timed out after {timeout}s")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

class FileTool(BaseTool):
    name: str = "file_io"
    description: str = "Read/Write/List files"

    def execute(self, operation: str, path: str, content: Optional[str] = None) -> ToolResult:
        try:
            if operation == "read":
                if not os.path.exists(path):
                    return ToolResult(success=False, output="", error="File not found")
                with open(path, 'r') as f:
                    return ToolResult(success=True, output=f.read())
            
            elif operation == "write":
                if content is None:
                    return ToolResult(success=False, output="", error="Content required for write")
                # Ensure dir exists
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
                with open(path, 'w') as f:
                    f.write(content)
                return ToolResult(success=True, output=f"Written to {path}")
            
            elif operation == "list":
                if not os.path.exists(path):
                     return ToolResult(success=False, output="", error="Directory not found")
                files = os.listdir(path)
                return ToolResult(success=True, output="\n".join(files))
            
            else:
                return ToolResult(success=False, output="", error=f"Unknown operation: {operation}")

        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

class WebTool(BaseTool):
    name: str = "web"
    description: str = "Make HTTP requests"

    def execute(self, url: str, method: str = "GET", data: Optional[Dict] = None) -> ToolResult:
        try:
            with httpx.Client(timeout=10.0) as client:
                if method.upper() == "GET":
                    resp = client.get(url)
                elif method.upper() == "POST":
                    resp = client.post(url, json=data)
                else:
                    return ToolResult(success=False, output="", error="Unsupported method")
                
                return ToolResult(success=True, output=resp.text[:2000]) # Limit output
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

def get_tools_map() -> Dict[str, BaseTool]:
    return {
        "shell": ShellTool(),
        "file_io": FileTool(),
        "web": WebTool()
    }

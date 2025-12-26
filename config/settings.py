"""
Configuration settings for the AI Agent
All defaults optimized for MacBook Air M4 (8-16GB RAM)
"""

from pathlib import Path
from pydantic import BaseModel
from typing import Optional


class ModelConfig(BaseModel):
    """LLM model configuration"""
    # ASSUMPTION: User will download a GGUF model compatible with M4
    # Recommended: Qwen2.5-7B-Instruct-Q4_K_M.gguf or similar 7B quantized model
    model_path: str = "./models/model.gguf"
    
    # Context window - 4096 is safe for 7B models on 8GB RAM
    n_ctx: int = 4096
    
    # GPU layers - Metal acceleration on M4
    # -1 means use all layers on GPU, adjust down if OOM
    n_gpu_layers: int = -1
    
    # Batch size for prompt processing - optimized for M4
    n_batch: int = 512
    
    # Number of threads - M4 has 10 cores (4 performance + 6 efficiency)
    n_threads: int = 6
    
    # Temperature for generation
    temperature: float = 0.7
    
    # Top-p sampling
    top_p: float = 0.9
    
    # Max tokens to generate per response
    max_tokens: int = 2048
    
    # Repeat penalty to reduce repetition
    repeat_penalty: float = 1.1


class MemoryConfig(BaseModel):
    """Memory store configuration"""
    # ChromaDB persistence directory
    persist_directory: str = "./memory_store"
    
    # Collection name for agent memory
    collection_name: str = "agent_memory"
    
    # Max memories to retrieve per query
    max_retrieval: int = 5
    
    # Similarity threshold for retrieval (0-1)
    similarity_threshold: float = 0.7


class VerificationConfig(BaseModel):
    """Verification and confidence settings"""
    # Confidence threshold to accept answer
    accept_threshold: float = 0.9
    
    # Confidence threshold below which to refuse
    refuse_threshold: float = 0.6
    
    # Max retry attempts for verification loop
    max_retries: int = 2
    
    # Enable code static analysis
    enable_static_analysis: bool = True
    
    # Enable numerical sanity checks
    enable_numerical_checks: bool = True


class APIConfig(BaseModel):
    """API server configuration"""
    host: str = "127.0.0.1"
    port: int = 8000
    
    # Enable request logging
    enable_logging: bool = True
    
    # Log directory
    log_dir: str = "./logs"


class AgentConfig(BaseModel):
    """Complete agent configuration"""
    model: ModelConfig = ModelConfig()
    memory: MemoryConfig = MemoryConfig()
    verification: VerificationConfig = VerificationConfig()
    api: APIConfig = APIConfig()
    
    # High-risk domains that trigger verification
    high_risk_domains: list[str] = [
        "security",
        "finance",
        "physics",
        "math",
        "medical",
        "legal",
        "systems"
    ]
    
    # Keywords that trigger deep thinking mode
    deep_thinking_triggers: list[str] = [
        "explain",
        "analyze",
        "design",
        "architect",
        "debug",
        "optimize",
        "security",
        "performance",
        "trade-off",
        "compare"
    ]


# Global config instance
config = AgentConfig()


def load_config(config_path: Optional[Path] = None) -> AgentConfig:
    """Load configuration from file or return defaults"""
    global config
    if config_path and config_path.exists():
        import json
        with open(config_path) as f:
            data = json.load(f)
        config = AgentConfig(**data)
    return config

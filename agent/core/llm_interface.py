"""
LLM Interface Module
Local model interface optimized for MacBook Air M4 using llama-cpp-python

DESIGN DECISIONS:
- Uses llama-cpp-python with Metal acceleration for M-series chips
- Supports GGUF quantized models (recommended: 7B-Q4_K_M for 8GB RAM)
- Lazy loading to minimize startup time
- Streaming support for better UX
"""

import os
from pathlib import Path
from typing import Optional, Generator, Any
from pydantic import BaseModel

# Will be imported when needed to avoid startup delay
_llama_module = None


class LLMResponse(BaseModel):
    """Response from the LLM"""
    text: str
    tokens_used: int
    finish_reason: str  # "stop", "length", "error"
    model_name: str


class LLMConfig(BaseModel):
    """Configuration for LLM"""
    model_path: str
    n_ctx: int = 4096
    n_gpu_layers: int = -1  # -1 = use all layers on GPU (Metal)
    n_batch: int = 512
    n_threads: int = 6
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    repeat_penalty: float = 1.1


class LLMInterface:
    """
    Interface to local LLM using llama-cpp-python.
    Optimized for MacBook Air M4 with Metal acceleration.
    
    USAGE:
    1. Download a GGUF model (e.g., Qwen2.5-7B-Instruct-Q4_K_M.gguf)
    2. Place in ./models/ directory
    3. Update config.model.model_path if needed
    """
    
    # Base system prompt - minimal, direct personality
    BASE_SYSTEM_PROMPT = """You are a precise, helpful AI assistant.

Core behaviors:
- Be direct and concise. No fluff.
- If uncertain, say exactly why. Never guess.
- For code: predict bugs, flag security issues, use diffs.
- For research: cite sources or label as estimates.
- If you can't answer confidently, refuse and explain.
- No "as an AI language model" phrases.
- No motivational speeches or excessive politeness.
- Have clear opinions when justified.

{mode_modifier}
"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig(model_path="./models/model.gguf")
        self._model = None
        self._model_loaded = False
    
    def _get_llama(self):
        """Lazy import of llama-cpp-python to speed up startup"""
        global _llama_module
        if _llama_module is None:
            try:
                from llama_cpp import Llama
                _llama_module = Llama
            except ImportError:
                raise ImportError(
                    "llama-cpp-python not installed. Run:\n"
                    "CMAKE_ARGS='-DLLAMA_METAL=on' pip install llama-cpp-python"
                )
        return _llama_module
    
    def load_model(self) -> bool:
        """
        Load the model into memory.
        Call this explicitly or it will be called on first generate().
        
        Returns True if successful, raises exception otherwise.
        """
        if self._model_loaded:
            return True
        
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Download a GGUF model and place it at {model_path.absolute()}\n"
                f"Recommended: Qwen2.5-7B-Instruct-Q4_K_M.gguf"
            )
        
        Llama = self._get_llama()
        
        # Load model with Metal acceleration
        self._model = Llama(
            model_path=str(model_path),
            n_ctx=self.config.n_ctx,
            n_gpu_layers=self.config.n_gpu_layers,
            n_batch=self.config.n_batch,
            n_threads=self.config.n_threads,
            verbose=False,  # Reduce noise
        )
        
        self._model_loaded = True
        return True
    
    def _ensure_model_loaded(self):
        """Ensure model is loaded before generation"""
        if not self._model_loaded:
            self.load_model()
    
    def build_prompt(
        self,
        user_message: str,
        mode_modifier: str = "",
        conversation_history: Optional[list[dict]] = None,
        context: Optional[str] = None
    ) -> str:
        """
        Build the full prompt with system prompt and conversation history.
        
        ASSUMPTION: Using ChatML format which works with most instruction-tuned models.
        Adjust if using a model with different prompt format.
        """
        system_prompt = self.BASE_SYSTEM_PROMPT.format(mode_modifier=mode_modifier)
        
        # Add context if provided (from memory retrieval)
        if context:
            system_prompt += f"\n\nRelevant context:\n{context}"
        
        # Build ChatML format
        messages = [{"role": "system", "content": system_prompt}]
        
        if conversation_history:
            messages.extend(conversation_history)
        
        messages.append({"role": "user", "content": user_message})
        
        # Convert to ChatML string
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        
        prompt_parts.append("<|im_start|>assistant\n")
        
        return "\n".join(prompt_parts)
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[list[str]] = None
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: Full formatted prompt
            max_tokens: Override default max tokens
            temperature: Override default temperature
            stop: Stop sequences
            
        Returns:
            LLMResponse with generated text and metadata
        """
        self._ensure_model_loaded()
        
        stop_sequences = stop or ["<|im_end|>", "<|im_start|>"]
        
        try:
            output = self._model(
                prompt,
                max_tokens=max_tokens or self.config.max_tokens,
                temperature=temperature or self.config.temperature,
                top_p=self.config.top_p,
                repeat_penalty=self.config.repeat_penalty,
                stop=stop_sequences,
                echo=False,
            )
            
            text = output["choices"][0]["text"].strip()
            finish_reason = output["choices"][0].get("finish_reason", "stop")
            tokens_used = output.get("usage", {}).get("total_tokens", 0)
            
            return LLMResponse(
                text=text,
                tokens_used=tokens_used,
                finish_reason=finish_reason,
                model_name=self.config.model_path.split("/")[-1]
            )
            
        except Exception as e:
            return LLMResponse(
                text=f"Error generating response: {str(e)}",
                tokens_used=0,
                finish_reason="error",
                model_name=self.config.model_path.split("/")[-1]
            )
    
    def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[list[str]] = None
    ) -> Generator[str, None, None]:
        """
        Stream tokens from the LLM.
        Yields tokens as they're generated for responsive UX.
        """
        self._ensure_model_loaded()
        
        stop_sequences = stop or ["<|im_end|>", "<|im_start|>"]
        
        try:
            stream = self._model(
                prompt,
                max_tokens=max_tokens or self.config.max_tokens,
                temperature=temperature or self.config.temperature,
                top_p=self.config.top_p,
                repeat_penalty=self.config.repeat_penalty,
                stop=stop_sequences,
                echo=False,
                stream=True,
            )
            
            for chunk in stream:
                token = chunk["choices"][0].get("text", "")
                if token:
                    yield token
                    
        except Exception as e:
            yield f"\n[Error: {str(e)}]"
    
    def unload_model(self):
        """Unload model from memory to free RAM"""
        if self._model is not None:
            del self._model
            self._model = None
            self._model_loaded = False
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        if not self._model_loaded:
            return {"loaded": False, "path": self.config.model_path}
        
        return {
            "loaded": True,
            "path": self.config.model_path,
            "context_size": self.config.n_ctx,
            "gpu_layers": self.config.n_gpu_layers,
        }


# Singleton instance
_llm: Optional[LLMInterface] = None


def get_llm_interface(config: Optional[LLMConfig] = None) -> LLMInterface:
    """Get or create singleton LLMInterface instance"""
    global _llm
    if _llm is None:
        _llm = LLMInterface(config)
    return _llm


def create_llm_interface(config: LLMConfig) -> LLMInterface:
    """Create a new LLMInterface with custom config (not singleton)"""
    return LLMInterface(config)

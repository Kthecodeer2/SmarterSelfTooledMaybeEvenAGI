# Local AI Agent

A production-grade, local-first AI agent optimized for MacBook Air M4. Features self-verification, confidence scoring, specialized coding/research modes, and long-term memory.

## Features

### Intelligence
- **Self-verification loop**: Automatically re-checks answers for high-stakes domains (security, finance, physics, math)
- **Confidence scoring**: 0-1 confidence with threshold-based accept/retry/refuse
- **Dual reasoning modes**: Fast mode for trivial tasks, Deep mode for complex analysis
- **Error intolerance**: No guessing - explains uncertainty when unsure

### Memory
- **Persistent memory** via ChromaDB
- **Tagged storage**: temporary, project, permanent
- **Categories**: environment, preference, constraint, goal, fact
- **User control**: inspect, edit, delete any memory

### Coding Mode
- **Static analysis** for security vulnerabilities
- **Bug prediction** before runtime
- **Git-style diffs** for code changes
- **Environment awareness** (OS, architecture, versions)

### Research Mode
- **Claim categorization**: fact, estimate, consensus, speculation
- **Source attribution** required for numerical claims
- **Assumption/limitation disclosure**
- **Reproducibility notes**

### Personality
- Minimal politeness, zero fluff
- Clear opinions when justified
- Direct communication
- No "as an AI" phrasing

## Requirements

- Python 3.11+
- MacBook Air M4 (or other Apple Silicon)
- 8GB+ RAM (16GB recommended for larger models)
- ~5GB disk space for model + dependencies

## Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
cd /path/to/local-ai-agent

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies with Metal support
CMAKE_ARGS="-DLLAMA_METAL=on" pip install -r requirements.txt
```

### 2. Download a Model

Download a GGUF-quantized model compatible with M4 (7B recommended for 8GB RAM):

```bash
# Create models directory
mkdir -p models

# Option A: Using huggingface-cli
pip install huggingface-hub
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF qwen2.5-7b-instruct-q4_k_m.gguf --local-dir models

# Option B: Manual download from HuggingFace
# Visit: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF
# Download: qwen2.5-7b-instruct-q4_k_m.gguf
# Place in: ./models/
```

Recommended models for M4:
- **Qwen2.5-7B-Instruct-Q4_K_M** - Best overall quality/speed
- **Mistral-7B-Instruct-v0.3-Q4_K_M** - Good for coding
- **Llama-2-7B-Chat-Q4_K_M** - Most compatible

### 3. Run the Server

```bash
# Start with default settings
python run.py --model ./models/qwen2.5-7b-instruct-q4_k_m.gguf

# Or with custom settings
python run.py \
    --model ./models/model.gguf \
    --port 8000 \
    --ctx 4096 \
    --threads 6
```

### 4. Make Requests

```bash
# Simple chat
curl -X POST http://127.0.0.1:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"message": "Explain binary search in Python"}'

# With detailed logs
curl -X POST http://127.0.0.1:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"message": "Review this code for security issues", "include_logs": true}'
```

## API Reference

### POST /chat
Main conversation endpoint.

**Request:**
```json
{
    "message": "Your question or task",
    "include_logs": false
}
```

**Response:**
```json
{
    "response": "Agent's response",
    "confidence": 0.92,
    "reasoning_mode": "deep",
    "task_type": "coding",
    "was_refused": false,
    "verification_log": [...],
    "memory_context": "...",
    "code_analysis": {...},
    "research_analysis": {...}
}
```

### GET /status
System status and model info.

### POST /memory
Add a memory entry.

```json
{
    "content": "User prefers TypeScript",
    "tag": "permanent",
    "category": "preference"
}
```

### GET /memory
List all memories (optional filters: `?tag=project&category=environment`).

### DELETE /memory/{id}
Delete a memory by ID.

### POST /session/clear
Clear temporary memories and conversation history.

### POST /model/load
Explicitly load model into memory.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Input                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Input Pipeline                          │
│  • Sanitize input                                           │
│  • Strip control tokens                                     │
│  • Extract code blocks                                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Task Classifier                          │
│  • Detect: trivial, coding, research, high-risk             │
│  • Identify risk domains: security, finance, etc.           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Mode Selector                           │
│  • FAST mode: trivial tasks                                 │
│  • DEEP mode: complex analysis                              │
│  • VERIFY mode: high-risk domains                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────┐        ┌─────────────────────────────────────┐
│    Memory    │◄──────►│              LLM Interface          │
│  (ChromaDB)  │        │  • llama-cpp-python                 │
│              │        │  • Metal acceleration               │
└──────────────┘        └─────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Verification Layer                        │
│  • Logical consistency                                      │
│  • Numerical sanity                                         │
│  • Code static analysis                                     │
│  • Adversarial self-critique                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Confidence Scoring                        │
│  • Score: 0.0 - 1.0                                         │
│  • Thresholds: accept (≥0.9), retry (0.6-0.9), refuse (<0.6)│
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
            ┌───────────┐        ┌─────────────┐
            │   Retry   │        │   Refusal   │
            │   Loop    │        │   Handler   │
            └───────────┘        └─────────────┘
                    │                   │
                    └─────────┬─────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Personality Filter                        │
│  • Remove fluff                                             │
│  • Apply style rules                                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       Response                              │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

Edit `config/settings.py` or pass CLI arguments:

| Setting | Default | Description |
|---------|---------|-------------|
| `model_path` | `./models/model.gguf` | Path to GGUF model |
| `n_ctx` | `4096` | Context window size |
| `n_gpu_layers` | `-1` | GPU layers (-1 = all) |
| `n_threads` | `6` | CPU threads (M4 optimal) |
| `temperature` | `0.7` | Generation temperature |
| `max_tokens` | `2048` | Max output tokens |
| `accept_threshold` | `0.9` | Confidence to accept |
| `refuse_threshold` | `0.6` | Confidence to refuse |
| `max_retries` | `2` | Verification retry limit |

## Performance Tuning for M4

### Memory Optimization
- Use Q4_K_M quantization (4-bit) for 8GB RAM
- Use Q5_K_M or Q6_K for 16GB RAM
- Context size of 4096 uses ~2GB for 7B model

### GPU Acceleration
- Metal is auto-enabled with `n_gpu_layers=-1`
- Falls back to CPU if Metal unavailable

### Thread Configuration
- M4 has 10 cores (4P + 6E)
- Default 6 threads balances performance and thermal

## File Structure

```
local-ai-agent/
├── agent/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── input_pipeline.py    # Input sanitization
│   │   ├── task_classifier.py   # Task type detection
│   │   ├── mode_selector.py     # Reasoning mode selection
│   │   ├── llm_interface.py     # LLM wrapper
│   │   ├── verification.py      # Response verification
│   │   ├── confidence.py        # Confidence scoring
│   │   ├── retry_loop.py        # Verification retry
│   │   ├── refusal.py           # Refusal handling
│   │   └── orchestrator.py      # Main coordinator
│   ├── memory/
│   │   ├── __init__.py
│   │   └── memory_store.py      # ChromaDB wrapper
│   ├── coding/
│   │   ├── __init__.py
│   │   └── code_analyzer.py     # Static analysis
│   ├── research/
│   │   ├── __init__.py
│   │   └── research_mode.py     # Research features
│   └── personality/
│       ├── __init__.py
│       └── response_filter.py   # Style filtering
├── api/
│   ├── __init__.py
│   └── server.py                # FastAPI server
├── config/
│   ├── __init__.py
│   └── settings.py              # Configuration
├── models/                      # GGUF models (gitignored)
├── memory_store/                # ChromaDB data (gitignored)
├── logs/                        # API logs (gitignored)
├── requirements.txt
├── run.py                       # Entry point
└── README.md
```

## Security Notes

- Model runs locally - no data leaves your machine
- Input sanitization removes injection attempts
- Code analysis flags security vulnerabilities
- No hardcoded credentials in responses
- Memory is stored locally in ChromaDB

## Troubleshooting

### Model won't load
- Ensure GGUF format (not GGML or safetensors)
- Check file isn't corrupted (re-download)
- Verify sufficient RAM

### Slow performance
- Reduce context size (`--ctx 2048`)
- Use smaller quantization (Q4_K_S)
- Check for thermal throttling

### Metal errors
- Reinstall with Metal flag: `CMAKE_ARGS="-DLLAMA_METAL=on" pip install --force-reinstall llama-cpp-python`
- Ensure macOS is updated

### Memory errors
- Reduce context size
- Use more aggressive quantization
- Close other applications

## License

MIT License - Use freely for any purpose.

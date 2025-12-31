#!/usr/bin/env python3
"""
Local AI Agent - Entry Point

Run the AI agent server optimized for MacBook Air M4.

Usage:
    python run.py                    # Start API server
    python run.py --port 8080        # Custom port
    python run.py --model ./my.gguf  # Custom model path
"""

import argparse
import sys
from pathlib import Path

# Ensure the project root is in the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="Local AI Agent - Optimized for MacBook Air M4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run.py                          # Start with defaults
    python run.py --port 8080              # Custom port
    python run.py --model ./models/qwen.gguf  # Custom model
    python run.py --no-gpu                 # CPU only mode
        """
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="./models/model.gguf",
        help="Path to GGUF model file"
    )
    
    parser.add_argument(
        "--ctx",
        type=int,
        default=4096,
        help="Context window size (default: 4096)"
    )
    
    parser.add_argument(
        "--threads",
        type=int,
        default=6,
        help="Number of CPU threads (default: 6 for M4)"
    )
    
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration (use CPU only)"
    )
    
    parser.add_argument(
        "--memory-path",
        type=str,
        default="./memory_store",
        help="Path for persistent memory storage"
    )
    
    parser.add_argument(
        "--no-logs",
        action="store_true",
        help="Disable logging"
    )
    
    args = parser.parse_args()
    
    # Update config
    from config.settings import config
    
    config.model.model_path = args.model
    config.model.n_ctx = args.ctx
    config.model.n_threads = args.threads
    config.model.n_gpu_layers = 0 if args.no_gpu else -1
    config.memory.persist_directory = args.memory_path
    config.api.host = args.host
    config.api.port = args.port
    config.api.enable_logging = not args.no_logs
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    MODEL NOT FOUND                           ║
╠══════════════════════════════════════════════════════════════╣
║ Model file not found at: {str(model_path):<33} ║
║                                                              ║
║ To get started:                                              ║
║ 1. Create a 'models' directory: mkdir models                 ║
║ 2. Download a GGUF model (7B Q4 recommended for M4):         ║
║                                                              ║
║    Recommended models:                                       ║
║    • Qwen2.5-7B-Instruct-Q4_K_M.gguf                        ║
║    • Mistral-7B-Instruct-v0.3-Q4_K_M.gguf                   ║
║    • Llama-2-7B-Chat-Q4_K_M.gguf                            ║
║                                                              ║
║ 3. Place in ./models/ and run:                               ║
║    python run.py --model ./models/your-model.gguf            ║
║                                                              ║
║ Download from: https://huggingface.co/                       ║
╚══════════════════════════════════════════════════════════════╝
""")
        sys.exit(1)
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║               LOCAL AI AGENT - Starting Up                   ║
╠══════════════════════════════════════════════════════════════╣
║ Model: {args.model:<53} ║
║ Context: {args.ctx:<51} ║
║ GPU: {'Enabled (Metal)' if not args.no_gpu else 'Disabled':<55} ║
║ Threads: {args.threads:<51} ║
║ Memory: {args.memory_path:<52} ║
║                                                              ║
║ Starting server at http://{args.host}:{args.port:<24} ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    # Start server
    import uvicorn
    
    uvicorn.run(
        "api.server:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info" if not args.no_logs else "warning"
    )


if __name__ == "__main__":
    main()

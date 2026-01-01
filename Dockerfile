FROM python:3.14-slim

# Using python:3.14-slim for smaller image size while maintaining
# compatibility with latest features.

WORKDIR /app

# Install system dependencies
# git: required for git-based operations in tools
# build-essential: required for compiling some python extensions (if needed)
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install dependencies
# Dependencies versions are pinned in requirements.txt to ensure reproducibility
# Key versions:
# - fastapi: 0.115.6 (Modern web framework)
# - uvicorn: 0.34.0 (ASGI server)
# - pydantic: 2.10.4 (Data validation)
# - chromadb: 0.5.23 (Vector store)
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set python path to allow imports from root
ENV PYTHONPATH=/app

# Default command (overridden in docker-compose.yml services)
CMD ["python", "run.py"]

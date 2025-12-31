FROM python:3.14-slim

WORKDIR /app

# Install system dependencies
# git for git operations, build-essential for some python packages
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set python path
ENV PYTHONPATH=/app

# Default command (overridden in docker-compose)
CMD ["python", "run.py"]

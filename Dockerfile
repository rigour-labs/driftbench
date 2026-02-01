FROM python:3.10-slim

# Install system dependencies (Git, Docker CLI, Node.js)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    gnupg \
    && curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g @rigour-labs/cli \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements and install
COPY runner/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn

# Copy application code
COPY . .

# Expose API port
EXPOSE 8080

# Run FastAPI app
# Default to 8080 if PORT is not set (for local testing)
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]

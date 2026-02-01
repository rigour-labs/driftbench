FROM python:3.10-slim

# Install system dependencies (Git, Docker CLI)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh \
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
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]

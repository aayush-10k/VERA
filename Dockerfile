# VERA — Verify-first Retrieval Architecture
# Samsung PRISM GenAI Hackathon · Theme 1: Agentic Code Intelligence (CoIR AppsRetrieval)
FROM python:3.11-slim

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install minimal OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency lockfile and install CPU wheels
COPY requirements.lock /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir gradio

# Copy application source code
COPY . /app

# Expose Gradio demo port
EXPOSE 7860

# Default command: run unit tests
CMD ["pytest", "tests/", "-q"]

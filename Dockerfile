# Base image
# We use Python 3.11 slim - smaller than full Python image
FROM python:3.11-slim

# Metadata
LABEL maintainer="VideoMind"
LABEL description="Multimodal video summarizer"

# System dependencies
# ffmpeg is needed for audio extraction
# build-essential is needed for some Python packages
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Working directory
# ALl subsequent commands run from /app inside the container
WORKDIR /app

# Python dependencies
# Copy requirements first - Docker caches this layer
# If requirements doesn't change, Docker skips reinstalling
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# App code
# Copy everything else after installing deps for better caching
COPY . .

# Install the packages in editable mode
RUN pip install --no-cache-dir -e .

# Runtime directories
RUN mkdir -p /tmp/videomind/uploads /tmp/videomind/outputs

# Enviroment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV APP_ENV=production

# Port
EXPOSE 8000

# Default command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
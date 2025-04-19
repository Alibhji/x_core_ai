FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . /app/

# Ensure the API directory is in the Python path
ENV PYTHONPATH=/app

# Set API directory as working directory
WORKDIR /app/api

# Create startup script
RUN echo '#!/bin/bash\n\
echo "Starting X-Core AI API server..."\n\
echo "API will be available at http://localhost:8000/docs"\n\
uvicorn app:app --host 0.0.0.0 --port 8000 --reload' > /app/api/start.sh && \
    chmod +x /app/api/start.sh

# Environment variables
ENV PYTHONUNBUFFERED=1

# Expose the port
EXPOSE 8000

# Set the entrypoint to the start script
ENTRYPOINT ["/app/api/start.sh"] 
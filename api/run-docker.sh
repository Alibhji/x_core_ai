#!/bin/bash

echo "=== X-Core AI API - Docker Environment Setup ==="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH."
    echo "Please install Docker from https://www.docker.com/get-started"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose is not installed or not in PATH."
    echo "Please install Docker Compose from https://docs.docker.com/compose/install/"
    exit 1
fi

echo "Building and starting the X-Core AI API container..."
echo ""

# Navigate to the script's directory
cd "$(dirname "$0")"

# Give execute permission to the start script (in case it wasn't set)
chmod +x start.sh

# Build and start the Docker containers
docker-compose up --build

echo ""
echo "API container has been stopped."
echo "To start it again, run 'docker-compose up' in the api directory." 
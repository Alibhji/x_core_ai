@echo off
echo === X-Core AI API - Docker Environment Setup ===
echo.

REM Check if Docker is installed
where docker >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: Docker is not installed or not in PATH.
    echo Please install Docker Desktop from https://www.docker.com/products/docker-desktop
    exit /b 1
)

echo Building and starting the X-Core AI API container...
echo.

REM Navigate to the api directory
cd /d %~dp0

REM Build and start the Docker containers
docker-compose up --build

echo.
echo API container has been stopped.
echo To start it again, run "docker-compose up" in the api directory. 
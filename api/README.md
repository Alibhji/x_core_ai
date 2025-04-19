# X-Core AI API

This API allows you to interact with the X-Core AI Framework for forecasting and validation operations.

## Setup and Installation

### Option 1: Local Installation

1. Install the requirements:
```bash
pip install -r requirements.txt
```

2. Run the API server:
```bash
python app.py
```

The server will start at http://localhost:8000

### Option 2: Using Docker (Recommended)

The easiest way to get started is using Docker, which handles all dependencies and environment setup automatically.

#### Prerequisites
- [Docker](https://www.docker.com/get-started) installed on your system
- [Docker Compose](https://docs.docker.com/compose/install/) (usually included with Docker Desktop)

#### Starting the API Server

**Windows:**
```
run-docker.bat
```

**Linux/macOS:**
```bash
chmod +x run-docker.sh
./run-docker.sh
```

Or manually with Docker Compose:
```bash
docker-compose up --build
```

This will:
1. Build the Docker image with all required dependencies
2. Start the API server at http://localhost:8000
3. Mount the source code for hot-reloading (any changes you make will be instantly reflected)

### Accessing the API

Once the server is running (either locally or with Docker), the API will be available at:
- API base URL: http://localhost:8000
- Interactive documentation: http://localhost:8000/docs
- ReDoc alternative documentation: http://localhost:8000/redoc

## API Endpoints

### Health Check
- `GET /health`: Check if the API is running

### Forecast
- `POST /forecast`: Make predictions using the model
  
  Example request body:
  ```json
  {
    "features": [[[0.1, 0.2, ...], [...], ...], ...],
    "config": {
      "model_name": "gated_cross_attention",
      "model_kwargs": {
        "input_shape": [40, 768],
        "num_layers": 2,
        "num_heads": 4,
        "hidden_dim": 512,
        "dropout": 0.1,
        "target_name": "cost_target"
      }
    }
  }
  ```

### Validation
- `POST /validation`: Evaluate model performance
  
  Example request body:
  ```json
  {
    "features": [[[0.1, 0.2, ...], [...], ...], ...],
    "targets": [10.5, 15.2, ...],
    "config": {
      "model_name": "gated_cross_attention",
      "model_kwargs": {
        "input_shape": [40, 768],
        "num_layers": 2,
        "num_heads": 4,
        "hidden_dim": 512,
        "dropout": 0.1,
        "target_name": "cost_target"
      },
      "metrics": ["mse", "mae", "rmse", "r2"]
    }
  }
  ```

### Demo Endpoints
For easy testing and demonstration, the following endpoints are available:

#### Get Demo Data
- `GET /demo/forecast-input?batch_size=2`: Get randomly generated input data for forecast testing
- `GET /demo/validation-input?batch_size=5`: Get randomly generated input data (with targets) for validation testing

  These endpoints return data in the exact format needed for the main endpoints, including features, targets (for validation), and configuration.

#### Run Demo
- `POST /demo/run-forecast?batch_size=2`: Run forecast with automatically generated dummy data
- `POST /demo/run-validation?batch_size=5`: Run validation with automatically generated dummy data

  These endpoints combine data generation and execution in a single call, making it easy to test the API functionality.

## Using the Interactive Swagger UI

The easiest way to test the API is through the Swagger UI at http://localhost:8000/docs:

1. Navigate to http://localhost:8000/docs in your browser
2. You'll see all available endpoints with documentation
3. To test an endpoint:
   - Click on the endpoint you want to try
   - Click the "Try it out" button
   - Fill in any parameters (for demo endpoints) or request body (for main endpoints)
   - Click "Execute"
   - Scroll down to see the response with model predictions or metrics

For demo endpoints:
- Use `/demo/run-forecast` for the quickest test of the forecast functionality
- Use `/demo/run-validation` for the quickest test of the validation functionality

## Example Usage with curl

### Using the Demo Endpoints

1. **Get demo data and inspect it:**
   ```bash
   curl -X GET "http://localhost:8000/demo/forecast-input?batch_size=3"
   ```

2. **Run a complete forecast demo:**
   ```bash
   curl -X POST "http://localhost:8000/demo/run-forecast?batch_size=3"
   ```

3. **Run a complete validation demo:**
   ```bash
   curl -X POST "http://localhost:8000/demo/run-validation?batch_size=6"
   ```

## Troubleshooting

### Docker Issues
- If you encounter permission issues, make sure you have Docker installed correctly
- If the container fails to start, check the Docker logs with `docker-compose logs`

### API Connection Issues
- Make sure port 8000 is not being used by another application
- Check if the API is running using the `/health` endpoint

### Windows curl Issues
- On Windows, you may need to use double quotes instead of single quotes:
  ```
  curl -X POST "http://localhost:8000/demo/run-forecast"
  ```

## Stopping the API

- If running locally: Press Ctrl+C in the terminal
- If running with Docker: Press Ctrl+C in the terminal or run `docker-compose down` in the api directory 
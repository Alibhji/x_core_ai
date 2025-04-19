import requests
import numpy as np
import json
import time

# Define API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("\nTesting Health Endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_forecast():
    """Test forecast endpoint"""
    print("\nTesting Forecast Endpoint...")
    
    # Create dummy input data with sequence length 40 and feature dimension 768
    sequence_length = 40
    feature_dim = 768
    batch_size = 2
    
    # Generate random features
    features = np.random.randn(batch_size, sequence_length, feature_dim).tolist()
    
    # Prepare request payload
    payload = {
        "features": features,
        "config": {
            "project_name": "api_test",
            "model_name": "gated_cross_attention",
            "model_kwargs": {
                "input_shape": [sequence_length, feature_dim],
                "num_layers": 2,
                "num_heads": 4,
                "hidden_dim": 512,
                "dropout": 0.1
            }
        }
    }
    
    # Make request
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/forecast", json=payload)
    elapsed = time.time() - start_time
    
    print(f"Status code: {response.status_code}")
    print(f"Elapsed time: {elapsed:.2f} seconds")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Predictions count: {len(result['predictions'])}")
        print(f"First 3 predictions: {result['predictions'][:3]}")
        print(f"Metadata: {result['metadata']}")
        return True
    else:
        print(f"Error: {response.text}")
        return False

def test_validation():
    """Test validation endpoint"""
    print("\nTesting Validation Endpoint...")
    
    # Create dummy input data
    sequence_length = 40
    feature_dim = 768
    batch_size = 10
    
    # Generate random features and targets
    features = np.random.randn(batch_size, sequence_length, feature_dim).tolist()
    targets = np.random.uniform(0, 100, batch_size).tolist()
    
    # Prepare request payload
    payload = {
        "features": features,
        "targets": targets,
        "config": {
            "project_name": "api_test",
            "model_name": "gated_cross_attention",
            "model_kwargs": {
                "input_shape": [sequence_length, feature_dim],
                "num_layers": 2,
                "num_heads": 4,
                "hidden_dim": 512,
                "dropout": 0.1,
                "target_name": "cost_target"
            },
            "metrics": ["mse", "mae", "rmse", "r2"]
        }
    }
    
    # Make request
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/validation", json=payload)
    elapsed = time.time() - start_time
    
    print(f"Status code: {response.status_code}")
    print(f"Elapsed time: {elapsed:.2f} seconds")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Metrics: {result['metrics']}")
        print(f"Metadata: {result['metadata']}")
        return True
    else:
        print(f"Error: {response.text}")
        return False

if __name__ == "__main__":
    print("=== X-Core AI API Test ===")
    
    # Wait for server to start
    print("Waiting for server to be available...")
    max_retries = 5
    for i in range(max_retries):
        try:
            health_response = requests.get(f"{BASE_URL}/health", timeout=2)
            if health_response.status_code == 200:
                print("Server is up and running!")
                break
        except requests.exceptions.RequestException:
            print(f"Attempt {i+1}/{max_retries}: Server not available yet...")
            if i < max_retries - 1:
                time.sleep(2)
    
    # Run tests
    health_ok = test_health()
    forecast_ok = test_forecast()
    validation_ok = test_validation()
    
    # Print summary
    print("\n=== Test Results ===")
    print(f"Health Check: {'✓' if health_ok else '✗'}")
    print(f"Forecast: {'✓' if forecast_ok else '✗'}")
    print(f"Validation: {'✓' if validation_ok else '✗'}")
    
    if health_ok and forecast_ok and validation_ok:
        print("\nAll tests passed successfully!")
    else:
        print("\nSome tests failed. Please check the logs for details.") 
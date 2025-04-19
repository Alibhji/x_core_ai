from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
import os
import sys
import logging
from typing import List, Dict, Any, Optional, Union

# Add parent directory to path to import our modules
paths = [r'C:\Users\alibh\Desktop\projects\python', r'C:\Users\alibh\Desktop\projects\python\x_core_ai']
for path in paths:
    if path not in sys.path:
        sys.path.insert(0, path)

# Add parent directory to path to import core modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import our modules
from src.core import Core, Forecast, Validation

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="X-Core AI API", 
             description="API for X-Core AI Framework - Forecast and Validation",
             version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request and response
class ForecastRequest(BaseModel):
    features: List[List[List[float]]]  # Shape: [batch_size, sequence_length, feature_dim]
    config: Optional[Dict[str, Any]] = None

class ValidationRequest(BaseModel):
    features: List[List[List[float]]]  # Shape: [batch_size, sequence_length, feature_dim]
    targets: List[float]  # Shape: [batch_size]
    config: Optional[Dict[str, Any]] = None

class PredictionResponse(BaseModel):
    predictions: List[float]
    metadata: Dict[str, Any]

class ValidationResponse(BaseModel):
    metrics: Dict[str, float]
    metadata: Dict[str, Any]

class DummyInputResponse(BaseModel):
    features: List[List[List[float]]]
    config: Dict[str, Any]
    targets: Optional[List[float]] = None

# Default configuration
def get_default_config():
    return {
        "project_name": "api_inference",
        "distributed": False,
        "gpus": [0],
        
        # Model configuration
        "model_name": "gated_cross_attention",
        "model_kwargs": {
            "input_shape": [40, 768],  # Using consistent sequence length of 40
            "num_layers": 2,
            "num_heads": 4,
            "hidden_dim": 512,
            "dropout": 0.1,
            "target_name": "cost_target"
        },
        
        # Metrics configuration for validation
        "metrics": ["mse", "mae", "rmse", "r2"]
    }

# Generate dummy input data for demonstration purposes
def generate_dummy_input(batch_size=2, include_targets=False):
    """Generate dummy input data for demo purposes"""
    # Fixed dimensions based on consistent sequence length
    sequence_length = 40
    feature_dim = 768
    
    # Generate random features
    features = np.random.randn(batch_size, sequence_length, feature_dim).tolist()
    
    # Get default config
    config = get_default_config()
    
    if include_targets:
        # Generate random targets (values between 0 and 100 for cost targets)
        targets = np.random.uniform(0, 100, batch_size).tolist()
        return features, targets, config
    
    return features, config

@app.get("/")
def read_root():
    return {"message": "Welcome to the X-Core AI API", "status": "active"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/forecast", response_model=PredictionResponse)
def run_forecast(request: ForecastRequest):
    try:
        # Get config from request or use default
        config = request.config if request.config else get_default_config()
        
        # Convert input features to tensor
        features = torch.tensor(request.features, dtype=torch.float32)
        logger.info(f"Input features shape: {features.shape}")
        
        # Initialize forecast
        forecast = Forecast(config)
        logger.info(f"Model loaded: {forecast.model.__class__.__name__}")
        
        # Make prediction
        predictions = forecast.predict(features)
        
        # Convert predictions to list
        if isinstance(predictions, np.ndarray):
            pred_list = predictions.flatten().tolist()
        elif isinstance(predictions, torch.Tensor):
            pred_list = predictions.cpu().detach().numpy().flatten().tolist()
        else:
            pred_list = predictions
            
        return {
            "predictions": pred_list,
            "metadata": {
                "model_name": config["model_name"],
                "input_shape": features.shape,
                "output_shape": predictions.shape if hasattr(predictions, "shape") else len(pred_list)
            }
        }
    except Exception as e:
        logger.error(f"Error in forecast endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validation", response_model=ValidationResponse)
def run_validation(request: ValidationRequest):
    try:
        # Get config from request or use default
        config = request.config if request.config else get_default_config()
        
        # Convert input features and targets to tensors
        features = torch.tensor(request.features, dtype=torch.float32)
        targets = torch.tensor(request.targets, dtype=torch.float32)
        logger.info(f"Input features shape: {features.shape}, targets shape: {targets.shape}")
        
        # Create dataset for validation
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, features, targets):
                self.features = features
                self.targets = targets
                
            def __len__(self):
                return len(self.features)
                
            def __getitem__(self, idx):
                # Return in the format expected by validation
                return {
                    "photo_feat": self.features[idx],
                    "cost_target": self.targets[idx]
                }
        
        dataset = SimpleDataset(features, targets)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)
        
        # Initialize validation
        validation = Validation(config)
        logger.info(f"Model loaded: {validation.model.__class__.__name__}")
        
        # Evaluate model
        metrics = validation.evaluate(dataloader)
        
        return {
            "metrics": metrics,
            "metadata": {
                "model_name": config["model_name"],
                "metrics_calculated": list(metrics.keys()),
                "samples_evaluated": len(features)
            }
        }
    except Exception as e:
        logger.error(f"Error in validation endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/demo/forecast-input", response_model=DummyInputResponse)
def get_forecast_demo_input(batch_size: int = 2):
    """Get dummy input data for forecast demo"""
    try:
        features, config = generate_dummy_input(batch_size, include_targets=False)
        return {
            "features": features,
            "config": config
        }
    except Exception as e:
        logger.error(f"Error generating dummy forecast input: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/demo/validation-input", response_model=DummyInputResponse)
def get_validation_demo_input(batch_size: int = 5):
    """Get dummy input data for validation demo"""
    try:
        features, targets, config = generate_dummy_input(batch_size, include_targets=True)
        return {
            "features": features,
            "targets": targets,
            "config": config
        }
    except Exception as e:
        logger.error(f"Error generating dummy validation input: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/demo/run-forecast")
def demo_forecast(batch_size: int = 2):
    """Run a forecast demo with automatically generated input data"""
    try:
        # Generate dummy input
        features, config = generate_dummy_input(batch_size, include_targets=False)
        
        # Create request
        request = ForecastRequest(features=features, config=config)
        
        # Run forecast
        return run_forecast(request)
    except Exception as e:
        logger.error(f"Error in forecast demo: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/demo/run-validation")
def demo_validation(batch_size: int = 5):
    """Run a validation demo with automatically generated input data"""
    try:
        # Generate dummy input
        features, targets, config = generate_dummy_input(batch_size, include_targets=True)
        
        # Create request
        request = ValidationRequest(features=features, targets=targets, config=config)
        
        # Run validation
        return run_validation(request)
    except Exception as e:
        logger.error(f"Error in validation demo: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 
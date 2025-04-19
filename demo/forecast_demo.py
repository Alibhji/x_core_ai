import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import our modules
paths = [r'C:\Users\alibh\Desktop\projects\python', r'C:\Users\alibh\Desktop\projects\python\x_core_ai']
for path in paths:
    if path not in sys.path:
        sys.path.insert(0, path)

# Import our modules
from src.core import Forecast
from sub_module.utilx.src.config import ConfigLoader

def forecast_demo():
    """Demonstrate the Forecast class for inference"""
    print("=== Forecast Demo ===")
    
    # Set consistent sequence length and feature dimension
    sequence_length = 40
    feature_dim = 768
    
    # Create config
    config = {
        "project_name": "forecast_demo",
        "distributed": False,
        "gpus": [0],
        
        # Model configuration
        "model_name": "gated_cross_attention",
        "model_kwargs": {
            "input_shape": [sequence_length, feature_dim],
            "num_layers": 2,
            "num_heads": 4,
            "hidden_dim": 512,
            "dropout": 0.1,
            "target_name": "cost_target"
        },
    }
    
    # Initialize forecast class
    forecast = Forecast(config)
    print(f"Initialized forecast with model: {forecast.model.__class__.__name__}")
    
    # Generate dummy test data
    num_samples = 10
    seq_length = config['model_kwargs']['input_shape'][0]
    feat_dim = config['model_kwargs']['input_shape'][1]
    
    test_data = torch.randn(num_samples, seq_length, feat_dim)
    
    # Run predictions
    print(f"\nGenerating predictions for {num_samples} samples...")
    predictions = forecast.predict(test_data)
    
    # Display results
    print("\nPrediction Results:")
    print(f"Shape: {predictions.shape}")
    print(f"Range: {predictions.min():.4f} to {predictions.max():.4f}")
    print(f"Mean: {predictions.mean():.4f}")
    print(f"Std Dev: {predictions.std():.4f}")
    
    # Create a simple visualization
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(predictions)), predictions.flatten())
    plt.title('Model Predictions')
    plt.xlabel('Sample Index')
    plt.ylabel('Predicted Value')
    plt.grid(True, alpha=0.3)
    plt.savefig('forecast_predictions.png')
    print("\nVisualization saved to 'forecast_predictions.png'")
    
    # Demonstrate batch prediction
    print("\nDemonstrating batch prediction...")
    
    # Create a simple dataset and dataloader
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, features):
            self.features = features
            
        def __len__(self):
            return len(self.features)
            
        def __getitem__(self, idx):
            return {'features': self.features[idx]}
    
    dataset = SimpleDataset(test_data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    
    # Run batch prediction
    batch_predictions, _ = forecast.batch_predict(dataloader)
    print(f"Batch predictions shape: {batch_predictions.shape}")
    
    print("\n=== Forecast Demo Completed ===")
    
if __name__ == "__main__":
    forecast_demo() 
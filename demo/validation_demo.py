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
from src.core import Validation
from sub_module.utilx.src.config import ConfigLoader

def validation_demo():
    """Demonstrate the Validation class for model evaluation"""
    print("=== Validation Demo ===")
    
    # Set consistent sequence length and feature dimension
    sequence_length = 40
    feature_dim = 768
    
    # Create config
    config = {
        "project_name": "validation_demo",
        "distributed": False,
        "gpus": [0],
        
        # Data configuration
        "data_name": "dummy_dataframe",
        "data_kwargs": {
            "photo_feature_sequence_length": sequence_length,
            "photo_feature_dim": feature_dim,
            "target_name": "cost_target",
            "target_type": "regression",
            "target_range": [0, 100],
            "number_of_samples": 1000
        },
        
        # Dataset configuration
        "dataset_name": "photo_feature_dataset",
        "dataset_kwargs": {
            "photo_feature_sequence_length": sequence_length,
            "photo_feature_dim": feature_dim,
            "target_name": "cost_target"
        },
        
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
        
        # Validation configuration
        "metrics": ["mse", "mae", "rmse", "r2"],
        
        # Dataloader configuration
        "dataloader_kwargs_val": {
            "batch_size": 32,
            "shuffle": False,
            "num_workers": 0
        }
    }
    
    # Initialize validation class
    validation = Validation(config)
    print(f"Initialized validation with model: {validation.model.__class__.__name__}")
    
    # Get validation dataloader
    try:
        dataloader = validation.get_dataloader()
        print(f"Created validation dataloader with {len(dataloader)} batches")
        
        # Evaluate model
        print("\nEvaluating model performance...")
        metrics = validation.evaluate(dataloader)
        
        # Display results
        print("\nEvaluation Results:")
        for metric_name, value in metrics.items():
            print(f"{metric_name.upper()}: {value:.6f}")
        
        # Create visualization of metrics
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics.keys(), metrics.values())
        
        # Add text labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.4f}', ha='center', va='bottom', rotation=0)
            
        plt.title('Model Evaluation Metrics')
        plt.ylabel('Metric Value')
        plt.grid(True, alpha=0.3, axis='y')
        plt.savefig('validation_metrics.png')
        print("\nMetrics visualization saved to 'validation_metrics.png'")
        
    except Exception as e:
        print(f"Error in validation: {e}")
    
    # Demonstrate custom metrics calculation
    print("\nDemonstrating custom metrics calculation...")
    
    # Create dummy predictions and targets
    predictions = np.random.rand(100) * 100
    targets = np.random.rand(100) * 100
    
    # Calculate metrics manually
    custom_metrics = validation.calculate_metrics(targets, predictions)
    print("\nCustom Metrics Results:")
    for metric_name, value in custom_metrics.items():
        print(f"{metric_name.upper()}: {value:.6f}")
    
    print("\n=== Validation Demo Completed ===")
    
if __name__ == "__main__":
    validation_demo() 
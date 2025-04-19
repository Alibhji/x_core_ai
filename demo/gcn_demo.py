import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import Core

def gcn_demo():
    """Demonstrate the Gated Cross-Attention Network (GCAN) model"""
    print("=== GCAN Demo ===")
    
    # Set consistent sequence length and feature dimension
    sequence_length = 40
    feature_dim = 768
    
    # Create config
    config = {
        "project_name": "gcn_demo",
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
    
    # Initialize Core class
    core = Core(config)
    print(f"Device: {core.device}")
    
    # Generate model
    core.model_generator()
    core.model_to_device()
    print(f"Model created: {core.model.__class__.__name__}")
    
    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, sequence_length, feature_dim)
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    print("\nPerforming forward pass...")
    with torch.no_grad():
        output = core.model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: {output.min().item():.4f} to {output.max().item():.4f}")
    
    print("\n=== GCAN Demo Completed ===")

if __name__ == "__main__":
    gcn_demo()

import sys
import os
import torch
import numpy as np
import time
import matplotlib.pyplot as plt

# Add parent directory to path to import our modules
paths = [r'C:\Users\alibh\Desktop\projects\python', r'C:\Users\alibh\Desktop\projects\python\x_core_ai']
for path in paths:
    if path not in sys.path:
        sys.path.insert(0, path)
# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from src.core import Core, Forecast, Validation, Training
from src.utils import get_logger

def create_dummy_config(enable_logging=True, log_level="INFO"):
    """Create a dummy configuration for testing with logging options"""
    # Use same sequence length (40) for both model and dataset
    sequence_length = 40
    feature_dim = 768
    
    config = {
        "project_name": "logging_demo",
        "distributed": False,
        "gpus": [0],
        
        # Logging configuration
        "logging": {
            "enable_logging": enable_logging,
            "log_to_file": True,
            "log_to_console": True,
            "log_level": log_level,
            "log_dir": "logs/demo",
            "log_filename": f"xcore_demo_{int(time.time())}.log"
        },
        
        # Data configuration
        "data_name": "dummy_dataframe",
        "dataframe_kwargs": {
            "photo_feature_sequence_length": sequence_length,
            "photo_feature_dim": feature_dim,
            "target_name": "cost_target",
            "target_type": "regression",
            "target_range": [0, 100],
            "number_of_samples": 500
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
        
        # Training configuration
        "epochs": 3,
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "loss": "mse",
        "optimizer": "adam",
        "scheduler": "cosine",
        "metrics": ["mse", "mae", "rmse"],
        "monitor_metric": "mse",
        "save_dir": "checkpoints/demo",
        "save_every": 1,
        "early_stopping_kwargs": {
            "patience": 3
        },
        
        # Dataloader configuration
        "dataloader_kwargs_train": {
            "batch_size": 16,
            "shuffle": True,
            "num_workers": 0
        },
        "dataloader_kwargs_val": {
            "batch_size": 32,
            "shuffle": False,
            "num_workers": 0
        }
    }
    
    return config

def demo_with_logging_enabled():
    """Run demonstration with logging enabled"""
    print("\n===== Demo with Logging Enabled =====")
    
    # Create config with logging enabled
    config = create_dummy_config(enable_logging=True, log_level="INFO")
    
    # Initialize logger
    logger = get_logger(config.get("logging", {}))
    logger.info("Starting demo with logging enabled")
    
    # Log the configuration
    logger.log_config(config)
    
    # Initialize Core class
    logger.info("Initializing Core module")
    core = Core(config)
    logger.info(f"Device: {core.device}")
    
    # Generate model
    logger.info("Generating model")
    core.model_generator()
    core.model_to_device()
    logger.info(f"Model created: {core.model.__class__.__name__}")
    
    # Log model summary
    logger.log_model_summary(core.model)
    
    # Run a short training session
    logger.info("Initializing Training module")
    training = Training(config)
    
    # Get dataloaders
    logger.info("Creating dataloaders")
    train_dataloader = training.get_train_dataloader()
    val_dataloader = training.get_dataloader()
    logger.info(f"Created training dataloader with {len(train_dataloader)} batches")
    
    # Train for a few epochs
    logger.info("Starting training")
    start_time = time.time()
    history = training.train(train_dataloader, val_dataloader)
    elapsed = time.time() - start_time
    
    # Log training results
    logger.info(f"Training completed in {elapsed:.2f} seconds")
    logger.info(f"Final training loss: {history['train_loss'][-1]:.6f}")
    
    # Log validation metrics
    final_metrics = history['val_metrics'][-1]
    logger.log_metrics(final_metrics, prefix="Final Validation")
    
    # Make a prediction with the trained model
    logger.info("Testing inference with trained model")
    forecast = Forecast(config)
    dummy_input = torch.randn(1, config['model_kwargs']['input_shape'][0], 
                             config['model_kwargs']['input_shape'][1])
    prediction = forecast.predict(dummy_input)
    logger.info(f"Prediction shape: {prediction.shape}")
    
    logger.info("Demo with logging enabled completed successfully")
    return logger.config['log_dir'], logger.config['log_filename']

def demo_with_logging_disabled():
    """Run demonstration with logging disabled"""
    print("\n===== Demo with Logging Disabled =====")
    
    # Create config with logging disabled
    config = create_dummy_config(enable_logging=False)
    
    # Initialize logger (which will be disabled)
    logger = get_logger(config.get("logging", {}))
    logger.info("This message should not appear anywhere")
    
    # Initialize Core class (no logs will be generated)
    core = Core(config)
    print(f"Device: {core.device}")
    
    # Generate model
    core.model_generator()
    core.model_to_device()
    print(f"Model created: {core.model.__class__.__name__}")
    
    # Run a very short training session
    training = Training(config)
    
    # Get dataloaders
    train_dataloader = training.get_train_dataloader()
    val_dataloader = training.get_dataloader()
    
    # Train for 1 epoch only (since we're just demonstrating disabled logging)
    config["epochs"] = 1
    history = training.train(train_dataloader, val_dataloader)
    
    print("Demo with logging disabled completed")
    print("No log file was generated")

def demo_debug_logging():
    """Run demonstration with DEBUG level logging"""
    print("\n===== Demo with DEBUG Level Logging =====")
    
    # Create config with debug logging enabled
    config = create_dummy_config(enable_logging=True, log_level="DEBUG")
    
    # Initialize logger
    logger = get_logger(config.get("logging", {}))
    logger.info("Starting demo with DEBUG level logging")
    logger.debug("This is a debug message that will be visible")
    
    # Initialize a simple model and log detailed information
    logger.debug("Creating a very simple model for demonstration")
    simple_model = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 1)
    )
    
    # Log detailed model info
    logger.log_model_summary(simple_model)
    
    # Generate random data and log tensor shapes
    logger.debug("Generating random tensor data")
    x = torch.randn(8, 10)
    logger.debug(f"Input tensor shape: {x.shape}, dtype: {x.dtype}")
    
    # Do a forward pass
    logger.debug("Performing forward pass")
    with torch.no_grad():
        y = simple_model(x)
    
    logger.debug(f"Output tensor shape: {y.shape}, range: {y.min().item():.4f} to {y.max().item():.4f}")
    logger.info("Demo with DEBUG level logging completed")
    
    return logger.config['log_dir'], logger.config['log_filename']

def main():
    """Main demo function"""
    print("=== X-Core AI Framework Advanced Logging Demo ===")
    
    # Run demo with logging enabled
    log_dir, log_filename = demo_with_logging_enabled()
    print(f"Log file created at: {os.path.join(log_dir, log_filename)}")
    
    # Run demo with DEBUG level logging
    debug_log_dir, debug_log_filename = demo_debug_logging()
    print(f"Debug log file created at: {os.path.join(debug_log_dir, debug_log_filename)}")
    
    # Run demo with logging disabled
    demo_with_logging_disabled()
    
    print("\n=== Logging Demo Completed Successfully ===")
    print("\nTry examining the log files to see the difference between INFO and DEBUG level logging!")

if __name__ == "__main__":
    main() 
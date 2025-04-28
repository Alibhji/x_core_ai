import sys
import torch
import argparse
import os

# Add parent directory to path to import our modules
paths = [r'C:\Users\alibh\Desktop\projects\python', r'C:\Users\alibh\Desktop\projects\python\x_core_ai']
for path in paths:
    if path not in sys.path:
        sys.path.insert(0, path)

# Import our modules
from src.core import Core, Forecast, Validation, Training
from sub_module.utilx.src.config import ConfigLoader



def demo_core(config):
    """Demonstrate Core functionality"""
    print("\n===== Core Demo =====")
    # Initialize Core class
    core = Core(config)
    print(f"Device: {core.device}")
    print(f"Root package path: {core.root_package_path}")
    
    # Generate model
    core.model_generator()
    core.model_to_device()
    print(f"Model created: {core.model.__class__.__name__}")
    
    return core

def demo_forecast(config):
    """Demonstrate Forecast functionality"""
    print("\n===== Forecast Demo =====")
    # Initialize Forecast class
    forecast = Forecast(config)
    
    # Create dummy input
    dummy_input = torch.randn(1, 
                             config['data_kwargs']['photo_feature_sequence_length'],
                             config['data_kwargs']['photo_feature_dim'])
    
    # Make prediction
    print("Making single prediction...")
    prediction = forecast.predict(dummy_input)
    print(f"Prediction shape: {prediction.shape}, Range: {prediction.min():.4f} to {prediction.max():.4f}")
    
    return forecast

def demo_validation(config):
    """Demonstrate Validation functionality"""
    print("\n===== Validation Demo =====")
    # Initialize Validation class
    validation = Validation(config)
    
    # Get validation dataloader
    dataloader = validation.get_dataloader()
    print(f"Created validation dataloader with {len(dataloader)} batches")
    
    # Evaluate model
    print("Evaluating model...")
    metrics = validation.evaluate(dataloader)
    print("Evaluation metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.6f}")
    
    return validation

def demo_training(config):
    """Demonstrate Training functionality"""
    print("\n===== Training Demo =====")
    # Initialize Training class (MLflow integration is built-in now)
    training = Training(config)
    
    # Display training setup
    print(f"Optimizer: {training.optimizer.__class__.__name__}")
    print(f"Loss function: {training.loss_fn.__class__.__name__}")
    if training.scheduler:
        print(f"Scheduler: {training.scheduler.__class__.__name__}")
    
    # Check if MLflow is enabled
    if hasattr(training, 'tracker') and training.tracker:
        print("MLflow experiment tracking is enabled")
    
    # Get dataloaders
    train_dataloader = training.get_train_dataloader()
    val_dataloader = training.get_dataloader()
    print(f"Created training dataloader with {len(train_dataloader)} batches")
    
    # Train for a few epochs
    print("Starting training...")
    history = training.train(train_dataloader, val_dataloader)
    
    # Save training visualization using the new method
    vis_path = training.save_training_visualization(history)
    print(f"Training visualization saved to: {vis_path}")
    
    return training


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='X-Core AI Training Demo')
    parser.add_argument('--config', type=str, default="configs/demo/dummy_train.yaml", 
                       help='Path to configuration file')
    parser.add_argument('--mlflow', action='store_true', help='Enable MLflow experiment tracking')
    parser.add_argument('--open-ui', action='store_true', help='Open MLflow UI after training')
    return parser.parse_args()

def open_mlflow_ui():
    """Open MLflow UI"""
    try:
        import subprocess
        import webbrowser
        import time
        import mlflow
        
        # Get tracking URI
        tracking_uri = mlflow.get_tracking_uri()
        print(f"\nOpening MLflow UI at {tracking_uri}")
        
        # Start MLflow UI server
        mlflow_cmd = [sys.executable, "-m", "mlflow", "ui", "--backend-store-uri", tracking_uri]
        print(f"Running command: {' '.join(mlflow_cmd)}")
        
        # Start server as subprocess
        server_process = subprocess.Popen(
            mlflow_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to start
        time.sleep(2)
        
        # Check if process is running
        if server_process.poll() is None:
            # Open browser
            webbrowser.open("http://localhost:5000")
            
            print("\nMLflow UI is now running. Press Ctrl+C to stop.")
            print("You can access it at http://localhost:5000")
            
            # Keep running until interrupted
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping MLflow UI...")
                server_process.terminate()
        else:
            # If process exited, print error
            stdout, stderr = server_process.communicate()
            print(f"MLflow UI failed to start: {stderr}")
    except Exception as e:
        print(f"Error launching MLflow UI: {e}")
        print(f"You can manually start the UI with: mlflow ui --backend-store-uri {mlflow.get_tracking_uri()}")

def main():
    """Main demo function"""
    print("=== X-Core AI Framework Demo ===")
    
    # Parse command line arguments
    args = parse_args()
    
    # Create dummy config
    config = ConfigLoader.load_config(args.config)
    print("Created dummy configuration")
    
    # Update config based on command line args
    if args.mlflow and "experiment_tracking" in config:
        if args.open_ui:
            config["experiment_tracking"]["open_ui"] = True
    
    # Demo each class
    core = demo_core(config)
    forecast = demo_forecast(config)
    validation = demo_validation(config)
    training = demo_training(config)
    
    print("\n=== Demo Completed Successfully ===")
    
    # Open MLflow UI if configured
    if (args.mlflow and args.open_ui and 
        "experiment_tracking" in config and 
        config["experiment_tracking"].get("open_ui", False)):
        open_mlflow_ui()

if __name__ == "__main__":
    main() 
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse
from pathlib import Path

paths = [r'C:\Users\alibh\Desktop\projects\python', r'C:\Users\alibh\Desktop\projects\python\x_core_ai']
for path in paths:
    if path not in sys.path:
        sys.path.insert(0, path)


from src.core import Core
from sub_module.utilx.src.config import ConfigLoader

# Try to import MLflow tracking
try:
    from src.experiment_tracker import create_experiment, integrate_tracker_with_core
    _has_mlflow = True
except ImportError:
    _has_mlflow = False
    print("MLflow tracking not available. Continuing without experiment tracking.")

def load_yaml_config(config_path):
    """Load YAML configuration file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        return None

def resolve_config_path(config_file):
    """Resolve config file path, handling both relative and absolute paths"""
    if os.path.isabs(config_file):
        # If it's an absolute path, use it directly
        if os.path.exists(config_file):
            return config_file
        else:
            print(f"Warning: Absolute path {config_file} does not exist")
            return None

def gcn_demo(config_file=None, enable_mlflow=False, open_ui=False):
    """Demonstrate the Gated Cross-Attention Network (GCAN) model"""
    print("=== GCAN Demo ===")
    
    # Load configuration from YAML file
    if config_file is None:
        config_file = "gcan_v1.0.0.yaml"
    
    # Resolve the config path
    config_path = resolve_config_path(config_file)
    if not config_path:
        print(f"Error: Could not locate config file {config_file}")
        return
    
    # Load configuration using ConfigLoader if available, otherwise use simple YAML loading
    try:
        config_loader = ConfigLoader()
        config = config_loader.load(config_path)
        print(f"Loaded configuration using ConfigLoader from {config_path}")
    except Exception as e:
        print(f"Falling back to simple YAML loading: {e}")
        config = load_yaml_config(config_path)
        
    if not config:
        print(f"Failed to load configuration from {config_path}")
        return
    
    # Add MLflow configuration if enabled
    if enable_mlflow and _has_mlflow:
        # Only add if not already present
        if "experiment_tracking" not in config:
            config["experiment_tracking"] = {
                "enable": True,
                "project_name": config.get("project_name", "gcn_demo"),
                "version": config.get("version", "v1.0"),
                "open_ui": open_ui,
                "log_artifacts": True,
                "log_model": True,
                "run_name": "gcn_demo_run"
            }
        else:
            # Update existing config
            config["experiment_tracking"]["enable"] = True
            if open_ui:
                config["experiment_tracking"]["open_ui"] = True
    
    # Extract key configuration parameters for the demo
    sequence_length = config.get("model_kwargs", {}).get("input_shape", [40, 768])[0]
    feature_dim = config.get("model_kwargs", {}).get("input_shape", [40, 768])[1]
    
    print(f"Using sequence length: {sequence_length}, feature dim: {feature_dim}")
    
    # Initialize Core class
    core = Core(config)
    print(f"Device: {core.device}")
    
    # Check if we need to set up transformers cache
    try:
        from transformers import AutoModel, AutoConfig
        import os
        
        # Set cache directory for transformers
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "storage",
            "transformers_cache"
        )
        print(f"Set transformers cache to: {os.environ.get('TRANSFORMERS_CACHE')}")
        
        # Check if we're using a BERT-based model
        if "bert" in config.get("model_name", "").lower() or "gated_cross_attention" in config.get("model_name", "").lower():
            print("BERT-based model detected, checking initialization...")
            # You might need to initialize the BERT model explicitly here if it's causing issues
    except ImportError:
        print("Transformers library not available, continuing without BERT support")
    
    # Setup MLflow tracking if enabled
    tracker = None
    if enable_mlflow and _has_mlflow:
        tracker = integrate_tracker_with_core(core)
        if tracker:
            tracker.start_run("gcn_demo_run")
            # Log model parameters
            if "model_kwargs" in config:
                tracker.log_params(config["model_kwargs"])
    
    # Generate model
    core.model_generator()
    core.model_to_device()
    print(f"Model created: {core.model.__class__.__name__}")
    
    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, sequence_length, feature_dim)
    print(f"Input shape: {dummy_input.shape}")
    
    # Move input to the same device as the model
    device = next(core.model.parameters()).device
    dummy_input = dummy_input.to(device)
    print(f"Input moved to device: {device}")
    
    # Forward pass
    print("\nPerforming forward pass...")
    try:
        with torch.no_grad():
            # Check if this is a GCAN model which might need special input handling
            model_name = config.get("model_name", "").lower()
            if "gated_cross_attention" in model_name:
                # GCAN might expect inputs in a specific format
                print("Using GCAN-specific input format")
                
                # Check if we need attention mask
                attention_mask = torch.ones(batch_size, sequence_length).to(device)
                
                # Try different input formats if the default one fails
                try:
                    # First try with photo_feat as the only input
                    output = core.model(photo_feat=dummy_input)
                except Exception as first_error:
                    print(f"First attempt failed: {first_error}. Trying alternative input format...")
                    try:
                        # Try with additional attention mask
                        output = core.model(photo_feat=dummy_input, attention_mask=attention_mask)
                    except Exception as second_error:
                        print(f"Second attempt failed: {second_error}. Trying standard format...")
                        # Fall back to standard format
                        output = core.model(dummy_input)
            else:
                # Standard forward pass
                output = core.model(dummy_input)
        
        # Handle output which might be a dictionary for GCAN
        if isinstance(output, dict):
            target_name = config.get("model_kwargs", {}).get("target_name", "cost_target")
            if target_name in output:
                output = output[target_name]
            else:
                output = next(iter(output.values()))
        
        print(f"Output shape: {output.shape}")
        print(f"Output range: {output.min().item():.4f} to {output.max().item():.4f}")
        
        # Create and save visualization
        if enable_mlflow and tracker:
            plt.figure(figsize=(10, 6))
            
            # Move tensors to CPU for visualization
            input_vis = dummy_input[0].detach().cpu().numpy()
            output_vis = output[0].detach().cpu().numpy()
            
            plt.subplot(1, 2, 1)
            plt.imshow(input_vis[:10, :10])
            plt.colorbar()
            plt.title("Input Sample (first 10x10)")
            
            plt.subplot(1, 2, 2)
            plt.plot(output_vis)
            plt.title("Model Output")
            plt.tight_layout()
            
            # Save figure
            output_path = "gcn_demo_output.png"
            plt.savefig(output_path)
            print(f"Saved visualization to {output_path}")
            
            # Log to MLflow
            tracker.log_artifact(output_path)
            
            # Log any error information
            tracker.log_params({
                "forward_pass": "success"
            })
            
            # End MLflow run
            tracker.end_run()
    except Exception as e:
        print(f"Error during forward pass: {e}")
        if enable_mlflow and tracker:
            # Log the error to MLflow
            tracker.log_params({
                "forward_pass": "failed",
                "error": str(e)
            })
            tracker.end_run()
        import traceback
        traceback.print_exc()
        print("\nTips to resolve this error:")
        print("1. Check that the model configuration matches the input dimensions")
        print("2. Ensure all required model components are installed")
        print("3. Verify CUDA/CPU compatibility if using GPU")
    
    print("\n=== GCAN Demo Completed ===")
    
    # Open MLflow UI if requested
    if enable_mlflow and open_ui and _has_mlflow:
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN Demo")
    parser.add_argument("--config", type=str, help="Path to configuration file", default="gcan_v1.0.0.yaml")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow tracking")
    parser.add_argument("--open-ui", action="store_true", help="Open MLflow UI after demo")
    args = parser.parse_args()
    
    gcn_demo(args.config, args.mlflow, args.open_ui)

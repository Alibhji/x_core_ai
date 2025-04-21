import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import time
import mlflow

paths = [r'C:\Users\alibh\Desktop\projects\python', r'C:\Users\alibh\Desktop\projects\python\x_core_ai']
for path in paths:
    if path not in sys.path:
        sys.path.insert(0, path)
# Ensure src is in the path


# For Windows compatibility with the module imports
if 'x_core_ai' not in sys.modules:
    import x_core_ai

from src.experiment_tracker import create_experiment
from src.core.core_base import Core

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_dummy_config():
    """Create a dummy configuration for testing"""
    # Calculate total input dimension for the model
    seq_len = 24
    feature_dim = 10
    input_dim = seq_len * feature_dim  # Input dimension is the flattened sequence
    
    return {
        "model_name": "SimpleModel",
        "model_kwargs": {
            "input_dim": input_dim,  # Update to use total flattened dimension
            "hidden_dim": 128,       # Increased hidden dimension
            "output_dim": 1,
            "dropout": 0.1
        },
        "data_kwargs": {
            "sequence_length": seq_len,
            "feature_dim": feature_dim,
            "num_samples": 5000      # Increased sample count for longer training
        },
        "training_kwargs": {
            "epochs": 20,            # Increased epochs
            "batch_size": 64,
            "learning_rate": 0.001,
            "weight_decay": 1e-5
        },
        "experiment_tracking": {
            "enable": True,
            "project_name": "mlflow_training_demo",  # Changed to specific folder
            "version": "v1.0",
            "open_ui": True          # Flag to open UI
        }
    }

def create_dummy_model(config):
    """Create a simple model for demonstration"""
    class SimpleModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
            super().__init__()
            print(f"Creating model with input_dim={input_dim}, hidden_dim={hidden_dim}")
            self.input_dim = input_dim
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, output_dim)
            )
        
        def forward(self, x):
            batch_size = x.size(0)
            # Verify input dimensions
            expected_features = self.input_dim
            actual_features = x.size(1)
            if actual_features != expected_features:
                raise ValueError(f"Expected input with {expected_features} features, but got {actual_features} features")
            return self.model(x)
    
    # Register the model
    setattr(sys.modules['x_core_ai.src.models.pool_auto.SimpleModel'], 
            'SimpleModel', SimpleModel)
    
    input_dim = config["model_kwargs"]["input_dim"]
    hidden_dim = config["model_kwargs"]["hidden_dim"]
    output_dim = config["model_kwargs"]["output_dim"]
    dropout = config["model_kwargs"]["dropout"]
    
    return SimpleModel(input_dim, hidden_dim, output_dim, dropout)

def create_dummy_data(config):
    """Create dummy data for training"""
    seq_len = config["data_kwargs"]["sequence_length"]
    feat_dim = config["data_kwargs"]["feature_dim"]
    input_dim = config["model_kwargs"]["input_dim"]  # The expected input dimension for the model
    num_samples = config["data_kwargs"]["num_samples"]
    
    # Validate dimensions
    if seq_len * feat_dim != input_dim:
        print(f"WARNING: Data dimensions ({seq_len}*{feat_dim}={seq_len*feat_dim}) do not match model input dim ({input_dim})")
    
    print(f"Creating data: {num_samples} samples with shape ({seq_len}, {feat_dim})")
    
    try:
        # Generate random input features
        X = torch.randn(num_samples, seq_len, feat_dim)
        
        # Generate target values (simple function of the input)
        y = torch.sum(X[:, :, 0:2], dim=(1, 2)).unsqueeze(1) / 10
        
        # Add some noise
        y += torch.randn_like(y) * 0.1
        
        # Reshape X for the simple model - flatten the sequence
        X_flat = X.reshape(num_samples, -1)
        print(f"Flattened features shape: {X_flat.shape}")
        
        # Split into train/val/test
        train_size = int(0.7 * num_samples)
        val_size = int(0.15 * num_samples)
        
        X_train = X_flat[:train_size]
        y_train = y[:train_size]
        
        X_val = X_flat[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        
        X_test = X_flat[train_size+val_size:]
        y_test = y[train_size+val_size:]
        
        # Create datasets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["training_kwargs"]["batch_size"],
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["training_kwargs"]["batch_size"],
            shuffle=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config["training_kwargs"]["batch_size"],
            shuffle=False
        )
        
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        print(f"Error creating data: {e}")
        raise

def train_with_mlflow(model, config, train_loader, val_loader, tracker=None):
    """Train the model while tracking metrics with MLflow"""
    # Setup training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training_kwargs"]["learning_rate"],
        weight_decay=config["training_kwargs"]["weight_decay"]
    )
    
    # Log model parameters if tracking is enabled
    if tracker and tracker.enabled:
        tracker.log_params(config["model_kwargs"])
        tracker.log_params(config["training_kwargs"])
    
    # Log model architecture summary
    print("\nModel Input Dimension:", config["model_kwargs"]["input_dim"])
    print("Data Shape (seq_len * feat_dim):", 
          config["data_kwargs"]["sequence_length"] * config["data_kwargs"]["feature_dim"])
    
    # Training loop
    epochs = config["training_kwargs"]["epochs"]
    start_time = time.time()
    min_train_time = 120  # Minimum training time in seconds (2 minutes)
    
    try:
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                # Debug first batch shape
                if epoch == 0 and batch_idx == 0:
                    print(f"Batch input shape: {inputs.shape}")
                
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Add a small delay every few batches to extend training time
                if batch_idx % 5 == 0:
                    time.sleep(0.01)  # 10ms delay
            
            train_loss /= len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Calculate elapsed time for this epoch
            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time
            remaining_time = min_train_time - total_time
            
            # Print metrics with timing information
            print(f"Epoch {epoch+1}/{epochs}: "
                f"Train Loss = {train_loss:.6f}, "
                f"Val Loss = {val_loss:.6f}, "
                f"Time: {epoch_time:.2f}s, "
                f"Total: {total_time:.2f}s")
            
            # Log metrics if tracking is enabled
            if tracker and tracker.enabled:
                metrics = {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "epoch_time": epoch_time
                }
                tracker.log_metrics(metrics, step=epoch)
            
            # Add a delay between epochs to ensure we meet the minimum training time
            if remaining_time > 0 and epoch < epochs - 1:
                # Calculate delay needed to reach min_train_time by the end of training
                epochs_left = epochs - epoch - 1
                if epochs_left > 0:
                    delay_per_epoch = min(2.0, remaining_time / epochs_left)
                    if delay_per_epoch > 0:
                        print(f"Adding delay of {delay_per_epoch:.2f}s to reach minimum training time")
                        time.sleep(delay_per_epoch)
    
        # Check if we've met the minimum training time
        total_time = time.time() - start_time
        if total_time < min_train_time:
            remaining = min_train_time - total_time
            print(f"Adding final delay of {remaining:.2f}s to reach minimum training time of {min_train_time}s")
            time.sleep(remaining)
        
        print(f"\nTotal training time: {time.time() - start_time:.2f} seconds")
    
    except Exception as e:
        print(f"\nError during training: {e}")
        print("Model training failed - check dimensions and data preparation")
        if hasattr(e, "__traceback__"):
            import traceback
            traceback.print_tb(e.__traceback__)
    
    return model

def evaluate_model(model, test_loader):
    """Evaluate the model on the test set"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    criterion = nn.MSELoss()
    test_loss = 0.0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    test_loss /= len(test_loader)
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Calculate MSE
    mse = np.mean((all_preds - all_targets) ** 2)
    
    print(f"\nTest Results - MSE: {mse:.6f}")
    
    return {
        "test_loss": test_loss,
        "test_mse": mse
    }

def run_mlflow_demo():
    """Run the MLflow training demo"""
    print("\n" + "="*50)
    print("MLflow Training Demo")
    print("="*50)
    
    # Create dummy configuration
    config = create_dummy_config()
    print("\nCreated dummy configuration")
    
    # Setup MLflow tracking
    tracking_enabled = config["experiment_tracking"]["enable"]
    mlflow_tracking_uri = None
    
    try:
        tracker = create_experiment(
            config["experiment_tracking"]["project_name"],
            config["experiment_tracking"]["version"],
            tracking_enabled
        )
        
        # Store the MLflow tracking URI for later
        if tracking_enabled and hasattr(tracker, 'mlflow_uri'):
            mlflow_tracking_uri = tracker.mlflow_uri
        
        print("\nExperiment tracking setup complete")
        print(f"Tracking enabled: {tracking_enabled}")
        if tracking_enabled and tracker.enabled:
            print(f"Project: {config['experiment_tracking']['project_name']}")
            print(f"Version: {config['experiment_tracking']['version']}")
    except Exception as e:
        print(f"\nError setting up MLflow: {e}")
        print("Continuing without experiment tracking")
        tracking_enabled = False
        # Create a dummy tracker that doesn't do anything
        class DummyTracker:
            def __init__(self):
                self.enabled = False
                self.active = False
            def start_run(self, run_name=None):
                print(f"[DUMMY] Started run: {run_name}")
                return self
            def end_run(self):
                print("[DUMMY] Ended run")
            def log_params(self, params):
                print(f"[DUMMY] Logged params: {list(params.keys())}")
            def log_metrics(self, metrics, step=None):
                print(f"[DUMMY] Logged metrics at step {step}: {list(metrics.keys())}")
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        tracker = DummyTracker()
    
    # Start a new run
    with tracker.start_run("training_demo"):
        print("\nStarted run")
        
        # Create model and data
        model = create_dummy_model(config)
        train_loader, val_loader, test_loader = create_dummy_data(config)
        
        print("\nCreated model and generated dummy data")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        
        # Train the model
        print("\nStarting model training...")
        model = train_with_mlflow(model, config, train_loader, val_loader, tracker)
        
        # Evaluate the model
        print("\nEvaluating model on test set...")
        metrics = evaluate_model(model, test_loader)
        
        # Log final test metrics
        if tracking_enabled:
            tracker.log_metrics(metrics)
            print("\nLogged test metrics")
    
    print("\nMLflow run complete")
    print("="*50)
    
    # Launch MLflow UI if required
    if tracking_enabled and config["experiment_tracking"].get("open_ui", False):
        try:
            import subprocess
            import sys
            import webbrowser
            import time
            
            # Get tracking URI from MLflow
            tracking_uri = mlflow.get_tracking_uri()
            print(f"\nOpening MLflow UI at {tracking_uri}")
            
            # Start MLflow UI server
            mlflow_cmd = [sys.executable, "-m", "mlflow", "ui", "--backend-store-uri", tracking_uri]
            print(f"Running command: {' '.join(mlflow_cmd)}")
            
            # Start the server as a subprocess
            server_process = subprocess.Popen(
                mlflow_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait a bit for the server to start
            time.sleep(2)
            
            # Check if the process is still running
            if server_process.poll() is None:
                # Open browser to the UI (typically at http://localhost:5000)
                webbrowser.open("http://localhost:5000")
                
                print("\nMLflow UI is now running. Press Ctrl+C to stop.")
                print("You can access it at http://localhost:5000")
                
                # Keep the script running until user interrupts
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\nStopping MLflow UI...")
                    server_process.terminate()
            else:
                # If the process exited, print the error
                stdout, stderr = server_process.communicate()
                print(f"MLflow UI failed to start: {stderr}")
                
        except Exception as e:
            print(f"Error launching MLflow UI: {e}")
            print("You can manually start the UI with: mlflow ui --backend-store-uri", tracking_uri)

if __name__ == "__main__":
    # Set random seed for reproducibility
    set_seed(42)
    
    # Make Core import work by creating a mock system module
    if not hasattr(sys.modules, 'x_core_ai.src.models.registry'):
        class MockRegistry:
            @staticmethod
            def get_model(name, *args, **kwargs):
                return getattr(sys.modules['x_core_ai.src.models.pool_auto.SimpleModel'], name)
                
        sys.modules['x_core_ai.src.models.registry'] = MockRegistry()
        sys.modules['x_core_ai.src.models.pool_auto'] = type('', (), {})()
        sys.modules['x_core_ai.src.models.pool_auto.SimpleModel'] = type('', (), {})()
    
    # Run the demo
    run_mlflow_demo() 
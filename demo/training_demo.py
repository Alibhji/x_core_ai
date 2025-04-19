import sys
import torch
import matplotlib.pyplot as plt

# Add parent directory to path to import our modules
paths = [r'C:\Users\alibh\Desktop\projects\python', r'C:\Users\alibh\Desktop\projects\python\x_core_ai']
for path in paths:
    if path not in sys.path:
        sys.path.insert(0, path)

# Import our modules
from src.core import Core, Forecast, Validation, Training
from sub_module.utilx.src.config import ConfigLoader

def create_dummy_config():
    """Create a dummy configuration for testing"""
    # Use same sequence length (40) for both model and dataset
    sequence_length = 40
    feature_dim = 768
    
    config = {
        "project_name": "training_demo",
        "distributed": False,
        "gpus": [0],
        
        # Data configuration
        "data_name": "dummy_dataframe",
        "dataframe_kwargs": {
            "photo_feature_sequence_length": sequence_length,  # Use the shared variable
            "photo_feature_dim": feature_dim,  # Use the shared variable
            "target_name": "cost_target",
            "target_type": "regression",
            "target_range": [0, 100],
            "number_of_samples": 1000
        },
        
        # Dataset configuration
        "dataset_name": "photo_feature_dataset",
        "dataset_kwargs": {
            "photo_feature_sequence_length": sequence_length,  # Add this to be explicit
            "photo_feature_dim": feature_dim,  # Add this to be explicit
            "target_name": "cost_target"
        },
        
        # Model configuration
        "model_name": "gated_cross_attention",
        "model_kwargs": {
            "input_shape": [sequence_length, feature_dim],  # Use the shared variable
            "num_layers": 2,
            "num_heads": 4,
            "hidden_dim": 512,
            "dropout": 0.1,
            "target_name": "cost_target"
        },
        
        # Training configuration
        "epochs": 5,
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "loss": "mse",
        "optimizer": "adam",
        "scheduler": "cosine",
        "metrics": ["mse", "mae", "rmse"],
        "monitor_metric": "mse",
        "save_dir": "checkpoints",
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
                             config['dataframe_kwargs']['photo_feature_sequence_length'],
                             config['dataframe_kwargs']['photo_feature_dim'])
    
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
    # Initialize Training class
    training = Training(config)
    
    # Display training setup
    print(f"Optimizer: {training.optimizer.__class__.__name__}")
    print(f"Loss function: {training.loss_fn.__class__.__name__}")
    if training.scheduler:
        print(f"Scheduler: {training.scheduler.__class__.__name__}")
    
    # Get dataloaders
    train_dataloader = training.get_train_dataloader()
    val_dataloader = training.get_dataloader()
    print(f"Created training dataloader with {len(train_dataloader)} batches")
    
    # Train for a few epochs
    print("Starting training...")
    history = training.train(train_dataloader, val_dataloader)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for metric in config['metrics']:
        plt.plot([epoch_metrics.get(metric, float('nan')) for epoch_metrics in history['val_metrics']], 
                 label=f'Val {metric}')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('Validation Metrics')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history saved to training_history.png")
    
    return training

def main():
    """Main demo function"""
    print("=== X-Core AI Framework Demo ===")
    
    # Create dummy config
    config = create_dummy_config()
    print("Created dummy configuration")
    
    # Demo each class
    core = demo_core(config)
    forecast = demo_forecast(config)
    validation = demo_validation(config)
    training = demo_training(config)
    
    print("\n=== Demo Completed Successfully ===")

if __name__ == "__main__":
    main() 
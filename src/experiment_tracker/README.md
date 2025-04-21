# MLflow Experiment Tracker

This module provides MLflow integration for tracking model training metrics, parameters, and artifacts.

## Features

- Track training and validation metrics during model training
- Log model parameters and hyperparameters
- Save artifacts (files, images, models)
- Organize experiments by project name and version
- Easy integration with Core class

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from src.experiment_tracker import create_experiment

# Create an experiment tracker
tracker = create_experiment(
    project_name="my_project",
    version="0.1.0",
    enable_tracking=True
)

# Start a run
with tracker.start_run("my_run"):
    # Log parameters
    tracker.log_params({
        "learning_rate": 0.001,
        "batch_size": 32
    })
    
    # Train your model and log metrics
    for epoch in range(10):
        # ... training code ...
        
        # Log metrics
        tracker.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss
        }, step=epoch)
```

### Integration with Core Class

```python
from src.experiment_tracker import integrate_tracker_with_core, track_training_epoch, track_model_params

# Add experiment tracking to your config
config = {
    # ... other config parameters ...
    "experiment_tracking": {
        "enable": True,
        "project_name": "my_project",
        "version": "0.1.0"
    }
}

# Create Core instance
core = Core(config)

# Initialize tracker
tracker = integrate_tracker_with_core(core)

# Start a run
with tracker.start_run("training"):
    # Log model parameters
    track_model_params(core)
    
    # Train your model and log metrics for each epoch
    for epoch in range(10):
        # ... training code ...
        
        # Track metrics
        track_training_epoch(
            core, 
            epoch, 
            train_metrics={"loss": train_loss}, 
            val_metrics={"loss": val_loss}
        )
```

### Data Storage

All experiment data is stored in the `storage/<project_name>/<version>` directory at the project root.

## Demo

See the `demo/mlflow_training_demo.py` file for a complete example of using the experiment tracker. 
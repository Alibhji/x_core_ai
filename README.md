# X-Core AI Framework

A comprehensive templating framework for training, validating, and deploying AI models with a consistent interface. 

## Overview

X-Core AI provides a standardized template for working with AI models throughout their lifecycle - from training to validation and deployment. It's designed to be modular, extensible, and easy to use with various model architectures and datasets.

Key capabilities:
- **Training**: Standardized training loops with metrics tracking
- **Validation**: Comprehensive evaluation with customizable metrics
- **Forecasting**: Simplified inference for trained models 
- **API Interface**: REST API for model deployment and integration
- **Consistent Sequence Length**: All components use a sequence length of 40 to ensure compatibility
- **Advanced Logging**: Configurable text logging system with enable/disable options

## Project Structure

```
x_core_ai/
├── api/                  # REST API for model deployment
├── checkpoints/          # Saved model checkpoints
├── configs/              # Configuration files
├── demo/                 # Demo scripts for each component
│   ├── forecast_demo.py  # Forecasting demonstration
│   ├── gcn_demo.py       # Gated Cross-Attention Network demo
│   ├── training_demo.py  # Training demonstration
│   └── validation_demo.py# Validation demonstration
├── logs/                 # Generated log files (when logging is enabled)
├── src/                  # Core source code
│   ├── core/             # Main framework components
│   │   ├── core_base.py  # Base class for all components
│   │   ├── forecast.py   # Model inference capabilities
│   │   ├── training.py   # Model training capabilities 
│   │   └── validation.py # Model evaluation capabilities
│   ├── datasets/         # Dataset implementations
│   ├── dataframes/       # Data loading and processing
│   ├── models/           # Model architectures
│   └── utils/            # Utility functions and classes
│       └── logger.py     # Advanced text logger implementation
└── sub_module/           # Supporting utilities
```

## Key Components

### Core Module

The heart of the framework is the `Core` module, which provides basic model initialization, loading, and device management.

```python
from src.core import Core

# Initialize with a configuration
core = Core(config)

# Load and prepare model
core.model_generator()
core.model_to_device()
```

### Training Module

The `Training` class provides a standardized training loop with metric tracking, checkpointing, and early stopping.

```python
from src.core import Training

# Initialize with a configuration
training = Training(config)

# Get dataloaders
train_dataloader = training.get_train_dataloader()
val_dataloader = training.get_dataloader()

# Train the model
history = training.train(train_dataloader, val_dataloader)
```

### Validation Module

The `Validation` class handles model evaluation with customizable metrics.

```python
from src.core import Validation

# Initialize with a configuration
validation = Validation(config)

# Get validation dataloader
dataloader = validation.get_dataloader()

# Evaluate model
metrics = validation.evaluate(dataloader)
```

### Forecast Module

The `Forecast` class simplifies inference with trained models.

```python
from src.core import Forecast

# Initialize with a configuration
forecast = Forecast(config)

# Make a prediction
prediction = forecast.predict(input_data)
```

### Advanced Logger

The framework includes a powerful logging system that can be enabled or disabled as needed:

```python
from src.utils import get_logger

# Get logger with default settings (enabled)
logger = get_logger()

# Log various message types
logger.info("Processing started")
logger.warning("Resource usage high")
logger.error("Failed to load file")

# Log structured data
logger.log_config(config)
logger.log_metrics(metrics, prefix="Validation")

# Disable logging with a configuration
disabled_config = {"enable_logging": False}
logger = get_logger(disabled_config)
```

Logging features:
- Enable/disable with a simple flag
- Log to console and/or file
- Multiple log levels (DEBUG, INFO, WARNING, ERROR)
- Automatic tracking of module and line numbers
- Specialized methods for logging configurations and metrics
- Timestamped log files

## API Interface

The project includes a ready-to-use REST API for model deployment:

```
cd api
python app.py
```

Or use Docker for simplified deployment:

```
cd api
docker-compose up --build
```

The API will be available at http://localhost:8000 with interactive documentation at http://localhost:8000/docs.

## Standard Configuration

The framework uses a consistent configuration format:

```python
config = {
    "project_name": "example_project",
    "distributed": False,
    "gpus": [0],
    
    # Model configuration
    "model_name": "gated_cross_attention",
    "model_kwargs": {
        "input_shape": [40, 768],  # Consistent sequence length of 40
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
    
    # Logging configuration
    "logging": {
        "enable_logging": True,
        "log_to_file": True,
        "log_to_console": True,
        "log_level": "INFO",
        "log_dir": "logs"
    }
}
```

## Getting Started

### Demo Scripts

The fastest way to understand how the framework works is through the demo scripts:

```
python demo/training_demo.py   # Demonstrates the training process
python demo/validation_demo.py # Demonstrates model evaluation
python demo/forecast_demo.py   # Demonstrates model inference
python demo/gcn_demo.py        # Demonstrates the GCAN model
```

### Logger Example

To see the logger in action:

```
python src/utils/logger_example.py
```

### API Quick Start

To quickly start the API for model deployment:

```
cd api
python app.py
```

Then navigate to http://localhost:8000/docs to explore the API.

## How to Use as a Template

This framework is designed to be used as a template for your own AI model training and inference workflows:

1. **Create your model**: Add your model architecture in `src/models/`
2. **Create your dataset**: Add your dataset handling in `src/datasets/`
3. **Configure**: Modify configuration settings for your specific needs
4. **Set up logging**: Enable or disable logging as needed
5. **Train**: Use the `Training` class to train your model
6. **Evaluate**: Use the `Validation` class to evaluate your model
7. **Deploy**: Use the API or `Forecast` class for inference

## Why Use This Template?

- **Consistency**: All components use a standardized interface and configuration
- **Modularity**: Easy to swap models, datasets, or training procedures
- **Scalability**: Designed to work with distributed training
- **Extensibility**: Simple to add custom metrics, models, or datasets
- **Deployment-Ready**: Includes a production-ready API for model serving
- **Complete Logging**: Advanced text logging with configurable options

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE) 
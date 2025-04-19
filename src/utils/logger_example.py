import sys
import os
import time
from typing import Dict, Any

# Add parent directory to path if running as script
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the logger
from src.utils import get_logger

def example_with_default_config():
    """Example using logger with default configuration."""
    print("\n=== EXAMPLE: Default Configuration ===")
    
    # Get logger with default settings
    logger = get_logger()
    
    # Log some messages
    logger.info("This is an info message")
    logger.debug("This is a debug message (won't show with default INFO level)")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Log some structured data
    config = {
        "model_name": "gated_cross_attention",
        "learning_rate": 0.001,
        "batch_size": 32,
        "model_kwargs": {
            "input_shape": [40, 768],
            "num_layers": 2
        }
    }
    logger.log_config(config)
    
    # Log metrics
    metrics = {
        "accuracy": 0.92,
        "precision": 0.89,
        "recall": 0.94,
        "f1_score": 0.91
    }
    logger.log_metrics(metrics, prefix="Validation")
    
    print(f"Log file created at: {os.path.join(logger.config['log_dir'], logger.config['log_filename'])}")

def example_with_custom_config():
    """Example using logger with custom configuration."""
    print("\n=== EXAMPLE: Custom Configuration ===")
    
    # Custom configuration
    config = {
        "enable_logging": True,
        "log_to_file": True,
        "log_to_console": True,
        "log_level": "DEBUG",  # Show debug messages
        "log_dir": "custom_logs",
        "log_filename": f"custom_log_{int(time.time())}.log"
    }
    
    # Get logger with custom settings
    logger = get_logger(config)
    
    # Log some messages
    logger.info("This is an info message with custom config")
    logger.debug("This debug message will now be visible")
    
    try:
        # Simulate an error
        result = 1 / 0
    except Exception as e:
        logger.exception("An error occurred during calculation")
    
    print(f"Custom log file created at: {os.path.join(logger.config['log_dir'], logger.config['log_filename'])}")

def example_with_disabled_logging():
    """Example with logging disabled."""
    print("\n=== EXAMPLE: Disabled Logging ===")
    
    # Configuration with logging disabled
    config = {
        "enable_logging": False
    }
    
    # Get logger with logging disabled
    logger = get_logger(config)
    
    # These logs won't be written anywhere
    logger.info("This message won't be logged")
    logger.error("This error won't be logged either")
    
    print("Logging is disabled - no messages will be saved")

def main():
    """Run all examples."""
    print("X-Core Logger Examples")
    
    # Run examples
    example_with_default_config()
    example_with_custom_config()
    example_with_disabled_logging()
    
    print("\nAll examples completed!")

if __name__ == "__main__":
    main() 
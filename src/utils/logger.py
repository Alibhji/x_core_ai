import os
import sys
import logging
import datetime
from pathlib import Path
import inspect
from typing import Optional, Dict, Any, Union

class XCoreLogger:
    """
    Advanced logger for X-Core AI Framework.
    
    Features:
    - Save logs to file with timestamp
    - Console output with colored formatting
    - Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - Enable/disable logging via configuration
    - Track module and line number for each log entry
    - Configurable log format and file path
    """
    
    # Class-level variables
    DEFAULT_LOG_DIR = "logs"
    DEFAULT_LOG_LEVEL = logging.INFO
    INSTANCE = None  # Singleton instance
    
    @classmethod
    def get_instance(cls, config: Optional[Dict[str, Any]] = None, reset: bool = False) -> 'XCoreLogger':
        """Get singleton instance of logger, create it if it doesn't exist."""
        if cls.INSTANCE is None or reset:
            cls.INSTANCE = XCoreLogger(config)
        elif config is not None:
            # Update existing instance with new config
            cls.INSTANCE.update_config(config)
        return cls.INSTANCE
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize logger with configuration.
        
        Args:
            config: Dictionary containing logger configuration
                - enable_logging: Enable/disable logging (default: True)
                - log_to_file: Enable/disable file logging (default: True)
                - log_to_console: Enable/disable console logging (default: True)
                - log_level: Logging level (default: INFO)
                - log_dir: Directory to store log files (default: "logs")
                - log_filename: Custom log filename (default: generated from date)
                - log_format: Custom log format (default: standard format)
        """
        self.logger = logging.getLogger("xcore")
        self.config = {
            "enable_logging": True,
            "log_to_file": True,
            "log_to_console": True,
            "log_level": self.DEFAULT_LOG_LEVEL,
            "log_dir": self.DEFAULT_LOG_DIR,
            "log_filename": None,
            "log_format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
        }
        
        # Update config with provided values
        if config:
            self.update_config(config)
            
        # Configure logger
        self.configure_logger()
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """Update logger configuration."""
        self.config.update(config)
        self.configure_logger()
    
    def configure_logger(self) -> None:
        """Configure logger based on current settings."""
        # Reset handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
            
        # Set log level
        self.logger.setLevel(self.config["log_level"])
        
        # If logging is disabled, stop here
        if not self.config["enable_logging"]:
            return
            
        # Create formatter
        formatter = logging.Formatter(self.config["log_format"])
        
        # Add console handler if enabled
        if self.config["log_to_console"]:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
        # Add file handler if enabled
        if self.config["log_to_file"]:
            # Create log directory if it doesn't exist
            log_dir = Path(self.config["log_dir"])
            log_dir.mkdir(exist_ok=True, parents=True)
            
            # Generate log filename if not provided
            if not self.config["log_filename"]:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                self.config["log_filename"] = f"xcore_{timestamp}.log"
                
            # Create file handler
            log_path = log_dir / self.config["log_filename"]
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            # Log the start of a new logging session
            self.logger.info(f"Logging session started. Log file: {log_path}")
    
    def _get_caller_info(self) -> str:
        """Get information about the caller (module, line number)."""
        # Get the frame 2 levels up (caller of the logging method)
        frame = inspect.currentframe().f_back.f_back
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        module = os.path.basename(filename)
        return f"{module}:{lineno}"
    
    def debug(self, message: str, *args, **kwargs) -> None:
        """Log debug message."""
        if self.config["enable_logging"]:
            caller = self._get_caller_info()
            self.logger.debug(f"[{caller}] {message}", *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs) -> None:
        """Log info message."""
        if self.config["enable_logging"]:
            caller = self._get_caller_info()
            self.logger.info(f"[{caller}] {message}", *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs) -> None:
        """Log warning message."""
        if self.config["enable_logging"]:
            caller = self._get_caller_info()
            self.logger.warning(f"[{caller}] {message}", *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs) -> None:
        """Log error message."""
        if self.config["enable_logging"]:
            caller = self._get_caller_info()
            self.logger.error(f"[{caller}] {message}", *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs) -> None:
        """Log critical message."""
        if self.config["enable_logging"]:
            caller = self._get_caller_info()
            self.logger.critical(f"[{caller}] {message}", *args, **kwargs)
    
    def exception(self, message: str, *args, exc_info=True, **kwargs) -> None:
        """Log exception with traceback."""
        if self.config["enable_logging"]:
            caller = self._get_caller_info()
            self.logger.exception(f"[{caller}] {message}", *args, exc_info=exc_info, **kwargs)
    
    def log_model_summary(self, model) -> None:
        """Log model architecture and parameter count."""
        if not self.config["enable_logging"]:
            return
            
        try:
            # Count total parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            self.info(f"Model Summary: {model.__class__.__name__}")
            self.info(f"Total parameters: {total_params:,}")
            self.info(f"Trainable parameters: {trainable_params:,}")
            self.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
            
            # Try to log model architecture if possible
            if hasattr(model, "__str__"):
                self.debug(f"Model architecture:\n{model}")
        except Exception as e:
            self.warning(f"Could not log model summary: {str(e)}")
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log configuration parameters."""
        if not self.config["enable_logging"]:
            return
            
        self.info("Configuration:")
        for key, value in config.items():
            if isinstance(value, dict):
                self.info(f"  {key}:")
                for k, v in value.items():
                    self.info(f"    {k}: {v}")
            else:
                self.info(f"  {key}: {value}")
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], prefix: str = "") -> None:
        """Log evaluation metrics."""
        if not self.config["enable_logging"]:
            return
            
        if prefix:
            self.info(f"{prefix} Metrics:")
        else:
            self.info("Metrics:")
            
        for metric_name, value in metrics.items():
            self.info(f"  {metric_name}: {value}")

# Convenient function to get logger instance
def get_logger(config: Optional[Dict[str, Any]] = None) -> XCoreLogger:
    """Get the XCoreLogger instance."""
    return XCoreLogger.get_instance(config) 
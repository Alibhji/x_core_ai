"""
Integration module to connect the experiment tracker with the Core class.
"""

from .mlflow_tracker import create_experiment

def integrate_tracker_with_core(core_instance):
    """
    Integrate experiment tracker with an existing Core instance.
    
    Args:
        core_instance: The Core instance to integrate with
        
    Returns:
        MLFlowTracker instance
    """
    config = core_instance.config
    
    # Check if experiment tracking is enabled
    tracking_config = config.get("experiment_tracking", {})
    enable_tracking = tracking_config.get("enable", False)
    
    if not enable_tracking:
        return None
    
    # Get project name and version from config
    project_name = tracking_config.get("project_name", "x_core_ai")
    version = tracking_config.get("version", "0.1.0")
    
    # Create the experiment tracker
    tracker = create_experiment(project_name, version, enable_tracking)
    
    # Attach to core instance
    core_instance.tracker = tracker
    
    return tracker

def track_training_epoch(core_instance, epoch, train_metrics, val_metrics=None, step=None):
    """
    Track training metrics for an epoch.
    
    Args:
        core_instance: The Core instance
        epoch: Current epoch number
        train_metrics: Dictionary of training metrics
        val_metrics: Dictionary of validation metrics (optional)
        step: Step number (optional, defaults to epoch)
    """
    tracker = getattr(core_instance, 'tracker', None)
    if not tracker or not tracker.enabled or not tracker.active:
        return
    
    # Use epoch as step if not provided
    if step is None:
        step = epoch
    
    # Log training metrics
    if train_metrics:
        prefixed_train_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
        tracker.log_metrics(prefixed_train_metrics, step=step)
    
    # Log validation metrics
    if val_metrics:
        prefixed_val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}
        tracker.log_metrics(prefixed_val_metrics, step=step)

def track_model_params(core_instance, params=None):
    """
    Track model parameters.
    
    Args:
        core_instance: The Core instance
        params: Additional parameters to log (optional)
    """
    tracker = getattr(core_instance, 'tracker', None)
    if not tracker or not tracker.enabled or not tracker.active:
        return
    
    # Log model configuration
    if "model_kwargs" in core_instance.config:
        tracker.log_params(core_instance.config["model_kwargs"])
    
    # Log training configuration
    if "training_kwargs" in core_instance.config:
        tracker.log_params(core_instance.config["training_kwargs"])
    
    # Log additional parameters
    if params:
        tracker.log_params(params) 
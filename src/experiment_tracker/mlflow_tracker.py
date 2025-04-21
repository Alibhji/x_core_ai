"""
MLflow experiment tracker for tracking model training metrics.
"""

import os
import mlflow
import logging
import platform
from datetime import datetime

logger = logging.getLogger(__name__)

def create_experiment(project_name, version, enable_tracking=True):
    """
    Create an MLflow experiment with proper storage path.
    
    Args:
        project_name (str): Name of the project
        version (str): Version of the project
        enable_tracking (bool): Flag to enable/disable tracking
        
    Returns:
        MLFlowTracker instance
    """
    if not enable_tracking:
        return MLFlowTracker(None, None, False)
    
    # Create storage directory based on project name and version
    storage_dir = os.path.abspath(os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "storage",
        project_name,
        version
    ))
    
    # Create directory if it doesn't exist
    os.makedirs(storage_dir, exist_ok=True)
    
    # Set MLflow tracking URI - handle Windows differently with SQLite
    is_windows = platform.system() == "Windows"
    if is_windows:
        # On Windows, use SQLite database for tracking
        db_path = os.path.join(storage_dir, "mlflow.db")
        mlflow_uri = f"sqlite:///{db_path}"
        artifact_location = storage_dir
    else:
        # On Unix-like systems, use file:// protocol
        mlflow_uri = f"file://{storage_dir}"
        artifact_location = os.path.join(mlflow_uri, "artifacts")
    
    mlflow.set_tracking_uri(mlflow_uri)
    logger.info(f"MLflow tracking URI set to: {mlflow_uri}")
    
    # Create or get experiment
    experiment_name = f"{project_name}_{version}"
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            experiment_id = experiment.experiment_id
        else:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_location
            )
    except mlflow.exceptions.MlflowException as e:
        logger.error(f"Error setting up MLflow experiment: {e}")
        # Fallback to memory store if there's an issue
        mlflow.set_tracking_uri("")
        mlflow_uri = ""
        experiment_id = mlflow.create_experiment(experiment_name)
    
    logger.info(f"MLflow experiment '{experiment_name}' initialized with ID: {experiment_id}")
    return MLFlowTracker(experiment_id, experiment_name, mlflow_uri, True)


class MLFlowTracker:
    """
    MLflow experiment tracker for logging metrics, parameters, and artifacts.
    """
    
    def __init__(self, experiment_id, experiment_name, mlflow_uri=None, enabled=True):
        """
        Initialize the MLflow tracker.
        
        Args:
            experiment_id (str): MLflow experiment ID
            experiment_name (str): Name of the experiment
            mlflow_uri (str): MLflow tracking URI
            enabled (bool): Flag to enable/disable tracking
        """
        self.experiment_id = experiment_id
        self.experiment_name = experiment_name
        self.mlflow_uri = mlflow_uri
        self.enabled = enabled
        self.run_id = None
        self.active = False
    
    def start_run(self, run_name=None):
        """
        Start a new MLflow run.
        
        Args:
            run_name (str, optional): Name for the run
        """
        if not self.enabled:
            return self
        
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        active_run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name
        )
        self.run_id = active_run.info.run_id
        self.active = True
        logger.info(f"Started MLflow run: {run_name} (ID: {self.run_id})")
        return self
    
    def end_run(self):
        """End the current MLflow run."""
        if not self.enabled or not self.active:
            return
        
        mlflow.end_run()
        logger.info(f"Ended MLflow run (ID: {self.run_id})")
        self.active = False
    
    def log_param(self, key, value):
        """
        Log a parameter.
        
        Args:
            key (str): Parameter name
            value: Parameter value
        """
        if not self.enabled or not self.active:
            return
        
        mlflow.log_param(key, value)
    
    def log_params(self, params_dict):
        """
        Log multiple parameters.
        
        Args:
            params_dict (dict): Dictionary of parameters
        """
        if not self.enabled or not self.active:
            return
        
        try:
            mlflow.log_params(params_dict)
        except Exception as e:
            logger.error(f"Error logging parameters: {e}")
            # Try logging each parameter individually
            for key, value in params_dict.items():
                try:
                    mlflow.log_param(key, value)
                except:
                    logger.error(f"Could not log parameter {key}={value}")
    
    def log_metric(self, key, value, step=None):
        """
        Log a metric.
        
        Args:
            key (str): Metric name
            value (float): Metric value
            step (int, optional): Step value
        """
        if not self.enabled or not self.active:
            return
        
        mlflow.log_metric(key, value, step=step)
    
    def log_metrics(self, metrics_dict, step=None):
        """
        Log multiple metrics.
        
        Args:
            metrics_dict (dict): Dictionary of metrics
            step (int, optional): Step value
        """
        if not self.enabled or not self.active:
            return
        
        try:
            mlflow.log_metrics(metrics_dict, step=step)
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")
            # Try logging each metric individually in case one is causing problems
            for key, value in metrics_dict.items():
                try:
                    mlflow.log_metric(key, value, step=step)
                except:
                    logger.error(f"Could not log metric {key}={value}")
    
    def log_artifact(self, local_path):
        """
        Log an artifact (file).
        
        Args:
            local_path (str): Path to the file
        """
        if not self.enabled or not self.active:
            return
        
        try:
            mlflow.log_artifact(local_path)
        except Exception as e:
            logger.error(f"Error logging artifact {local_path}: {e}")
    
    def log_dict(self, dictionary, artifact_file):
        """
        Log a dictionary as a JSON file.
        
        Args:
            dictionary (dict): Dictionary to log
            artifact_file (str): Name of the JSON file
        """
        if not self.enabled or not self.active:
            return
        
        try:
            mlflow.log_dict(dictionary, artifact_file)
        except Exception as e:
            logger.error(f"Error logging dictionary as {artifact_file}: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        if not self.active:
            self.start_run()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.end_run() 
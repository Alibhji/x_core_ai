import torch
import numpy as np
from sklearn import metrics
from .forecast import Forecast

class Validation(Forecast):
    def __init__(self, config, package_name='x_core_ai.src'):
        """Validation class for model evaluation"""
        super().__init__(config, package_name)
        self.metrics_functions = self._get_metrics_functions()
        
    def _get_metrics_functions(self):
        """Get metric functions based on config"""
        metric_dict = {}
        
        # Get metrics from config or use defaults
        metric_names = self.config.get('metrics', ['mse', 'mae'])
        
        for metric in metric_names:
            if metric.lower() == 'mse':
                metric_dict['mse'] = metrics.mean_squared_error
            elif metric.lower() == 'mae':
                metric_dict['mae'] = metrics.mean_absolute_error
            elif metric.lower() == 'rmse':
                metric_dict['rmse'] = lambda y_true, y_pred: np.sqrt(metrics.mean_squared_error(y_true, y_pred))
            elif metric.lower() == 'r2':
                metric_dict['r2'] = metrics.r2_score
            elif metric.lower() == 'accuracy':
                metric_dict['accuracy'] = metrics.accuracy_score
            elif metric.lower() == 'f1':
                metric_dict['f1'] = metrics.f1_score
            elif metric.lower() == 'precision':
                metric_dict['precision'] = metrics.precision_score
            elif metric.lower() == 'recall':
                metric_dict['recall'] = metrics.recall_score
                
        return metric_dict
    
    def evaluate(self, dataloader=None):
        """
        Evaluate model on a dataloader
        Args:
            dataloader: DataLoader with validation data (uses default val dataloader if None)
        Returns:
            Dictionary of evaluation metrics
        """
        # Get dataloader if not provided
        if dataloader is None:
            dataloader = self.get_val_dataloader()
            
        if dataloader is None:
            raise ValueError("No validation dataloader available")
            
        all_predictions = []
        all_targets = []
        
        # Get predictions and targets
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                # Extract targets from batch
                if isinstance(batch, dict):
                    target_name = self.config.get('target_name', 'cost_target')
                    if target_name in batch:
                        targets = batch[target_name]
                        # For GCAN models, ensure we use photo_feat as input
                        if 'photo_feat' in batch and self.config.get('model_name') == 'gated_cross_attention':
                            inputs = batch['photo_feat']
                        else:
                            inputs = {k: v for k, v in batch.items() if k != target_name}
                    else:
                        # Try common fallbacks
                        targets = batch.get('target', None)
                        inputs = batch.get('photo_feat', batch)
                        if targets is None:
                            # Try first non-photo_feat key as target
                            for k, v in batch.items():
                                if k != 'photo_feat':
                                    targets = v
                                    break
                else:
                    # Handle tuple of (inputs, targets)
                    inputs, targets = batch
                
                # Move inputs to device
                if isinstance(inputs, dict):
                    inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                             for k, v in inputs.items()}
                elif isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(self.device)
                
                # Handle targets
                if targets is None:
                    print("Warning: No targets found in batch, skipping")
                    continue
                    
                if isinstance(targets, torch.Tensor):
                    targets = targets.to(self.device)
                
                # Get predictions
                outputs = self.model(inputs)
                predictions = self.process_outputs(outputs)
                
                # Convert targets to numpy if needed
                if isinstance(targets, torch.Tensor):
                    targets = targets.cpu().numpy()
                
                # Ensure predictions and targets have consistent shapes
                if len(predictions.shape) == 1:
                    predictions = predictions.reshape(-1, 1)
                if len(targets.shape) == 1:
                    targets = targets.reshape(-1, 1)
                
                all_predictions.append(predictions)
                all_targets.append(targets)
        
        # Combine predictions and targets
        if all(isinstance(p, np.ndarray) for p in all_predictions):
            try:
                all_predictions = np.vstack(all_predictions)
            except ValueError as e:
                print(f"Warning: Unable to vstack predictions due to inconsistent shapes: {e}")
                # Try concatenating along axis 0 without requiring same shape along other dimensions
                all_predictions = np.concatenate([p.reshape(p.shape[0], -1) for p in all_predictions], axis=0)
                
        if all(isinstance(t, np.ndarray) for t in all_targets):
            try:
                all_targets = np.vstack(all_targets)
            except ValueError as e:
                print(f"Warning: Unable to vstack targets due to inconsistent shapes: {e}")
                # Try concatenating along axis 0 without requiring same shape along other dimensions
                all_targets = np.concatenate([t.reshape(t.shape[0], -1) for t in all_targets], axis=0)
            
        # Calculate metrics
        metrics_results = self.calculate_metrics(all_targets, all_predictions)
        return metrics_results
    
    def calculate_metrics(self, targets, predictions):
        """
        Calculate evaluation metrics
        Args:
            targets: Ground truth values
            predictions: Model predictions
        Returns:
            Dictionary of metric values
        """
        results = {}
        
        for metric_name, metric_fn in self.metrics_functions.items():
            try:
                # Handle different metric requirements
                if metric_name in ['accuracy', 'f1', 'precision', 'recall']:
                    # Classification metrics might need class predictions
                    if predictions.ndim > 1 and predictions.shape[1] > 1:
                        # Convert probabilities to class predictions
                        class_predictions = np.argmax(predictions, axis=1)
                    else:
                        # Binary classification with threshold 0.5
                        class_predictions = (predictions > 0.5).astype(int)
                    
                    if metric_name == 'f1' or metric_name == 'precision' or metric_name == 'recall':
                        # These need average parameter
                        result = metric_fn(targets, class_predictions, average='macro')
                    else:
                        result = metric_fn(targets, class_predictions)
                else:
                    # Regression metrics
                    # Ensure consistent shapes for regression metrics
                    if predictions.ndim > 1 and targets.ndim == 1:
                        predictions = predictions.squeeze()
                    elif targets.ndim > 1 and predictions.ndim == 1:
                        targets = targets.squeeze()
                    
                    # If shapes still don't match, flatten both
                    if predictions.shape != targets.shape:
                        predictions = predictions.flatten()
                        targets = targets.flatten()
                        
                    result = metric_fn(targets, predictions)
                    
                results[metric_name] = result
            except Exception as e:
                print(f"Error calculating {metric_name}: {e}")
                results[metric_name] = float('nan')
                
        return results
        
    def get_dataloader(self, dataset=None):
        """Get validation dataloader"""
        if dataset is None:
            # Load validation dataset from config
            if self.config.get('dataset_name') and self.config.get('dataset_kwargs'):
                df_val = None
                if self.config.get('data_name'):
                    dataframe = self.get_dataframe(self.config['data_name'], 
                                                  **self.config.get('dataframe_kwargs', {}))
                    df_val = dataframe.df_val
                    
                dataset = self.get_dataset(self.config['dataset_name'], 
                                          df=df_val, 
                                          train=False, 
                                          **self.config.get('dataset_kwargs', {}))
                
        if dataset is None:
            raise ValueError("No validation dataset provided")
            
        # Create dataloader
        batch_size = self.config.get('dataloader_kwargs_val', {}).get('batch_size', 32)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            **{k: v for k, v in self.config.get('dataloader_kwargs_val', {}).items() if k != 'batch_size'}
        )
        
        return dataloader 
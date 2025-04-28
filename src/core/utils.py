import torch
import numpy as np
import sklearn.metrics as metrics

def custom_collate_fn(batch):
        collated_batch = {}

        # Get all keys in the dataset
        keys = batch[0].keys()

        for key in keys:
            values = [sample[key] for sample in batch]

            if isinstance(values[0], np.ndarray):
                try:
                    collated_batch[key]= torch.as_tensor(np.stack(values))
                except:
                    print('issue in photofeatures')
            elif isinstance(values[0], list):
                collated_batch[key] = torch.stack([torch.tensor(v) for v in values])

            elif isinstance(values[0], int) or isinstance(values[0], float):  # Handle numerical values
                collated_batch[key] = torch.tensor(values)
                
            elif isinstance(values[0], torch.Tensor) or isinstance(values[0], np.ndarray):
                values = [torch.tensor(v) if isinstance(v, np.ndarray) else v for v in values]
                collated_batch[key] = torch.stack(values)  # Stack tensors

            elif isinstance(values[0], str) or isinstance(values[0], list):  # Handle strings/lists
                collated_batch[key] = values  # Keep them as lists
                
            elif isinstance(values[0], dict):
                collated_batch[key] = custom_collate_fn(values)
            else:
                raise TypeError(f"Unsupported data type for key '{key}': {type(values[0])}")

        return collated_batch

def get_metrics_functions(metric_names = ['mse', 'mae']):
    """Get metric functions based on config
    Args:
        metrics: List of metrics to get functions for (default is ['mse', 'mae']) 
        possible metrics: ['mse', 'mae', 'rmse', 'r2', 'accuracy', 'f1', 'precision', 'recall']
    Returns:
        Dictionary of metric functions
    """
    metric_dict = {}

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